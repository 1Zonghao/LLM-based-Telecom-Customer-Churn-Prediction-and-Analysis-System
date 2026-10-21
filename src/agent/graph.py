import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from .state import AgentState, init_state
from .llm_client import LLMClient
from .nodes import (
    make_reasoning_node,
    make_action_node,
    make_checkpoint_fe_node,
    make_checkpoint_model_node,
    make_pre_submit_node,
    make_verifier_node,
    pre_submit_router,
    router,
    verifier_router,
)
from sandbox.kernel_manager import KernelManager
from sandbox.compressor import OutputCompressor


def build_agent(
    llm: LLMClient,
    km: KernelManager,
    compressor: OutputCompressor | None = None,
) -> StateGraph:
    """构建 LangGraph 智能体（含断点节点）。

    Args:
        llm: LLM 客户端
        km: 已启动的 KernelManager
        compressor: 输出压缩器

    Returns:
        编译好的 CompiledStateGraph（含 InMemorySaver checkpointer）
    """
    if compressor is None:
        compressor = OutputCompressor()

    reasoning = make_reasoning_node(llm)
    action = make_action_node(km, compressor)
    checkpoint_fe = make_checkpoint_fe_node()
    checkpoint_model = make_checkpoint_model_node()
    pre_submit = make_pre_submit_node(km)
    verifier = make_verifier_node()

    graph = StateGraph(AgentState)
    graph.add_node("reasoning", reasoning)
    graph.add_node("action", action)
    graph.add_node("checkpoint_fe", checkpoint_fe)
    graph.add_node("checkpoint_model", checkpoint_model)
    graph.add_node("pre_submit", pre_submit)
    graph.add_node("verifier", verifier)

    graph.set_entry_point("reasoning")

    # 智能体声称完成时，先跑强制自检，再进入交付验证
    graph.add_conditional_edges("reasoning", router, {
        "execute": "action",
        "checkpoint_fe": "checkpoint_fe",
        "checkpoint_model": "checkpoint_model",
        "finish": "pre_submit",
    })

    # 强制自检：失败回推理修复，通过则进入 verifier
    graph.add_conditional_edges("pre_submit", pre_submit_router, {
        "repair": "reasoning",
        "verify": "verifier",
    })

    # 验证器：通过则结束，失败则回到推理修复
    graph.add_conditional_edges("verifier", verifier_router, {
        "repair": "reasoning",
        "finish": END,
    })

    # 执行后回到推理，断点确认后也回到推理
    graph.add_edge("action", "reasoning")
    graph.add_edge("checkpoint_fe", "reasoning")
    graph.add_edge("checkpoint_model", "reasoning")

    # 使用 interrupt_before 在关键节点前自动挂起，实现 Human-in-the-Loop
    checkpointer = InMemorySaver()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["checkpoint_fe", "checkpoint_model"],
    )


def make_thread_config() -> dict:
    """为每次对话生成唯一的 thread 配置。"""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def run_agent(
    task: str,
    data_path: str,
    llm: LLMClient,
    km: KernelManager,
    compressor: OutputCompressor | None = None,
    thread_config: dict | None = None,
    domain: str = "generic",
    artifact_contract: dict | None = None,
):
    """运行智能体并返回最终状态（同步，无中断处理）。"""
    app = build_agent(llm, km, compressor)
    state = init_state(
        task,
        data_path,
        domain=domain,
        artifact_contract=artifact_contract,
    )
    if thread_config is None:
        thread_config = make_thread_config()
    return app.invoke(state, thread_config)


