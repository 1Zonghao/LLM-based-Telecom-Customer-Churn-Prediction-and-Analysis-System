"""可复用的单任务 episode runner。

把“建图 → 跑 → 自动确认断点 → 收集状态”的循环抽出来，
供 competition_eval、experiment_runner 和测试共用。
"""
from __future__ import annotations

import time
import traceback
from typing import Callable

from .graph import build_agent, make_thread_config
from .state import init_state
from sandbox.compressor import OutputCompressor
from sandbox.kernel_manager import KernelManager


def run_episode(
    llm,
    task: str,
    data_path: str,
    domain: str = "generic",
    artifact_contract: dict | None = None,
    max_iter: int = 20,
    max_reasoning_calls: int | None = None,
    skill_context: str = "",
    kernel_factory: Callable[[], KernelManager] | None = None,
) -> dict:
    """执行一次完整分析 episode，自动确认静态断点。

    Args:
        llm: 具备 chat() 的客户端（LLMClient / AnthropicLLMClient / MockLLMClient）。
        task: 任务描述。
        data_path: 主数据集路径。
        domain: 领域插件名。
        artifact_contract: 交付物契约；为空时 verifier 自动通过。
        max_iter: 动作预算（传给 nodes.MAX_ITERATIONS）。
        max_reasoning_calls: 硬上限；默认 max_iter + 4（给验证重试留余量）。
        skill_context: 注入的过往经验文本。
        kernel_factory: 自定义 kernel 工厂（测试可传 FakeKernel）。

    Returns:
        结构化结果字典。
    """
    from agent import nodes

    nodes.MAX_ITERATIONS = max_iter
    hard_cap = max_reasoning_calls or (max_iter + 4)

    factory = kernel_factory or KernelManager
    km = factory()
    if hasattr(km, "start"):
        km.start()

    compressor = OutputCompressor()
    app = build_agent(llm, km, compressor)
    config = make_thread_config()
    state = init_state(
        task,
        data_path,
        domain=domain,
        artifact_contract=artifact_contract,
        skill_context=skill_context,
    )

    transcript: list[dict] = []
    reasoning_calls = 0
    iteration = 0
    started = time.time()
    finished_normally = False
    error = ""
    hit_cap = False

    try:
        inputs = state
        while True:
            for event in app.stream(inputs, config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    if node_name == "__interrupt__":
                        continue

                    if node_name == "reasoning":
                        reasoning_calls += 1
                    if isinstance(node_output, dict):
                        iteration = max(
                            iteration,
                            int(node_output.get("iteration", 0) or 0),
                        )

                    transcript.append({
                        "node": node_name,
                        "output": _truncate_output(node_output),
                        "t": round(time.time() - started, 1),
                    })

                    if reasoning_calls >= hard_cap:
                        hit_cap = True
                        break
                if hit_cap:
                    break

            if hit_cap:
                break

            snapshot = app.get_state(config)
            next_nodes = tuple(getattr(snapshot, "next", ()) or ())
            if next_nodes and any(str(n).startswith("checkpoint") for n in next_nodes):
                # 静态 interrupt_before 用 None 恢复
                inputs = None
                continue

            finished_normally = True
            break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        duration = round(time.time() - started, 2)
        try:
            final_values = app.get_state(config).values
        except Exception:
            final_values = state
        if hasattr(km, "shutdown"):
            try:
                km.shutdown()
            except Exception:
                pass

    if not isinstance(final_values, dict):
        final_values = state

    return {
        "duration_sec": duration,
        "reasoning_calls": reasoning_calls,
        "iterations": iteration,
        "finished_normally": finished_normally,
        "hit_cap": hit_cap,
        "error": error,
        "token_usage": final_values.get("token_usage", {}),
        "started_at": final_values.get("started_at", ""),
        "domain": final_values.get("domain", domain),
        "phase": final_values.get("phase", ""),
        "data_profile": final_values.get("data_profile", {}),
        "final_report": final_values.get("final_report", ""),
        "code_history": final_values.get("code_history", []),
        "verification_history": final_values.get("verification_history", []),
        "verification_result": final_values.get("verification_result", {}),
        "verification_passed": final_values.get("verification_passed", None),
        "verification_retries": final_values.get("verification_retries", 0),
        "skill_context": skill_context,
        "transcript": transcript,
    }


def _truncate_output(output, limit: int = 1200):
    if not isinstance(output, dict):
        return str(output)[:limit]
    truncated = {}
    for key, value in output.items():
        if isinstance(value, (int, float, bool, type(None))):
            truncated[key] = value
        elif isinstance(value, list):
            truncated[key] = f"[list len={len(value)}] {str(value)[:limit]}"
        else:
            truncated[key] = str(value)[:limit]
    return truncated
