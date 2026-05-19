"""Phase 3: Gradio 前端 — 流式对话、图表展示、断点确认、进度观测。"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure src/ is on path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
from langgraph.types import Command

from agent.llm_client import LLMClient
from agent.state import init_state
from agent.graph import build_agent, make_thread_config
from sandbox.kernel_manager import KernelManager
from sandbox.compressor import OutputCompressor

DEFAULT_TASK = "分析客户流失因素，构建预测模型，关注召回率优化"


def _format_chatbot(messages: list[dict]) -> list[dict]:
    """将内部 messages 转为 Gradio Chatbot 格式 (role, content)。"""
    rows = []
    for m in messages:
        role = m.get("role", "user")
        if role == "assistant":
            rows.append({"role": "assistant", "content": m["content"]})
        elif role == "user":
            rows.append({"role": "user", "content": m["content"]})
        elif role == "system":
            rows.append({"role": "assistant", "content": f"*[System]* {m['content']}"})
    return rows


def _get_pending_checkpoint(app, config) -> dict | None:
    """通过 app.get_state 检查是否因 interrupt_before 处于挂起状态。

    若 next_action 指向 checkpoint 节点且 pending_proposal 非空，返回中断信息。
    """
    try:
        snapshot = app.get_state(config)
        values = snapshot.values if hasattr(snapshot, "values") else snapshot
    except Exception:
        return None

    next_action = values.get("next_action", "finish")
    if next_action in ("checkpoint_fe", "checkpoint_model"):
        proposal = values.get("pending_proposal", "")
        cp_type = "feature_engineering" if next_action == "checkpoint_fe" else "model_training"
        return {
            "type": cp_type,
            "proposal": proposal,
            "data_profile": values.get("data_profile", {}),
        }
    return None


def _format_progress(values: dict) -> str:
    """将当前状态格式化为进度 Markdown。"""
    phase = values.get("phase", "等待开始")
    iteration = values.get("iteration", 0)
    profile = values.get("data_profile", {})

    phase_map = {
        "eda": "🔍 数据探索 (EDA)",
        "feature_engineering": "🔧 特征工程",
        "model_training": "🤖 模型训练",
        "reporting": "📊 报告生成",
    }
    phase_label = phase_map.get(phase, phase)

    lines = [
        f"**当前阶段**: {phase_label}",
        f"**迭代轮次**: {iteration} / 10",
    ]

    if profile:
        metric_lines = []
        key_map = {
            "n_rows": "样本数",
            "n_cols": "特征数",
            "best_f1": "最佳 F1",
            "best_auc": "最佳 AUC",
            "best_recall": "最佳 Recall",
        }
        for k, v in profile.items():
            label = key_map.get(k, k)
            metric_lines.append(f"- {label}: {v}")
        if metric_lines:
            lines.append("**关键指标**:\n" + "\n".join(metric_lines))

    return "\n\n".join(lines)


def _export_code_history(code_history: list[str], task: str = "") -> tuple[str, str]:
    """将代码执行历史导出为 .py 脚本与 .ipynb Notebook。

    返回 (py_path, ipynb_path)。
    """
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    py_path = output_dir / f"analysis_{timestamp}.py"
    py_lines = [
        '"""Auto-generated data analysis script.',
        f'Task: {task}',
        f'Generated: {datetime.now().isoformat()}',
        '"""',
        "",
    ]
    for i, code in enumerate(code_history, 1):
        py_lines.append(f"# ---- Step {i} ----")
        py_lines.append(code)
        py_lines.append("")
    py_path.write_text("\n".join(py_lines), encoding="utf-8")

    ipynb_path = output_dir / f"analysis_{timestamp}.ipynb"
    notebook = {
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    if task:
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# 数据分析任务\n\n{task}"],
        })
    for code in code_history:
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [code],
        })
    ipynb_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return str(py_path), str(ipynb_path)


class AgentSession:
    """封装 agent 运行时的状态，跨 Gradio 事件持久化。"""

    def __init__(self):
        self.app = None
        self.config = None
        self.km: KernelManager | None = None
        self.state: dict | None = None
        self.interrupt_info: dict | None = None
        # 若设为 True，后续所有 checkpoint 将自动确认通过
        self.auto_confirm: bool = False

    def reset(self):
        if self.km:
            try:
                self.km.shutdown()
            except Exception:
                pass
        self.km = KernelManager()
        self.km.start()
        self.app = None
        self.config = None
        self.state = None
        self.interrupt_info = None
        self.auto_confirm = False


def run_agent_stream(
    dataset_path: str,
    task: str,
    session_state: dict,
):
    """流式运行 agent 的生成器函数。

    从初始状态开始，逐步执行 agent 节点，yield UI 更新。
    遇到 interrupt 时停止，将状态保存到 session_state 供后续 resume。
    """
    session: AgentSession = session_state.get("_session")
    if session is None:
        session = AgentSession()
        session_state["_session"] = session

    session.reset()

    llm = LLMClient()
    compressor = OutputCompressor()
    session.app = build_agent(llm, session.km, compressor)
    session.config = make_thread_config()
    session.state = init_state(task, dataset_path)

    # 维护一份本地状态副本，用于实时渲染进度（避免反复调用 get_state）
    latest_values = dict(session.state)
    latest_values.setdefault("messages", [])
    latest_values.setdefault("code_history", [])

    chatbot = []
    status = "正在启动..."
    progress_md = _format_progress(latest_values)

    yield (
        chatbot,                    # 0 chatbot
        status,                     # 1 status
        gr.update(visible=False),   # 2 checkpoint_panel
        "",                         # 3 checkpoint_proposal
        gr.update(value=""),        # 4 feedback_input
        progress_md,                # 5 progress_display
        session_state,              # 6 session_state
    )

    try:
        for event in session.app.stream(session.state, session.config, stream_mode="updates"):
            for node_name, node_output in event.items():
                # LangGraph 1.x 中 interrupt_before 会产生 __interrupt__ 元数据事件
                if node_name == "__interrupt__":
                    continue
                # 增量更新本地状态
                for key, value in node_output.items():
                    if key in ("messages", "code_history"):
                        latest_values.setdefault(key, [])
                        if isinstance(value, list):
                            latest_values[key].extend(value)
                        else:
                            latest_values[key].append(value)
                    else:
                        latest_values[key] = value

                msgs = node_output.get("messages", [])
                if msgs:
                    chatbot = _format_chatbot(msgs)
                    status = f"节点: {node_name}"

                progress_md = _format_progress(latest_values)
                yield (
                    chatbot,
                    status,
                    gr.update(visible=False),
                    "",
                    gr.update(value=""),
                    progress_md,
                    session_state,
                )
    except Exception as e:
        status = f"错误: {str(e)}"
        progress_md = _format_progress(latest_values)
        yield (
            chatbot,
            status,
            gr.update(visible=False),
            "",
            gr.update(value=""),
            progress_md,
            session_state,
        )
        return

    # Stream 结束后检查是否因 interrupt_before 处于挂起状态
    interrupt_info = _get_pending_checkpoint(session.app, session.config)
    if interrupt_info:
        session.interrupt_info = interrupt_info
        proposal = interrupt_info.get("proposal", "")
        checkpoint_type = interrupt_info.get("type", "checkpoint")

        status = f"已暂停 — 等待确认 {checkpoint_type} 方案"
        progress_md = _format_progress(latest_values)
        yield (
            chatbot,
            status,
            gr.update(visible=True),
            proposal,
            gr.update(value=""),
            progress_md,
            session_state,
        )
        return  # 停止流，等待用户操作

    # Agent finished — 导出代码历史并展示最终报告
    snapshot = session.app.get_state(session.config)
    values = snapshot.values if hasattr(snapshot, "values") else snapshot
    final_report = values.get("final_report", "分析完成。")
    code_history = values.get("code_history", [])
    task_name = values.get("task", "")
    py_path, ipynb_path = _export_code_history(code_history, task_name)
    export_text = f"\n\n---\n\n**代码执行记录已导出**\n- Python: `{py_path}`\n- Notebook: `{ipynb_path}`"
    chatbot.append({"role": "assistant", "content": f"### 最终报告\n\n{final_report}{export_text}"})
    status = "分析完成"
    progress_md = _format_progress(latest_values)

    yield (
        chatbot,
        status,
        gr.update(visible=False),
        "",
        gr.update(value=""),
        progress_md,
        session_state,
    )


def resume_agent_stream(session_state: dict, feedback: str = ""):
    """从中断点恢复的生成器函数（支持 auto_confirm 自动通过）。"""
    session: AgentSession = session_state.get("_session")
    if session is None or session.app is None or session.interrupt_info is None:
        yield (
            [],
            "无中断状态可恢复，请重新开始",
            gr.update(visible=False),
            "",
            gr.update(value=""),
            "等待开始",
            session_state,
        )
        return

    # 如有用户反馈，先写入图状态（供 checkpoint 节点读取并注入 messages）
    if feedback.strip():
        try:
            session.app.update_state(session.config, {"user_feedback": feedback})
        except Exception:
            pass

    session.interrupt_info = None
    chatbot = []
    status = "正在恢复执行..."
    latest_values = dict(session.state) if session.state else {}
    latest_values.setdefault("messages", [])
    latest_values.setdefault("code_history", [])
    progress_md = _format_progress(latest_values)

    yield (
        chatbot,
        status,
        gr.update(visible=False),
        "",
        gr.update(value=""),
        progress_md,
        session_state,
    )

    while True:
        try:
            for event in session.app.stream(Command(resume="confirmed"), session.config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    # LangGraph 1.x 中 interrupt_before 会产生 __interrupt__ 元数据事件
                    if node_name == "__interrupt__":
                        continue
                    for key, value in node_output.items():
                        if key in ("messages", "code_history"):
                            latest_values.setdefault(key, [])
                            if isinstance(value, list):
                                latest_values[key].extend(value)
                            else:
                                latest_values[key].append(value)
                        else:
                            latest_values[key] = value

                    msgs = node_output.get("messages", [])
                    if msgs:
                        chatbot = _format_chatbot(msgs)
                        status = f"节点: {node_name}"

                    progress_md = _format_progress(latest_values)
                    yield (
                        chatbot,
                        status,
                        gr.update(visible=False),
                        "",
                        gr.update(value=""),
                        progress_md,
                        session_state,
                    )
        except Exception as e:
            status = f"错误: {str(e)}"
            progress_md = _format_progress(latest_values)
            yield (
                chatbot,
                status,
                gr.update(visible=False),
                "",
                gr.update(value=""),
                progress_md,
                session_state,
            )
            return

        # Stream 结束后检查是否因 interrupt_before 再次挂起
        interrupt_info = _get_pending_checkpoint(session.app, session.config)
        if interrupt_info:
            if session.auto_confirm:
                # 自动确认并继续下一循环
                session.interrupt_info = interrupt_info
                status = f"自动确认 — {interrupt_info.get('type', 'checkpoint')}"
                progress_md = _format_progress(latest_values)
                yield (
                    chatbot,
                    status,
                    gr.update(visible=False),
                    "",
                    gr.update(value=""),
                    progress_md,
                    session_state,
                )
                continue
            else:
                # 显示确认面板，等待用户操作
                session.interrupt_info = interrupt_info
                proposal = interrupt_info.get("proposal", "")
                checkpoint_type = interrupt_info.get("type", "checkpoint")

                status = f"已暂停 — 等待确认 {checkpoint_type} 方案"
                progress_md = _format_progress(latest_values)
                yield (
                    chatbot,
                    status,
                    gr.update(visible=True),
                    proposal,
                    gr.update(value=""),
                    progress_md,
                    session_state,
                )
                return

        # 无中断，正常结束
        break

    snapshot = session.app.get_state(session.config)
    values = snapshot.values if hasattr(snapshot, "values") else snapshot
    final_report = values.get("final_report", "分析完成。")
    code_history = values.get("code_history", [])
    task_name = values.get("task", "")
    py_path, ipynb_path = _export_code_history(code_history, task_name)
    export_text = f"\n\n---\n\n**代码执行记录已导出**\n- Python: `{py_path}`\n- Notebook: `{ipynb_path}`"
    if final_report and final_report != "分析完成。":
        chatbot.append({"role": "assistant", "content": f"### 最终报告\n\n{final_report}{export_text}"})
    else:
        chatbot.append({"role": "assistant", "content": f"### 分析完成{export_text}"})
    status = "分析完成"
    progress_md = _format_progress(latest_values)

    yield (
        chatbot,
        status,
        gr.update(visible=False),
        "",
        gr.update(value=""),
        progress_md,
        session_state,
    )


def create_ui() -> gr.Blocks:
    """创建 Gradio 前端界面。"""

    with gr.Blocks(title="自主数据科学智能体") as app:
        gr.Markdown("# 自主数据科学智能体")
        gr.Markdown("基于 LLM 的电信客户流失预测分析系统。上传数据集，描述分析目标，智能体将自主完成数据探索、特征工程、模型训练和报告生成。")
        gr.Markdown("---")

        session_state = gr.State({})

        with gr.Row():
            # 左侧面板
            with gr.Column(scale=1, min_width=300):
                dataset_input = gr.File(
                    label="上传数据集 (CSV)",
                    file_types=[".csv"],
                    type="filepath",
                )
                task_input = gr.Textbox(
                    label="分析任务",
                    value=DEFAULT_TASK,
                    lines=3,
                    placeholder="描述你想让智能体完成的分析任务...",
                )
                with gr.Row():
                    start_btn = gr.Button("开始分析", variant="primary", size="lg")
                    stop_btn = gr.Button("停止", variant="stop")

                status_display = gr.Textbox(
                    label="状态",
                    value="就绪 — 请上传数据集并点击「开始分析」",
                    interactive=False,
                )

                # 新增：进度观测面板
                progress_display = gr.Markdown(
                    label="运行进度",
                    value="**当前阶段**: 等待开始\n\n**迭代轮次**: 0 / 10",
                )

                gr.Markdown("---")
                gr.Markdown("### 断点说明")
                gr.Markdown("""
                智能体会在两个关键节点暂停并请求确认：
                1. **特征工程前** — 确认数据清洗和编码方案
                2. **模型训练前** — 确认算法和重采样策略
                """)

            # 右侧主区域
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="对话历史",
                    height=500,
                )

                # 断点确认面板（默认隐藏）
                with gr.Group(visible=False) as checkpoint_panel:
                    gr.Markdown("### 断点 — 方案确认")
                    checkpoint_proposal = gr.Markdown(
                        "智能体将在执行前等待你的确认。",
                        label="提案内容",
                    )
                    # 新增：用户反馈输入框
                    feedback_input = gr.Textbox(
                        label="修改意见 / 补充提示（可选）",
                        lines=3,
                        placeholder="如：请补充交叉验证、改用 SMOTE 而非 class_weight、对 TotalCharges 做对数变换...",
                    )
                    with gr.Row():
                        confirm_btn = gr.Button("确认执行", variant="primary")
                        skip_btn = gr.Button("跳过后续确认 / 自动完成")

        # 事件绑定
        outputs = [
            chatbot,              # 0
            status_display,       # 1
            checkpoint_panel,     # 2
            checkpoint_proposal,  # 3
            feedback_input,       # 4
            progress_display,     # 5
            session_state,        # 6
        ]

        start_event = start_btn.click(
            fn=run_agent_stream,
            inputs=[dataset_input, task_input, session_state],
            outputs=outputs,
        )

        confirm_event = confirm_btn.click(
            fn=resume_agent_stream,
            inputs=[session_state, feedback_input],
            outputs=outputs,
        )

        def skip_and_resume_stream(session_state: dict):
            """设置 auto_confirm 并恢复执行。"""
            session: AgentSession = session_state.get("_session")
            if session:
                session.auto_confirm = True
            yield from resume_agent_stream(session_state, feedback="")

        # 跳过按钮：设置 auto-complete 信号后恢复
        skip_btn.click(
            fn=skip_and_resume_stream,
            inputs=[session_state],
            outputs=outputs,
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
