"""Phase 3: Gradio 前端 — 流式对话、图表展示、断点确认、进度观测。
v2 — 会话持久化、侧边栏历史、消息层次化渲染。
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
from langgraph.types import Command

from agent.llm_client import LLMClient
from agent.state import init_state
from agent.graph import build_agent, make_thread_config
from sandbox.kernel_manager import KernelManager
from sandbox.compressor import OutputCompressor
from ui.session_store import SessionStore
from ui.message_formatter import format_chatbot, build_final_report_message

DEFAULT_TASK = "分析客户流失因素，构建预测模型，关注召回率优化"


# ============================================================================
# Session data helpers
# ============================================================================

def _refresh_session_list() -> list[list]:
    """Return [[session_id, title, time], ...] — session_id hidden via CSS."""
    store = SessionStore()
    return [
        [s.session_id, s.title, s.updated_at[:16] if s.updated_at else ""]
        for s in store.list_sessions()
    ]


def _save_session_from_state(
    session_state: dict,
    chatbot_msgs: list[dict],
    values: dict,
) -> None:
    session_id = session_state.get("active_session_id")
    if not session_id:
        return
    store = SessionStore()
    rec = store.get_session(session_id)
    if rec is None:
        return
    rec.messages = chatbot_msgs
    rec.task = values.get("task", rec.task)
    rec.code_history = values.get("code_history", [])
    rec.final_report = values.get("final_report", "")
    rec.metrics = {
        k: values.get("data_profile", {}).get(k)
        for k in ("best_f1", "best_auc", "best_recall")
    }
    rec.completed = values.get("next_action") == "finish"
    store.save_session(rec)


# ============================================================================
# Formatting (thin wrappers over message_formatter)
# ============================================================================

def _format_chatbot(messages: list[dict]) -> list[dict]:
    return format_chatbot(messages)


def _format_progress(values: dict, step_times: list | None = None) -> str:
    """全面升级为美观的高级 HTML 数据卡片与动态状态进度条"""
    phase = values.get("phase", "等待开始")
    iteration = values.get("iteration", 0)
    profile = values.get("data_profile", {})

    phase_map = {
        "eda": "🔍 核心数据探索 (EDA)",
        "feature_engineering": "🔧 极简特征工程",
        "model_training": "🤖 协同模型训练",
        "reporting": "📊 演化报告生成",
        "已完成": "✅ 流程审计完成",
        "进行中": "⚡ 引擎持续推演中"
    }
    phase_label = phase_map.get(phase, phase)
    progress_percentage = min(iteration * 10, 100)

    html = f"""
    <div class="crypto-progress-card">
        <div class="progress-header">
            <span class="progress-title">CURRENT OPERATING PHASE</span>
            <span class="progress-subtitle">迭代轮次: {iteration} / 10</span>
        </div>
        <div class="progress-label-main">{phase_label}</div>
        <div class="progress-bar-container">
            <div class="progress-bar-fill" style="width: {progress_percentage}%;"></div>
        </div>
    """

    if profile:
        html += """
        <div class="metrics-divider"></div>
        <div class="metrics-grid-title">SANDBOX DATA SEGMENT PROFILE</div>
        <div class="metrics-grid">
        """
        key_map = {
            "n_rows": "样本基数",
            "n_cols": "特征维度",
            "best_f1": "最优 F1",
            "best_auc": "最优 AUC",
            "best_recall": "最优召回率",
        }
        for k, v in profile.items():
            label = key_map.get(k, k)
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            html += f"""
            <div class="metric-item">
                <span class="metric-label">{label}</span>
                <span class="metric-value">{val_str}</span>
            </div>
            """
        html += "</div>"

    # ── Token / Timing 资源面板 ──
    token_usage = values.get("token_usage", {})
    started_at = values.get("started_at", "")
    elapsed_str = ""
    if started_at:
        try:
            start_dt = datetime.fromisoformat(started_at)
            elapsed = (datetime.now(timezone.utc) - start_dt).total_seconds()
            elapsed_str = _format_duration(elapsed)
        except Exception:
            pass

    show_resource = token_usage or elapsed_str or step_times
    if show_resource:
        html += '<div class="metrics-divider"></div>'
        html += '<div class="metrics-grid-title">RESOURCE TRACE</div>'
        html += '<div class="resource-grid">'

        if token_usage:
            pt = _format_token_count(token_usage.get("prompt_tokens", 0))
            ct = _format_token_count(token_usage.get("completion_tokens", 0))
            tt = _format_token_count(token_usage.get("total_tokens", 0))
            html += f"""
            <div class="resource-item">
                <span class="resource-label">TOKEN</span>
                <span class="resource-value">P:{pt} C:{ct} Σ:{tt}</span>
            </div>"""

        if elapsed_str:
            html += f"""
            <div class="resource-item">
                <span class="resource-label">ELAPSED</span>
                <span class="resource-value">{elapsed_str}</span>
            </div>"""

        html += "</div>"

        if step_times:
            html += '<div class="step-trace"><span class="step-label">TRACE</span><div class="step-chips">'
            node_labels = {
                "reasoning": "R",
                "action": "A",
                "checkpoint_fe": "FE",
                "checkpoint_model": "M",
            }
            for st in step_times[-10:]:
                node = st.get("node", "?")
                dur = st.get("duration", 0)
                label = node_labels.get(node, node[:2])
                html += f'<span class="step-chip step-{node}">{label} {dur:.1f}s</span>'
            html += "</div></div>"

    html += "</div>"
    return html


def _format_duration(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS 或 MM:SS 字符串。"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_token_count(n: int) -> str:
    """格式化 token 数量，过千时用 k 表示。"""
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def _export_code_history(code_history: list[str], task: str = "") -> tuple[str, str]:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    py_path = output_dir / f"analysis_{timestamp}.py"
    py_lines = [
        '"""Auto-generated data analysis script.',
        f"Task: {task}",
        f"Generated: {datetime.now().isoformat()}",
        '"""',
        "",
    ]
    for i, code in enumerate(code_history, 1):
        py_lines.append(f"# ---- Step {i} ----")
        py_lines.append(code)
        py_lines.append("")
    py_path.write_text("\n".join(py_lines), encoding="utf-8")

    ipynb_path = output_dir / f"analysis_{timestamp}.ipynb"
    notebook: dict = {
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


def _get_pending_checkpoint(app, config) -> dict | None:
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


# ============================================================================
# AgentSession
# ============================================================================

class AgentSession:
    """Encapsulates runtime state persisted across Gradio events."""

    def __init__(self):
        self.app = None
        self.config = None
        self.km: KernelManager | None = None
        self.state: dict | None = None
        self.interrupt_info: dict | None = None
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


# ============================================================================
# Sidebar event handlers
# ============================================================================

def on_new_chat(session_state: dict):
    """Create a fresh session and clear the main panel."""
    store = SessionStore()
    rec = store.create_session()
    session_state["active_session_id"] = rec.session_id
    session_state["_session"] = None

    session_list = _refresh_session_list()
    init_progress = _format_progress({"phase": "等待开始", "iteration": 0})
    init_status = (
        "<div class='status-html-card status-ready'>"
        "⚡ 核心就绪 — 请载入标准数据集并激活「开始分析」管道"
        "</div>"
    )
    return (
        session_list,           # 7 session_list
        [],                     # 0 chatbot
        init_status,            # 1 status
        gr.update(visible=False),  # 2 checkpoint_panel
        "",                     # 3 checkpoint_proposal
        gr.update(value=""),    # 4 feedback_input
        init_progress,          # 5 progress
        session_state,          # 6 session_state
    )


def _ensure_active_session(session_state: dict) -> str:
    sid = session_state.get("active_session_id")
    if not sid:
        store = SessionStore()
        rec = store.create_session()
        session_state["active_session_id"] = rec.session_id
        return rec.session_id
    return sid


def on_session_selected(evt: gr.SelectData, session_state: dict, session_list_data):
    """【已修复问题 3】精确映射 gr.SelectData 签名，通过行索引安全读取会话 ID"""
    if evt.index is None or session_list_data is None:
        return [gr.update()] * 8

    try:
        row_idx = evt.index[0]
        # Dataframe value 可能是 list[list] 或者 pandas DataFrame
        if hasattr(session_list_data, "iloc"):
            session_id = session_list_data.iloc[row_idx, 0]
        else:
            session_id = session_list_data[row_idx][0]
    except Exception:
        return [gr.update()] * 8

    store = SessionStore()
    record = store.get_session(session_id)
    if record is None:
        return [gr.update()] * 8

    session_state["active_session_id"] = record.session_id

    profile = record.metrics or {}
    progress = _format_progress({
        "phase": "已完成" if record.completed else "进行中",
        "data_profile": profile,
    })
    
    loaded_status = f"<div class='status-html-card status-loaded'>📁 归档反序列化完成: {record.title}</div>"

    return (
        _refresh_session_list(),          # 7
        record.messages,                   # 0
        loaded_status,                     # 1
        gr.update(visible=False),          # 2
        "",                                # 3
        gr.update(value=""),               # 4
        progress,                          # 5
        session_state,                     # 6
    )


def on_delete_session(session_state: dict):
    """Delete the currently selected session."""
    session_id = session_state.get("active_session_id")
    if session_id:
        store = SessionStore()
        store.delete_session(session_id)
        session_state["active_session_id"] = None
        session_state["_session"] = None

    deleted_status = "<div class='status-html-card status-deleted'>🗑️ 当前活跃会话及其磁盘存盘已被安全物理销毁</div>"
    init_progress = _format_progress({"phase": "等待开始", "iteration": 0})

    return (
        _refresh_session_list(),          # 7
        [],                               # 0
        deleted_status,                    # 1
        gr.update(visible=False),          # 2
        "",                               # 3
        gr.update(value=""),              # 4
        init_progress,                     # 5
        session_state,                    # 6
    )


# ============================================================================
# Streaming generators
# ============================================================================

def run_agent_stream(
    dataset_path: str,
    task: str,
    session_state: dict,
):
    """Stream agent execution from initial state, yielding UI updates."""
    session: AgentSession = session_state.get("_session")
    if session is None:
        session = AgentSession()
        session_state["_session"] = session

    _ensure_active_session(session_state)
    session.reset()

    llm = LLMClient()
    compressor = OutputCompressor()
    session.app = build_agent(llm, session.km, compressor)
    session.config = make_thread_config()
    session.state = init_state(task, dataset_path)

    latest_values = dict(session.state)
    latest_values.setdefault("messages", [])
    latest_values.setdefault("code_history", [])

    chatbot: list[dict] = []
    status = "<div class='status-html-card status-running'>⚙️ 容器算力冷启动，正在分配拓扑管道...</div>"
    progress_md = _format_progress(latest_values)

    yield (
        chatbot,                    # 0
        status,                     # 1
        gr.update(visible=False),   # 2
        "",                         # 3
        gr.update(value=""),        # 4
        progress_md,                # 5
        session_state,              # 6
        _refresh_session_list(),    # 7
    )

    step_times: list[dict] = []
    last_event_time = time.time()

    try:
        for event in session.app.stream(session.state, session.config, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "__interrupt__":
                    continue

                now = time.time()
                step_duration = now - last_event_time
                step_times.append({"node": node_name, "duration": round(step_duration, 2)})
                last_event_time = now

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
                    chatbot = _format_chatbot(latest_values["messages"])
                    status = f"<div class='status-html-card status-running'>🚀 活跃计算拓扑节点: <code>{node_name}</code></div>"

                progress_md = _format_progress(latest_values, step_times)
                yield (
                    chatbot,
                    status,
                    gr.update(visible=False),
                    "",
                    gr.update(value=""),
                    progress_md,
                    session_state,
                    gr.update(),  # no sidebar refresh during streaming
                )
    except Exception as e:
        status = f"<div class='status-html-card status-error'>❌ 容器内核崩溃故障: {str(e)}</div>"
        progress_md = _format_progress(latest_values, step_times)
        yield (
            chatbot,
            status,
            gr.update(visible=False),
            "",
            gr.update(value=""),
            progress_md,
            session_state,
            _refresh_session_list(),
        )
        return

    # After stream: check for interrupt
    interrupt_info = _get_pending_checkpoint(session.app, session.config)
    if interrupt_info:
        session.interrupt_info = interrupt_info
        proposal = interrupt_info.get("proposal", "")
        checkpoint_type = interrupt_info.get("type", "checkpoint")

        _save_session_from_state(session_state, chatbot, latest_values)

        status = f"<div class='status-html-card status-paused'>🚧 触发人工确认关卡: <code>{checkpoint_type}</code></div>"
        progress_md = _format_progress(latest_values, step_times)
        yield (
            chatbot,
            status,
            gr.update(visible=True),
            proposal,
            gr.update(value=""),
            progress_md,
            session_state,
            _refresh_session_list(),
        )
        return

    # Analysis finished
    snapshot = session.app.get_state(session.config)
    values = snapshot.values if hasattr(snapshot, "values") else snapshot
    final_report = values.get("final_report", "分析完成。")
    code_history = values.get("code_history", [])
    task_name = values.get("task", "")
    py_path, ipynb_path = _export_code_history(code_history, task_name)
    export_text = (
        f"\n\n---\n\n💡 **自动化数据科学流水线记录已安全落盘**\n- 🐍 Python 独立脚本: `{py_path}`\n- 📓 Jupyter Notebook: `{ipynb_path}`"
    )

    report_msg = build_final_report_message(final_report, export_text)
    chatbot.append(report_msg)

    _save_session_from_state(session_state, chatbot, values)

    status = "<div class='status-html-card status-success'>✅ 完整数据推演流程审计圆满结束</div>"
    progress_md = _format_progress(latest_values, step_times)

    yield (
        chatbot,
        status,
        gr.update(visible=False),
        "",
        gr.update(value=""),
        progress_md,
        session_state,
        _refresh_session_list(),
    )


def resume_agent_stream(session_state: dict, feedback: str = ""):
    """Resume agent from a checkpoint interrupt, yielding UI updates."""
    session: AgentSession = session_state.get("_session")
    if session is None or session.app is None or session.interrupt_info is None:
        yield (
            [],
            "<div class='status-html-card status-error'>⚠️ 未发现挂起的断点拓扑，请重新初始化任务</div>",
            gr.update(visible=False),
            "",
            gr.update(value=""),
            _format_progress({}),
            session_state,
            _refresh_session_list(),
        )
        return

    if feedback.strip():
        try:
            session.app.update_state(session.config, {"user_feedback": feedback})
        except Exception:
            pass

    session.interrupt_info = None
    status = "<div class='status-html-card status-running'>🔄 正在解冻上下文并唤醒推演核心...</div>"
    latest_values = dict(session.state) if session.state else {}
    latest_values.setdefault("messages", [])
    latest_values.setdefault("code_history", [])

    # 从 checkpointer 恢复 started_at / token_usage，否则 resume 后这些值会丢失
    try:
        cp_snap = session.app.get_state(session.config)
        if cp_snap:
            cp_vals = cp_snap.values if hasattr(cp_snap, "values") else cp_snap
            for k in ("started_at", "token_usage"):
                if cp_vals.get(k):
                    latest_values[k] = cp_vals[k]
    except Exception:
        pass

    chatbot: list[dict] = _format_chatbot(latest_values["messages"])
    step_times: list[dict] = []
    last_event_time = time.time()
    progress_md = _format_progress(latest_values, step_times)

    yield (
        chatbot,
        status,
        gr.update(visible=False),
        "",
        gr.update(value=""),
        progress_md,
        session_state,
        gr.update(),
    )

    while True:
        try:
            for event in session.app.stream(Command(resume="confirmed"), session.config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    if node_name == "__interrupt__":
                        continue

                    now = time.time()
                    step_duration = now - last_event_time
                    step_times.append({"node": node_name, "duration": round(step_duration, 2)})
                    last_event_time = now

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
                        chatbot = _format_chatbot(latest_values["messages"])
                        status = f"<div class='status-html-card status-running'>🚀 活跃计算拓扑节点: <code>{node_name}</code></div>"

                    progress_md = _format_progress(latest_values, step_times)
                    yield (
                        chatbot,
                        status,
                        gr.update(visible=False),
                        "",
                        gr.update(value=""),
                        progress_md,
                        session_state,
                        gr.update(),
                    )
        except Exception as e:
            status = f"<div class='status-html-card status-error'>❌ 恢复推演时内核崩溃: {str(e)}</div>"
            progress_md = _format_progress(latest_values, step_times)
            yield (
                chatbot,
                status,
                gr.update(visible=False),
                "",
                gr.update(value=""),
                progress_md,
                session_state,
                _refresh_session_list(),
            )
            return

        interrupt_info = _get_pending_checkpoint(session.app, session.config)
        if interrupt_info:
            if session.auto_confirm:
                session.interrupt_info = interrupt_info
                status = f"<div class='status-html-card status-running'>⚡ 联动机制流转中 — 自动核准: <code>{interrupt_info.get('type', 'checkpoint')}</code></div>"
                progress_md = _format_progress(latest_values, step_times)
                yield (
                    chatbot,
                    status,
                    gr.update(visible=False),
                    "",
                    gr.update(value=""),
                    progress_md,
                    session_state,
                    gr.update(),
                )
                continue
            else:
                session.interrupt_info = interrupt_info
                proposal = interrupt_info.get("proposal", "")
                checkpoint_type = interrupt_info.get("type", "checkpoint")

                _save_session_from_state(session_state, chatbot, latest_values)

                status = f"<div class='status-html-card status-paused'>🚧 持续挂起中 — 等待确认 <code>{checkpoint_type}</code> 方案</div>"
                progress_md = _format_progress(latest_values, step_times)
                yield (
                    chatbot,
                    status,
                    gr.update(visible=True),
                    proposal,
                    gr.update(value=""),
                    progress_md,
                    session_state,
                    _refresh_session_list(),
                )
                return

        break

    snapshot = session.app.get_state(session.config)
    values = snapshot.values if hasattr(snapshot, "values") else snapshot
    final_report = values.get("final_report", "分析完成。")
    code_history = values.get("code_history", [])
    task_name = values.get("task", "")
    py_path, ipynb_path = _export_code_history(code_history, task_name)
    export_text = (
        f"\n\n---\n\n💡 **自动化数据科学流水线记录已安全落盘**\n- 🐍 Python 独立脚本: `{py_path}`\n- 📓 Jupyter Notebook: `{ipynb_path}`"
    )

    report_msg = build_final_report_message(final_report, export_text)
    chatbot.append(report_msg)

    _save_session_from_state(session_state, chatbot, values)

    status = "<div class='status-html-card status-success'>✅ 完整数据推演流程审计圆满结束</div>"
    progress_md = _format_progress(latest_values, step_times)

    yield (
        chatbot,
        status,
        gr.update(visible=False),
        "",
        gr.update(value=""),
        progress_md,
        session_state,
        _refresh_session_list(),
    )


# ============================================================================
# Gradio UI Design Layout
# ============================================================================

PREMIUM_CSS = """
body, .gradio-container { background-color: #f8fafc !important; font-family: 'Inter', system-ui, sans-serif !important; }
#main-row { gap: 20px !important; overflow-y: auto !important; }
#sidebar-column { background: #ffffff !important; border-right: 1px solid #e2e8f0 !important; padding: 16px !important; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }
#main-column { gap: 16px !important; }
#session-list { border: none !important; background: transparent !important; }
#session-list .table-wrap { border: none !important; }
#session-list table { border-collapse: collapse !important; width: 100%; }
#session-list th { display: none !important; }
#session-list td { padding: 10px 12px !important; border-bottom: 1px solid #f1f5f9 !important; font-size: 13px !important; color: #334155 !important; cursor: pointer; transition: background 0.2s; }
#session-list tr:hover td { background-color: #f1f5f9 !important; border-radius: 6px; }
#input-group, #checkpoint-panel { background: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 12px !important; padding: 20px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important; }
#checkpoint-panel { border-left: 4px solid #f59e0b !important; }
#checkpoint-panel h3 { margin-top: 0; color: #b45309; font-size: 15px; font-weight: 600; letter-spacing: 0.02em; }
.status-html-card { padding: 10px 14px; border-radius: 6px; font-size: 13px; font-family: monospace; font-weight: 500; border: 1px solid transparent; }
.status-ready { background-color: #f1f5f9; color: #475569; border-color: #e2e8f0; }
.status-running { background-color: #f0fdf4; color: #166534; border-color: #bbf7d0; }
.status-paused { background-color: #fffbeb; color: #92400e; border-color: #fef3c7; }
.status-success { background-color: #e0e7ff; color: #3730a3; border-color: #c7d2fe; }
.status-error { background-color: #fee2e2; color: #991b1b; border-color: #fecaca; }
.status-loaded { background-color: #f8fafc; color: #0f172a; border-color: #e2e8f0; }
.status-deleted { background-color: #fafafa; color: #737373; border-color: #e5e5e5; }
.crypto-progress-card { background: #ffffff; padding: 18px; border-radius: 8px; border: 1px solid #e2e8f0; }
.progress-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.progress-title { font-size: 10px; color: #64748b; font-weight: 600; letter-spacing: 0.08em; }
.progress-subtitle { font-size: 11px; color: #475569; font-weight: 500; }
.progress-label-main { font-size: 16px; color: #0f172a; font-weight: 600; margin-bottom: 10px; }
.progress-bar-container { background: #f1f5f9; border-radius: 10px; height: 5px; overflow: hidden; }
.progress-bar-fill { background: #0f172a; height: 100%; border-radius: 10px; transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1); }
.metrics-divider { margin: 14px 0 10px 0; border-top: 1px dashed #e2e8f0; }
.metrics-grid-title { font-size: 10px; color: #64748b; font-weight: 600; letter-spacing: 0.08em; margin-bottom: 8px; }
.metrics-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
.metric-item { background: #f8fafc; padding: 6px 10px; border-radius: 4px; border: 1px solid #f1f5f9; display: flex; justify-content: space-between; align-items: center; }
.metric-label { font-size: 11px; color: #64748b; }
.metric-value { font-size: 12px; font-family: monospace; font-weight: 600; color: #0f172a; }
.resource-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-bottom: 8px; }
.resource-item { background: #f0fdf4; padding: 6px 10px; border-radius: 4px; border: 1px solid #bbf7d0; display: flex; justify-content: space-between; align-items: center; }
.resource-label { font-size: 10px; color: #166534; font-weight: 600; letter-spacing: 0.06em; }
.resource-value { font-size: 11px; font-family: monospace; font-weight: 600; color: #14532d; }
.step-trace { display: flex; align-items: flex-start; gap: 8px; margin-top: 4px; }
.step-label { font-size: 10px; color: #64748b; font-weight: 600; letter-spacing: 0.06em; white-space: nowrap; padding-top: 2px; }
.step-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.step-chip { font-size: 10px; font-family: monospace; padding: 2px 6px; border-radius: 3px; font-weight: 500; white-space: nowrap; }
.step-chip.step-reasoning { background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; }
.step-chip.step-action { background: #f1f5f9; color: #334155; border: 1px solid #e2e8f0; }
.step-chip.step-checkpoint_fe { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
.step-chip.step-checkpoint_model { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
"""

def create_ui() -> gr.Blocks:
    css_path = Path(__file__).resolve().parent / "static" / "style.css"
    try:
        css_content = css_path.read_text(encoding="utf-8") + "\n" + PREMIUM_CSS
    except Exception:
        css_content = PREMIUM_CSS

    # 【已修复问题 2】准备将学术风格的骨架主题传给最后的 launch()
    academic_theme = gr.themes.Soft(
        primary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
    ).set(
        button_primary_background_fill="*neutral_900",
        button_primary_background_fill_hover="*neutral_800",
        block_title_text_weight="600",
        block_label_text_size="11px"
    )

    with gr.Blocks(title="自主数据科学智能体") as app:
        app._custom_css = css_content
        app._custom_theme = academic_theme
        session_state = gr.State({})

        gr.HTML("""
        <div style='border-left: 4px solid #0f172a; padding-left: 14px; margin-bottom: 18px; margin-top: 8px;'>
            <h1 style='font-size: 24px; font-weight: 700; color: #0f172a; margin: 0; letter-spacing: -0.02em;'>Autonomous Data Science Engine</h1>
            <p style='font-size: 13px; color: #64748b; margin-top: 4px; margin-bottom: 0;'>
                基于双核解耦拓扑的多模型流式演化沙箱 — 覆盖全链路自动 EDA、极简特征工程与决策树/高阶增强算法集成
            </p>
        </div>
        """)

        with gr.Row(elem_id="main-row"):
            # ===================== SIDEBAR =====================
            with gr.Column(scale=1, min_width=280, elem_id="sidebar-column"):
                gr.HTML(
                    '<div class="sidebar-header" style="margin-bottom:12px;">'
                    "<h3 style='font-size:14px; font-weight:600; color:#0f172a; margin:0;'>ARCHIVE SESSIONS</h3>"
                    "</div>"
                )

                new_chat_btn = gr.Button(
                    "+ 开启全新推演沙箱",
                    elem_id="new-chat-btn",
                    variant="primary",
                    size="sm"
                )

                initial_list = _refresh_session_list()
                session_list = gr.Dataframe(
                    value=initial_list,
                    headers=["ID", "会话任务归档", "时间步"],
                    interactive=False,
                    elem_id="session-list",
                    datatype=["str", "str", "str"],
                    wrap=True,
                    max_height=420,
                )

                delete_btn = gr.Button(
                    "物理销毁当前存盘",
                    elem_id="delete-session-btn",
                    size="sm",
                    variant="stop"
                )

                with gr.Accordion("关于系统运行指标", open=False, elem_id="sidebar-about"):
                    store = SessionStore()
                    count = len(store.list_sessions())
                    gr.Markdown(
                        f"本地就绪存盘: **{count}** 组\n\n"
                        f"持久化底座: `data/sessions/`\n\n"
                        "提示：点击历史行可直接热加载历史计算快照。"
                    )

            # ===================== MAIN PANEL =====================
            with gr.Column(scale=3, elem_id="main-column"):
                status_display = gr.HTML(
                    value="<div class='status-html-card status-ready'>⚡ 核心就绪 — 请载入标准数据集并激活「开始分析」管道</div>",
                    elem_id="status-container"
                )
                
                with gr.Group(visible=False, elem_id="checkpoint-panel") as checkpoint_panel:
                    gr.Markdown("### 🚧 拓扑阻断确认")
                    checkpoint_proposal = gr.Markdown(
                        "智能体将在执行前等待你的确认。",
                    )
                    feedback_input = gr.Textbox(
                        label="下达干预指令 / 修改偏置（可选）",
                        lines=2,
                        placeholder="如：请补充交叉验证、改用 SMOTE 而非 class_weight、对 TotalCharges 做对数变换...",
                    )
                    with gr.Row():
                        confirm_btn = gr.Button("准予核准并向下推进", variant="primary", size="sm")
                        skip_btn = gr.Button("信任后续步骤 (全自动跳过确认)", variant="secondary", size="sm")

                chatbot = gr.Chatbot(
                    label="智能体推演及交互跟踪迹",
                    height=460,
                    elem_id="main-chatbot",
                    show_label=True
                )

                

                with gr.Group(elem_id="input-group"):
                    dataset_input = gr.File(
                        label="上传数据集 (仅支持标准 CSV 文件格式)",
                        file_types=[".csv"],
                        type="filepath",
                    )
                    with gr.Row():
                        with gr.Column(scale=3):
                            task_input = gr.Textbox(
                                label="数据科学推演指令 (Task Prompt)",
                                value=DEFAULT_TASK,
                                lines=2,
                                placeholder="描述你想让智能体完成的分析任务...",
                            )
                        with gr.Column(scale=1):
                            start_btn = gr.Button(
                                "激活管道",
                                variant="primary",
                                size="lg",
                                elem_id="start-btn",
                            )
                            stop_btn = gr.Button("强行熔断", variant="stop", size="lg")

                progress_display = gr.HTML(
                    value=_format_progress({"phase": "等待开始", "iteration": 0}),
                    elem_id="progress-display-html",
                )

        # ====================================================================
        # Event bindings
        # ====================================================================
        outputs = [
            chatbot,              # 0
            status_display,       # 1
            checkpoint_panel,     # 2
            checkpoint_proposal,  # 3
            feedback_input,       # 4
            progress_display,     # 5
            session_state,        # 6
            session_list,         # 7
        ]

        start_btn.click(
            fn=run_agent_stream,
            inputs=[dataset_input, task_input, session_state],
            outputs=outputs,
        )

        confirm_btn.click(
            fn=resume_agent_stream,
            inputs=[session_state, feedback_input],
            outputs=outputs,
        )

        def skip_and_resume(session_state: dict):
            session: AgentSession = session_state.get("_session")
            if session:
                session.auto_confirm = True
            yield from resume_agent_stream(session_state, feedback="")

        skip_btn.click(
            fn=skip_and_resume,
            inputs=[session_state],
            outputs=outputs,
        )

        new_chat_btn.click(
            fn=on_new_chat,
            inputs=[session_state],
            outputs=[
                session_list,
                chatbot,
                status_display,
                checkpoint_panel,
                checkpoint_proposal,
                feedback_input,
                progress_display,
                session_state,
            ],
        )

        # 【已修复问题 3】绑定 inputs 时加入 session_list，使之能正确投递给它的 handler
        session_list.select(
            fn=on_session_selected,
            inputs=[session_state, session_list],
            outputs=[
                session_list,
                chatbot,
                status_display,
                checkpoint_panel,
                checkpoint_proposal,
                feedback_input,
                progress_display,
                session_state,
            ],
        )

        delete_btn.click(
            fn=on_delete_session,
            inputs=[session_state],
            outputs=[
                session_list,
                chatbot,
                status_display,
                checkpoint_panel,
                checkpoint_proposal,
                feedback_input,
                progress_display,
                session_state,
            ],
        )

    return app


if __name__ == "__main__":
    ui = create_ui()
    # 【已修复问题 2】将 theme 与 css 参数统一移动到 launch() 方法内挂载
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=getattr(ui, "_custom_css", ""),
        theme=getattr(ui, "_custom_theme", gr.themes.Soft()),
    )
    