"""Message type detection and structured formatting for visual hierarchy in the chatbot."""

import base64
import re
from pathlib import Path


# Patterns for detecting message types
_EXEC_RESULT_RE = re.compile(r"^\[执行结果\]")
_EXEC_ERROR_RE = re.compile(r"^\[执行错误\]")
_TRACEBACK_RE = re.compile(r"Traceback\s*\(most recent call last\):", re.IGNORECASE)
_FINAL_REPORT_RE = re.compile(r"###\s*最终报告|##\s*最终报告|最终报告", re.IGNORECASE)
_CHECKPOINT_RE = re.compile(r"\[CHECKPOINT:(FE|MODEL)\]")


def detect_message_type(content: str, role: str) -> str:
    """Classify a message into one of the visual-type categories.

    Returns one of: normal, exec_result, exec_error, system, final_report, checkpoint.
    """
    if role == "system":
        return "system"

    if content.startswith("[执行结果]"):
        return "exec_result"

    if content.startswith("[执行错误]") or _TRACEBACK_RE.search(content):
        return "exec_error"

    if _FINAL_REPORT_RE.search(content):
        return "final_report"

    if _CHECKPOINT_RE.search(content):
        return "checkpoint"

    return "normal"


def format_message(msg: dict) -> dict:
    """Tag a single message with its visual type for CSS targeting."""
    content = msg.get("content", "")
    role = msg.get("role", "user")
    msg_type = detect_message_type(content, role)

    # Embed type tag as HTML comment — invisible to rendering, targetable via CSS
    # We prefix it so the CSS :has() or adjacent selector can match
    if msg_type != "normal":
        content = f'<span class="msg-type msg-{msg_type}" style="display:none"></span>\n{content}'

    return {"role": role, "content": content}


_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def _convert_images_to_base64(content: str) -> str:
    """Replace local image markdown references with base64 data URIs."""

    def _replace(m: re.Match) -> str:
        alt = m.group(1)
        src = m.group(2)
        if src.startswith("data:") or src.startswith("http"):
            return m.group(0)
        path = Path(src)
        if not path.exists():
            return m.group(0)
        try:
            img_bytes = path.read_bytes()
            b64 = base64.b64encode(img_bytes).decode()
            ext = path.suffix.lower()
            mime = "image/png" if ext == ".png" else "image/jpeg" if ext in (".jpg", ".jpeg") else "image/svg+xml" if ext == ".svg" else "image/png"
            return f"![{alt}](data:{mime};base64,{b64})"
        except Exception:
            return m.group(0)

    return _IMG_RE.sub(_replace, content)


def format_chatbot(messages: list[dict]) -> list[dict]:
    """Convert internal messages to Gradio Chatbot format with type annotations.

    Handles role normalization: system messages render with assistant role.
    """
    rows: list[dict] = []
    for m in messages:
        role_raw = m.get("role", "user")
        content = m.get("content", "")

        # Normalize role for Gradio Chatbot (only "user" / "assistant" supported)
        if role_raw == "system":
            role = "assistant"
            content = f"**[System]** {content}"
        elif role_raw == "assistant":
            role = "assistant"
        else:
            role = "user"

        # Detect type and annotate
        msg_type = detect_message_type(content, role_raw)
        if msg_type != "normal":
            content = f'<span class="msg-type-badge msg-type-{msg_type}"></span>{content}'

        content = _convert_images_to_base64(content)
        rows.append({"role": role, "content": content})

    return rows


def build_final_report_message(final_report: str, export_text: str = "") -> dict:
    """Build a rich final-report message for the chatbot."""
    parts = []
    if final_report and final_report != "分析完成。":
        parts.append(final_report)
    if export_text:
        parts.append(export_text.strip())
    content = "\n\n---\n\n".join(parts) if parts else "分析完成。"
    return {"role": "assistant", "content": content}
