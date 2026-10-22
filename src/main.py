"""自主数据科学智能体 — 应用主入口。

启动 Gradio 前端服务，提供数据集上传、自然语言指令输入、
流式思考展示、关键节点断点确认等人机协同界面。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui.app import create_ui


def launch_ui(host: str = "0.0.0.0", port: int = 7860) -> None:
    """启动 Gradio 前端服务。"""
    app = create_ui()
    app.launch(
        server_name=host,
        server_port=port,
        css=getattr(app, "_custom_css", ""),
        theme=getattr(app, "_custom_theme", "soft"),
    )


if __name__ == "__main__":
    launch_ui()
