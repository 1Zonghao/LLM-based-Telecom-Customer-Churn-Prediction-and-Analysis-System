import base64
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from jupyter_client import KernelManager as JupyterKernelManager


@dataclass
class ExecutionResult:
    text: str = ""
    images: list[str] = field(default_factory=list)
    error: Optional[str] = None
    execution_count: int = 0

    @property
    def has_error(self) -> bool:
        return self.error is not None

    @property
    def summary(self) -> str:
        parts = []
        if self.text.strip():
            parts.append(self.text.strip())
        if self.images:
            parts.append(f"[{len(self.images)} image(s)]")
        if self.error:
            parts.append(f"[Error] {self.error}")
        return "\n".join(parts) if parts else "(empty output)"


class KernelManager:
    """管理有状态 Jupyter Kernel，支持代码执行与富文本捕获。"""

    def __init__(self, image_dir: str = "data/images"):
        self._jkm: Optional[JupyterKernelManager] = None
        self._kc = None
        self._image_dir = Path(image_dir)
        self._image_dir.mkdir(parents=True, exist_ok=True)
        self._execution_count = 0

    def start(self, kernel_name: str = "python3") -> None:
        if self._jkm is not None:
            return
        self._jkm = JupyterKernelManager(kernel_name=kernel_name)
        self._jkm.start_kernel()
        self._kc = self._jkm.blocking_client()
        self._kc.start_channels()
        try:
            self._kc.wait_for_ready(timeout=30)
        except RuntimeError:
            pass

    def shutdown(self) -> None:
        if self._kc:
            self._kc.stop_channels()
            self._kc = None
        if self._jkm:
            self._jkm.shutdown_kernel(now=True)
            self._jkm = None

    @property
    def is_alive(self) -> bool:
        if self._jkm is None:
            return False
        return self._jkm.is_alive()

    def execute(self, code: str, timeout: int = 120) -> ExecutionResult:
        if not self.is_alive:
            raise RuntimeError("Kernel is not running. Call start() first.")

        msg_id = self._kc.execute(code)
        result = ExecutionResult()

        stdout_parts = []
        stderr_parts = []

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = self._kc.get_iopub_msg(timeout=0.5)
            except Exception:
                continue

            msg_type = msg["header"]["msg_type"]
            content = msg["content"]
            parent_msg_id = msg.get("parent_header", {}).get("msg_id")

            if parent_msg_id != msg_id:
                continue

            if msg_type == "stream":
                if content["name"] == "stdout":
                    stdout_parts.append(content["text"])
                elif content["name"] == "stderr":
                    stderr_parts.append(content["text"])

            elif msg_type == "execute_result":
                self._execution_count = content.get("execution_count", self._execution_count)
                result.execution_count = self._execution_count
                text, images = self._extract_mime_data(content.get("data", {}))
                if text:
                    stdout_parts.append(text)
                result.images.extend(images)

            elif msg_type == "display_data":
                text, images = self._extract_mime_data(content.get("data", {}))
                if text:
                    stdout_parts.append(text)
                result.images.extend(images)

            elif msg_type == "error":
                result.error = self._format_error(content)

            elif msg_type == "status" and content.get("execution_state") == "idle":
                self._execution_count += 1
                result.execution_count = self._execution_count
                break

        result.text = "".join(stdout_parts)
        if stderr_parts:
            stderr_text = "".join(stderr_parts)
            if result.text:
                result.text += f"\n[stderr]\n{stderr_text}"
            else:
                result.text = stderr_text

        return result

    def _extract_mime_data(self, data: dict) -> tuple[str, list[str]]:
        text = ""
        images = []

        if "text/plain" in data:
            text = data["text/plain"]

        if "image/png" in data:
            img_data = data["image/png"]
            if isinstance(img_data, str):
                img_bytes = base64.b64decode(img_data)
            else:
                img_bytes = img_data
            filename = f"img_{uuid.uuid4().hex[:12]}.png"
            filepath = self._image_dir / filename
            filepath.write_bytes(img_bytes)
            images.append(str(filepath))
            # Append markdown reference so LLM can reference it
            if text:
                text += f"\n![chart]({filepath})"
            else:
                text = f"![chart]({filepath})"

        if "text/html" in data and not text:
            # Strip HTML tags for LLM consumption
            html = data["text/html"]
            text = re.sub(r"<[^>]+>", "", html).strip()

        return text, images

    def _format_error(self, content: dict) -> str:
        ename = content.get("ename", "Error")
        evalue = content.get("evalue", "")
        traceback_lines = content.get("traceback", [])
        # Only keep the last few relevant lines
        relevant = []
        for line in traceback_lines:
            clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
            if clean:
                relevant.append(clean)
        return f"{ename}: {evalue}" if evalue else ename

    def restart(self) -> None:
        """重启内核，清除所有变量状态。"""
        self.shutdown()
        time.sleep(0.5)
        self.start()
