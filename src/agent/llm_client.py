import json
import os
import time
from pathlib import Path
from typing import Optional

import urllib.request
import urllib.error

# 自动加载项目根目录的 .env 文件
def _load_dotenv():
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass

_load_dotenv()


class LLMClient:
    """OpenAI 兼容的 LLM API 客户端。"""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model or os.getenv("LLM_MODEL", "deepseek-v4-pro")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        self.max_retries = 2

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        """发送聊天请求，返回 LLM 响应文本。"""
        if not self.is_configured:
            raise RuntimeError(
                "LLM_API_KEY not set. Set the environment variable or pass api_key parameter."
            )

        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }).encode("utf-8")

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    return body["choices"][0]["message"]["content"]

            except urllib.error.HTTPError as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
            except (urllib.error.URLError, OSError, Exception) as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        raise RuntimeError(f"LLM API request failed after {self.max_retries + 1} attempts: {last_error}")


class MockLLMClient(LLMClient):
    """Mock LLM 客户端，用于测试图结构时模拟 LLM 响应。"""

    def __init__(self, responses: Optional[list[str]] = None):
        super().__init__(model="mock", api_key="mock", base_url="http://localhost")
        self.responses = responses or []
        self.call_count = 0

    @property
    def is_configured(self) -> bool:
        return True

    def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        # Default: return a finish response
        return "数据分析完成。"
