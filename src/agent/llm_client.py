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

    def chat(self, messages: list[dict], temperature: float = 0.1) -> tuple[str, dict]:
        """发送聊天请求，返回 (LLM 响应文本, token_usage 字典)。"""
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
                    content = body["choices"][0]["message"]["content"]
                    usage_raw = body.get("usage", {})
                    usage = {
                        "prompt_tokens": usage_raw.get("prompt_tokens", 0),
                        "completion_tokens": usage_raw.get("completion_tokens", 0),
                        "total_tokens": usage_raw.get("total_tokens", 0),
                    }
                    return content, usage

            except urllib.error.HTTPError as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
            except (urllib.error.URLError, OSError, Exception) as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        raise RuntimeError(f"LLM API request failed after {self.max_retries + 1} attempts: {last_error}")


class AnthropicLLMClient(LLMClient):
    """Anthropic Messages API 适配器（用于 DeepSeek 的 /anthropic 端点）。

    与 LLMClient 保持相同的 chat() 接口，返回 (文本, token_usage)。
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 8192,
    ):
        self.model = model or os.getenv("ANTHROPIC_MODEL", "deepseek-v4-pro[1m]")
        self.api_key = api_key or os.getenv("ANTHROPIC_AUTH_TOKEN", "")
        self.base_url = (base_url or os.getenv("ANTHROPIC_BASE_URL", "")).rstrip("/")
        self.max_tokens = max_tokens
        self.max_retries = 2

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url)

    def chat(self, messages: list[dict], temperature: float = 0.1) -> tuple[str, dict]:
        system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
        convo: list[dict] = []
        for message in messages:
            role = message.get("role")
            if role not in ("user", "assistant"):
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    str(item.get("text", item)) if isinstance(item, dict) else str(item)
                    for item in content
                )
            if convo and convo[-1]["role"] == role:
                convo[-1]["content"] += "\n\n" + content
            else:
                convo.append({"role": role, "content": content})
        if not convo:
            convo = [{"role": "user", "content": ""}]

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": convo,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                request = urllib.request.Request(
                    self.base_url + "/v1/messages",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=180) as response:
                    body = json.loads(response.read().decode("utf-8"))
                text = "".join(
                    block.get("text", "")
                    for block in body.get("content", [])
                    if block.get("type") == "text"
                )
                usage_raw = body.get("usage", {})
                prompt_tokens = int(usage_raw.get("input_tokens", 0) or 0)
                completion_tokens = int(usage_raw.get("output_tokens", 0) or 0)
                if not text.strip():
                    raise RuntimeError(
                        "Anthropic response contained no text "
                        f"(stop_reason={body.get('stop_reason')})"
                    )
                return text, {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            except urllib.error.HTTPError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"Anthropic API request failed after {self.max_retries + 1} attempts: {last_error}"
        )


def build_llm_client(provider: str = "auto") -> LLMClient:
    """按 provider 构建客户端。

    auto: 如果配置了 ANTHROPIC_AUTH_TOKEN，优先使用 Anthropic 兼容端点；
          否则使用 OpenAI 兼容的 LLM_API_KEY。
    """
    if provider == "auto":
        provider = "anthropic" if os.getenv("ANTHROPIC_AUTH_TOKEN") else "openai"
    if provider == "anthropic":
        return AnthropicLLMClient()
    return LLMClient()


class MockLLMClient(LLMClient):
    """Mock LLM 客户端，用于测试图结构时模拟 LLM 响应。"""

    def __init__(self, responses: Optional[list[str]] = None):
        super().__init__(model="mock", api_key="mock", base_url="http://localhost")
        self.responses = responses or []
        self.call_count = 0

    @property
    def is_configured(self) -> bool:
        return True

    def chat(self, messages: list[dict], temperature: float = 0.1) -> tuple[str, dict]:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        # Default: return a finish response
        return "数据分析完成。", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
