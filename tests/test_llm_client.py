"""LLM 客户端模型选择测试（不发起网络请求）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.llm_client import AnthropicLLMClient


def test_anthropic_prefers_llm_model_over_anthropic_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "deepseek-flash[1m]")
    monkeypatch.setenv("ANTHROPIC_MODEL", "deepseek-v4-pro[1m]")
    monkeypatch.delenv("AGENT_MODEL", raising=False)

    client = AnthropicLLMClient(api_key="test", base_url="http://localhost")

    assert client.model == "deepseek-flash[1m]"


def test_agent_model_has_highest_priority(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "agent-model")
    monkeypatch.setenv("LLM_MODEL", "llm-model")
    monkeypatch.setenv("ANTHROPIC_MODEL", "anthropic-model")

    client = AnthropicLLMClient(api_key="test", base_url="http://localhost")

    assert client.model == "agent-model"
