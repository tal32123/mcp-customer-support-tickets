from unittest.mock import MagicMock
import pytest
from mcp_cst.llm.protocol import LlmClient
from mcp_cst.llm.anthropic_client import AnthropicClient
from mcp_cst.llm.openai_client import OpenAIClient


def test_anthropic_client_uses_messages_api(monkeypatch):
    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.content = [MagicMock(text="drafted reply")]
    fake_client.messages.create.return_value = fake_resp
    monkeypatch.setattr("mcp_cst.llm.anthropic_client._make_sdk_client", lambda: fake_client)

    client = AnthropicClient(model="claude-opus-4-7")
    out = client.complete(system="sys", user="usr")
    assert out == "drafted reply"
    args = fake_client.messages.create.call_args
    assert args.kwargs["model"] == "claude-opus-4-7"
    assert args.kwargs["system"] == "sys"
    assert args.kwargs["messages"][0]["role"] == "user"
    assert args.kwargs["messages"][0]["content"] == "usr"


def test_openai_client_uses_chat_api(monkeypatch):
    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content="drafted gpt"))]
    fake_client.chat.completions.create.return_value = fake_resp
    monkeypatch.setattr("mcp_cst.llm.openai_client._make_sdk_client", lambda: fake_client)

    client = OpenAIClient(model="gpt-4o")
    out = client.complete(system="sys", user="usr")
    assert out == "drafted gpt"
    args = fake_client.chat.completions.create.call_args
    assert args.kwargs["model"] == "gpt-4o"
    messages = args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[1] == {"role": "user", "content": "usr"}


def test_both_clients_satisfy_protocol():
    assert hasattr(AnthropicClient, "complete")
    assert hasattr(OpenAIClient, "complete")
