import json

import pytest
from pydantic import BaseModel

from common.llm_client import LLMError, LLMParseError, LLMSettings, OllamaClient


class Answer(BaseModel):
    value: int


def scripted(*contents):
    """Fake Ollama transport returning the given contents in order; records payloads."""
    calls = []

    def transport(url, payload, timeout):
        calls.append(payload)
        return {"message": {"content": contents[len(calls) - 1]}, "prompt_eval_count": 10, "eval_count": 3}

    return transport, calls


def test_settings_match_course_baseline_and_are_sent():
    transport, calls = scripted('{"value": 1}')
    client = OllamaClient(LLMSettings(host="http://h:1"), transport=transport)
    client.chat_json([{"role": "user", "content": "x"}], Answer, "t")
    payload = calls[0]
    assert payload["model"] == "qwen3.5:9b"
    assert payload["options"] == {"temperature": 0.6, "num_predict": 8192, "seed": 42, "num_ctx": 32768}
    assert payload["format"] == "json" and payload["stream"] is False


def test_ollama_host_env_is_used_and_normalized(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11434")
    assert LLMSettings().host == "http://0.0.0.0:11434"


def test_valid_json_logs_telemetry():
    transport, _ = scripted('{"value": 7}')
    client = OllamaClient(transport=transport)
    assert client.chat_json([], Answer, "plan").value == 7
    assert client.telemetry.summary() == {"llm_calls": 1, "input_tokens": 10, "output_tokens": 3, "latency_ms": client.telemetry.calls[0].latency_ms}


def test_parse_failure_retries_once_with_error_feedback():
    transport, calls = scripted("not json", '{"value": 2}')
    client = OllamaClient(transport=transport)
    assert client.chat_json([{"role": "user", "content": "q"}], Answer, "plan").value == 2
    assert len(calls) == 2
    assert "not valid" in calls[1]["messages"][-1]["content"]
    assert [c.attempt for c in client.telemetry.calls] == [1, 2]


def test_second_parse_failure_raises():
    transport, _ = scripted('{"value": "x"}', '{"wrong": 1}')
    with pytest.raises(LLMParseError):
        OllamaClient(transport=transport).chat_json([], Answer, "plan")


def test_transport_failure_propagates_no_mock_fallback():
    def broken(url, payload, timeout):
        raise LLMError("connection refused")

    with pytest.raises(LLMError):
        OllamaClient(transport=broken).chat([], "plan")


def test_malformed_envelope_raises():
    with pytest.raises(LLMError):
        OllamaClient(transport=lambda *a: {"error": "boom"}).chat([], "plan")


def test_telemetry_reset():
    transport, _ = scripted(json.dumps({"value": 1}))
    client = OllamaClient(transport=transport)
    client.chat_json([], Answer, "p")
    client.telemetry.reset()
    assert client.telemetry.summary()["llm_calls"] == 0
