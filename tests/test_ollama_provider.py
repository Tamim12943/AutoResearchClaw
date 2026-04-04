"""Tests for Ollama provider integration.

Covers: provider preset, factory wiring, no-API-key usage,
config validation bypass, and custom base_url override.
"""

from __future__ import annotations

import json
import urllib.request
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from researchclaw.llm import PROVIDER_PRESETS, create_llm_client
from researchclaw.llm.client import LLMClient, LLMConfig, LLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyHTTPResponse:
    """Minimal stub for ``urllib.request.urlopen`` results."""

    def __init__(self, payload: Mapping[str, Any]):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> _DummyHTTPResponse:
        return self

    def __exit__(self, *a: object) -> None:
        return None


def _make_ollama_client(
    *,
    api_key: str = "",
    primary_model: str = "llama3.2",
    fallback_models: list[str] | None = None,
    base_url: str = "http://localhost:11434/v1",
) -> LLMClient:
    config = LLMConfig(
        base_url=base_url,
        api_key=api_key,
        primary_model=primary_model,
        fallback_models=fallback_models or [],
    )
    return LLMClient(config)


# ---------------------------------------------------------------------------
# Unit tests — provider preset
# ---------------------------------------------------------------------------


class TestOllamaPreset:
    """Verify Ollama is registered in PROVIDER_PRESETS."""

    def test_ollama_in_provider_presets(self):
        assert "ollama" in PROVIDER_PRESETS

    def test_ollama_base_url(self):
        assert PROVIDER_PRESETS["ollama"]["base_url"] == "http://localhost:11434/v1"

    def test_ollama_in_cli_provider_choices(self):
        from researchclaw.cli import _PROVIDER_CHOICES

        found = any(v[0] == "ollama" for v in _PROVIDER_CHOICES.values())
        assert found, "ollama not found in _PROVIDER_CHOICES"

    def test_ollama_provider_models_defaults(self):
        from researchclaw.cli import _PROVIDER_MODELS

        primary, fallbacks = _PROVIDER_MODELS["ollama"]
        assert primary == "qwen2.5:14b"
        assert "qwen3.5:4b" in fallbacks


# ---------------------------------------------------------------------------
# Unit tests — from_rc_config wiring
# ---------------------------------------------------------------------------


class TestOllamaFromRCConfig:
    """Verify that LLMClient.from_rc_config resolves the Ollama preset."""

    def _make_rc_config(
        self,
        *,
        base_url: str = "",
        api_key: str = "",
        api_key_env: str = "",
        primary_model: str = "llama3.2",
        fallback_models: tuple[str, ...] = (),
    ) -> SimpleNamespace:
        return SimpleNamespace(
            llm=SimpleNamespace(
                provider="ollama",
                base_url=base_url,
                api_key=api_key,
                api_key_env=api_key_env,
                primary_model=primary_model,
                fallback_models=fallback_models,
            ),
        )

    def test_from_rc_config_sets_ollama_base_url(self):
        rc_config = self._make_rc_config()
        client = LLMClient.from_rc_config(rc_config)
        assert client.config.base_url == "http://localhost:11434/v1"

    def test_from_rc_config_empty_api_key_allowed(self):
        """Ollama does not require an API key."""
        rc_config = self._make_rc_config(api_key="", api_key_env="")
        client = LLMClient.from_rc_config(rc_config)
        assert client.config.api_key == ""

    def test_from_rc_config_custom_base_url_overrides_preset(self):
        """Users can point to a non-default Ollama endpoint."""
        rc_config = self._make_rc_config(
            base_url="http://192.168.1.10:11434/v1",
        )
        client = LLMClient.from_rc_config(rc_config)
        assert client.config.base_url == "http://192.168.1.10:11434/v1"

    def test_from_rc_config_model_and_fallbacks(self):
        rc_config = self._make_rc_config(
            primary_model="mistral",
            fallback_models=("llama3.2",),
        )
        client = LLMClient.from_rc_config(rc_config)
        assert client.config.primary_model == "mistral"
        assert client.config.fallback_models == ["llama3.2"]

    def test_no_anthropic_adapter_for_ollama(self):
        """Ollama should not trigger the Anthropic adapter."""
        rc_config = self._make_rc_config()
        client = LLMClient.from_rc_config(rc_config)
        assert client._anthropic is None


# ---------------------------------------------------------------------------
# Unit tests — config validation bypass
# ---------------------------------------------------------------------------


class TestOllamaConfigValidation:
    """Verify that config validation skips base_url and api_key_env for Ollama."""

    def _minimal_data(self, **llm_overrides: Any) -> dict[str, Any]:
        llm: dict[str, Any] = {
            "provider": "ollama",
            "primary_model": "llama3.2",
            "fallback_models": [],
        }
        llm.update(llm_overrides)
        return {
            "project": {"name": "test", "mode": "full-auto"},
            "research": {"topic": "test topic"},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "console"},
            "knowledge_base": {"backend": "markdown", "root": "docs/kb"},
            "llm": llm,
        }

    def test_no_base_url_required(self):
        from researchclaw.config import validate_config

        data = self._minimal_data()
        # No base_url or api_key_env — should not produce a missing-field error
        result = validate_config(data, check_paths=False)
        field_errors = [e for e in result.errors if "llm.base_url" in e or "llm.api_key_env" in e]
        assert field_errors == []

    def test_no_api_key_env_required(self):
        from researchclaw.config import validate_config

        data = self._minimal_data()
        result = validate_config(data, check_paths=False)
        field_errors = [e for e in result.errors if "llm.api_key_env" in e]
        assert field_errors == []

    def test_explicit_base_url_accepted(self):
        from researchclaw.config import validate_config

        data = self._minimal_data(base_url="http://192.168.1.10:11434/v1")
        result = validate_config(data, check_paths=False)
        field_errors = [e for e in result.errors if "llm.base_url" in e]
        assert field_errors == []


# ---------------------------------------------------------------------------
# Unit tests — HTTP request
# ---------------------------------------------------------------------------


class TestOllamaHTTPRequest:
    """Verify that requests are routed to the correct Ollama endpoint."""

    def test_request_goes_to_ollama_endpoint(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, Any] = {}

        def fake_urlopen(
            req: urllib.request.Request, timeout: int
        ) -> _DummyHTTPResponse:
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _DummyHTTPResponse(
                {
                    "choices": [
                        {"message": {"content": "hello"}, "finish_reason": "stop"}
                    ]
                }
            )

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        client = _make_ollama_client()
        client._raw_call(
            "llama3.2",
            [{"role": "user", "content": "hi"}],
            512,
            0.7,
            False,
        )

        assert captured["url"] == "http://localhost:11434/v1/chat/completions"
        assert captured["body"]["model"] == "llama3.2"

    def test_empty_api_key_sends_bearer_with_empty_token(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """When api_key is empty the Authorization header is 'Bearer '."""
        captured: dict[str, Any] = {}

        def fake_urlopen(
            req: urllib.request.Request, timeout: int
        ) -> _DummyHTTPResponse:
            captured["auth"] = req.get_header("Authorization")
            return _DummyHTTPResponse(
                {
                    "choices": [
                        {"message": {"content": "ok"}, "finish_reason": "stop"}
                    ]
                }
            )

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        client = _make_ollama_client(api_key="")
        client._raw_call(
            "llama3.2",
            [{"role": "user", "content": "hi"}],
            512,
            0.7,
            False,
        )
        assert captured["auth"] == "Bearer "
