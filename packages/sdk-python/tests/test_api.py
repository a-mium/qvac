"""Unit tests for qvac.api's hand-written wrappers: pure request-shaping and
response-validation logic, so these run against a fake transport rather than
a spawned worker (see test_poc_progress.py / test_poc_smoke.py for the real
end-to-end coverage)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent / "src"))

from qvac import api


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.sent = None

    def call(self, payload):
        self.sent = payload
        return self.response

    def call_stream(self, payload):
        self.sent = payload
        yield from self.response

    def call_duplex(self, payload, up):
        raise NotImplementedError


def test_cancel_by_request_id():
    transport = FakeTransport({"type": "cancel", "success": True})
    api.cancel(transport, request_id="req-1")
    assert transport.sent == {
        "type": "cancel",
        "operation": "request",
        "requestId": "req-1",
    }


def test_cancel_by_request_id_with_clear_cache():
    transport = FakeTransport({"type": "cancel", "success": True})
    api.cancel(transport, request_id="req-1", clear_cache=True)
    assert transport.sent == {
        "type": "cancel",
        "operation": "request",
        "requestId": "req-1",
        "clearCache": True,
    }


def test_cancel_broad_by_model_id():
    transport = FakeTransport({"type": "cancel", "success": True})
    api.cancel(transport, model_id="model-1", kind="completion")
    assert transport.sent == {
        "type": "cancel",
        "operation": "broad",
        "modelId": "model-1",
        "kind": "completion",
    }


def test_cancel_requires_request_id_or_model_id():
    transport = FakeTransport({"type": "cancel", "success": True})
    with pytest.raises(ValueError):
        api.cancel(transport)


def test_cancel_raises_on_failure():
    transport = FakeTransport({"type": "cancel", "success": False, "error": "nope"})
    with pytest.raises(api.CancelFailedError):
        api.cancel(transport, request_id="req-1")


def test_unload_model_success():
    transport = FakeTransport({"type": "unloadModel", "success": True})
    api.unload_model(transport, "model-1")
    assert transport.sent == {
        "type": "unloadModel",
        "modelId": "model-1",
        "clearStorage": False,
    }


def test_unload_model_raises_on_failure():
    transport = FakeTransport({"type": "unloadModel", "success": False})
    with pytest.raises(api.ModelUnloadFailedError):
        api.unload_model(transport, "model-1")


def test_invoke_plugin_unwraps_result():
    transport = FakeTransport({"type": "pluginInvoke", "result": {"ok": True}})
    result = api.invoke_plugin(transport, "model-1", "vlaRun", params={"x": 1})
    assert result == {"ok": True}
    assert transport.sent == {
        "type": "pluginInvoke",
        "modelId": "model-1",
        "handler": "vlaRun",
        "params": {"x": 1},
    }


def test_invoke_plugin_stream_skips_done_chunk():
    transport = FakeTransport(
        [
            {"type": "pluginInvokeStream", "result": "a"},
            {"type": "pluginInvokeStream", "result": "b"},
            {"type": "pluginInvokeStream", "result": None, "done": True},
        ]
    )
    results = list(api.invoke_plugin_stream(transport, "model-1", "handler"))
    assert results == ["a", "b"]


REGISTRY_MODEL_ITEM = {
    "name": "TEST_MODEL",
    "registryPath": "p",
    "registrySource": "hf",
    "blobCoreKey": "0" * 64,
    "blobBlockOffset": 0,
    "blobBlockLength": 1,
    "blobByteOffset": 0,
    "modelId": "m",
    "addon": "llm",
    "expectedSize": 1,
    "sha256Checksum": "0" * 64,
    "engine": "llamacpp-completion",
    "quantization": "q4",
    "params": "1B",
}


def test_model_registry_list_returns_models():
    transport = FakeTransport(
        {"type": "modelRegistryList", "success": True, "models": [REGISTRY_MODEL_ITEM]}
    )
    models = api.model_registry_list(transport)
    assert len(models) == 1
    assert models[0].registry_path == "p"


def test_model_registry_list_raises_on_failure():
    transport = FakeTransport(
        {"type": "modelRegistryList", "success": False, "error": "boom"}
    )
    with pytest.raises(api.ModelRegistryQueryFailedError, match="boom"):
        api.model_registry_list(transport)


def test_model_registry_search_model_type_aliases_addon():
    transport = FakeTransport(
        {"type": "modelRegistrySearch", "success": True, "models": []}
    )
    api.model_registry_search(transport, model_type="llm")
    assert transport.sent == {"type": "modelRegistrySearch", "addon": "llm"}


def test_model_registry_search_model_type_wins_over_addon():
    transport = FakeTransport(
        {"type": "modelRegistrySearch", "success": True, "models": []}
    )
    api.model_registry_search(transport, model_type="llm", addon="whisper")
    assert transport.sent["addon"] == "llm"


def test_model_registry_search_passes_filters():
    transport = FakeTransport(
        {"type": "modelRegistrySearch", "success": True, "models": []}
    )
    api.model_registry_search(
        transport, filter="qwen", engine="llamacpp-completion", quantization="q4"
    )
    assert transport.sent == {
        "type": "modelRegistrySearch",
        "filter": "qwen",
        "engine": "llamacpp-completion",
        "quantization": "q4",
    }


def test_model_registry_get_model_uses_fallback_error():
    transport = FakeTransport(
        {"type": "modelRegistryGetModel", "success": False, "error": None}
    )
    with pytest.raises(
        api.ModelRegistryQueryFailedError, match="Model not found: hf/p"
    ):
        api.model_registry_get_model(transport, "p", "hf")


def test_delete_cache_all():
    transport = FakeTransport({"type": "deleteCache", "success": True})
    result = api.delete_cache(transport, all=True)
    assert transport.sent == {"type": "deleteCache", "all": True}
    assert result == {"success": True}


def test_delete_cache_by_kv_cache_key():
    transport = FakeTransport({"type": "deleteCache", "success": True})
    api.delete_cache(transport, kv_cache_key="key-1", model_id="model-1")
    assert transport.sent == {
        "type": "deleteCache",
        "kvCacheKey": "key-1",
        "modelId": "model-1",
    }


def test_delete_cache_requires_all_or_kv_cache_key():
    transport = FakeTransport({"type": "deleteCache", "success": True})
    with pytest.raises(api.InvalidDeleteCacheParamsError):
        api.delete_cache(transport)


def test_delete_cache_raises_only_when_error_message_present():
    transport = FakeTransport(
        {"type": "deleteCache", "success": False, "error": "boom"}
    )
    with pytest.raises(api.DeleteCacheFailedError):
        api.delete_cache(transport, all=True)


def test_delete_cache_silent_failure_without_error_message():
    transport = FakeTransport({"type": "deleteCache", "success": False})
    result = api.delete_cache(transport, all=True)
    assert result == {"success": False}
