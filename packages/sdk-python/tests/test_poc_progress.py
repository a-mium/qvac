"""Real end-to-end test of a progress-capable method against a running SDK
worker: loads an actual cached model over a local HTTP server (so the
worker's `http` model-src resolver — the one that actually emits
`modelProgress` events — is exercised for real, not a plain local-path
resolve which never fires progress at all) and validates every event the
worker streams back against the generated pydantic models.

Needs the SDK's Bare worker built (same QVAC_POC_SDK_DIR requirement as
test_poc_smoke.py) and a real GGUF model on disk (QVAC_POC_MODEL, defaulting
to a small model already present in the local `~/.qvac/models` cache);
skipped when either is missing.
"""

from __future__ import annotations

import functools
import http.server
import os
import sys
import threading
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent / "src"))

DEFAULT_MODEL = str(
    Path.home() / ".qvac" / "models" / "5b8aae816570a09d_Qwen3-0.6B-Q4_0.gguf"
)
MODEL_PATH = Path(os.environ.get("QVAC_POC_MODEL", DEFAULT_MODEL))

pytestmark = [
    pytest.mark.skipif(
        "QVAC_POC_SDK_DIR" not in os.environ,
        reason="set QVAC_POC_SDK_DIR to a built SDK checkout to run the PoC progress test",
    ),
    pytest.mark.skipif(
        not MODEL_PATH.is_file(),
        reason=f"no model at {MODEL_PATH}; set QVAC_POC_MODEL to a real .gguf file",
    ),
]


# The worker caches downloads by a hash of the source URL (createHttpDownloadKey
# in server/rpc/handlers/load-model/http.ts), not by content — a fixed port
# keeps that cache key stable across test runs so repeat runs hit the "already
# cached" branch instead of writing a fresh multi-hundred-MB copy every time.
_LOCAL_HTTP_PORT = 47681


@pytest.fixture
def model_url():
    """Serves MODEL_PATH's directory over local HTTP so the worker's `http`
    model-src resolver runs for real — the plain-local-path resolver never
    threads a progress callback through at all, so it's the only way to
    observe a genuine `modelProgress` event end to end."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(MODEL_PATH.parent)
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", _LOCAL_HTTP_PORT), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/{MODEL_PATH.name}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture
def worker():
    from poc_heartbeat import QvacWorker

    with QvacWorker() as w:
        yield w


@pytest.fixture
def transport(worker):
    from poc_transport import PocTransport

    return PocTransport(worker)


def test_load_model_with_progress_streams_real_progress_then_terminal_reply(
    transport, model_url
) -> None:
    from qvac._generated import (
        LoadModelRequest,
        LoadModelResponse,
        ModelProgressResponse,
    )
    from qvac._generated.methods import load_model_with_progress

    params = LoadModelRequest.model_validate(
        {
            "type": "loadModel",
            "modelSrc": model_url,
            "modelType": "llamacpp-completion",
            "modelConfig": {},
        }
    )

    events = list(load_model_with_progress(transport, params))

    assert events, "expected at least one event from a progress-capable call"

    *progress_events, terminal = events
    assert isinstance(
        terminal, LoadModelResponse
    ), f"last event must be the terminal LoadModelResponse, got {type(terminal).__name__}"
    assert terminal.type == "loadModel"
    assert terminal.success is True, f"loadModel failed: {terminal.error}"
    assert terminal.model_id

    assert progress_events, (
        "expected at least one modelProgress event before the terminal reply — "
        "the http model-src resolver emits one even on a cache hit"
    )
    for event in progress_events:
        assert isinstance(
            event, ModelProgressResponse
        ), f"non-terminal event must be ModelProgressResponse, got {type(event).__name__}"
        assert event.type == "modelProgress"
        assert 0 <= event.percentage <= 100
