"""Real end-to-end test of a progress-capable method against a running SDK
worker: loads an actual cached model via a `registry://` modelSrc and
validates every event the worker streams back against the generated
pydantic models.

`registry://` is the only modelSrc shape (besides `http://`/hyperdrive) that
threads a progress callback through at all — a plain local-path modelSrc
never emits `modelProgress` (see resolve.ts's filesystem branch). Pointed at
a registryPath already present in the SDK's built-in catalog
(packages/sdk/models/registry/models.ts, QWEN3_600M_INST_Q4) whose cached
file already exists locally, this hits the registry resolver's cache-hit
branch — a real code path, purely local disk I/O and checksum verification,
no network — which still emits a real synthetic 100% `modelProgress` event
before the terminal reply.

Needs the SDK's Bare worker built (same QVAC_POC_SDK_DIR requirement as
test_poc_smoke.py) and that model already cached locally (QVAC_POC_MODEL,
defaulting to `~/.qvac/models/5b8aae816570a09d_Qwen3-0.6B-Q4_0.gguf`);
skipped when either is missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from qvac.models import QWEN3_600M_INST_Q4

DEFAULT_MODEL = str(
    Path.home() / ".qvac" / "models" / "5b8aae816570a09d_Qwen3-0.6B-Q4_0.gguf"
)
MODEL_PATH = Path(os.environ.get("QVAC_POC_MODEL", DEFAULT_MODEL))

# QWEN3_600M_INST_Q4's registryPath hashes (server/utils/formatting.ts's
# generateShortHash) to `5b8aae816570a09d`, matching MODEL_PATH's cache
# filename exactly.
REGISTRY_MODEL_SRC = QWEN3_600M_INST_Q4.src

pytestmark = [
    pytest.mark.skipif(
        "QVAC_POC_SDK_DIR" not in os.environ,
        reason="set QVAC_POC_SDK_DIR to a built SDK checkout to run the PoC progress test",
    ),
    pytest.mark.skipif(
        not MODEL_PATH.is_file(),
        reason=f"no cached model at {MODEL_PATH}; set QVAC_POC_MODEL to a real .gguf file",
    ),
]


@pytest.fixture
async def worker():
    from poc_heartbeat import QvacWorker

    async with QvacWorker() as w:
        yield w


@pytest.fixture
def transport(worker):
    from poc_transport import PocTransport

    return PocTransport(worker)


async def test_load_model_with_progress_streams_real_progress_then_terminal_reply(
    transport,
) -> None:
    from qvac.schemas import (
        LoadModelRequest,
        LoadModelResponse,
        ModelProgressResponse,
    )
    from qvac.methods import load_model_with_progress

    params = LoadModelRequest.model_validate(
        {
            "type": "loadModel",
            "modelSrc": REGISTRY_MODEL_SRC,
            "modelType": "llamacpp-completion",
            "modelConfig": {},
        }
    )

    events = [e async for e in load_model_with_progress(transport, params)]

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
        "the registry model-src resolver emits one even on a cache hit"
    )
    for event in progress_events:
        assert isinstance(
            event, ModelProgressResponse
        ), f"non-terminal event must be ModelProgressResponse, got {type(event).__name__}"
        assert event.type == "modelProgress"
        assert 0 <= event.percentage <= 100
