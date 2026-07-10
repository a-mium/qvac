"""QVAC-21806 PoC: same real-worker rigor as test_poc_smoke.py/test_poc_progress.py,
but the wire encode/decode is bare_rpc.RPC instead of poc_heartbeat.QvacWorker's
hand-rolled framing. Validates responses against the same generated pydantic
models either way -- proving bare_rpc is a drop-in replacement at the framing
layer without changing anything above it.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from qvac.models import QWEN3_600M_INST_Q4
from qvac.schemas import (
    CompletionStreamRequest,
    HeartbeatRequest,
    LoadModelRequest,
)
from qvac.methods import completion_stream, heartbeat, load_model

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        "QVAC_POC_SDK_DIR" not in os.environ,
        reason="set QVAC_POC_SDK_DIR to a built SDK checkout to run the bare_rpc PoC",
    ),
]


@pytest_asyncio.fixture
async def worker():
    from poc_bare_rpc_transport import BareRpcWorker

    async with BareRpcWorker() as w:
        yield w


async def test_heartbeat_unary_via_bare_rpc(worker) -> None:
    """BareRpcWorker.call/.call_stream already match qvac._transport.Transport's
    async shape exactly, so it plugs straight into the real generated stubs --
    no PocTransport-style adapter needed, unlike poc_heartbeat.QvacWorker."""
    response = await heartbeat(worker, HeartbeatRequest(type="heartbeat"))
    assert response.type == "heartbeat"
    assert isinstance(response.number, float)


async def test_load_model_and_completion_stream_via_bare_rpc(worker) -> None:
    load_request = LoadModelRequest.model_validate(
        {
            "type": "loadModel",
            "modelSrc": QWEN3_600M_INST_Q4.src,
            "modelType": "llamacpp-completion",
            "modelConfig": {},
        }
    )
    load_response = await load_model(worker, load_request)
    assert load_response.success, load_response.error
    model_id = load_response.model_id

    completion_request = CompletionStreamRequest.model_validate(
        {
            "type": "completionStream",
            "modelId": model_id,
            "history": [{"role": "user", "content": "Say hello in five words."}],
            "stream": True,
        }
    )

    text = ""
    async for chunk in completion_stream(worker, completion_request):
        for event in chunk.events:
            if event.type == "contentDelta":
                text += event.text
    assert text.strip(), "expected real completion text via the bare_rpc server-stream"
