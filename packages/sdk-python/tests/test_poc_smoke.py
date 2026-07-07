"""Smoke-tests the generated typed surface against a real running SDK worker,
via the hand-written PoC transport (poc_heartbeat.py / poc_transport.py) —
the production socket transport isn't built yet.

Needs the SDK's Bare worker built (`bun run build` in packages/sdk) and the
Bare runtime prebuild available; skipped unless QVAC_POC_SDK_DIR points at a
built SDK checkout, so it never blocks a normal `pytest` run or CI.

Only exercises request-reply methods that need no loaded model (`heartbeat`,
`state`) — server-stream/duplex methods need a downloaded model, which this
environment doesn't have.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent / "src"))

pytestmark = pytest.mark.skipif(
    "QVAC_POC_SDK_DIR" not in os.environ,
    reason="set QVAC_POC_SDK_DIR to a built SDK checkout to run the PoC smoke test",
)


@pytest.fixture
def worker():
    from poc_heartbeat import QvacWorker

    with QvacWorker() as w:
        yield w


@pytest.fixture
def transport(worker):
    from poc_transport import PocTransport

    return PocTransport(worker)


def test_heartbeat_reply_round_trips_through_generated_stub(transport) -> None:
    from qvac._generated import HeartbeatRequest
    from qvac._generated.methods import heartbeat

    response = heartbeat(transport, HeartbeatRequest(type="heartbeat"))
    assert response.type == "heartbeat"
    assert isinstance(response.number, float)


def test_state_reply_round_trips_through_generated_stub(transport) -> None:
    from qvac._generated import StateRequest
    from qvac._generated.methods import state

    response = state(transport, StateRequest(type="state"))
    assert response.type == "state"
    assert response.state is not None
