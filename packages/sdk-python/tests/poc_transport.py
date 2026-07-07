"""Adapts the hand-written PoC's `QvacWorker` (poc_heartbeat.py) to the
`qvac._transport.Transport` protocol, so the generated stubs in
`qvac._generated.methods` can be smoke-tested against a real running worker
ahead of the production `bare-rpc-python` transport (not yet built).
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator

from poc_heartbeat import QvacWorker


class PocTransport:
    def __init__(self, worker: QvacWorker) -> None:
        self._worker = worker

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._worker.call(payload)

    def call_stream(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        yield from self._worker.call_stream(payload)

    def call_duplex(
        self, payload: dict[str, Any], up: Iterable[bytes]
    ) -> Iterator[dict[str, Any]]:
        yield from self._worker._duplex_call(payload, up)
