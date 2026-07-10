"""QVAC-21806 PoC: wire bare_rpc.RPC (github.com/holepunchto/bare-rpc-python)
to a real spawned SDK worker, replacing poc_heartbeat.py's hand-rolled frame
encode/decode with the real library.

Kept deliberately separate from poc_heartbeat.py/poc_transport.py (used by
PR #3100) rather than replacing them -- this is exploratory, async (bare_rpc
is asyncio-native), and not yet wired into qvac._transport.Transport's sync
protocol. See the PR/commit notes for what's proven here vs. still open.

Wire-level notes:
- Our worker ignores bare-rpc's `command` field entirely and routes purely
  on the JSON payload's own `type` field (see poc_heartbeat.py's comment:
  "command == id: the server ignores `command` and routes on payload.type").
  So every call below passes command=0 -- bare_rpc.RPC's own internal
  monotonic id (not `command`) is what correlates request/response/stream
  frames on the wire, and that's handled entirely inside the library.
- Server-stream replies arrive as newline-delimited JSON, but not
  necessarily one JSON document per wire STREAM-DATA frame (the worker may
  coalesce or split them) -- so chunks from `IncomingStream` still need the
  same buffer-and-split-on-newline handling poc_heartbeat.py's call_stream
  does, just fed from bare_rpc's discrete async-iterated chunks instead of
  raw socket bytes.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator

import bare_rpc

SDK = os.environ.get(
    "QVAC_POC_SDK_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "sdk"),
)
BARE = f"{SDK}/node_modules/bare-runtime-darwin-arm64/bin/bare"
WORKER = f"{SDK}/dist/server/worker.js"


def _json_or_raise(data: bytes) -> Any:
    """Parse a JSON payload; the SDK reports failures in-band as {"type":"error"}."""
    obj = json.loads(data.decode("utf-8"))
    if isinstance(obj, dict) and obj.get("type") == "error":
        raise RuntimeError("worker: " + str(obj.get("message", "unknown error")))
    return obj


class BareRpcWorker:
    """Spawns the real Bare worker and speaks to it via bare_rpc.RPC instead
    of hand-rolled framing. asyncio-native, unlike poc_heartbeat.QvacWorker."""

    def __init__(self) -> None:
        self._sock_path = os.path.join(
            tempfile.gettempdir(), f"qvac-bare-rpc-poc-{os.getpid()}.sock"
        )
        self._log_path = os.path.join(
            tempfile.gettempdir(), f"qvac-bare-rpc-poc-worker-{os.getpid()}.log"
        )
        self._server: asyncio.AbstractServer | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._log_fh = None
        self._writer: asyncio.StreamWriter | None = None
        self._read_task: asyncio.Task | None = None
        self.rpc: bare_rpc.RPC | None = None

    async def start(self) -> "BareRpcWorker":
        if os.path.exists(self._sock_path):
            os.unlink(self._sock_path)

        connected: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        async def on_client(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            self._writer = writer
            self._read_task = asyncio.current_task()
            if not connected.done():
                connected.set_result(None)
            try:
                while True:
                    chunk = await reader.read(65536)
                    if not chunk:
                        break
                    await self.rpc.receive(chunk)
            except asyncio.CancelledError:
                pass

        self._server = await asyncio.start_unix_server(on_client, path=self._sock_path)

        config = json.dumps(
            {
                "QVAC_IPC_SOCKET_PATH": self._sock_path,
                "HOME_DIR": os.path.expanduser("~"),
            }
        )
        self._log_fh = open(self._log_path, "wb")
        self._proc = await asyncio.create_subprocess_exec(
            BARE,
            WORKER,
            config,
            cwd=SDK,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
        )

        def send(frame: bytes) -> None:
            self._writer.write(frame)

        self.rpc = bare_rpc.RPC(send=send)
        await asyncio.wait_for(connected, timeout=30)
        return self

    async def close(self) -> None:
        if self._read_task:
            self._read_task.cancel()
        if self.rpc:
            self.rpc.close()
        if self._proc:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
        if self._writer:
            self._writer.close()
        if self._server:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=5)
            except asyncio.TimeoutError:
                pass
        if self._log_fh:
            self._log_fh.close()
        if os.path.exists(self._sock_path):
            os.unlink(self._sock_path)

    async def __aenter__(self) -> "BareRpcWorker":
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def worker_logs(self) -> str:
        try:
            with open(self._log_path, "r", errors="replace") as f:
                return f.read()
        except OSError:
            return ""

    # ---- the two call shapes exercised so far --------------------------

    async def call(self, payload: dict) -> dict:
        """Unary, via bare_rpc.RPC.request -- no hand-rolled framing at all."""
        data = await self.rpc.request(
            command=0, data=json.dumps(payload).encode("utf-8")
        )
        return _json_or_raise(data)

    async def call_stream(self, payload: dict) -> AsyncIterator[dict]:
        """Server-stream, via bare_rpc.RPC.request_with_response_stream."""
        stream = await self.rpc.request_with_response_stream(
            command=0, data=json.dumps(payload).encode("utf-8")
        )
        buffer = ""
        async for chunk in stream:
            buffer += chunk.decode("utf-8")
            lines = buffer.split("\n")
            buffer = lines.pop()
            for line in lines:
                if line.strip():
                    yield _json_or_raise(line.encode("utf-8"))
        if buffer.strip():
            yield _json_or_raise(buffer.encode("utf-8"))
