#!/usr/bin/env python3
"""
QVAC RPC proof-of-concept — talk to the SDK worker from Python.

Copied verbatim (path constants only made worktree-portable via env vars)
from the hand-written PoC used to design the Python client. Used here as
the test harness for the generated typed surface: `poc_transport.py` adapts
`QvacWorker` below to the `qvac._transport.Transport` protocol so the
generated stubs can be smoke-tested against a real running worker ahead of
the production transport (a separate, not-yet-built task).

A hand-written, minimal stand-in for the eventual generated Python client +
`qvac._transport` module. It shows how a Python program:

  1. starts the SDK's Bare worker and connects over a Unix socket,
  2. sends a `bare-rpc` request and reads the reply  (unary  — heartbeat, loadModel),
  3. reads a `bare-rpc` response stream            (stream — completion tokens).

No HTTP. The worker already runs inference; we just speak its socket protocol.

RUN:
  python3 qvac_poc_heartbeat.py                                   # heartbeat only
  QVAC_POC_MODEL="/path/to/model.gguf" python3 qvac_poc_heartbeat.py   # + loadModel + completion

Wire format (bare-rpc: lib/messages.js, lib/constants.js):
  frame          = uint32(len, little-endian) + body
  body(REQUEST)  = uint(1) uint(id) uint(command) uint(0) uint(len) <json>
  body(RESPONSE) = uint(2) uint(id) bool(err) uint(stream) [<err> | uint(len) <json>]
  body(STREAM)   = uint(3) uint(id) uint(flags) [uint(len) <json> if flags & DATA]
  `uint` is a varint; payloads are UTF-8 JSON. Application errors come back
  in-band as a normal reply {"type":"error","message":...}.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import traceback
import array
import wave

# ============================================================================
# 0. Where the worker and the Bare runtime live
# ============================================================================

SDK = os.environ.get(
    "QVAC_POC_SDK_DIR",
    "/Users/lauri/noxtton/qvac-new/packages/sdk",
)
BARE = f"{SDK}/node_modules/bare-runtime-darwin-arm64/bin/bare"
WORKER = f"{SDK}/dist/server/worker.js"
DEFAULT_AUDIO = (
    f"{SDK}/e2e/assets/audio/transcription-short-wav.wav"  # an SDK e2e fixture
)

REQUEST, RESPONSE, STREAM = 1, 2, 3  # bare-rpc message types

# stream flags (a STREAM frame's `flags` field OR's these together)
S_OPEN, S_CLOSE, S_PAUSE, S_RESUME = 0x1, 0x2, 0x4, 0x8
S_DATA, S_END, S_DESTROY, S_ERROR = 0x10, 0x20, 0x40, 0x80
S_REQUEST, S_RESPONSE = 0x100, 0x200  # which half of the call the stream belongs to

DEBUG = bool(os.environ.get("QVAC_POC_DEBUG"))  # dump raw frame headers


# ============================================================================
# 1. compact-encoding — only the few primitives the envelope uses
# ============================================================================


def enc_uint(n: int) -> bytes:
    if n <= 0xFC:
        return bytes([n])
    if n <= 0xFFFF:
        return b"\xfd" + n.to_bytes(2, "little")
    if n <= 0xFFFFFFFF:
        return b"\xfe" + n.to_bytes(4, "little")
    return b"\xff" + n.to_bytes(8, "little")


def dec_uint(buf: bytes, pos: int):
    a = buf[pos]
    pos += 1
    if a < 0xFD:
        return a, pos
    if a == 0xFD:
        return int.from_bytes(buf[pos : pos + 2], "little"), pos + 2
    if a == 0xFE:
        return int.from_bytes(buf[pos : pos + 4], "little"), pos + 4
    return int.from_bytes(buf[pos : pos + 8], "little"), pos + 8


def dec_utf8(buf: bytes, pos: int):
    n, pos = dec_uint(buf, pos)
    return buf[pos : pos + n].decode("utf-8"), pos + n


def dec_int(buf: bytes, pos: int):
    u, pos = dec_uint(buf, pos)
    return (u >> 1) ^ -(u & 1), pos  # zigzag


def _short(s: str, n: int = 400) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + " …[truncated]"


# ============================================================================
# 2. Minimal bare-rpc client over a Unix socket
#    The worker connects back to *us*, so we are the socket server.
# ============================================================================


class QvacWorker:
    def __init__(self):
        self._sock_path = os.path.join(
            tempfile.gettempdir(), f"qvac-poc-{os.getpid()}.sock"
        )
        self._server = None
        self._conn = None
        self._proc = None
        self._id = 0
        self._rx = b""  # bytes read past a frame boundary
        self._log_path = os.path.join(
            tempfile.gettempdir(), f"qvac-poc-worker-{os.getpid()}.log"
        )
        self._log_fh = None

    # ---- lifecycle -------------------------------------------------------

    def start(self):
        if os.path.exists(self._sock_path):
            os.unlink(self._sock_path)
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self._sock_path)
        self._server.listen(1)
        self._server.settimeout(30)

        # The worker reads its socket path (+ home dir) from a JSON arg,
        # parsed as Bare.argv[2] (server/env.ts).
        config = json.dumps(
            {
                "QVAC_IPC_SOCKET_PATH": self._sock_path,
                "HOME_DIR": os.path.expanduser("~"),
            }
        )
        # worker stdout+stderr -> a file, so reading its logs never blocks on a live pipe
        self._log_fh = open(self._log_path, "wb")
        self._proc = subprocess.Popen(
            [BARE, WORKER, config],
            cwd=SDK,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
        )
        self._conn, _ = self._server.accept()  # worker dials back in
        self._conn.settimeout(180)  # model load + generation can take a while
        return self

    def close(self):
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._conn:
            self._conn.close()
        if self._server:
            self._server.close()
        if self._log_fh:
            self._log_fh.close()
        if os.path.exists(self._sock_path):
            os.unlink(self._sock_path)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    def worker_logs(self) -> str:
        try:
            with open(self._log_path, "r", errors="replace") as f:
                return f.read()
        except OSError:
            return ""

    # ---- framing ---------------------------------------------------------

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _send_frame(self, body: bytes):
        self._conn.sendall(len(body).to_bytes(4, "little") + body)

    def _send_request(self, payload: dict) -> int:
        req_id = self._next_id()
        data = json.dumps(payload).encode("utf-8")
        # command == id: the server ignores `command` and routes on payload.type
        body = (
            enc_uint(REQUEST)
            + enc_uint(req_id)
            + enc_uint(req_id)
            + enc_uint(0)
            + enc_uint(len(data))
            + data
        )
        self._send_frame(body)
        return req_id

    def _send_stream_ctrl(self, req_id: int, flags: int):
        self._send_frame(enc_uint(STREAM) + enc_uint(req_id) + enc_uint(flags))

    def _send_request_open(self, req_id: int):
        # open the client->server request stream: a REQUEST msg with stream=OPEN
        # and no data (mirrors bare-rpc OutgoingStream._open for a REQUEST stream).
        self._send_frame(
            enc_uint(REQUEST) + enc_uint(req_id) + enc_uint(req_id) + enc_uint(S_OPEN)
        )

    def _send_stream_data(self, req_id: int, flags: int, data: bytes):
        self._send_frame(
            enc_uint(STREAM)
            + enc_uint(req_id)
            + enc_uint(flags)
            + enc_uint(len(data))
            + data
        )

    def _recv(self, n: int) -> bytes:
        while len(self._rx) < n:
            chunk = self._conn.recv(65536)
            if not chunk:
                raise RuntimeError("worker closed the socket")
            self._rx += chunk
        out, self._rx = self._rx[:n], self._rx[n:]
        return out

    def _read_message(self) -> dict:
        frame_len = int.from_bytes(self._recv(4), "little")
        body = self._recv(frame_len)
        if DEBUG:
            print(f"[frame] len={frame_len} head={body[:24].hex()}", file=sys.stderr)
        mtype, pos = dec_uint(body, 0)
        msg_id, pos = dec_uint(body, pos)

        if mtype == RESPONSE:
            err = body[pos]
            pos += 1  # bool: 1 byte
            stream, pos = dec_uint(body, pos)
            if err:
                raise RuntimeError(self._decode_error(body, pos))
            # stream == 0 -> a unary reply carrying data.
            # stream != 0 -> a "the response is a stream" marker, no data; the
            #               actual payloads arrive as STREAM frames after this.
            data = None
            if stream == 0:
                data_len, pos = dec_uint(body, pos)
                data = body[pos : pos + data_len]
            return {"kind": "response", "id": msg_id, "stream": stream, "data": data}

        if mtype == STREAM:
            flags, pos = dec_uint(body, pos)
            if flags & S_ERROR:
                raise RuntimeError(self._decode_error(body, pos))
            data = None
            if flags & S_DATA:
                data_len, pos = dec_uint(body, pos)
                data = body[pos : pos + data_len]
            return {"kind": "stream", "id": msg_id, "flags": flags, "data": data}

        return {
            "kind": "request",
            "id": msg_id,
        }  # server->client callback (phase-2 tools)

    @staticmethod
    def _decode_error(body: bytes, pos: int) -> str:
        msg, pos = dec_utf8(body, pos)
        code, pos = dec_utf8(body, pos)
        errno, pos = dec_int(body, pos)
        return f"bare-rpc error frame: {msg} (code={code!r} errno={errno})"

    @staticmethod
    def _json_or_raise(data: bytes):
        """Parse a JSON payload; the SDK reports failures in-band as {"type":"error"}."""
        obj = json.loads(data.decode("utf-8"))
        if isinstance(obj, dict) and obj.get("type") == "error":
            raise RuntimeError("worker: " + _short(obj.get("message", "unknown error")))
        return obj

    # ---- the two call shapes -----------------------------------------------

    def call(self, payload: dict) -> dict:
        """Unary: send a request, wait for its single reply, return parsed JSON."""
        req_id = self._send_request(payload)
        while True:
            msg = self._read_message()
            if msg["kind"] == "response" and msg["id"] == req_id:
                return self._json_or_raise(msg["data"])

    def call_stream(self, payload: dict):
        """Server-stream: send the request, OPEN + RESUME the response stream, then
        yield the newline-delimited JSON objects the worker pushes, until it ENDs."""
        req_id = self._send_request(payload)
        self._send_stream_ctrl(req_id, S_RESPONSE | S_OPEN)
        self._send_stream_ctrl(req_id, S_RESPONSE | S_RESUME)

        buffer = ""
        while True:
            msg = self._read_message()
            if msg["id"] != req_id:
                continue
            if msg["kind"] == "response":
                if msg["stream"] == 0:  # a real unary-style terminal reply
                    if msg["data"]:
                        yield self._json_or_raise(msg["data"])
                    return
                continue  # "response is a stream" marker; DATA frames follow
            if msg["kind"] != "stream":
                continue
            if msg["data"]:
                buffer += msg["data"].decode("utf-8")
                lines = buffer.split("\n")
                buffer = lines.pop()  # keep the partial line
                for line in lines:
                    if line.strip():
                        yield self._json_or_raise(line.encode("utf-8"))
            if msg["flags"] & (S_END | S_CLOSE):
                if buffer.strip():
                    yield self._json_or_raise(buffer.encode("utf-8"))
                return

    # ---- high-level convenience (what a generated client would expose) -----

    def heartbeat(self) -> dict:
        return self.call({"type": "heartbeat"})

    def load_model(
        self, model_src, model_type="llamacpp-completion", model_config=None
    ) -> str:
        # `modelType` is the CANONICAL engine type (the "llm" alias is normalized
        # client-side before the wire). Returns the loaded model's id.
        req = {"type": "loadModel", "modelSrc": model_src, "modelType": model_type}
        if model_config:
            req["modelConfig"] = model_config
        result = self.call(req)
        if not result.get("success"):
            raise RuntimeError(f"loadModel failed: {result.get('error')}")
        return result["modelId"]

    def completion(self, model_id, messages, **generation_params):
        # stream-type method; yields raw completionStream events
        # ({"type":"completionStream","events":[...],"done":bool}).
        req = {
            "type": "completionStream",
            "modelId": model_id,
            "history": messages,
            "stream": True,
        }
        if generation_params:
            req["generationParams"] = generation_params
        yield from self.call_stream(req)

    def embed(self, model_id, text):
        # reply-type method; returns the embedding vector (a single string ->
        # number[]; a list of strings -> number[][]).
        result = self.call({"type": "embed", "modelId": model_id, "text": text})
        if not result.get("success"):
            raise RuntimeError(f"embed failed: {result.get('error')}")
        return result["embedding"]

    def transcribe(self, model_id, audio_path):
        # stream-type method: yields transcribe events ({text, segment, done, ...}).
        # audioChunk accepts a local file path directly (no base64 needed).
        req = {
            "type": "transcribe",
            "modelId": model_id,
            "audioChunk": {"type": "filePath", "value": audio_path},
        }
        yield from self.call_stream(req)

    def _duplex_call(self, payload_obj, up_chunks):
        # DUPLEX: open a client->server request stream (first chunk = the JSON
        # payload, then `up_chunks`), open the server->client response stream, and
        # yield the response events. Both halves share one req id; REQUEST-masked
        # STREAM frames go up, RESPONSE-masked frames come down.
        req_id = self._next_id()
        payload = json.dumps(payload_obj).encode("utf-8")
        self._send_request_open(req_id)  # open client->server stream
        self._send_stream_ctrl(
            req_id, S_RESPONSE | S_OPEN
        )  # open server->client stream
        self._send_stream_ctrl(req_id, S_RESPONSE | S_RESUME)
        self._send_stream_data(
            req_id, S_REQUEST | S_DATA, payload
        )  # 1st chunk = request JSON
        for chunk in up_chunks:
            self._send_stream_data(req_id, S_REQUEST | S_DATA, chunk)
        self._send_stream_ctrl(req_id, S_REQUEST | S_END)  # done sending

        buffer = ""
        while True:
            msg = self._read_message()
            if msg["id"] != req_id:
                continue
            if msg["kind"] == "response":
                if msg["stream"] == 0 and msg["data"]:
                    yield self._json_or_raise(msg["data"])
                    return
                continue  # response-stream-open marker
            if msg["kind"] != "stream":
                continue
            if msg["data"]:
                buffer += msg["data"].decode("utf-8")
                lines = buffer.split("\n")
                buffer = lines.pop()
                for line in lines:
                    if line.strip():
                        yield self._json_or_raise(line.encode("utf-8"))
            if msg["flags"] & (S_END | S_CLOSE):
                if buffer.strip():
                    yield self._json_or_raise(buffer.encode("utf-8"))
                return

    def transcribe_stream(self, model_id, pcm_chunks, parakeet_config=None):
        payload = {"type": "transcribeStream", "modelId": model_id}
        if parakeet_config:
            payload["parakeetStreamingConfig"] = parakeet_config
        yield from self._duplex_call(
            payload, pcm_chunks
        )  # audio chunks up, transcripts down

    def text_to_speech_stream(self, model_id, text):
        # duplex TTS: the text to speak is streamed up the request stream; audio
        # ({buffer:number[]}) comes back down the response stream.
        payload = {"type": "textToSpeechStream", "modelId": model_id}
        yield from self._duplex_call(payload, [text.encode("utf-8")])


# ============================================================================
# 3. Demo
# ============================================================================


def _dump_failure(w, label, e):
    print(f"\n[poc] {label} failed: {e}", file=sys.stderr)
    traceback.print_exc()
    logs = w.worker_logs()
    if logs.strip():
        print("---- worker logs (tail) ----\n" + _short(logs, 2000), file=sys.stderr)


def demo_completion(w, model):
    print(f"[loadModel] loading LLM {model} ...")
    model_id = w.load_model(model)  # canonical llamacpp-completion
    print(f"[loadModel] -> modelId={model_id!r}\n")
    print("[completion] streaming 'Say hello in five words.':")
    text = ""
    for resp in w.completion(
        model_id, [{"role": "user", "content": "Say hello in five words."}]
    ):
        for e in resp.get("events", []):
            if e.get("type") == "contentDelta":
                text += e["text"]
                sys.stdout.write(e["text"])
                sys.stdout.flush()
    print(f"\n[completion] full text -> {text!r}")


def demo_embed(w, model):
    print(f"[loadModel] loading embedding model {model} ...")
    model_id = w.load_model(model, model_type="llamacpp-embedding")
    print(f"[loadModel] -> modelId={model_id!r}")
    vec = w.embed(model_id, "hello world")
    print(
        f"[embed] 'hello world' -> dim={len(vec)}, first 5={[round(x, 4) for x in vec[:5]]}"
    )


def demo_transcribe(w, model):
    audio = os.environ.get("QVAC_POC_AUDIO", DEFAULT_AUDIO)
    print(f"[loadModel] loading transcription model {model} ...")
    model_id = w.load_model(model, model_type="parakeet-transcription")
    print(f"[loadModel] -> modelId={model_id!r}")
    print(f"[transcribe] {audio}:")
    text = ""
    for resp in w.transcribe(model_id, audio):
        if resp.get("text"):
            text += resp["text"]
    print(f"[transcribe] -> {text!r}")


def _wav_to_pcm_16k_mono(path, chunk_ms=100, fmt=None):
    """Decode a 16-bit PCM wav to 16 kHz mono, as int16 or float32 (QVAC_POC_PCM)."""
    fmt = fmt or os.environ.get("QVAC_POC_PCM", "i16")
    with wave.open(path, "rb") as wf:
        ch, width, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise RuntimeError(f"PoC expects 16-bit PCM wav (sampwidth=2), got {width}")
    samples = array.array("h")
    samples.frombytes(raw)
    mono = array.array("h", samples[0::ch]) if ch > 1 else samples  # take first channel
    target = 16000
    if rate != target and rate % target == 0:
        mono = array.array("h", mono[0 :: (rate // target)])  # crude decimation
    if fmt == "f32":
        data = array.array("f", (s / 32768.0 for s in mono)).tobytes()
        sample_bytes = 4
    else:  # int16 (default)
        data = mono.tobytes()
        sample_bytes = 2
    per_chunk = int(target * chunk_ms / 1000) * sample_bytes
    return (
        [data[i : i + per_chunk] for i in range(0, len(data), per_chunk)],
        target,
        fmt,
    )


def demo_transcribe_stream(w, model):
    # parakeet duplex needs: a TRUE 16 kHz mono f32le stream (resample non-16k with
    # ffmpeg first), 1 s chunks, `emitPartials` so it emits per-chunk text, and
    # ~1.5 s of trailing silence so the stream finalizes.
    audio = os.environ.get("QVAC_POC_AUDIO", DEFAULT_AUDIO)
    chunk_ms = 1000
    chunks, rate, fmt = _wav_to_pcm_16k_mono(audio, chunk_ms=chunk_ms, fmt="f32")
    per_chunk = int(rate * chunk_ms / 1000) * 4
    silence = bytes(int(rate * 1.5) * 4)
    chunks += [silence[i : i + per_chunk] for i in range(0, len(silence), per_chunk)]
    cfg = {"chunkMs": chunk_ms, "emitPartials": True}
    print(f"[loadModel] loading transcription model {model} ...")
    model_id = w.load_model(model, model_type="parakeet-transcription")
    print(f"[loadModel] -> modelId={model_id!r}")
    print(
        f"[transcribeStream] DUPLEX: {rate}Hz mono {fmt}, {len(chunks)} chunks, config={cfg}:"
    )
    text = ""
    for resp in w.transcribe_stream(model_id, chunks, parakeet_config=cfg):
        if DEBUG:
            print(f"[event] {resp}", file=sys.stderr)
        piece = resp.get("text") or (resp.get("segment") or {}).get("text")
        if piece:
            text += piece
            sys.stdout.write(piece)
            sys.stdout.flush()
    print(f"\n[transcribeStream] -> {text!r}")


def _write_wav(path, samples, rate):
    peak = max((abs(s) for s in samples), default=0)
    if peak <= 4:  # float [-1, 1]
        ints = array.array(
            "h", (max(-32768, min(32767, int(s * 32767))) for s in samples)
        )
    else:  # already ~int16 scale
        ints = array.array("h", (max(-32768, min(32767, int(s))) for s in samples))
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(ints.tobytes())


def demo_tts_stream(w, model):
    text = os.environ.get(
        "QVAC_POC_TTS_TEXT", "Hello from QVAC. This is streaming text to speech."
    )
    print(f"[loadModel] loading TTS model {model} ...")
    model_id = w.load_model(
        model,
        model_type="tts-ggml",
        model_config={"ttsEngine": "supertonic", "language": "en"},
    )
    print(f"[loadModel] -> modelId={model_id!r}")
    print(f"[textToSpeechStream] DUPLEX: synthesizing {text!r}")
    samples, rate, events = [], None, 0
    for resp in w.text_to_speech_stream(model_id, text):
        events += 1
        if DEBUG:
            print(
                f"[event] keys={list(resp)} buf={len(resp.get('buffer', []))} done={resp.get('done')}",
                file=sys.stderr,
            )
        samples.extend(resp.get("buffer", []))
        st = resp.get("stats") or {}
        if st.get("totalSamples") and st.get("audioDuration"):
            cand = st["totalSamples"] / (
                st["audioDuration"] / 1000
            )  # audioDuration is ms
            if 8000 <= cand <= 96000:
                rate = round(cand)
    rate = rate or 44100
    print(
        f"[textToSpeechStream] -> {events} events, {len(samples)} audio samples "
        f"(~{len(samples) / rate:.2f}s @ {rate}Hz)"
    )
    if samples:
        out = os.path.join(tempfile.gettempdir(), "qvac-poc-tts.wav")
        _write_wav(out, samples, rate)
        print(f"[textToSpeechStream] wrote {out} — play it to verify real speech")


# capability -> (env var pointing at a model, demo fn). Add cases here one at a time.
CASES = [
    ("completion", "QVAC_POC_MODEL", demo_completion),
    ("embed", "QVAC_POC_EMBED_MODEL", demo_embed),
    ("transcribe", "QVAC_POC_STT_MODEL", demo_transcribe),
    ("transcribe-stream", "QVAC_POC_STT_STREAM_MODEL", demo_transcribe_stream),
    ("tts-stream", "QVAC_POC_TTS_MODEL", demo_tts_stream),
]


def main():
    with QvacWorker() as w:
        print("[poc] worker connected\n")
        print(f"[heartbeat] -> {w.heartbeat()}\n")

        ran = False
        for label, env_var, fn in CASES:
            model = os.environ.get(env_var)
            if not model:
                continue
            ran = True
            try:
                fn(w, model)
                print()
            except Exception as e:
                _dump_failure(w, label, e)
                raise
        if not ran:
            print(
                "[poc] set QVAC_POC_MODEL (LLM) / QVAC_POC_EMBED_MODEL (embeddings) to run cases."
            )


if __name__ == "__main__":
    main()
