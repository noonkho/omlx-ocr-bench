#!/usr/bin/env python3
"""Three-panel Unlimited-OCR bench. `uv run server.py` then open http://127.0.0.1:4700

Left: server, prompt, sampling knobs and a drop zone. Middle: the document as it really is.
Right: what the model returned — raw, rendered, or drawn back onto the page as layout boxes.

Everything goes through this process rather than straight from the browser, because the oMLX box
sets no CORS headers and a browser would refuse the call. The `<|det|>` parsing also lives here
and not in the page, so there is exactly one parser and the live view cannot disagree with the
finished one.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import config
import ocr
from ocr import COLOURS, JUNK, build_body, data_url, http_detail, parse, render_pages, request

HERE = Path(__file__).parent
JOBS: dict[str, dict] = {}
#: A run's settings, POSTed and then quoted back by the `EventSource`. `EventSource` cannot set
#: headers, so without this the API key would ride in a URL — into the browser's history, and
#: into the access log of any proxy that ever sits in front of this.
RUNS: dict[str, dict] = {}
PORT = 4700
JOB_TTL = 3600

#: Sampling knobs passed straight through to `/v1/chat/completions`, and how to read them.
#:
#: oMLX accepts every one of these (measured — none returns 400). Two caveats worth knowing:
#: `no_repeat_ngram_size` is accepted and then **ignored** (identical output with and without it,
#: see README), and the context window is not a request field at all — oMLX takes the model's own
#: `max_model_len`, so the bench only shows it.
KNOBS: dict[str, type] = {
    "temperature": float, "top_p": float, "min_p": float, "repetition_penalty": float,
    "presence_penalty": float, "frequency_penalty": float,
    "max_tokens": int, "top_k": int, "seed": int, "no_repeat_ngram_size": int,
}

#: What the page starts with, and what `ocr_page` uses, from one place.
DEFAULT_KNOBS = {"max_tokens": 4096, "temperature": 0,
                 "frequency_penalty": config.FREQUENCY_PENALTY}

#: How often a streaming answer is pushed to the browser. One frame per token would cost a JSON
#: encode, an HTTP chunk and a flush each; at 60 ms the answer still appears to stream.
FLUSH_SECONDS = 0.06
FLUSH_CHARS = 64


class Stopped(Exception):
    """Raised to abandon a streaming response early — the page ran away, or the browser left."""


class RepeatWatch:
    """Is the answer turning into the same line over and over?

    Measured behaviour, not a guess — see README. On a near-empty page this model emits one block,
    or `1. 2. 3. 4. …`, until it hits the token cap. The test is cheap and incremental: count how
    many completed lines are exact repeats of a line already seen. A real document repeats words;
    it does not repeat whole LINES. `<|det|>` markers are stripped first, because their
    coordinates differ on every repeat even when the text they wrap is identical.
    """

    # Deliberately conservative. A real form legitimately repeats lines — twenty rows of
    # "HK$0.00", a signature block per party — so the bar is a third distinct over a long
    # answer, which a document does not reach by accident.
    MIN_LINES = 30
    MAX_REPEATS = 0.34          # distinct / total, below which it is not a document

    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.total = 0
        self._pending = ""
        self.tripped = False

    def feed(self, piece: str) -> bool:
        """Add the next piece of the answer. Returns True once it has clearly run away."""
        self._pending += piece
        if "\n" not in piece:
            return self.tripped
        *done, self._pending = self._pending.split("\n")
        for line in done:
            line = ocr.strip_markers(line).strip()
            if not line:
                continue
            self.total += 1
            self.seen.add(line)
        if self.total >= self.MIN_LINES and len(self.seen) / self.total < self.MAX_REPEATS:
            self.tripped = True
        return self.tripped


def stream_call(base_url: str, api_key: str, body: dict, on_delta,
                timeout: float = 1800) -> tuple[str, dict, str]:
    """Stream one completion. `on_delta(piece, text_so_far)` may raise `Stopped` to cut it short.

    Returns (the text, usage, finish_reason). A `Stopped` is not an error — the caller asked.
    """
    body = dict(body, stream=True, stream_options={"include_usage": True})
    parts: list[str] = []
    usage, finish = {}, ""
    with urllib.request.urlopen(request(base_url, api_key, body), timeout=timeout) as r:
        try:
            for line in r:
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content") or ""
                    if piece:
                        parts.append(piece)
                        on_delta(piece, parts)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        except Stopped:
            finish = finish or "aborted_runaway"
    return "".join(parts), usage, finish


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):            # quiet; the browser is the log
        pass

    # ---------------------------------------------------------------- helpers
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # ---------------------------------------------------------------- routes
    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(200, (HERE / "ui.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/defaults":
            # Everything the page would otherwise have to hardcode: the server, the measured
            # knob defaults, and the block taxonomy the parser uses.
            self._json(200, {"base_url": config.BASE_URL, "api_key": config.API_KEY,
                             "model": config.MODEL, "prompt": config.PROMPT, "dpi": config.DPI,
                             "knobs": DEFAULT_KNOBS, "colours": COLOURS, "junk": sorted(JUNK)})
        elif url.path == "/api/samples":
            # The bundled synthetic pages, so the bench demonstrates itself without the user
            # having to find a document first. Synthetic on purpose: no real documents, ever.
            self._json(200, {"ok": True, "samples": [
                {"name": f.name, "label": label} for f, label in (
                    (HERE / "samples" / "synthetic_hk.pdf", "HK writ, 2 pages, EN + a chop"),
                    (HERE / "samples" / "synthetic_zh.png", "Traditional Chinese notice, 1 page"),
                ) if f.exists()]})
        elif url.path == "/api/sample":
            name = (parse_qs(url.query).get("name") or [""])[0]
            path = (HERE / "samples" / Path(name).name)
            if not path.exists() or path.parent != HERE / "samples":
                self._json(404, {"ok": False, "error": "no such sample"})
                return
            self._json(200, {"ok": True, "name": path.name,
                             "data": base64.b64encode(path.read_bytes()).decode()})
        elif url.path == "/api/stream":
            self.stream(parse_qs(url.query))
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        url = urlparse(self.path)
        prune()
        if url.path == "/api/models":
            self.list_models()
        elif url.path == "/api/upload":
            self.upload()
        elif url.path == "/api/run":
            # Park this run's settings and hand back a ticket. See `RUNS`.
            ticket = uuid.uuid4().hex
            RUNS[ticket] = dict(self._read_json(), at=time.time())
            self._json(200, {"ok": True, "run": ticket})
        else:
            self._send(404, b"not found", "text/plain")

    # ---------------------------------------------------------------- actions
    def list_models(self) -> None:
        """Connection test. Names every model the server has, so the right one can be picked."""
        body = self._read_json()
        base = (body.get("base_url") or config.BASE_URL).rstrip("/")
        key = body.get("api_key") or config.API_KEY
        req = urllib.request.Request(base + "/models",
                                     headers={"Authorization": f"Bearer {key}"})
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read())
            models = [{"id": m.get("id"), "max_model_len": m.get("max_model_len")}
                      for m in (data.get("data") or []) if m.get("id")]
            self._json(200, {"ok": True, "models": models,
                             "ms": round((time.time() - t0) * 1000)})
        except Exception as exc:                                    # noqa: BLE001
            self._json(200, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def upload(self) -> None:
        """A PDF or image arrives as base64. Rendered here; the pages stay in memory.

        The `data:` URL is built once and kept, because both the page and every OCR request need
        it and base64 of a 300 dpi render is not free.
        """
        body = self._read_json()
        raw = base64.b64decode((body.get("data") or "").split(",")[-1])
        name = body.get("name") or "document"
        dpi = int(body.get("dpi") or config.DPI)
        tmp = HERE / "out" / f".upload-{uuid.uuid4().hex}{Path(name).suffix or '.pdf'}"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(raw)
        try:
            rendered = render_pages(tmp, dpi)
        except Exception as exc:                                    # noqa: BLE001
            self._json(200, {"ok": False, "error": f"could not render: {exc}"})
            return
        finally:
            tmp.unlink(missing_ok=True)

        pages = [{"index": i, "w": size[0], "h": size[1], "url": data_url(png)}
                 for i, (png, size) in enumerate(rendered)]
        job = uuid.uuid4().hex
        JOBS[job] = {"name": name, "pages": pages, "dpi": dpi, "at": time.time()}
        self._json(200, {"ok": True, "job": job, "name": name, "dpi": dpi, "pages": pages})

    # ---------------------------------------------------------------- the run
    def stream(self, q: dict) -> None:
        """Server-sent events: `delta` as the answer arrives, then one `page` per finished unit."""
        settings = RUNS.pop((q.get("run") or [""])[0], None)
        if settings is None:
            self._json(404, {"ok": False, "error": "unknown run — press Run OCR again"})
            return
        meta = JOBS.get(settings.get("job") or "")
        if not meta:
            self._json(404, {"ok": False, "error": "unknown job — upload again"})
            return

        def one(name: str, fallback):
            value = settings.get(name)
            return fallback if value in (None, "") else value

        prompt = one("prompt", config.PROMPT)
        model = one("model", config.MODEL)
        base = one("base_url", config.BASE_URL)
        key = one("api_key", config.API_KEY)
        guard = str(one("guard", "1")) == "1"
        whole = str(one("mode", "page")) == "whole"

        knobs = dict(DEFAULT_KNOBS)
        for name, cast in KNOBS.items():
            raw = one(name, "")
            if raw != "":
                knobs[name] = cast(float(raw))


        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(event: str, payload: dict) -> bool:
            try:
                self.wfile.write(f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()
                return True
            except Exception:                                       # browser went away
                return False

        # One request per page, or the whole document in one request. The second is what the
        # model is built for and what oMLX cannot yet do: only the FIRST image reaches the model,
        # so the answer covers page 1 alone. Kept in the bench so the bug can be reproduced.
        urls = [p["url"] for p in meta["pages"]]
        groups = [urls] if whole else [[u] for u in urls]
        emit("start", {"pages": len(urls), "requests": len(groups), "model": model,
                       "prompt": prompt, "mode": "whole" if whole else "page", "knobs": knobs})

        run_t0 = time.time()
        for i, group in enumerate(groups):
            if not emit("page_start", {"index": i, "images": len(group)}):
                return
            if not self.run_one(i, group, emit, base, key, model, prompt, knobs, guard):
                return
        emit("done", {"seconds": round(time.time() - run_t0, 2)})

    def run_one(self, index: int, images: list[str], emit, base: str, key: str, model: str,
                prompt: str, knobs: dict, guard: bool) -> bool:
        """One request, streamed. Returns False if the browser has gone and the run should stop."""
        watch = RepeatWatch()
        pulse = {"at": 0.0, "chars": 0, "closed": False}
        t0 = time.time()

        def flush(parts: list[str], force: bool = False) -> None:
            """Push what has arrived since the last push, with blocks when a new one closed."""
            if not force and pulse["chars"] < FLUSH_CHARS and time.time() - pulse["at"] < FLUSH_SECONDS:
                return
            text = "".join(parts)
            payload = {"index": index, "text": text}
            if pulse["closed"]:
                # Only re-parse when a `<|det|>` actually closed; otherwise the blocks cannot
                # have changed and the parse would be wasted work on every frame.
                payload["blocks"] = as_json(parse(text))
                pulse["closed"] = False
            pulse["at"], pulse["chars"] = time.time(), 0
            if not emit("delta", payload):
                raise Stopped                       # the browser closed the tab

        def on_delta(piece: str, parts: list[str]) -> None:
            pulse["chars"] += len(piece)
            pulse["closed"] = pulse["closed"] or "<|/det|>" in piece
            # Always watch, so a runaway is reported even when the guard is off; only STOP
            # the answer when the guard is on.
            if watch.feed(piece) and guard:
                raise Stopped                       # cut it short rather than pay for the cap
            flush(parts)

        text, usage, finish, err = "", {}, "", ""
        try:
            text, usage, finish = stream_call(base, key, build_body(model, prompt, images, knobs),
                                              on_delta)
        except Exception as exc:                                    # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}{http_detail(exc)}"

        runaway = watch.tripped
        if runaway and guard:
            # See README: a near-empty page makes this model repeat until the cap. The official
            # pipeline suppresses it with `no_repeat_ngram_size=35`, which oMLX accepts and
            # ignores — so a caller can only notice and stop paying. What HAS arrived is kept
            # and flagged: the guard exists to save tokens, not to hide the page from the user.
            finish = finish or "aborted_runaway"
        blocks = parse(text) if text else []
        return emit("page", {
            "index": index, "seconds": round(time.time() - t0, 2), "error": err, "finish": finish,
            "runaway": runaway, "raw": text, "usage": usage, "images": len(images),
            "blocks": as_json(blocks)})


def prune(now: float | None = None) -> None:
    """Drop stale uploads and unclaimed tickets. Called on every POST, so an idle bench that has
    finished with a 100-page render does not hold it for the rest of the day."""
    now = now or time.time()
    for store in (JOBS, RUNS):
        for key, meta in list(store.items()):
            if now - meta.get("at", 0) > JOB_TTL:
                store.pop(key, None)


def as_json(blocks) -> list[dict]:
    return [{"label": b.label, "box": list(b.box), "content": b.content} for b in blocks]


def main() -> int:
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Unlimited-OCR bench  ->  http://127.0.0.1:{PORT}")
    print(f"  default server: {config.BASE_URL}  model: {config.MODEL}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
