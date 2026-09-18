#!/usr/bin/env python3
"""Three-panel Unlimited-OCR bench. `uv run server.py` then open http://127.0.0.1:4700

Left: server settings, the prompt, and a drop zone. Middle: what the model returned. Right: the
page with the model's own layout boxes drawn on it.

Everything goes through this process rather than straight from the browser, because the oMLX box
sets no CORS headers and a browser would refuse the call.
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import config
from ocr import parse, render_pages

HERE = Path(__file__).parent
JOBS: dict[str, dict] = {}
PORT = 4700


def _post(base_url: str, api_key: str, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):            # quiet; the browser is the log
        pass

    # ---------------------------------------------------------------- helpers
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
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
            self._json(200, {"base_url": config.BASE_URL, "api_key": config.API_KEY,
                             "model": config.MODEL, "prompt": config.PROMPT, "dpi": config.DPI})
        elif url.path == "/api/samples":
            # The bundled synthetic pages, so the bench demonstrates itself without the user
            # having to find a document first. Synthetic on purpose: no case data, ever.
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
        if url.path == "/api/models":
            self.list_models()
        elif url.path == "/api/upload":
            self.upload()
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
            ids = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
            self._json(200, {"ok": True, "models": ids, "ms": round((time.time() - t0) * 1000)})
        except Exception as exc:                                    # noqa: BLE001
            self._json(200, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def upload(self) -> None:
        """A PDF or image arrives as base64. Rendered here; the pages stay in memory."""
        body = self._read_json()
        raw = base64.b64decode((body.get("data") or "").split(",")[-1])
        name = body.get("name") or "document"
        dpi = int(body.get("dpi") or config.DPI)
        tmp = HERE / "out" / f".upload-{uuid.uuid4().hex}{Path(name).suffix or '.pdf'}"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(raw)
        try:
            pages = render_pages(tmp, dpi)
        except Exception as exc:                                    # noqa: BLE001
            self._json(200, {"ok": False, "error": f"could not render: {exc}"})
            return
        finally:
            tmp.unlink(missing_ok=True)

        job = uuid.uuid4().hex
        JOBS[job] = {"name": name, "pages": pages, "dpi": dpi, "at": time.time()}
        for old, meta in list(JOBS.items()):                        # one hour is plenty
            if time.time() - meta["at"] > 3600:
                JOBS.pop(old, None)
        self._json(200, {"ok": True, "job": job, "name": name, "dpi": dpi,
                         "pages": [{"index": i, "w": s[0], "h": s[1],
                                    "png": base64.b64encode(p).decode()}
                                   for i, (p, s) in enumerate(pages)]})

    def stream(self, q: dict) -> None:
        """Server-sent events: one `page` event per page, as each finishes."""
        job = (q.get("job") or [""])[0]
        meta = JOBS.get(job)
        if not meta:
            self._json(404, {"ok": False, "error": "unknown job — upload again"})
            return
        prompt = (q.get("prompt") or [config.PROMPT])[0]
        model = (q.get("model") or [config.MODEL])[0]
        base = (q.get("base_url") or [config.BASE_URL])[0]
        key = (q.get("api_key") or [config.API_KEY])[0]
        max_tokens = int((q.get("max_tokens") or ["4096"])[0])
        guard = (q.get("guard") or ["1"])[0] == "1"

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
            except Exception:
                return False                                        # browser went away

        pages = meta["pages"]
        emit("start", {"pages": len(pages), "model": model, "prompt": prompt})
        run_t0 = time.time()
        for i, (png, size) in enumerate(pages):
            if not emit("page_start", {"index": i}):
                return
            body = {"model": model, "max_tokens": max_tokens, "temperature": 0,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": "data:image/png;base64," + base64.b64encode(png).decode()}}]}]}
            t0 = time.time()
            try:
                data = _post(base, key, body)
                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage") or {}
                finish = (data["choices"][0].get("finish_reason") or "")
                err = ""
            except Exception as exc:                                # noqa: BLE001
                text, usage, finish, err = "", {}, "", f"{type(exc).__name__}: {exc}"
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        err += " " + exc.read()[:200].decode("utf-8", "replace")
                    except Exception:
                        pass
            took = time.time() - t0
            blocks = parse(text) if text else []
            runaway = looks_runaway(text)
            if guard and runaway:
                # See README: a near-empty page makes this model repeat until the cap. The
                # official pipeline suppresses it with `no_repeat_ngram_size=35`, which mlx-vlm
                # does not implement — so the only defence a CALLER has is to notice and refuse.
                blocks, text = [], text[:400]
            if not emit("page", {
                "index": i, "seconds": round(took, 2), "error": err, "finish": finish,
                "runaway": runaway, "raw": text, "usage": usage,
                "blocks": [{"label": b.label, "box": list(b.box), "content": b.content}
                           for b in blocks]}):
                return
        emit("done", {"seconds": round(time.time() - run_t0, 2)})


_WORD = re.compile(r"\S+")


def looks_runaway(text: str) -> bool:
    """Has the model fallen into a repeat loop? (`1. 2. 3. 4. …` on a near-empty page.)

    Measured behaviour, not a guess — see README. The test is deliberately crude and cheap: if a
    long answer is mostly made of very few DISTINCT tokens, it is not a document.
    """
    words = _WORD.findall(text or "")
    if len(words) < 120:
        return False
    return len(set(words)) / len(words) < 0.18


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
