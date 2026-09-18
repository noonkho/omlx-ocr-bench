"""Talk to Unlimited-OCR on oMLX, and make sense of what comes back."""

from __future__ import annotations

import base64
import html
import io
import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import config

#: `<|det|>LABEL [x0, y0, x1, y1]<|/det|>` followed by that block's content.
#: Coordinates are 0-1000 NORMALISED to the page, not pixels — so they survive any render DPI.
_DET = re.compile(r"<\|det\|>\s*(\w+)\s*\[([\d\s,]+)\]\s*<\|/det\|>")

#: One colour per block kind, for the overlay.
COLOURS = {
    "title": "#d92b2b", "text": "#2b6cd9", "table": "#1a9e5c", "image": "#b453d9",
    "formula": "#d98c1a", "header": "#e2760c", "footer": "#8a8a8a", "caption": "#00a0a8",
}


@dataclass
class Block:
    label: str
    box: tuple[int, int, int, int]      # 0-1000 normalised
    content: str = ""

    def pixels(self, w: int, h: int) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.box
        return (round(x0 * w / 1000), round(y0 * h / 1000),
                round(x1 * w / 1000), round(y1 * h / 1000))


@dataclass
class PageResult:
    index: int
    png: bytes
    size: tuple[int, int]
    raw: str = ""
    blocks: list[Block] = field(default_factory=list)
    seconds: float = 0.0
    usage: dict = field(default_factory=dict)
    error: str = ""

    @property
    def plain_text(self) -> str:
        """Just the words, markers stripped — what a downstream pipeline would actually read."""
        return "\n".join(b.content for b in self.blocks if b.content.strip())


def parse(raw: str) -> list[Block]:
    """`<|det|>` markers into blocks. Text before the first marker is kept as an unlabelled block.

    The model sometimes emits the SAME block twice in a row (seen on a stamp). Consecutive exact
    duplicates are collapsed — a repeated read is one fact, and counting it twice would make a
    stamp look like two stamps.
    """
    out: list[Block] = []
    marks = list(_DET.finditer(raw))
    if not marks:
        return [Block("text", (0, 0, 1000, 1000), raw.strip())] if raw.strip() else []

    preamble = raw[:marks[0].start()].strip()
    if preamble:
        # Not inside any box the model drew. Usually a header it read loosely — or, in testing,
        # an invented date. Kept and flagged rather than silently dropped.
        out.append(Block("unboxed", (0, 0, 1000, 20), preamble))

    for i, m in enumerate(marks):
        nums = [int(n) for n in re.findall(r"\d+", m.group(2))]
        if len(nums) != 4:
            continue
        end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        content = raw[m.end():end].strip()
        block = Block(m.group(1).lower(), tuple(nums), content)  # type: ignore[arg-type]
        if out and out[-1].label == block.label and out[-1].content == block.content:
            continue                                     # the same read, twice
        out.append(block)
    return out


def render_pages(path: Path, dpi: int = config.DPI) -> list[tuple[bytes, tuple[int, int]]]:
    """Every page of a PDF as PNG bytes — or the image itself, if it is already an image."""
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return [(buf.getvalue(), im.size)]

    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    try:
        pages = []
        for i in range(len(doc)):
            image = doc[i].render(scale=dpi / 72).to_pil()
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            pages.append((buf.getvalue(), image.size))
        return pages
    finally:
        doc.close()


def ocr_page(png: bytes, size: tuple[int, int], index: int,
             prompt: str = config.PROMPT, max_tokens: int = 8192) -> PageResult:
    """One page, one request. Never raises — a page that fails carries its own error."""
    result = PageResult(index=index, png=png, size=size)
    body = {
        "model": config.MODEL, "max_tokens": max_tokens, "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
        ]}],
    }
    req = urllib.request.Request(
        config.BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {config.API_KEY}"})
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            data = json.loads(response.read())
        result.raw = data["choices"][0]["message"]["content"]
        result.usage = data.get("usage") or {}
        result.blocks = parse(result.raw)
    except Exception as exc:                               # noqa: BLE001 - reported, not raised
        detail = ""
        if hasattr(exc, "read"):
            try:
                detail = exc.read()[:300].decode("utf-8", "replace")
            except Exception:
                pass
        result.error = f"{type(exc).__name__}: {exc} {detail}".strip()
    result.seconds = time.time() - started
    return result


def escape(text: str) -> str:
    return html.escape(text, quote=False)
