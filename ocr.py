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

#: A run of bare numbers — `1. 2. 3. 4. …`, `2. 2. 2. …`, `1.2.3.4.5.6.` — which this model emits
#: when a page holds almost no text for it to read. Never document content.
_COUNTER = re.compile(r"^(?:\d+\.\s*){4,}$")

#: The model talking about its own annotation rules instead of reading the page. Seen on a sparse
#: page: "The Ground Truth image displays a single, solid horizontal line. According to Rule 2
#: (UNDERSCORE & LINE RULES)…". It is training-harness text, never document content — but it is
#: inside a `<|det|>` box, so it needs catching separately from the counter.
_LEAK = re.compile(r"ground truth image|according to rule \d|\b(?:RULE|RULES)\s*\d+\s*\(",
                   re.IGNORECASE)

#: Below this share of dark pixels a page carries no text worth sending. Measured: a truly blank
#: page renders at 0.0000, and the sparsest real page tested (four short lines) at 0.0073 — so
#: this sits an order of magnitude below anything a caller would want read.
BLANK_INK = 0.0008

#: One colour per block kind, for the overlay.
COLOURS = {
    "title": "#d92b2b", "text": "#2b6cd9", "table": "#1a9e5c", "image": "#b453d9",
    "formula": "#d98c1a", "header": "#e2760c", "footer": "#8a8a8a", "caption": "#00a0a8",
    "page_number": "#7a7a7a", "unboxed": "#b3261e", "noise": "#b3261e",
}

#: Block kinds a caller should not put into a markdown file. See `parse`.
JUNK = {"noise", "unboxed"}


def ink(png: bytes) -> float:
    """How much of the page is dark, 0 to 1. A blank page is not worth a request.

    Measured, not guessed — see README. On a blank page this model has nothing to read and
    counts instead, burning seconds and the whole token cap. The cheapest fix is to never ask.
    """
    from PIL import Image

    with Image.open(io.BytesIO(png)) as im:
        grey = im.convert("L")
        small = grey.resize((max(grey.width // 4, 1), max(grey.height // 4, 1)))
        dark = sum(1 for pixel in small.get_flattened_data() if pixel < 200)
        return dark / (small.width * small.height)


def strip_markers(text: str) -> str:
    """The words without the `<|det|>` markers. Their coordinates differ on every repeat, so a
    repeat test has to look at what the markers WRAP, not at the markers themselves."""
    return _DET.sub("", text or "")


def build_body(model: str, prompt: str, images: list[str], knobs: dict) -> dict:
    """One request body, for every caller. `images` are ready-made `data:` URLs."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": url}} for url in images]
    return {"model": model, "messages": [{"role": "user", "content": content}], **knobs}


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def request(base_url: str, api_key: str, body: dict) -> urllib.request.Request:
    return urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})


def http_detail(exc: Exception) -> str:
    """The server's own words when it returns an error, which are the useful part."""
    if not hasattr(exc, "read"):
        return ""
    try:
        return " " + exc.read()[:300].decode("utf-8", "replace")
    except Exception:                                      # noqa: BLE001 - best effort
        return ""


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
        return "\n".join(b.content for b in self.blocks
                         if b.content.strip() and b.label not in JUNK)


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
        # Not inside any box the model drew, and measured to be untrustworthy: on a near-empty
        # page it is `1. 2. 3. 4. …`, and on a dense one it has produced a date that is not on
        # the page at all. Flagged rather than silently dropped, and kept OUT of the markdown
        # a caller would export. `noise` is the counter form; `unboxed` is everything else.
        label = "noise" if _COUNTER.match(preamble) else "unboxed"
        out.append(Block(label, (0, 0, 1000, 20), preamble))

    for i, m in enumerate(marks):
        nums = [int(n) for n in re.findall(r"\d+", m.group(2))]
        if len(nums) != 4:
            continue
        end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        content = raw[m.end():end].strip()
        label = m.group(1).lower()
        if _COUNTER.match(content.strip()) or _LEAK.search(content):
            label = "noise"                  # the counter, or the model's own rulebook
        block = Block(label, tuple(nums), content)  # type: ignore[arg-type]
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
             prompt: str = config.PROMPT, max_tokens: int = 4096) -> PageResult:
    """One page, one request. Never raises — a page that fails carries its own error."""
    result = PageResult(index=index, png=png, size=size)
    body = build_body(config.MODEL, prompt, [data_url(png)],
                      {"max_tokens": max_tokens, "temperature": 0,
                       "frequency_penalty": config.FREQUENCY_PENALTY})
    req = request(config.BASE_URL, config.API_KEY, body)
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            data = json.loads(response.read())
        result.raw = data["choices"][0]["message"]["content"]
        result.usage = data.get("usage") or {}
        result.blocks = parse(result.raw)
    except Exception as exc:                               # noqa: BLE001 - reported, not raised
        result.error = f"{type(exc).__name__}: {exc}{http_detail(exc)}".strip()
    result.seconds = time.time() - started
    return result


def escape(text: str) -> str:
    return html.escape(text, quote=False)
