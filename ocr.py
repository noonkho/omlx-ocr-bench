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
    "seal": "#c0392b", "signature": "#6c3483", "image_caption": "#00a0a8",
}

#: Block kinds a caller should not put into a markdown file. See `parse`.
JUNK = {"noise", "unboxed"}


#: The prompts each model family was actually trained on, in Python so the bench, the page and
#: the one-shot CLI cannot drift apart. `match` is a case-insensitive regex against the model id;
#: the last entry matches everything and is the fallback.
FAMILIES = [
    {"name": "Unlimited-OCR", "match": r"unlimited", "prompts": [
        ("Multi page parsing.", "the one the model card documents; best on dense pages"),
        ("document parsing.", "calmer on sparse pages, but has invented text on dense ones"),
        ("Free OCR.", "text only, no layout"),
        ("Extract the text in the image.", "text only, plainer wording"),
    ]},
    {"name": "chandra", "match": r"chandra", "prompts": [
        ("Convert this page to markdown.",
         "anything works — measured: this model's output does not change with the prompt"),
        ("OCR this image to HTML, arranged as layout blocks.",
         "the opening line of chandra's own OCR_LAYOUT_PROMPT"),
        ("OCR this image to HTML.", "the opening line of chandra's own OCR_PROMPT"),
    ]},
    {"name": "PaddleOCR-VL / Qianfan", "match": r"paddle|qianfan|ernie", "prompts": [
        ("document parsing.", "the documented prompt for both"),
        ("OCR:", "text only"),
    ]},
    {"name": "dots.ocr / MinerU", "match": r"dots|mineru|logics", "prompts": [
        ("Parse the layout of this document.", "layout-first, returns JSON"),
        ("document parsing.", "worth trying if the first returns prose"),
    ]},
    # The last entry is the fallback: no pattern, matches by being last.
    {"name": "Anything else", "match": None, "prompts": [
        ("Convert this page to markdown.", "the safest opening move on an unknown model"),
        ("Extract all text from this document.", "text only"),
        ("Read this page. Return each block of text with a bounding box.",
         "asks for coordinates; most general VLMs will not give them"),
    ]},
]


#: The sampling knobs every caller starts from. Measured — see README on `frequency_penalty`.
DEFAULT_KNOBS = {"max_tokens": 4096, "temperature": 0,
                 "frequency_penalty": config.FREQUENCY_PENALTY}


def ink(png: bytes) -> float:
    """How much of the page is dark, 0 to 1. A blank page is not worth a request.

    Measured, not guessed — see README. On a blank page this model has nothing to read and
    counts instead, burning seconds and the whole token cap. The cheapest fix is to never ask.
    """
    from PIL import Image

    with Image.open(io.BytesIO(png)) as im:
        grey = im.convert("L")
        small = grey.resize((max(grey.width // 4, 1), max(grey.height // 4, 1)))
        counts = small.histogram()          # 256 buckets, counted in C
        return sum(counts[:200]) / (small.width * small.height)


def stitch(pngs: list[bytes]) -> tuple[bytes, list[int]]:
    """Several page renders as ONE tall image, plus each page's height in it.

    This is the workaround for oMLX dropping every image after the first: one image it cannot
    drop. Measured — see README — two pages stitched come back in one request, in the time one
    page takes, with both pages' text present and `prompt_tokens` up from 909 to 1539, which is
    the proof that the second page was actually read.
    """
    from PIL import Image

    pages = [Image.open(io.BytesIO(p)).convert("RGB") for p in pngs]
    width = max(p.width for p in pages)
    heights = [p.height for p in pages]
    tall = Image.new("RGB", (width, sum(heights)), "white")
    y = 0
    for page in pages:
        tall.paste(page, (0, y))
        y += page.height
    buf = io.BytesIO()
    tall.save(buf, format="PNG")
    return buf.getvalue(), heights


def split_blocks(blocks: list[Block], heights: list[int]) -> list[list[Block]]:
    """Blocks drawn on a stitched image, put back on the pages they came from.

    A block is assigned to the page its centre falls on, and its box is re-expressed in that
    page's own 0-1000 space — so the overlay and the answer key both work unchanged.
    """
    total = sum(heights) or 1
    edges, run = [], 0
    for h in heights:
        edges.append((run, run + h))
        run += h

    pages: list[list[Block]] = [[] for _ in heights]
    for b in blocks:
        if not b.boxed:
            pages[0].append(b)
            continue
        x0, y0, x1, y1 = b.box
        top, bottom = y0 * total / 1000, y1 * total / 1000
        middle = (top + bottom) / 2
        index = next((i for i, (a, z) in enumerate(edges) if a <= middle < z), len(heights) - 1)
        a, z = edges[index]
        height = max(z - a, 1)
        pages[index].append(Block(
            b.label,
            (x0, round(max(top - a, 0) * 1000 / height),
             x1, round(min(bottom - a, height) * 1000 / height)),
            b.content, b.boxed))
    return pages


def kind_of(content: str, default: str) -> str:
    """A block's kind, downgraded to `noise` when its text is something the model made up.

    Every parser asks this, because every format can carry the same two failures: a bare counter,
    and the model reciting its own annotation rulebook. See `parse`.
    """
    return "noise" if _COUNTER.match(content.strip()) or _LEAK.search(content) else default


def prompt_for(model: str) -> str:
    """The prompt this model was trained on. One table, used by the bench, the page and the CLI."""
    for fam in FAMILIES:
        if fam["match"] and re.search(fam["match"], model, re.IGNORECASE):
            return fam["prompts"][0][0]
    return FAMILIES[-1]["prompts"][0][0]


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
    #: False when the model gave no coordinates and the box is a stand-in for the whole page.
    #: Scoring must not credit such a block with placing anything.
    boxed: bool = True

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
    """Whatever an OCR model returned, as blocks. Three formats, tried in order.

    1. `<|det|>LABEL [x0, y0, x1, y1]<|/det|>` markers — Unlimited-OCR and its relatives.
    2. `<div data-bbox="x0 y0 x1 y1" data-label="Text">…</div>` — chandra and the other models
       that answer in annotated HTML. Already 0-1000 normalised.
    3. A JSON list of `{"bbox": [...], "category": ..., "text": ...}` — dots.ocr, PaddleOCR-VL
       and most of the layout-first models. Pixel boxes are rescaled to 0-1000 when they are
       plainly out of range.
    4. Plain markdown, which many models return with no coordinates at all. Those blocks are
       marked `boxed=False`, so the bench scores their words and reports no position.

    Text before the first marker is kept as its own block and flagged — see below.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    if _DET.search(raw):
        return _parse_det(raw)
    if _BBOX.search(raw):
        return _parse_bbox_html(raw)
    blocks = _parse_json(raw)
    return blocks if blocks else _parse_markdown(raw)


def _parse_det(raw: str) -> list[Block]:
    """The `<|det|>` format.

    The model sometimes emits the SAME block twice in a row (seen on a stamp). Consecutive exact
    duplicates are collapsed — a repeated read is one fact, and counting it twice would make a
    stamp look like two stamps.
    """
    out: list[Block] = []
    marks = list(_DET.finditer(raw))

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
        block = Block(kind_of(content, m.group(1).lower()), tuple(nums), content)  # type: ignore[arg-type]
        if out and out[-1].label == block.label and out[-1].content == block.content:
            continue                                     # the same read, twice
        out.append(block)
    return out


#: `<div data-bbox="62 36 140 100" data-label="Image">` — chandra's annotated HTML. Matched as a
#: marker rather than a tag pair, so a nested tag of the same name cannot end a block early.
_BBOX = re.compile(r"""<\w+[^>]*?\sdata-bbox=["']\s*([\d\s.]+?)["']([^>]*)>""", re.IGNORECASE)
_LABEL = re.compile(r"""data-label=["']([^"']*)["']""", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


def _parse_bbox_html(raw: str) -> list[Block]:
    """Annotated HTML. The coordinates are already in this bench's 0-1000 space."""
    out: list[Block] = []
    marks = list(_BBOX.finditer(raw))
    for i, m in enumerate(marks):
        nums = [float(n) for n in re.findall(r"[\d.]+", m.group(1))]
        if len(nums) != 4:
            continue
        end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        content = raw[m.end():end].strip()
        if content.endswith("</div>"):
            content = content[: -len("</div>")].rstrip()
        if "<table" not in content.lower():
            # Keep table markup; drop the wrapper tags around ordinary text.
            content = html.unescape(_TAG.sub(" ", re.sub(r"<br\s*/?>", "\n", content))).strip()
            content = re.sub(r"[ \t]{2,}", " ", content)
        label = (_LABEL.search(m.group(2)) or [None, m.group(0).split()[0].lstrip("<")])[1]
        out.append(Block(kind_of(content, str(label).lower()),
                         tuple(round(n) for n in nums), content))  # type: ignore[arg-type]
    return out


def _parse_json(raw: str) -> list[Block]:
    """A JSON list of regions, the shape most layout-first OCR models return."""
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        items = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list) or not items or not isinstance(items[0], dict):
        return []

    # The page's own scale, worked out once rather than per region. One stray pixel-scale box on
    # an otherwise normalised page must not shrink everything else, so the majority decides.
    reach = [max(float(b[2]), float(b[3])) for b in map(_box, items) if b]
    pixels = sum(1 for r in reach if r > 1000) * 2 >= len(reach) if reach else False
    widest = max(reach) if pixels else 0.0

    out: list[Block] = []
    for item in items:
        box = _box(item)
        content = item.get("text") or item.get("content") or item.get("caption") or ""
        label = str(item.get("category") or item.get("label") or item.get("type") or "text")
        if not box:
            out.append(Block(label.lower(), (0, 0, 1000, 1000), str(content), boxed=False))
            continue
        nums = [float(n) for n in box]
        if widest:                           # pixels, not the 0-1000 space this bench works in
            nums = [min(max(n * 1000 / widest, 0), 1000) for n in nums]
        out.append(Block(label.lower(), tuple(round(n) for n in nums),  # type: ignore[arg-type]
                         str(content)))
    return out


def _box(item: dict):
    """The one key a model used for its coordinates, whichever of the three it picked."""
    box = item.get("bbox") or item.get("box") or item.get("boundingBox")
    return box if isinstance(box, (list, tuple)) and len(box) == 4 else None


def _parse_markdown(raw: str) -> list[Block]:
    """No coordinates at all. Paragraphs, and an honest `boxed=False` on each."""
    out: list[Block] = []
    for chunk in re.split(r"\n\s*\n", raw):
        chunk = chunk.strip()
        if not chunk:
            continue
        label = "title" if chunk.startswith("#") else "table" if chunk.startswith("<table") \
            else "text"
        out.append(Block(kind_of(chunk, label), (0, 0, 1000, 1000),
                         chunk.lstrip("# ").strip(), boxed=False))
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
             prompt: str | None = None, max_tokens: int = 4096) -> PageResult:
    """One page, one request. Never raises — a page that fails carries its own error.

    With no prompt given, the one this model was trained on is used — see `FAMILIES`. Sending
    Unlimited-OCR's prompt to chandra is the single cheapest way to lose accuracy.
    """
    prompt = prompt or prompt_for(config.MODEL)
    result = PageResult(index=index, png=png, size=size)
    body = build_body(config.MODEL, prompt, [data_url(png)],
                      dict(DEFAULT_KNOBS, max_tokens=max_tokens))
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
