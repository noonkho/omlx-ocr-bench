"""Score a model's answer against the answer key a drawn sample carries.

Two numbers, because they fail independently:

* **Text** — did it read the words? Length-weighted, so getting a heading right does not excuse
  losing a paragraph.
* **Position** — did it put those words in the right place on the page? A model can transcribe a
  page perfectly and still be useless for a pipeline that crops regions or checks a signature
  block, which is exactly what layout-aware OCR is sold for.

Both are deliberately forgiving about *how* a model groups text. One model returns a block per
line, another a block per paragraph, a third one blob of markdown. Scoring per line and asking
only whether the line's place on the page falls inside whatever block carried it keeps those
three comparable.
"""

from __future__ import annotations

import html
import re
from difflib import SequenceMatcher

from ocr import JUNK

#: Truth entries that are graphics, not words. A model is judged on marking them, not reading them.
GRAPHIC = {"image", "seal", "signature"}

#: Below this a line is "not found at all" rather than "found and misread".
FOUND = 0.80
#: Below this there is no sensible block to ask a position question about.
MATCHED = 0.55

_SPACE = re.compile(r"\s+")
_TAGS = re.compile(r"<[^>]+>")


def norm(s: str) -> str:
    """Whitespace and HTML entities differ between models and mean nothing. Level them."""
    return _SPACE.sub(" ", html.unescape(_TAGS.sub(" ", s or ""))).strip()


def lines_of(blocks) -> list[tuple[str, list[int]]]:
    """Every line a model returned, with the box of the block that carried it."""
    out = []
    for b in blocks:
        label = b["label"] if isinstance(b, dict) else b.label
        if label in JUNK:
            continue                       # invented text is scored separately, as `spurious`
        content = b["content"] if isinstance(b, dict) else b.content
        boxed = b.get("boxed", True) if isinstance(b, dict) else b.boxed
        # A model that returned no coordinates gets no position score, rather than a free one.
        box = list(b["box"] if isinstance(b, dict) else b.box) if boxed else None
        raw = content or ""
        pieces = re.split(r"</t[dh]>|</tr>|\n", raw) if "<t" in raw else raw.split("\n")
        for piece in pieces:
            line = norm(piece)
            if line:
                out.append((line, box))
    return out


def best(line: str, candidates, near: list[int] | None = None) -> tuple[float, list[int] | None]:
    """The closest line a model returned, and where it put it.

    A page repeats short strings — "Not permitted" four times in one table — so an equal-scoring
    tie is broken by whichever candidate sits nearest to where the line actually is. Without that,
    every repeat is charged to the first one and a perfect answer scores less than perfect.
    """
    top, where = 0.0, None

    def away(box) -> float:
        if not near or not box:
            return 0.0
        return abs((box[0] + box[2]) / 2 - (near[0] + near[2]) / 2) + \
               abs((box[1] + box[3]) / 2 - (near[1] + near[3]) / 2)

    for text, box in candidates:
        # A cheap length filter first: SequenceMatcher on every pair is the slow part here.
        if not 0.4 <= len(text) / max(len(line), 1) <= 2.5:
            continue
        ratio = 1.0 if line == text else SequenceMatcher(None, line, text).ratio()
        if ratio > top or (ratio == top and away(box) < away(where)):
            top, where = ratio, box
    return top, where


def inside(box: list[int], target: list[int], slack: int = 30) -> bool:
    """Is the truth line's centre inside the block the model put its words in?"""
    cx, cy = (target[0] + target[2]) / 2, (target[1] + target[3]) / 2
    return (box[0] - slack <= cx <= box[2] + slack
            and box[1] - slack <= cy <= box[3] + slack)


def score_page(blocks, truth: list[dict]) -> dict:
    """One page: how much of it was read, and how much of it was placed."""
    model = lines_of(blocks)
    wanted = [t for t in truth
              if t["label"] not in GRAPHIC and t["label"] != "table" and norm(t["text"])]

    weight = read = 0.0
    found = placed = matched = 0
    misses: list[str] = []
    for item in wanted:
        line = norm(item["text"])
        ratio, box = best(line, model, item["box"])
        weight += len(line)
        read += len(line) * ratio
        if ratio >= FOUND:
            found += 1
        elif len(misses) < 8:
            misses.append(item["text"][:60])
        if ratio >= MATCHED and box:
            matched += 1
            if inside(box, item["box"]):
                placed += 1

    # Graphics: the model should mark a region there, whatever it calls it.
    graphics = [t for t in truth if t["label"] in GRAPHIC]
    boxes = [list(b["box"] if isinstance(b, dict) else b.box) for b in blocks
             if (b.get("boxed", True) if isinstance(b, dict) else b.boxed)]
    marked = sum(1 for g in graphics if any(inside(box, g["box"], 60) for box in boxes))

    # Text the model produced that matches nothing on the page. High means it is inventing.
    truth_lines = [(norm(t["text"]), t["box"]) for t in truth if norm(t["text"])]
    spurious = sum(1 for line, _ in model
                   if len(line) > 12 and best(line, truth_lines)[0] < MATCHED)

    junk = sum(1 for b in blocks
               if (b["label"] if isinstance(b, dict) else b.label) in JUNK)
    return {
        "text": round(read / weight, 4) if weight else None,
        "position": round(placed / matched, 4) if matched else None,
        "found": found, "lines": len(wanted),
        "graphics_marked": marked, "graphics": len(graphics),
        "spurious": spurious, "junk": junk, "misses": misses,
    }


def combine(pages: list[dict]) -> dict:
    """A document's score: pages weighted by how much was on them."""
    scored = [p for p in pages if p and p.get("lines")]
    if not scored:
        return {}
    total = sum(p["lines"] for p in scored)
    def mean(key):
        have = [p for p in scored if p.get(key) is not None]
        return (round(sum(p[key] * p["lines"] for p in have) / sum(p["lines"] for p in have), 4)
                if have else None)
    return {
        "text": mean("text"), "position": mean("position"),
        "found": sum(p["found"] for p in scored), "lines": total,
        "graphics_marked": sum(p["graphics_marked"] for p in scored),
        "graphics": sum(p["graphics"] for p in scored),
        "spurious": sum(p["spurious"] for p in scored),
        "junk": sum(p["junk"] for p in scored),
    }
