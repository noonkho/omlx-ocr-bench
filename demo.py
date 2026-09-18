#!/usr/bin/env python3
"""Run Unlimited-OCR over a PDF or image and write a page you can look at.

    uv run demo.py samples/synthetic_hk.pdf
    uv run demo.py some.pdf --dpi 200 --open

Writes `out/<name>.html` — each page as the rendered image with the model's own layout boxes
drawn on it, the plain text beside it, and the timings. The point is to be able to SEE whether
the model read the page, rather than to trust a number.
"""

from __future__ import annotations

import argparse
import base64
import subprocess
import sys
from pathlib import Path

import config
from ocr import COLOURS, PageResult, escape, ocr_page, render_pages

HERE = Path(__file__).parent


def page_html(page: PageResult) -> str:
    w, h = page.size
    boxes = []
    for i, block in enumerate(page.blocks):
        if block.label == "unboxed":
            continue
        x0, y0, x1, y1 = block.box
        colour = COLOURS.get(block.label, "#888")
        boxes.append(
            f'<div class="box" style="left:{x0/10:.2f}%;top:{y0/10:.2f}%;'
            f'width:{(x1-x0)/10:.2f}%;height:{(y1-y0)/10:.2f}%;border-color:{colour}" '
            f'data-i="{i}" title="{escape(block.label)}">'
            f'<span style="background:{colour}">{escape(block.label)}</span></div>')

    rows = []
    for i, block in enumerate(page.blocks):
        colour = COLOURS.get(block.label, "#888")
        warn = ' <em class="warn">outside every box the model drew</em>' if block.label == "unboxed" else ""
        rows.append(
            f'<div class="blk" data-i="{i}"><span class="tag" style="background:{colour}">'
            f'{escape(block.label)}</span>{warn}<div class="txt">{escape(block.content)}</div></div>')

    u = page.usage
    stat = (f"{page.seconds:.1f}s · {u.get('completion_tokens', 0)} out / "
            f"{u.get('prompt_tokens', 0)} in tokens"
            + (f" · {u.get('completion_tokens', 0)/page.seconds:.0f} tok/s" if page.seconds else ""))
    err = f'<p class="err">{escape(page.error)}</p>' if page.error else ""
    return f"""<section>
  <h2>Page {page.index + 1} <small>{stat} · {w}&times;{h}px · {len(page.blocks)} blocks</small></h2>
  {err}
  <div class="pair">
    <div class="sheet"><img src="data:image/png;base64,{base64.b64encode(page.png).decode()}">{''.join(boxes)}</div>
    <div class="side">{''.join(rows) or '<p class="err">nothing came back</p>'}</div>
  </div>
</section>"""


def build(source: Path, out: Path, dpi: int) -> Path:
    pages = render_pages(source, dpi)
    print(f"{source.name}: {len(pages)} page(s) at {dpi} dpi -> {config.MODEL}", flush=True)
    results = []
    for i, (png, size) in enumerate(pages):
        print(f"  page {i+1}/{len(pages)} ...", end="", flush=True)
        page = ocr_page(png, size, i)
        print(f" {page.seconds:.1f}s  {len(page.blocks)} blocks"
              + (f"  ERROR {page.error[:80]}" if page.error else ""), flush=True)
        results.append(page)

    total = sum(p.seconds for p in results)
    tokens = sum((p.usage.get("completion_tokens") or 0) for p in results)
    head = (f"{len(results)} page(s) · {total:.1f}s total · "
            f"{total/len(results):.1f}s per page · {tokens} output tokens"
            + (f" · {tokens/total:.0f} tok/s" if total else ""))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.format(
        title=escape(source.name), model=escape(config.MODEL),
        prompt=escape(config.PROMPT), head=escape(head),
        body="\n".join(page_html(p) for p in results),
    ), encoding="utf-8")

    txt = out.with_suffix(".txt")
    txt.write_text("\n\n".join(f"--- page {p.index+1} ---\n{p.plain_text}" for p in results),
                   encoding="utf-8")
    print(f"\n{head}\nwrote {out}\nwrote {txt}")
    return out


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — Unlimited-OCR</title><style>
:root{{--bg:#f6f6f4;--panel:#fff;--ink:#1a1a18;--muted:#6b6b66;--line:#e2e2dd;--warn:#b3261e}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#16161a;--panel:#1e1e24;
--ink:#ececf0;--muted:#9a9aa4;--line:#303038}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:24px 16px}}
.wrap{{max-width:1500px;margin:0 auto}}
h1{{font-size:26px;margin:0 0 4px;letter-spacing:-.02em}}
.sub{{color:var(--muted);font-size:13px;margin:0 0 6px}}
.head{{font-size:15px;font-weight:600;margin:0 0 22px}}
h2{{font-size:17px;margin:30px 0 10px;font-weight:600}}
h2 small{{font-weight:400;color:var(--muted);font-size:12.5px;margin-left:8px}}
.pair{{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:16px;align-items:start}}
@media(max-width:900px){{.pair{{grid-template-columns:1fr}}}}
.sheet{{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;line-height:0}}
.sheet img{{width:100%;display:block}}
.box{{position:absolute;border:2px solid;border-radius:2px;pointer-events:auto;cursor:pointer}}
.box span{{position:absolute;top:-1px;left:-1px;color:#fff;font-size:9px;line-height:1.35;padding:0 4px;border-radius:2px 0 4px 0;white-space:nowrap;opacity:.9}}
.box.on{{box-shadow:0 0 0 3px rgba(255,200,0,.75)}}
.side{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px;max-height:82vh;overflow:auto}}
.blk{{padding:7px 8px;border-radius:7px;border:1px solid transparent}}
.blk.on{{border-color:var(--line);background:rgba(255,200,0,.12)}}
.tag{{display:inline-block;color:#fff;font-size:10px;padding:1px 6px;border-radius:999px;vertical-align:2px}}
.txt{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13.5px;margin-top:3px}}
.err{{color:var(--warn);font-size:13px}} .warn{{color:var(--warn);font-size:11px;margin-left:6px}}
code{{background:var(--line);padding:1px 5px;border-radius:4px;font-size:12.5px}}
</style></head><body><div class="wrap">
<h1>{title}</h1>
<p class="sub">model <code>{model}</code> &nbsp; prompt <code>{prompt}</code></p>
<p class="head">{head}</p>
{body}
<script>
// Hovering a box highlights its text, and the other way round.
function pair(sel, other){{
  document.querySelectorAll(sel).forEach(el => {{
    const mark = on => el.closest("section").querySelectorAll(other + '[data-i="'+el.dataset.i+'"], '
      + sel + '[data-i="'+el.dataset.i+'"]').forEach(x => x.classList.toggle("on", on));
    el.addEventListener("mouseenter", () => mark(true));
    el.addEventListener("mouseleave", () => mark(false));
  }});
}}
pair(".box", ".blk"); pair(".blk", ".box");
</script>
</div></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("--dpi", type=int, default=config.DPI)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--open", action="store_true", help="open the result when it is written")
    args = ap.parse_args()
    if not args.source.exists():
        print(f"no such file: {args.source}", file=sys.stderr)
        return 2
    out = args.out or (HERE / "out" / (args.source.stem + ".html"))
    built = build(args.source, out, args.dpi)
    if args.open:
        subprocess.run(["open", str(built)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
