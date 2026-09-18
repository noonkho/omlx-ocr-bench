# Does `baidu/Unlimited-OCR` on oMLX actually work?

A small, throwaway harness to answer that with pictures rather than opinions. Separate from the
Feenote repo on purpose — nothing here touches a case folder, and every sample is synthetic.

```bash
uv run server.py        # the three-panel bench -> http://127.0.0.1:4700
uv run demo.py samples/synthetic_hk.pdf --open     # one-shot, writes a static HTML page
uv run test_multipage.py                           # can one request carry several pages?
```

**The bench** (`server.py` + `ui.html`) is the thing to use. Three panels:

* **left** — base URL, API key and a **Test connection** button that lists every model the server
  has, so the right one can be picked and a wrong one is called out; the prompt, with presets;
  drag-and-drop or click to upload a PDF or image; DPI, max tokens and the runaway guard.
* **middle** — what the model returned, as plain text, rendered, or raw with the `<|det|>` markers.
* **right** — each page with the model's own layout boxes drawn on it. Hover a box to highlight
  its text, and the other way round.

Pages stream in one at a time, so a long document shows progress instead of hanging. Everything
goes through the local process because the oMLX box sets no CORS headers.

Two synthetic samples are built in — click them in the left panel.

Server and model come from env vars (`config.py`): `OMLX_BASE_URL`, `OMLX_API_KEY`,
`OMLX_OCR_MODEL`, `OMLX_OCR_DPI`, `OMLX_OCR_PROMPT`.

`demo.py` writes `out/<name>.html` — the rendered page with the model's own layout boxes drawn on
it, the text beside it, and timings. Hover a box to highlight its text, and the reverse.

---

## What was measured (2026-09-18, `Unlimited-OCR-bf16` on oMLX 0.6.4, M3 Ultra over VPN)

### 1. The documented prompt returns HTTP 500

The model card writes the prompt as `<image>Multi page parsing.` **That literal `<image>` makes
oMLX return 500.** oMLX inserts its own image placeholder when it renders the chat template, so a
second one in your text gives two placeholders for one image and the render fails.

Send the bare instruction — `Multi page parsing.` — and the same request succeeds in ~4s.

### 2. It is about twice as fast as `chandra-ocr-2`, on the same page

| Model | Page 1 of the synthetic HK writ |
|---|---|
| `chandra-ocr-2-8bit-mlx` | 10.5 s |
| `Unlimited-OCR-bf16` | **4.7 s** (~126 tok/s) |

### 3. Quality is good, and it sees stamps

On a synthetic HK High Court page it returned every heading, both parties, the action number, the
body text, **a real HTML `<table>`** for the costs table, and — importantly for a fee note — it
read the red chop, `FILED 20 FEB 2025`, as its own block.

Traditional Chinese came back almost perfectly, including the chop (`已存案 / 2025年2月20日`).

**One flaw that matters in Hong Kong:** it returned **黄** where the page said **黃**. That is the
simplified variant of a very common HK surname. Any name it produces has to be checked against the
document, not trusted.

### 4. ⚠️ Multi-page in ONE request does NOT work on oMLX

This was the main reason to want this model, and it does not hold up. `prompt_tokens` is
**identical at 909 whether the request carries one image, two, or three** — oMLX passes only the
FIRST image to the model and silently drops the rest.

| Images in the request | `prompt_tokens` |
|---|---|
| 1 | 909 |
| 2 | 909 |
| 3 | 909 |

A "speed-up" measured this way is not a speed-up: the second page was never read. **Until oMLX
supports multi-image messages, one page per request stands.**

### 5. ⚠️ A near-empty page runs away

The second page of the sample holds four short lines. The model emits `1. 2. 3. 4. 5. …` until it
hits the output cap — **29 s and 8,192 wasted tokens on a nearly blank page**.

**The cause is known.** The official pipeline runs the model with `no_repeat_ngram_size=35` and
`ngram_window=128` — an n-gram logits processor that forbids repeating any 35-token sequence.
**`mlx-vlm` does not implement it.** The only knob it exposes is `repetition_penalty`, which is
the wrong tool (a document legitimately repeats words) and does not work here:

| Setting | Result |
|---|---|
| none | `1. 2. 3. 4. 5. …` to the cap |
| `presence_penalty: 0.4` | unchanged |
| `repetition_penalty: 1.08` | becomes `1.2.3. 2.3.4. 2.3.5. …`, still to the cap |

So until `mlx-vlm` gains the n-gram processor, a caller needs its own guard. The bench ships one
(`server.looks_runaway`): if a long answer is made of very few DISTINCT tokens, it is not a
document, and the page is refused rather than stored. Plus a modest `max_tokens` — 4096, not
16384 — so a page that does run away costs seconds instead of a minute.

In a real bundle — separator pages, blank backs of scans, exhibit dividers — this would otherwise
be a large share of the run.

### 6. Yes, it reads plain images

`samples/synthetic_zh.png` is a PNG, not a PDF, and went through unchanged. It also labels
picture regions (`[Non-Text]`), so a figure is marked rather than silently skipped.

---

## What this means for Feenote

* **Worth switching** — roughly half the time per page, better tables, and it reads chops, which
  is what `clause.proof` needs.
* **Not for the reason I first gave.** One-shot multi-page does not work on oMLX, so it does not
  remove the per-page loop.
* **Needs a runaway guard before it is safe** on real bundles.
* Keep the 500-causing `<image>` prefix out of the prompt template.

## The oMLX bug

`OMLX-BUG-REPORT.md` is a ready-to-post issue for https://github.com/jundot/omlx/issues, with the
`prompt_tokens` table, the 500 matrix and a minimal reproduction. oMLX's README claims multi-image
chat support, so this is a bug rather than a missing feature — worth filing, because if it is
fixed the per-page loop goes away.
