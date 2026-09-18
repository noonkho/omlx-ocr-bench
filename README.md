# Does `baidu/Unlimited-OCR` on oMLX actually work?

A small, throwaway harness to answer that with pictures rather than opinions. Separate from the
Feenote repo on purpose — nothing here touches a case folder, and every sample is synthetic.

```bash
uv run demo.py samples/synthetic_hk.pdf --open     # PDF, 2 pages
uv run demo.py samples/synthetic_zh.png --open     # Traditional Chinese, as an image
uv run test_multipage.py                           # can one request carry several pages?
```

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

Sampling penalties do not fix it:

| Setting | Result |
|---|---|
| none | `1. 2. 3. 4. 5. …` to the cap |
| `presence_penalty: 0.4` | unchanged |
| `repetition_penalty: 1.08` | becomes `1.2.3. 2.3.4. 2.3.5. …`, still to the cap |

So a caller needs its own guard: a modest `max_tokens` for OCR, and a check that discards output
which is mostly one repeated pattern. In a real bundle — separator pages, blank backs of scans,
exhibit dividers — this would otherwise be a large share of the run.

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
