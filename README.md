# Unlimited-OCR bench

A three-panel workbench for driving [`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR)
through an [oMLX](https://github.com/jundot/omlx) server, and for finding out what the pair can and
cannot actually do.

It answers one question with pictures rather than opinions: **can this model turn a PDF — with
tables, stamps, signatures and logos — into markdown, over an OpenAI-compatible endpoint?**

The short answer is yes for a page, no for a whole document, and there are two settings you must
get right. Everything below was measured, not assumed.

![Settings on the left, the document in the middle, the model's answer on the right](docs/screenshot.jpg)

## Run it

```bash
uv run server.py
```

Then open <http://127.0.0.1:4700>. Nothing else is needed — no Docker, no local model. The bench
talks to an oMLX server that has already downloaded and loaded the model.

```bash
uv run samples/make_samples.py                 # draw the sample pack (do this first)
uv run demo.py samples/legal_deed.pdf --open   # one-shot, writes a static HTML page to out/
uv run test_multipage.py                       # can one request carry several pages?
```

Server and defaults come from env vars, read in `config.py`:

| variable | default |
|---|---|
| `OMLX_BASE_URL` | `http://100.77.164.75:8000/v1` |
| `OMLX_API_KEY` | `testing` |
| `OMLX_OCR_MODEL` | `Unlimited-OCR-bf16` |
| `OMLX_OCR_PROMPT` | `Multi page parsing.` |
| `OMLX_OCR_DPI` | `150` |
| `OMLX_OCR_FREQUENCY_PENALTY` | `0.8` |

## The three panels

**Left — set up.** Drag and drop a PDF or an image, or click one of the drawn samples. Then: the
prompt, with presets; base URL, API key and a **Test connection** button that lists every model on
the server, shows the chosen one's context window, and says so plainly if it is not an
Unlimited-OCR model; every sampling knob oMLX accepts; render DPI; and the choice between one
request per page and the whole document in one request.

**Middle — the document.** Every page as the model sees it, at the DPI you chose, scrolling.

**Right — what came back.** Tokens appear as they arrive, and the panel follows them down the page
until you scroll up to read something. Three views:

* **Rendered** — the answer as a document. The model's own HTML tables are kept as tables.
* **Raw** — exactly what the model emitted, `<|det|>` markers and all, dimmed so the words stand out.
* **Boxes** — the page with the model's own layout boxes drawn on it, coloured by kind, with the
  text beside it. Hover a box to highlight its text, and the other way round.

**Copy** and **.md** take the answer out as a markdown file — the thing a downstream pipeline keeps.

Everything goes through the local Python process rather than straight from the browser, because
the oMLX box sets no CORS headers and a browser would refuse the call.

## The samples

`uv run samples/make_samples.py` draws a pack that between them exercise what the model claims:

| file | what it is there to test |
|---|---|
| `legal_deed.pdf` | a deed: numbered clauses, a schedule table, three handwritten signatures, an inked chop and a dry embossed seal, then a near-blank exhibit divider |
| `gov_permit.pdf` | a premises licence: a drawn crest, a two-axis hours table, a conditions box, a figure, a barcode, a council seal |
| `school_transcript.pdf` | a transcript: a grade table, free-text remarks, two signatures, an embossed school seal |
| `commercial_invoice.pdf` | an invoice: a logo, line items, totals, a barcode, a PAID chop |
| `notice_zh.pdf` | Traditional Chinese: a table, a signature, a chop |
| `scan_permit.png`, `scan_notice_zh.png` | the same two pages as bad phone photos — skewed, noisy, lit from one side |

**Every page is drawn from scratch and every name is invented.** No document taken from the web, no
real person, firm, school or authority. That is not only a licence question: a drawn page has a
known ground truth, so "did the model read it right?" has an answer.

All seven went through cleanly. Tables came back as real HTML tables; the crest, seals, signatures
and barcode were each marked as picture regions rather than guessed at; captions were labelled as
captions; and the two phone-photo versions lost nothing measurable against the clean renders. The
one reading error worth naming: **Bramwell** where the page said **Branwell**.

---

## What was measured

`Unlimited-OCR-bf16` on oMLX 0.6.4, Mac Studio M3 Ultra, over a VPN. September 2026. The long run
below is a real 47-page Chinese national standard (GB/T 48000.3—2026), not a sample.

### 1. Skip blank pages before you send them

A blank page is where this model falls apart: with nothing to read it **counts** —
`1. 2. 3. 4. 5. …` — until it hits the token cap. 29 s and 8,192 tokens for a page with nothing on
it. In a real bundle — separator sheets, backs of scans, exhibit dividers — that is a large share
of a run.

So the bench measures the ink on each page as it renders it (`ocr.ink`) and never sends a page that
is under 0.08% dark. Measured: a truly blank page renders at **0.0000**, and the sparsest real page
tested — four short lines — at **0.0073**, an order of magnitude clear of the line.

On the 47-page standard this caught pages 2, 45, 46 and 47 and skipped them outright: **four
requests never made**, and four counting runs never paid for.

### 2. Set `frequency_penalty: 0.8`

For pages that do have text, but not much, the same loop appears. The knob that ends it:

| setting | near-empty page | dense page |
|---|---|---|
| none | 17.4 s, 4096 tokens, `finish_reason: length` | 5.1 s, 413 tokens, 14 blocks |
| `repetition_penalty: 1.08` | still runs to the cap | — |
| `repetition_penalty: 1.15` | 0.9 s, 103 tokens, stops | **damages layout**: merges two parties into a bogus one-row `<table>` and loses a word |
| `frequency_penalty: 0.3` / `0.5` | still runs to the cap | — |
| **`frequency_penalty: 0.8`** | **1.5 s, 340 tokens, `finish_reason: stop`** | **byte-identical to none — same tokens, same 14 blocks** |

`frequency_penalty: 0.8` ends the loop and, on the page that matters, changes nothing at all. It is
the default here. `repetition_penalty` is the wrong tool: it is a blunt ban on any token seen
before, and a legal document legitimately repeats words.

The official pipeline uses neither — it runs an n-gram logits processor with
`no_repeat_ngram_size=35`. **oMLX accepts `no_repeat_ngram_size` and then ignores it:** the same
request with and without it returns byte-identical output. That is a bug report of its own, below.

Belt and braces, for when you turn the penalty down to see what happens: `server.RepeatWatch`
watches the answer as it streams and abandons it once more than two thirds of its lines are exact
repeats. On the 47-page run it fired on two pages, both genuine loops — the same `<|det|>` block
written over and over to the cap — and on none of the other 41. What already arrived is kept and
flagged; the guard exists to save tokens, not to hide the page.

### 3. Do not trust anything the model writes outside a box

Two kinds of text come back that are not on the page.

**Before the first `<|det|>` marker**, on every single one of the 47 pages, there is a loose line
nobody asked for. Sometimes it is the running title read badly; on the contents page it was
`2017年1月1日`, **a date that is nowhere in the document**; elsewhere `10:00 AM` and
`6/1/2000 3-2026`.

**Inside a box, on sparse pages**, the model sometimes writes about its own annotation rules
instead of reading the page: *"The Ground Truth image displays a single, solid horizontal line.
According to Rule 2 (UNDERSCORE & LINE RULES)…"*. That is training-harness text that escaped into
the output.

`ocr.parse` labels the first kind `unboxed`, and a bare counter or a leaked rule `noise`. Both are
shown in the right panel with a warning, and **both are kept out of the exported markdown**.

### 4. Which prompt? `Multi page parsing.` — but know the trade

Tested page by page on the 47-page standard:

| | `Multi page parsing.` | `document parsing.` |
|---|---|---|
| sparse page 8 | 14.1 s, ran to the cap | 3.8 s, 39 blocks, clean stop |
| sparse page 11 | 14.1 s, ran to the cap | 3.3 s, 25 blocks, clean stop |
| dense page 40 | 5.7 s, **40 blocks** | 2.5 s, **4 blocks** — and it invented a passage about battery cells that is nowhere in this standard |

So `document parsing.` is calmer on thin pages and **hallucinates on dense ones**. With blank pages
now skipped and `frequency_penalty` handling the loop, `Multi page parsing.` is the safer default
and the one shipped. Both are one click apart in the panel — try them on your own documents.

### 5. Multi-page in ONE request does NOT work on oMLX

This is the reason to want this model, and it does not hold up today. `prompt_tokens` is
**identical at 909 whether the request carries one image, two, or three** — oMLX passes only the
first image to the model and silently drops the rest.

| images in the request | `prompt_tokens` | what came back |
|---|---|---|
| 1 | 909 | the first page |
| 2 | 909 | the first page only |
| 3 | 909 | the first page only |

A "speed-up" measured this way is not a speed-up: the second page was never read. The cause is not
a mystery — the oMLX pull request that added this model
([#2328](https://github.com/jundot/omlx/pull/2328)) states that it "reproduces the single-`<image>`
multi-page prompt semantics", i.e. exactly one placeholder is rendered however many images arrive.
That one decision explains both this and the 500 below.

The bench keeps the mode anyway (**Run → Requests → whole document in one**) so the bug can be
reproduced in one click, and so it starts working the day oMLX fixes it. Until then, one page per
request.

### 6. The documented prompt returns HTTP 500

The model card writes the prompt as `<image>Multi page parsing.` **That literal `<image>` makes
oMLX return 500.** oMLX inserts its own image placeholder when it renders the chat template, so a
second one in your text gives two placeholders for one image and the render fails. Send the bare
instruction and the same request succeeds in ~4 s.

### 7. Speed, on real pages

The 47-page standard, one request per page, at 150 dpi: **131.7 s end to end**, about **3 s a
page**, four pages skipped as blank and two cut short as loops. On a single page, against the
other OCR model on the same box:

| model | one page |
|---|---|
| `chandra-ocr-2-8bit-mlx` | 10.5 s |
| `Unlimited-OCR-bf16` | **4.7 s** (~126 tok/s) |

### 8. Quality, and where it slips

It reads chops and stamps as their own blocks, returns real HTML `<table>`s, marks logos, seals,
signatures and barcodes as picture regions rather than inventing text for them, and handles
Traditional Chinese almost perfectly.

Two slips worth carrying into any pipeline built on it:

* **It substitutes similar characters.** It returned **黄** where a page said **黃** — the
  simplified variant of a common Hong Kong surname — and **Bramwell** for **Branwell**. Any name it
  produces has to be checked against the document, not trusted.
* **On a thin page it invents.** See §3.

---

## Which Unlimited-OCR checkpoint?

`mlx-community/Unlimited-OCR-bf16` is the right default and there is no reason to move off it on a
Mac Studio.

| checkpoint | size | why / why not |
|---|---|---|
| [`mlx-community/Unlimited-OCR-bf16`](https://huggingface.co/mlx-community/Unlimited-OCR-bf16) | 6.22 GiB | **use this.** Full precision, and 6 GB is nothing on a 200 GB box. |
| [`mlx-community/Unlimited-OCR-8bit`](https://huggingface.co/mlx-community/Unlimited-OCR-8bit) | 3.66 GiB | only if memory is tight. Affine, group size 64. |
| [`mlx-community/Unlimited-OCR-mxfp8`](https://huggingface.co/mlx-community/Unlimited-OCR-mxfp8) | — | same reasoning as 8-bit. |
| [`sahilchachra/unlimited-ocr-4bit-mlx`](https://huggingface.co/sahilchachra/unlimited-ocr-4bit-mlx) / `mxfp4` | — | avoid for documents. |

The reasoning is §8: the model already confuses character pairs at **full** precision. Quantisation
is exactly the pressure that makes rare-glyph choices worse, and it buys you memory you are not
short of. If you do want to compare, this bench is the tool — same page, same prompt, swap the
model in the dropdown.

## The oMLX bugs

`OMLX-BUG-REPORT.md` holds two ready-to-post issues for
<https://github.com/jundot/omlx/issues>, with the evidence tables and minimal reproductions:

1. **Only the first image of a multi-image request reaches the model**, and a literal `<image>`
   returns a bare 500. One cause, two symptoms. oMLX's own README claims multi-image chat support,
   so this is a bug rather than a missing feature — and if it is fixed, the per-page loop goes away.
2. **`no_repeat_ngram_size` is accepted and silently ignored.** It should either work or be
   rejected with a 400. It is the knob the model's official pipeline relies on.

Neither is filed yet — search the tracker before posting, in case someone got there first.

## Layout

| file | what it is |
|---|---|
| `server.py` | the bench: static files, upload, and one SSE stream per run |
| `ui.html` | the three panels. No build step, no dependencies, no CDN |
| `ocr.py` | page rendering, the ink test, the `<|det|>` parser, one-shot OCR for `demo.py` |
| `config.py` | where the model lives |
| `demo.py` | one-shot: a PDF in, a static HTML page out |
| `test_multipage.py` | the multi-image experiment on its own |
| `samples/make_samples.py` | draws the sample pack |

## Licence and credit

MIT — see `LICENSE`. The model is Baidu's; the MLX conversions are the Hugging Face community's;
oMLX is [jundot/omlx](https://github.com/jundot/omlx). The panel idea comes from
[GMfatcat/Unlimited-OCR-Local](https://github.com/GMfatcat/Unlimited-OCR-Local), which does the
same job with a local model instead of a server. This repo is only the bench.
