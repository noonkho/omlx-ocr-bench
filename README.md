# Unlimited-OCR bench

A three-panel workbench for driving [`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR)
through an [oMLX](https://github.com/jundot/omlx) server, and for finding out what the pair can and
cannot actually do.

It answers one question with pictures rather than opinions: **can this model turn a PDF — with
tables, stamps, signatures and logos — into markdown, over an OpenAI-compatible endpoint?**

The short answer is yes for a page, no for a whole document, and there is a knob you must set.
Everything below was measured, not assumed. Every sample in the repo is synthetic.

## Run it

```bash
uv run server.py
```

Then open <http://127.0.0.1:4700>. Nothing else is needed — no Docker, no local model. The bench
talks to an oMLX server that has already downloaded and loaded the model.

```bash
uv run demo.py samples/synthetic_hk.pdf --open   # one-shot, writes a static HTML page to out/
uv run test_multipage.py                         # can one request carry several pages?
```

Server and defaults come from env vars, read in `config.py`:

| variable | default |
|---|---|
| `OMLX_BASE_URL` | `http://100.77.164.75:8000/v1` |
| `OMLX_API_KEY` | `testing` |
| `OMLX_OCR_MODEL` | `Unlimited-OCR-bf16` |
| `OMLX_OCR_PROMPT` | `Multi page parsing.` |
| `OMLX_OCR_DPI` | `150` |

## The three panels

**Left — set up.** Drag and drop a PDF or an image, or click one of the built-in synthetic pages.
Then: the prompt, with presets; base URL, API key and a **Test connection** button that lists every
model on the server, shows the chosen one's context window, and says so plainly if it is not an
Unlimited-OCR model; every sampling knob oMLX accepts; render DPI; and the choice between one
request per page and the whole document in one request.

**Middle — the document.** Every page as the model sees it, at the DPI you chose, scrolling.

**Right — what came back.** Tokens appear as they arrive. Three views:

* **Rendered** — the answer as a document. The model's own HTML tables are kept as tables.
* **Raw** — exactly what the model emitted, `<|det|>` markers and all, dimmed so the words stand out.
* **Boxes** — the page with the model's own layout boxes drawn on it, coloured by kind, with the
  text beside it. Hover a box to highlight its text, and the other way round.

**Copy** and **.md** take the answer out as a markdown file — the thing a downstream pipeline keeps.

Everything goes through the local Python process rather than straight from the browser, because
the oMLX box sets no CORS headers and a browser would refuse the call.

---

## What was measured

`Unlimited-OCR-bf16` on oMLX 0.6.4, Mac Studio M3 Ultra, over a VPN. 2026-09-18.

### 1. Set `frequency_penalty: 0.8`, or a near-empty page will cost you a minute

This is the single most important finding, and the bench ships with it on by default.

On a page holding four short lines, the model repeats the same block until it hits the token cap —
**29 s and 8,192 wasted tokens on a nearly blank page.** In a real bundle, separator pages, blank
backs of scans and exhibit dividers make that a large share of a run.

| setting | near-empty page 2 | dense page 1 |
|---|---|---|
| none | 17.4 s, 4096 tokens, `finish_reason: length` | 5.1 s, 413 tokens, 14 blocks |
| `repetition_penalty: 1.08` | still runs to the cap | — |
| `repetition_penalty: 1.15` | 0.9 s, 103 tokens, stops | **damages layout**: merges the parties into a bogus one-row `<table>` and loses a word (0.90 similarity) |
| `frequency_penalty: 0.3` / `0.5` | still runs to the cap | — |
| **`frequency_penalty: 0.8`** | **1.5 s, 340 tokens, `finish_reason: stop`** | **byte-identical to none — 413 tokens, the same 14 blocks, similarity 1.000** |

So `frequency_penalty: 0.8` ends the loop and, on the page that matters, changes nothing at all.
`repetition_penalty` is the wrong tool: it is a blunt ban on any token seen before, and a legal
document legitimately repeats words.

The official pipeline does not use either — it runs an n-gram logits processor with
`no_repeat_ngram_size=35`. **oMLX accepts `no_repeat_ngram_size` and then ignores it:** the same
request with and without it returns byte-identical output. That is worth a bug report of its own
(below).

The bench also keeps a belt-and-braces guard (`server.RepeatWatch`), for when you turn the penalty
down to see what happens: if the answer becomes the same line over and over, the stream is
abandoned mid-flight rather than paid for to the cap. What already arrived is kept and flagged —
the guard exists to save tokens, not to hide the page.

### 2. The `1. 2. 3. 4. 5. …` is a preamble, and it should be thrown away

On a near-empty page the answer starts with a run of bare numbers **before the first `<|det|>`
marker**. It is not inside any box the model drew, and it survives every prompt tried
(`Multi page parsing.`, `document parsing.`, `Free OCR.`, `Extract the text in the image.`, `OCR.`).

Anything before the first marker is untrustworthy in general, not just here: on a dense page,
`Multi page parsing.` produced `2014年1月1日` — **a date that is nowhere on the page.** With
`document parsing.` the same page produced no invented preamble at all.

So the bench classifies it rather than trusting it. `ocr.parse` labels a numeric run `noise` and any
other loose text `unboxed`; both are shown in the right panel with a warning stripe, and both are
**excluded from the exported markdown**. If you only ever want one prompt, prefer
`document parsing.` for single-page requests.

### 3. The documented prompt returns HTTP 500

The model card writes the prompt as `<image>Multi page parsing.` **That literal `<image>` makes
oMLX return 500.** oMLX inserts its own image placeholder when it renders the chat template, so a
second one in your text gives two placeholders for one image and the render fails.

Send the bare instruction — `Multi page parsing.` — and the same request succeeds in ~4 s.

### 4. Multi-page in ONE request does NOT work on oMLX

This is the reason to want this model, and it does not hold up today. `prompt_tokens` is
**identical at 909 whether the request carries one image, two, or three** — oMLX passes only the
first image to the model and silently drops the rest.

| images in the request | `prompt_tokens` | what came back |
|---|---|---|
| 1 | 909 | the English page |
| 2 | 909 | the English page only |
| 3 | 909 | the English page only |

A "speed-up" measured this way is not a speed-up: the second page was never read. The cause is not
a mystery — the oMLX pull request that added this model
([#2328](https://github.com/jundot/omlx/pull/2328)) states that it "reproduces the single-`<image>`
multi-page prompt semantics", i.e. exactly one placeholder is rendered however many images arrive.
That one decision explains both this and the 500 above.

The bench keeps the mode anyway (**Requests → whole document in one**) so the bug can be reproduced
in one click, and so it starts working the day oMLX fixes it. Until then, one page per request.

### 5. It is about twice as fast as `chandra-ocr-2`, on the same page

| model | page 1 of the synthetic HK writ |
|---|---|
| `chandra-ocr-2-8bit-mlx` | 10.5 s |
| `Unlimited-OCR-bf16` | **4.7 s** (~126 tok/s) |

### 6. Quality is good, and it sees stamps

On a synthetic Hong Kong High Court page it returned every heading, both parties, the action
number, the body text, **a real HTML `<table>`** for the costs table, and — importantly — it read
the red chop, `FILED 20 FEB 2025`, as its own block. Traditional Chinese came back almost perfectly,
including the chop (`已存案 / 2025年2月20日`).

**One flaw that matters in Hong Kong:** it returned **黄** where the page said **黃** — the
simplified variant of a very common HK surname. Any name it produces has to be checked against the
document, not trusted.

It also reads plain images (`samples/synthetic_zh.png` is a PNG) and labels picture regions, so a
figure is marked rather than silently skipped.

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

The reasoning is the 黄/黃 error above: the model already confuses a character pair at **full**
precision. Quantisation is exactly the pressure that makes rare-glyph choices worse, and it buys
you memory you are not short of. If you ever do compare them, this bench is the tool — same page,
same prompt, swap the model in the dropdown.

---

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
| `server.py` | the bench: static files, an upload endpoint, and one SSE stream per run |
| `ui.html` | the three panels. No build step, no dependencies, no CDN |
| `ocr.py` | rendering pages, the `<|det|>` parser, and one-shot OCR for `demo.py` |
| `config.py` | where the model lives |
| `demo.py` | one-shot: a PDF in, a static HTML page out |
| `test_multipage.py` | the multi-image experiment on its own |
| `samples/` | two synthetic pages — an English HK writ with a chop, a Chinese notice |

## Licence and credit

The model is Baidu's; the MLX conversions are the Hugging Face community's; oMLX is
[jundot/omlx](https://github.com/jundot/omlx). The panel idea comes from
[GMfatcat/Unlimited-OCR-Local](https://github.com/GMfatcat/Unlimited-OCR-Local), which does the
same job with a local model instead of a server. This repo is only the bench.
