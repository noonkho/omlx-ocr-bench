# Handoff: using PaddleOCR-VL through an oMLX server

Paste the block below into the other project's session. Everything in it was measured on
[noonkho/omlx-ocr-bench](https://github.com/noonkho/omlx-ocr-bench) against drawn pages with a
known answer key, on `PaddleOCR-VL-1.6` served by oMLX 0.6.4 over an OpenAI-compatible endpoint.

---

I want to use **PaddleOCR-VL** (served by an **oMLX** server, OpenAI-compatible
`/v1/chat/completions`) to turn PDFs into markdown **plus** the position of every piece of text,
so I can render the result as HTML with the boxes overlaid on the page image.

## First: verify all of this against the source before you write code

Do not trust the notes below without checking them. Read, in this order:

1. The model card and its files: <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6>
   (also the 0.9B base card, <https://huggingface.co/PaddlePaddle/PaddleOCR-VL>). Look for the
   documented prompt strings, the chat template, and any special tokens.
2. The GitHub repo: <https://github.com/PaddlePaddle/PaddleOCR> — in particular
   `docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md`, and search the source for the prompt
   strings and for how the pipeline post-processes the model's output.
3. The paper, *PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact
   Vision-Language Model*, <https://arxiv.org/pdf/2510.14528>.

Report back anything that contradicts what follows — these notes are from one measured setup,
and the model has point releases.

## The one thing that matters most

**For this model the prompt is a mode switch, not phrasing.** The wrong prompt does not produce a
worse answer, it produces a confidently wrong one with no error. Measured on a page of prose,
`Table Recognition:` scored **0%** against the answer key and reported nothing wrong. Choosing the
right prompt was worth **31 accuracy points** over a sensible-sounding guess.

## The prompts it documents

| prompt | what it is for |
|---|---|
| `Spotting:` | a whole page: every line of text **with its bounding box** |
| `OCR:` | text of one region, no coordinates |
| `Table Recognition:` | a cropped table → OTSL markup |
| `Formula Recognition:` | a cropped formula → LaTeX |
| `Chart Recognition:` | a cropped chart |
| `Seal Recognition:` | a cropped stamp or chop |

**PaddleOCR-VL is designed as the recognition half of a two-stage pipeline.** A separate layout
model (PP-DocLayoutV2) detects regions, their types and the reading order; each crop then goes to
the VLM with its task named. Using it whole-page is off-label — it just happens to work very well.

Two ways to build, and you should decide which you need:

* **One pass, `Spotting:` on the whole page.** Simple, fast (~2–4 s a page), gives you text and
  position for every line. **It does not give you structure** — no table markup, no headings, no
  reading order beyond the order the lines arrive in.
* **Two stage, as designed.** Layout model first, then `Table Recognition:` on table crops,
  `Formula Recognition:` on formulas, `OCR:` or `Spotting:` on text. More moving parts, but this
  is the only way to get real tables.

Measured, one page, whole-page mode:

| prompt | grounded | text accuracy | lines found |
|---|---|---|---|
| **`Spotting:`** | **yes** | **99.5%** | **63/63** |
| `OCR:` | no | 80.1% | 32/63 |
| `Seal Recognition:` | no | 70.8% | 20/63 |
| `Chart Recognition:` | no | 19.2% | 0/63 |
| `Table Recognition:` | no | 0.0% | 0/63 |
| `Formula Recognition:` | no | 0.0% | 0/63 |

## The output format — this is what you have to parse

### `Spotting:` — text with positions

One **line per line of text**. The words come first, then **exactly eight** `<|LOC_n|>` tokens:

```
BRANWELL COUNTY COUNCIL<|LOC_160|><|LOC_43|><|LOC_499|><|LOC_43|><|LOC_499|><|LOC_58|><|LOC_160|><|LOC_58|>
Licensing and Environmental Health Service<|LOC_160|><|LOC_62|><|LOC_469|><|LOC_62|><|LOC_469|><|LOC_74|><|LOC_160|><|LOC_74|>
```

Rules, all verified:

* The eight numbers are **four corner points**, `x0 y0  x1 y1  x2 y2  x3 y3`, clockwise from the
  top left. Read them as pairs, not as two corners.
* **They are a quadrilateral, not a rectangle.** Most are axis-aligned, but on one page 3 of 62
  were not. If you assume a rectangle you will mis-place text on skewed or photographed pages.
  Draw the polygon, or take the bounding box of the four points if a rectangle is all you need.
* Coordinates are **normalised 0–1000** against the image you sent, both axes. To get pixels:
  `x_px = x / 1000 * image_width`, `y_px = y / 1000 * image_height`. Nothing depends on DPI.
* Strip the `<|LOC_n|>` tokens to get the text; what is left is plain text, **not markdown** —
  no `##`, no `|` tables, no `<table>`, no `$$`.
* A line may arrive with **no** LOC tokens. Keep the text and record that it has no position,
  rather than inventing one.
* Suggested regex: `<\|LOC_(\d+)\|>`. Split each line on it: the even parts are the text, the odd
  parts are the numbers — one pass, no second scan.

### `Table Recognition:` — structure, on a crop

Returns **OTSL**, not HTML and not markdown:

```
<fcel>Activity<fcel>Monday to Thursday<fcel>Friday and Saturday<fcel>Sunday<nl><fcel>Sale of alcohol, on sales<fcel>11:00 – 23:00<fcel>…<nl>
```

`<fcel>` starts a cell, `<nl>` ends a row. Check the repo for the full tag set — there are tags
for spanning and empty cells (`<ecel>`, `<lcel>`, `<ucel>`, `<xcel>`) that this one table did not
exercise. Convert OTSL to `<table>` yourself; the accuracy on the cells was perfect.

## Settings

* `temperature: 0`. Deterministic, which is what you want for a document.
* `max_tokens`: 8192 is comfortable for a dense A4 page under `Spotting:` (~1,100 output tokens
  observed). Watch `finish_reason` — `length` means you truncated.
* **Do not set `frequency_penalty` or `repetition_penalty`.** This matters: `frequency_penalty:
  0.8` is the standard cure for a different OCR model's repeat loop, and applying it here took one
  page from 99.3% down to 56.0%. Leave the penalties alone unless you measure a reason.
* Render pages at **150 dpi**. Measured on a related model, dropping to 96 dpi halves the input
  tokens and saves no time at all, because the time goes on generating the answer.
* If your oMLX server is remote, send the image as a `data:image/png;base64,…` URL in an
  `image_url` content part.

## Rendering it as HTML

The shape that worked:

1. Render each PDF page to PNG at a known pixel size (`pypdfium2`, `page.render(scale=dpi/72)`).
2. Send the page, parse the lines into `{text, points[4], hasPosition}`.
3. In the browser, put the page image in a `position: relative` container at whatever width you
   like, and place each box as a **percentage** — `left: x/10 %`, `top: y/10 %`, since the
   coordinates are 0–1000. Percentages mean the overlay scales with the image and you never
   convert to pixels in the page.
4. For a true quadrilateral use an SVG `<polygon>` over the image rather than a `<div>`; a `div`
   can only be a rectangle.
5. Link each box to its text both ways — hovering either should highlight the other. That is what
   makes the overlay useful for checking rather than just decorative.

## Known failure modes to design around

* **Blank pages.** Some OCR models emit `1. 2. 3. 4. …` until they hit the token cap on a page
  with nothing on it. Measure the ink on the render and skip pages below roughly 0.08% dark
  pixels — a truly blank page measures 0.0000, the sparsest real page tested 0.0073. Cheap, and
  it removes the failure entirely.
* **Silent truncation.** Always check `finish_reason`.
* **No structure from `Spotting:`.** If your documents are table-heavy, plan for the two-stage
  pipeline from the start rather than trying to reconstruct tables from line positions.
* **Verify against a known answer.** The reason these numbers exist is that the pages were drawn,
  so their text and boxes were known in advance. If accuracy matters to your project, make a few
  pages whose content you control and score against them — a wrong prompt or a stray sampling
  knob is invisible without that.

---

*Findings from [noonkho/omlx-ocr-bench](https://github.com/noonkho/omlx-ocr-bench), September
2026: PaddleOCR-VL-1.6 on oMLX 0.6.4, scored against drawn pages with an answer key. It beat
`Unlimited-OCR-bf16` and `chandra-ocr-2-8bit-mlx` on all seven samples, with 100% position
accuracy on every one, at 0.9B parameters.*
