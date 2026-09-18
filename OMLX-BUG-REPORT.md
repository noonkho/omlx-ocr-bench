# Two issues to file at https://github.com/jundot/omlx/issues

Each section below is one complete issue. Copy everything under its rule. Replace the version
numbers if yours differ, and **search the tracker first** in case someone got there before you.

**Status:** Issue 1 is filed as [omlx#3740](https://github.com/jundot/omlx/issues/3740). Issue 2
was not in the tracker as of 2026-09-19. Related and already open:
[omlx#2331](https://github.com/jundot/omlx/issues/2331) asks for the VLM processor settings that
Unlimited-OCR's own multi-page mode needs, and
[omlx#2424](https://github.com/jundot/omlx/issues/2424) reports garbled Chinese from the same
model — which matches the 黄/黃 substitution measured here.

---
---

# Issue 1

## Title

Unlimited-OCR: only the first image of a multi-image chat request reaches the model, and a literal `<image>` in the text returns HTTP 500

## Summary

On `/v1/chat/completions` with `Unlimited-OCR`, a message whose `content` carries **several**
`image_url` parts is accepted, but only the **first** image is given to the model. The extra
images are dropped silently — no error, no warning — and `usage.prompt_tokens` does not change at
all.

Separately, putting the model's own image placeholder token (`<image>`) in the text returns
**HTTP 500 Internal Server Error** with no detail.

I believe these are one cause, and that #2328 names it. That PR says it "reproduces the
single-`<image>` multi-page prompt semantics" — exactly one placeholder is rendered however many
images the request carries. So a second placeholder in the user's text gives two placeholders for
one image and the template render raises (→ 500); and two images with one placeholder means the
second image has nowhere to go (→ silently dropped).

The README says oMLX "Supports multi-image chat, base64/URL/file image inputs", so from a caller's
side this reads as a bug rather than an unsupported feature.

## Environment

| | |
|---|---|
| oMLX | 0.6.4 (the .dmg app) |
| Host | Apple Silicon Mac (M3 generation) |
| Model | `Unlimited-OCR-bf16` (an MLX conversion of `baidu/Unlimited-OCR`) |
| Client | plain `urllib` POST to `/v1/chat/completions` |

## Evidence — `prompt_tokens` never changes

The same request, with one, two and three **different** images in the one user message. Each image
is a full page render, ~130 KB PNG, 1241×1754.

| images in the request | `usage.prompt_tokens` | content of the answer |
|---|---|---|
| 1 (English page) | **909** | the English page |
| 2 (English + Chinese page) | **909** | the English page only |
| 3 (English + Chinese + page 2) | **909** | the English page only |

909 for one image and 909 for three is the whole report: the extra images are never tokenised, so
they were never passed to the model. The answer confirms it — with an English page first and a
Traditional Chinese page second, nothing from the Chinese page appears in the output.

## Evidence — `<image>` returns 500

`baidu/Unlimited-OCR`'s own model card documents the prompt as `<image>Multi page parsing.`
Sending that exact string fails:

```
HTTP 500
{"error":{"message":"Internal server error","type":"server_error","param":null,"code":null}}
```

Removing the `<image>` prefix and sending only `Multi page parsing.` succeeds in ~4 s.

| prompt text | images | result |
|---|---|---|
| `Multi page parsing.` | 1 | 200 OK, ~4 s |
| `<image>Multi page parsing.` | 1 | **500** |
| `<image>Free OCR.` | 1 | **500** |
| `<image><image>Multi page parsing.` | 2 | **500** |

## Minimal reproduction

```python
import base64, json, urllib.request

BASE, KEY, MODEL = "http://YOUR-HOST:8000/v1", "YOUR-KEY", "Unlimited-OCR-bf16"

def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()

def ask(paths, text="Multi page parsing."):
    content = [{"type": "text", "text": text}]
    for p in paths:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + b64(p)}})
    body = {"model": MODEL, "max_tokens": 512, "temperature": 0,
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    print(len(paths), "image(s) ->", d["usage"]["prompt_tokens"], "prompt tokens")

ask(["page1.png"])                           # 909
ask(["page1.png", "page2.png"])              # 909  <-- expected to be larger
ask(["page1.png", "page2.png", "page3.png"]) # 909  <-- expected to be larger
```

## Expected

* One `<image>` placeholder is rendered **per image** in the message, so `prompt_tokens` grows with
  each image and every image reaches the model.
* A literal `<image>` in the text is either handled (treated as the placeholder for the next image)
  or rejected with a **400 that says so** — not a bare 500.

## Why it matters

`baidu/Unlimited-OCR` is built for "one-shot long-horizon parsing" — a whole multi-page document in
one request, up to roughly 40 pages in its 32K context. With only the first image reaching the
model, that capability cannot be used through oMLX at all, and callers must fall back to one
request per page. It is the single reason to choose this model over a page-at-a-time OCR model.

## Possibly related

* #2314 / #2328 — the issue and PR that added this model.
* #3508 / #3512 — multi-image vision cache concatenation, which suggests multi-image works for
  other VLMs (e.g. gemma-4) and that this is specific to the Unlimited-OCR path.

---
---

# Issue 2

## Title

`no_repeat_ngram_size` is accepted on `/v1/chat/completions` and then silently ignored

## Summary

`no_repeat_ngram_size` is accepted by the API — no 400, no warning — and has **no effect
whatsoever**. The same request with and without it returns byte-identical output, in the same
number of tokens, with the same `finish_reason`.

This matters because it is the knob `baidu/Unlimited-OCR`'s own official pipeline relies on: it
generates with `no_repeat_ngram_size=35` and `ngram_window=128`. Without it, the model falls into a
repeat loop on any near-empty page and burns the whole `max_tokens` budget.

Either implement it, or reject unknown sampling fields with a 400. Silently accepting a knob that
does nothing is the worst of the three, because a caller cannot tell the difference between "the
guard is on and this page still loops" and "the guard was never applied".

## Environment

| | |
|---|---|
| oMLX | 0.6.4 (the .dmg app) |
| Host | Apple Silicon Mac (M3 generation) |
| Model | `Unlimited-OCR-bf16` |

## Evidence

One page holding four short lines, `max_tokens: 1024`, `temperature: 0`:

| request | time | `completion_tokens` | `finish_reason` | distinct lines |
|---|---|---|---|---|
| baseline | 7.0 s | 1024 | `length` | 130 / 382 |
| `no_repeat_ngram_size: 35` | 3.7 s | 1024 | `length` | 130 / 382 |
| `no_repeat_ngram_size: 35`, `ngram_window: 128` | 3.7 s | 1024 | `length` | 130 / 382 |

The three answers are identical strings. The tail of each is the same line repeated to the cap:

```
<|det|>text [50, 154, 320, 172]<|/det|>To: The Registrar, High Court
<|det|>text [50, 154, 320, 172]<|/det|>To: The Registrar, High Court
<|det|>text [50, ...
```

With `no_repeat_ngram_size=35` honoured, that 35-token sequence could not repeat.

For contrast, the knobs that **are** implemented do change the result on the same page:

| request | time | `completion_tokens` | `finish_reason` |
|---|---|---|---|
| `frequency_penalty: 0.8` | 1.5 s | 340 | `stop` |
| `repetition_penalty: 1.15` | 0.9 s | 103 | `stop` |

So this is specific to `no_repeat_ngram_size`, not to penalties in general.

## Minimal reproduction

```python
import base64, json, urllib.request

BASE, KEY, MODEL = "http://YOUR-HOST:8000/v1", "YOUR-KEY", "Unlimited-OCR-bf16"

def ask(extra):
    img = base64.b64encode(open("near_empty_page.png", "rb").read()).decode()
    body = {"model": MODEL, "max_tokens": 1024, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Multi page parsing."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}}]}]}
    body.update(extra)
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"]

a = ask({})
b = ask({"no_repeat_ngram_size": 35})
print("identical:", a == b)          # True
```

Use any page with very little text on it — a separator page, a blank scan back, an exhibit divider.

## Expected

Either:

* `no_repeat_ngram_size` (and `ngram_window`) are applied as an n-gram logits processor, matching
  `transformers` and the model's official pipeline; or
* the request is rejected with a 400 naming the unsupported field.

## Why it matters

Document models are the models most likely to loop, because blank and near-blank pages are normal
in a real bundle: separator sheets, backs of scans, exhibit dividers. On this model, one such page
costs 29 s and 8,192 wasted tokens. The n-gram guard is the tool the model's authors chose for
exactly that, and today there is no way to reach it through oMLX.
