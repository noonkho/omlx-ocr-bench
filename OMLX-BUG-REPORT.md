# Bug report to file at https://github.com/jundot/omlx/issues

Copy everything below the line. Replace the version numbers if yours differ.

---

## Title

Only the first image of a multi-image chat request reaches the model (VLM); a literal `<image>` in the text returns HTTP 500

## Summary

On `/v1/chat/completions` with a vision model, a message whose `content` carries **several**
`image_url` parts is accepted, but only the **first** image is actually given to the model. The
extra images are dropped silently — no error, no warning — and `usage.prompt_tokens` does not
change at all.

Separately, putting the model's own image placeholder token (`<image>`) in the text returns
**HTTP 500 Internal Server Error** with no detail.

The README says oMLX "Supports multi-image chat, base64/URL/file image inputs", so this looks
like a bug rather than an unsupported feature.

## Environment

| | |
|---|---|
| oMLX | 0.6.4 (the .dmg app) |
| Host | Mac Studio, M3 Ultra, 200 GB |
| Model | `Unlimited-OCR-bf16` (an MLX conversion of `baidu/Unlimited-OCR`) |
| Client | plain `urllib` POST to `/v1/chat/completions` |

## Evidence — `prompt_tokens` never changes

The same request, with one, two and three **different** images in the one user message. Each
image is a full page render, ~130 KB PNG, 1241×1754.

| images in the request | `usage.prompt_tokens` | content of the answer |
|---|---|---|
| 1 (English page) | **909** | the English page |
| 2 (English + Chinese page) | **909** | the English page only |
| 3 (English + Chinese + page 2) | **909** | the English page only |

909 for one image and 909 for three is the whole report: the extra images are never tokenised,
so they were never passed to the model. The answer confirms it — with an English page first and a
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

The two symptoms look like one cause: the chat template inserts **exactly one** image placeholder
regardless of how many images the request carries. A second placeholder in the user's text then
makes two placeholders for one image and the render raises (→ 500); and two images with one
placeholder means the second image has nowhere to go (→ silently dropped).

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

ask(["page1.png"])                          # 909
ask(["page1.png", "page2.png"])             # 909  <-- expected to be larger
ask(["page1.png", "page2.png", "page3.png"])# 909  <-- expected to be larger
```

## Expected

* `prompt_tokens` grows with each image, and every image in the message is given to the model.
* A literal `<image>` in the text is either handled (treated as the placeholder for the next
  image) or rejected with a **400 and a message saying so** — not a bare 500.

## Why it matters

`baidu/Unlimited-OCR` is built for "one-shot long-horizon parsing" — a whole multi-page document
in one request. With only the first image reaching the model, that capability cannot be used
through oMLX at all, and callers must fall back to one request per page.

## Possibly related

Several `mlx-vlm` issues and PRs touch this model's prompt formatting, e.g.
`Blaizzy/mlx-vlm#2164` and `#2169`. If oMLX delegates template rendering to `mlx-vlm`, the
placeholder count may be decided there.
