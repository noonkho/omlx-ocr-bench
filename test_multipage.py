#!/usr/bin/env python3
"""Can ONE request carry several page images? That is the whole reason to care about this model.

"Multi page parsing" in the model's own prompt suggests it can take a whole document at once.
This asks the server directly and compares against the same pages sent one at a time.

The answer on oMLX 0.6.4 is no, and `prompt_tokens` is the proof: it does not change when more
images are added, so the extra images were never tokenised. Any "speed-up" printed below is
therefore fake — the other pages were never read. See README and OMLX-BUG-REPORT.md.
"""
import base64, json, sys, time, urllib.request
from pathlib import Path

import config
from ocr import render_pages

HERE = Path(__file__).parent


def send(images, prompt=config.PROMPT, max_tokens=8192):
    content = [{"type": "text", "text": prompt}]
    for png in images:
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(png).decode()}})
    body = {"model": config.MODEL, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(config.BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {config.API_KEY}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.loads(r.read())
        return time.time() - t0, d["choices"][0]["message"]["content"], d.get("usage", {}), ""
    except Exception as e:
        detail = ""
        if hasattr(e, "read"):
            try: detail = e.read()[:200].decode()
            except Exception: pass
        return time.time() - t0, "", {}, f"{type(e).__name__}: {e} {detail}"


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "samples" / "synthetic_hk.pdf"
    pages = [png for png, _ in render_pages(src)]
    print(f"{src.name}: {len(pages)} pages\n")

    print("A. one request per page")
    one_total = 0.0
    for i, png in enumerate(pages, 1):
        dt, text, usage, err = send([png])
        one_total += dt
        print(f"   page {i}: {dt:5.1f}s  {usage.get('completion_tokens',0):4d} out tok"
              f"  {err or repr(text[:60])}")
    print(f"   TOTAL {one_total:.1f}s\n")

    print("B. all pages in ONE request")
    dt, text, usage, err = send(pages)
    if err:
        print(f"   FAILED after {dt:.1f}s: {err}")
    else:
        print(f"   {dt:5.1f}s  {usage.get('completion_tokens',0)} out tok / "
              f"{usage.get('prompt_tokens',0)} in tok")
        print(f"   returned {len(text)} chars")
        print("   " + text[:400].replace("\n", "\n   "))
        if dt:
            print(f"\n   apparent speed-up vs one-per-page: {one_total/dt:.2f}x")
            print("   ^ NOT real. Compare prompt_tokens above against a single-image request:")
            print("     if it is unchanged, only the first page was ever read.")


if __name__ == "__main__":
    main()
