"""Where the model lives. Override with env vars rather than editing this file."""
import os

BASE_URL = os.environ.get("OMLX_BASE_URL", "http://100.77.164.75:8000/v1")
API_KEY = os.environ.get("OMLX_API_KEY", "testing")
MODEL = os.environ.get("OMLX_OCR_MODEL", "Unlimited-OCR-bf16")

#: The prompt that actually works on oMLX.
#:
#: The model card writes it as "<image>Multi page parsing." — and that literal `<image>` makes
#: oMLX return **HTTP 500**. oMLX inserts its own image placeholder when it builds the chat
#: template, so a second one in your text produces two placeholders for one image and the
#: template render fails. Measured, not guessed: the same request with the prefix removed
#: succeeds in ~4s. Use the bare instruction.
PROMPT = os.environ.get("OMLX_OCR_PROMPT", "Multi page parsing.")

#: Render DPI. 150 is enough for typed English; Traditional Chinese benefits from more.
DPI = int(os.environ.get("OMLX_OCR_DPI", "150"))

#: The knob that stops a near-empty page repeating itself to the token cap.
#:
#: Measured, not guessed — see README. At 0.8 a blank-ish page ends in 1.5s instead of running to
#: the cap, and a dense page comes back BYTE-IDENTICAL to no penalty at all. The model's official
#: pipeline uses `no_repeat_ngram_size=35` instead, which oMLX accepts and then ignores.
FREQUENCY_PENALTY = float(os.environ.get("OMLX_OCR_FREQUENCY_PENALTY", "0.8"))
