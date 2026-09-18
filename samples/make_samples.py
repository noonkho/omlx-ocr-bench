#!/usr/bin/env python3
"""Draw the sample pack: `uv run samples/make_samples.py`

Every page here is **drawn from scratch**, not taken from the web. That is on purpose:

* a document found online carries someone's copyright and, worse, someone's real names, and this
  repo is public;
* a drawn page has a known ground truth, so "did the model read it right?" has an answer;
* it is reproducible — run this file and get the same pages back.

Every name, number, address and reference below is invented. None of them belongs to a real
person, firm, school or authority, and none is reused from anywhere.

Between them the pages cover the document types worth testing — a legal instrument, a government
permit, a school transcript, a commercial invoice, a Chinese-language notice — and the features
worth testing: body text, headings, numbered clauses, real tables, a figure the model should mark
as a picture, boxed callouts, drawn logos and crests, handwritten signatures, embossed and inked
seals, a near-blank divider page, Traditional Chinese, and two of the pages again as bad phone
photos: skewed, noisy, lit from one side.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).parent
TRUTH_DIR = HERE / "truth"

#: What is on the page being drawn, filled in as it is drawn: one entry per line of text and one
#: per graphic. This is the whole reason to draw the samples rather than download them — it is the
#: answer key the bench scores a model against. Boxes are 0-1000 normalised, like the model's own.
TRUTH: list[dict] = []

W, H = 1241, 1755                        # A4 at 150 dpi, matching the bench's default render
INK, GREY, RED, BLUE = (26, 26, 26), (110, 110, 110), (178, 34, 34), (30, 60, 140)
GREEN, GOLD = (24, 92, 60), (150, 118, 20)

FONTS = {
    "serif": ["/System/Library/Fonts/Supplemental/Times New Roman.ttf"],
    "serif_bold": ["/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"],
    "sans": ["/System/Library/Fonts/Supplemental/Arial.ttf"],
    "sans_bold": ["/System/Library/Fonts/Supplemental/Arial Bold.ttf"],
    "mono": ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Courier.ttc"],
    "cjk": ["/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"],
}


def note(label: str, x0, y0, x1, y1, body: str = "") -> None:
    """Record one thing on the page, in the model's own 0-1000 coordinate space."""
    TRUTH.append({"label": label, "text": body,
                  "box": [round(x0 * 1000 / W), round(y0 * 1000 / H),
                          round(x1 * 1000 / W), round(y1 * 1000 / H)]})


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for path in FONTS[kind]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W, H), "white")
    return im, ImageDraw.Draw(im)


def text(d, xy, s, kind="serif", size=22, fill=INK, label="text", **kw):
    f = font(kind, size)
    d.text(xy, s, font=f, fill=fill, **kw)
    x, y = xy
    if kw.get("anchor", "").startswith("r"):              # right-aligned, as on a page number
        x -= d.textlength(s, font=f)
    note(label, x, y, x + d.textlength(s, font=f), y + size * 1.25, s)


def centred(d, y, s, kind="serif_bold", size=30, fill=INK, label="title"):
    f = font(kind, size)
    x = (W - d.textlength(s, font=f)) / 2
    d.text((x, y), s, font=f, fill=fill)
    note(label, x, y, x + d.textlength(s, font=f), y + size * 1.25, s)


def paragraph(d, x, y, body, width=1080, size=21, kind="serif", leading=31, fill=INK) -> int:
    """Greedy wrap, one word at a time for Latin and one glyph at a time for CJK."""
    f = font(kind, size)
    cjk = any(ord(ch) > 0x2E80 for ch in body)
    units = list(body) if cjk else body.split()
    join = "" if cjk else " "
    line = ""
    for unit in units:
        trial = line + join + unit if line else unit
        if d.textlength(trial, font=f) > width and line:
            d.text((x, y), line, font=f, fill=fill)
            note("text", x, y, x + d.textlength(line, font=f), y + size * 1.2, line)
            y += leading
            line = unit
        else:
            line = trial
    if line:
        d.text((x, y), line, font=f, fill=fill)
        note("text", x, y, x + d.textlength(line, font=f), y + size * 1.2, line)
        y += leading
    return y


def table(d, x, y, rows, widths, size=20, header=True, row_h=40) -> int:
    """A ruled table — the thing the model is supposed to give back as a real `<table>`."""
    top = y
    for r, row in enumerate(rows):
        cx = x
        if header and r == 0:
            d.rectangle([x, y, x + sum(widths), y + row_h], fill=(238, 238, 234))
        for cell, w in zip(row, widths):
            d.rectangle([cx, y, cx + w, y + row_h], outline=(150, 150, 150), width=1)
            kind = "sans_bold" if header and r == 0 else "sans"
            if any(ord(ch) > 0x2E80 for ch in str(cell)):
                kind = "cjk"
            text(d, (cx + 9, y + (row_h - size) / 2 - 2), str(cell), kind, size)
            cx += w
        y += row_h
    # …and the table as a whole, so "did it see a table here?" can be asked separately.
    note("table", x, top, x + sum(widths), y,
         " ".join(str(c) for row in rows for c in row))
    return y


def signature(d, x, y, seed=1, width=230, colour=(20, 30, 90)):
    """A scrawl: several loops of different size, a break, and an underline.

    Handwriting is the thing OCR models most often drop, so it has to look written rather than
    plotted — one clean sine wave is neither.
    """
    rng = random.Random(seed)
    loops = rng.uniform(3.5, 6.5)
    stroke, lifted = [], []
    for t in range(0, 70):
        f = t / 69
        px = x + f * width
        py = (y + math.sin(f * loops * math.pi + seed) * rng.uniform(13, 21)
              - f * rng.uniform(6, 20) + math.sin(f * 23) * 2.5 + rng.uniform(-2.5, 2.5))
        (lifted if 0.55 < f < 0.62 else stroke).append((px, py))
        if 0.55 < f < 0.62 and len(stroke) > 1:          # the pen leaves the paper once
            d.line(stroke, fill=colour, width=rng.choice([2, 3]), joint="curve")
            stroke = []
    if len(stroke) > 1:
        d.line(stroke, fill=colour, width=3, joint="curve")
    d.arc([x - 6, y - 26, x + 54, y + 24], 120, 340, fill=colour, width=3)   # an initial capital
    d.line([(x + 12, y + 36), (x + width - 14, y + 28 + rng.uniform(-4, 4))],
           fill=colour, width=2)
    note("signature", x - 6, y - 28, x + width, y + 40)


def signature_block(d, x, y, name, role, seed, width=310):
    text(d, (x, y), "Signed", "sans", 16, GREY)
    signature(d, x, y + 56, seed=seed, width=width - 70)
    d.line([(x, y + 104), (x + width, y + 104)], fill=GREY, width=1)
    text(d, (x, y + 112), name, "sans_bold", 18)
    text(d, (x, y + 136), role, "sans", 17, GREY)


def chop(im, x, y, lines, seed=3, radius=78, colour=(200, 30, 30)):
    """A round inked seal, drawn on its own layer so it sits crooked and semi-transparent —
    which is how a real chop lands on a page, and what makes it hard to read."""
    layer = Image.new("RGBA", (radius * 2 + 40, radius * 2 + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    c = radius + 20
    r, g, b = colour
    d.ellipse([c - radius, c - radius, c + radius, c + radius], outline=(r, g, b, 215), width=5)
    d.ellipse([c - radius + 12, c - radius + 12, c + radius - 12, c + radius - 12],
              outline=(r, g, b, 170), width=2)
    for i, (line, size) in enumerate(lines):
        f = font("cjk" if any(ord(ch) > 0x2E80 for ch in line) else "sans_bold", size)
        d.text((c - d.textlength(line, font=f) / 2, c - 34 + i * 32), line, font=f,
               fill=(r, g, b, 225))
    layer = layer.rotate(random.Random(seed).uniform(-14, 14), resample=Image.BICUBIC)
    im.paste(layer, (x, y), layer)
    note("seal", x, y, x + radius * 2 + 40, y + radius * 2 + 40,
         " ".join(line for line, _ in lines))


def emboss(im, x, y, lines, radius=70):
    """A dry seal — pressed into the paper, almost no colour. The hardest mark to read, and the
    one a school or a notary actually uses."""
    layer = Image.new("RGBA", (radius * 2 + 30, radius * 2 + 30), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    c = radius + 15
    d.ellipse([c - radius, c - radius, c + radius, c + radius], outline=(150, 150, 150, 140),
              width=3)
    d.ellipse([c - radius + 10, c - radius + 10, c + radius - 10, c + radius - 10],
              outline=(170, 170, 170, 110), width=2)
    for i, (line, size) in enumerate(lines):
        f = font("sans_bold", size)
        d.text((c - d.textlength(line, font=f) / 2, c - 26 + i * 26), line, font=f,
               fill=(150, 150, 150, 160))
    im.paste(layer, (x, y), layer)
    note("seal", x, y, x + radius * 2 + 30, y + radius * 2 + 30,
         " ".join(line for line, _ in lines))


def crest(d, x, y, initials="BC", colour=GREEN):
    """A drawn coat of arms. A graphic the model must notice and not try to read as a word."""
    d.polygon([(x, y), (x + 92, y), (x + 92, y + 66), (x + 46, y + 104), (x, y + 66)],
              outline=colour, width=4)
    d.line([(x + 14, y + 28), (x + 78, y + 28)], fill=colour, width=3)
    f = font("serif_bold", 34)
    d.text((x + 46 - d.textlength(initials, font=f) / 2, y + 40), initials, font=f, fill=colour)
    note("image", x, y, x + 92, y + 104, initials)


def logo(d, x, y, name, tag, colour=BLUE):
    d.ellipse([x, y, x + 62, y + 62], outline=colour, width=5)
    d.polygon([(x + 18, y + 40), (x + 31, y + 17), (x + 44, y + 40)], fill=colour)
    d.rectangle([x + 18, y + 43, x + 44, y + 47], fill=colour)
    note("image", x, y, x + 62, y + 62)
    text(d, (x + 78, y + 8), name, "sans_bold", 30, colour, label="title")
    text(d, (x + 79, y + 42), tag, "sans", 15, GREY)


def figure(d, x, y, w=460, h=260, title="Applications received by quarter"):
    """A drawn chart. The model should mark this region as a picture, not invent numbers."""
    d.rectangle([x, y, x + w, y + h], outline=(120, 120, 120), width=2)
    d.line([(x + 55, y + 30), (x + 55, y + h - 45), (x + w - 30, y + h - 45)], fill=INK, width=2)
    for i, value in enumerate([0.42, 0.68, 0.51, 0.86, 0.74]):
        bx = x + 78 + i * 68
        d.rectangle([bx, y + h - 45 - value * (h - 95), bx + 44, y + h - 45], fill=(70, 110, 190))
        text(d, (bx + 6, y + h - 38), f"Q{i+1}", "sans", 15, GREY)
    note("image", x, y, x + w, y + h, title)
    text(d, (x + 16, y + 10), title, "sans_bold", 16, INK, label="caption")


def callout(d, x, y, w, title, body) -> int:
    """A boxed note — a text box, which is a different layout block from a paragraph."""
    inner = y + 44
    for line in body.split("\n"):
        inner = paragraph(d, x + 12, inner, line, width=w - 24, size=18, kind="sans", leading=26)
    h = inner - y + 12
    d.rectangle([x, y, x + w, y + h], outline=BLUE, width=2)
    d.rectangle([x, y, x + w, y + 34], fill=(232, 238, 250))
    text(d, (x + 12, y + 7), title, "sans_bold", 19, BLUE)
    inner = y + 44                                   # the box is drawn under the words, so again
    for line in body.split("\n"):
        inner = paragraph(d, x + 12, inner, line, width=w - 24, size=18, kind="sans", leading=26)
    return y + h


def barcode(d, x, y, digits, w=300, h=64):
    """A code the model should report as a picture, with its number read from underneath."""
    rng = random.Random(len(digits))
    cx = x
    while cx < x + w:
        bar = rng.choice([2, 2, 3, 5])
        if rng.random() > 0.35:
            d.rectangle([cx, y, cx + bar, y + h], fill=INK)
        cx += bar + rng.choice([2, 3])
    note("image", x, y, x + w, y + h, digits)
    text(d, (x, y + h + 6), digits, "mono", 17, INK)


# ---------------------------------------------------------------------------- legal
def deed_page_one() -> Image.Image:
    im, d = page()
    centred(d, 92, "DEED OF ASSIGNMENT", size=34)
    centred(d, 140, "of goodwill and trade marks", "serif", 23, GREY)
    d.line([(150, 186), (1091, 186)], fill=(180, 180, 176), width=2)

    text(d, (150, 220), "THIS DEED is made on the 11th day of February 2026", "serif", 22)
    text(d, (150, 264), "BETWEEN", "serif_bold", 22)
    y = paragraph(d, 190, 300,
        "(1) ARDWICK & NOLL TRADING LIMITED, a company incorporated in Hong Kong with company "
        "number 6612940, whose registered office is at Room 1804, Hoi Ming Commercial Centre, "
        "118 Rutland Street, Sheung Wan (the \"Assignor\"); and", width=900)
    y = paragraph(d, 190, y + 12,
        "(2) PELLMORE HOLDINGS (ASIA) LIMITED, a company incorporated in Hong Kong with company "
        "number 7204418, whose registered office is at 27/F, Cavendale Tower, 9 Ferrier Road, "
        "Quarry Bay (the \"Assignee\").", width=900)

    y = paragraph(d, 150, y + 30,
        "WHEREAS the Assignor carries on the business of specialty coffee roasting under the "
        "name Ardwick Roastery and is the registered proprietor of the trade marks listed in "
        "the Schedule, and has agreed to assign the same to the Assignee for the consideration "
        "set out below.", width=940)

    text(d, (150, y + 26), "NOW THIS DEED WITNESSES as follows:", "serif_bold", 22)
    y += 70
    for n, clause in enumerate([
        "In consideration of the sum of HK$4,850,000 (receipt of which the Assignor "
        "acknowledges) the Assignor assigns to the Assignee absolutely all its right, title and "
        "interest in the trade marks listed in the Schedule, together with the goodwill of the "
        "business to which they relate.",
        "The Assignor shall at the Assignee's cost execute every further document reasonably "
        "required to give effect to this Deed, including any form required by the Registrar.",
        "The Assignor warrants that the trade marks are free of any charge, licence or adverse "
        "claim, and that no third party has asserted a right in them within the past six years.",
        "This Deed is governed by the laws of the Hong Kong Special Administrative Region.",
    ], start=1):
        text(d, (150, y), f"{n}.", "serif", 21)
        y = paragraph(d, 190, y, clause, width=900) + 14

    text(d, (1091, 1690), "Page 1 of 2", "serif", 17, GREY, anchor="rs")
    return im


def deed_page_two() -> Image.Image:
    im, d = page()
    text(d, (150, 90), "THE SCHEDULE", "serif_bold", 24)
    text(d, (150, 128), "Registered trade marks assigned", "serif", 20, GREY)
    table(d, 150, 176, [
        ["Mark", "Registration no.", "Classes", "Renewal due"],
        ["ARDWICK ROASTERY", "304812996", "30, 43", "08 Jun 2029"],
        ["ARDWICK (device)", "304813011", "30", "08 Jun 2029"],
        ["SLOW HARVEST", "305226704", "30", "22 Nov 2031"],
    ], [420, 260, 140, 220])

    y = 400
    y = callout(d, 150, y, 940, "Stamp duty",
                "Adjudication has been sought under section 13 of the Stamp Duty Ordinance. "
                "This counterpart is not to be relied on as evidence of title until stamped.")

    text(d, (150, y + 60), "EXECUTED AS A DEED by the parties on the date first written above.",
         "serif", 21)

    signature_block(d, 150, y + 130, "Imogen T. Ardwick", "Director, Ardwick & Noll Trading "
                                                          "Limited", seed=41)
    signature_block(d, 660, y + 130, "R. S. Pellmore", "Director, Pellmore Holdings (Asia) "
                                                       "Limited", seed=57)

    text(d, (150, y + 320), "In the presence of", "sans", 17, GREY)
    signature_block(d, 150, y + 360, "Amara Baptiste-Rowe", "Solicitor, Quennell Vasseur LLP",
                    seed=63)
    text(d, (150, y + 512), "Address: 31/F, Lyndhurst House, 4 Aberdeen Street, Central",
         "sans", 17, GREY)
    text(d, (150, y + 538), "Occupation: Solicitor of the High Court", "sans", 17, GREY)

    # The chop lands half over the witness block, the way a real one does.
    chop(im, 700, y + 400, [("QUENNELL", 22), ("VASSEUR LLP", 18), ("11 FEB 2026", 14)],
         seed=8, radius=86)
    emboss(im, 940, y + 150, [("NOTARIAL", 17), ("SEAL", 17)])
    text(d, (1091, 1690), "Page 2 of 2", "serif", 17, GREY, anchor="rs")
    return im


def deed_divider() -> Image.Image:
    """Four words on an otherwise empty page — the exhibit divider that makes the model count."""
    im, d = page()
    centred(d, 800, "EXHIBIT \"C\"", size=36)
    centred(d, 856, "Correspondence, 2024 to 2025", "serif", 22, GREY)
    return im


# ---------------------------------------------------------------------------- government
def permit() -> Image.Image:
    im, d = page()
    crest(d, 80, 66, "BC", GREEN)
    text(d, (200, 74), "BRANWELL COUNTY COUNCIL", "sans_bold", 28, GREEN)
    text(d, (201, 110), "Licensing and Environmental Health Service", "sans", 19, GREY)
    text(d, (201, 138), "Marden Civic Offices, 2 Halloway Row, Branwell BR1 4QA", "sans", 16, GREY)
    d.line([(80, 182), (1161, 182)], fill=GREEN, width=3)

    centred(d, 214, "PREMISES LICENCE", "sans_bold", 34, INK)
    centred(d, 258, "Issued under the Licensing Act 2003, Part 3", "sans", 19, GREY)

    y = table(d, 80, 310, [
        ["Licence number", "PRM/2026/04417"],
        ["Premises", "The Wheelwright's Arms, 14 Stallard Lane, Branwell BR2 7NX"],
        ["Licence holder", "Dunleavy Hospitality Group Limited (company no. 09841277)"],
        ["Designated supervisor", "Marguerite O. Falkenrath"],
        ["Issued", "03 March 2026"],
        ["Expires", "02 March 2027"],
    ], [300, 781], header=False, row_h=44)

    text(d, (80, y + 34), "Licensable activities and permitted hours", "sans_bold", 22)
    y = table(d, 80, y + 72, [
        ["Activity", "Monday to Thursday", "Friday and Saturday", "Sunday"],
        ["Sale of alcohol, on sales", "11:00 – 23:00", "11:00 – 00:30", "12:00 – 22:30"],
        ["Recorded music, indoors", "11:00 – 23:00", "11:00 – 00:30", "12:00 – 22:30"],
        ["Live music, indoors", "Not permitted", "19:00 – 23:00", "Not permitted"],
        ["Late night refreshment", "Not permitted", "23:00 – 00:30", "Not permitted"],
    ], [340, 260, 260, 221], size=18)

    top = y + 34
    bottom = callout(d, 80, top, 640, "Conditions attached",
                     "1. A digital CCTV system shall be operated at all times the premises are "
                     "open.\n2. No drinks in open glass containers shall leave the premises.\n"
                     "3. A refusals log shall be kept and produced on request by a constable.")
    figure(d, 760, top, 400, 240, "Complaints received, by quarter")
    text(d, (760, top + 250), "Figure 1 — Noise complaints, this ward.", "sans", 16, GREY)
    y = max(bottom, top + 282)

    y = paragraph(d, 80, y + 40,
        "The holder shall notify the Service in writing within fourteen days of any change to "
        "the designated premises supervisor, and shall surrender this licence for endorsement. "
        "A copy of the plan deposited with the application is held under the same reference.",
        width=1080, size=19, kind="sans", leading=28)

    barcode(d, 80, y + 60, "PRM 2026 04417 8")
    signature_block(d, 660, y + 46, "H. Osafo-Lindqvist",
                    "Head of Licensing, Branwell County Council", seed=77)
    chop(im, 420, y + 26, [("BRANWELL", 20), ("COUNTY", 20), ("COUNCIL", 16)], seed=5,
         radius=80, colour=(24, 92, 60))

    text(d, (80, 1620), "This licence must be displayed at the premises. It is an offence under "
                        "section 136 of the Act to carry on a", "sans", 16, GREY)
    text(d, (80, 1646), "licensable activity otherwise than in accordance with it. Appeals lie "
                        "to the magistrates' court within 21 days.", "sans", 16, GREY)
    return im


# ---------------------------------------------------------------------------- school
def transcript() -> Image.Image:
    im, d = page()
    crest(d, 560, 62, "KH", (86, 28, 92))
    centred(d, 182, "KESTREL HOUSE SCHOOL", "serif_bold", 32)
    centred(d, 226, "Founded 1911 · 40 Ivyhurst Road, Branwell BR3 2LE", "serif", 19, GREY)
    d.line([(80, 266), (1161, 266)], fill=(86, 28, 92), width=2)
    centred(d, 292, "OFFICIAL ACADEMIC TRANSCRIPT", "sans_bold", 24)

    y = table(d, 80, 344, [
        ["Student", "Tobiah N. Arkwright-Mensah", "Student number", "KH-2021-0388"],
        ["Date of birth", "14 September 2008", "Year group", "Upper Sixth"],
        ["Programme", "A Level, three subjects", "Issued", "26 June 2026"],
    ], [180, 430, 220, 251], header=False, row_h=42)

    text(d, (80, y + 32), "Results", "sans_bold", 22)
    y = table(d, 80, y + 70, [
        ["Subject", "Code", "Grade", "UMS", "Session"],
        ["Mathematics", "9MA0", "A*", "276 / 300", "Summer 2026"],
        ["Physics", "9PH0", "A", "252 / 300", "Summer 2026"],
        ["Design & Technology", "9DT0", "A", "248 / 300", "Summer 2026"],
        ["Extended Project", "7993", "B", "42 / 50", "Autumn 2025"],
    ], [420, 160, 130, 200, 171], size=19)

    y = table(d, 80, y + 20, [["Overall", "", "AAA*", "", "Rank 4 of 118"]],
              [420, 160, 130, 200, 171], size=19, header=False)

    text(d, (80, y + 36), "Form tutor's remarks", "sans_bold", 20)
    y = paragraph(d, 80, y + 72,
        "Tobiah has been an exceptional student of the sciences and a patient collaborator in "
        "the workshop. The extended project, on passive cooling in school buildings, was "
        "presented to the governors in March and is held in the library. He leaves with our "
        "warm regards and a place confirmed to read engineering.", width=1080, size=20,
        leading=30)

    y = callout(d, 80, y + 26, 660, "Verification",
                "This transcript is issued directly by the school. Any copy not bearing the "
                "embossed seal and the registrar's signature should be treated as unverified. "
                "Queries: registrar@kestrelhouse.example")

    signature_block(d, 80, 1400, "Perpetua Saoirse Blackwood", "Registrar", seed=91)
    signature_block(d, 620, 1400, "Dr N. A. Okonkwo-Reilly", "Head Teacher", seed=95)
    emboss(im, 980, 1360, [("KESTREL", 15), ("HOUSE", 15), ("SCHOOL", 13)], radius=76)
    text(d, (80, 1670), "Page 1 of 1 · Transcript reference KH/TR/2026/0388", "serif", 16, GREY)
    return im


# ---------------------------------------------------------------------------- commercial
def invoice() -> Image.Image:
    im, d = page()
    logo(d, 80, 70, "MERIDIAN", "SHIPPING & LOGISTICS")
    text(d, (880, 74), "INVOICE", "sans_bold", 40, INK)
    text(d, (880, 124), "No. MS-2026-0417", "sans", 20, GREY)
    text(d, (880, 152), "Issued 14 March 2026", "sans", 20, GREY)

    text(d, (80, 200), "BILL TO", "sans_bold", 16, GREY)
    for i, line in enumerate(["Harbourline Freight Limited", "Unit 2203, Kwai Chung Plaza",
                              "Kwai Chung, New Territories", "Hong Kong SAR"]):
        text(d, (80, 226 + i * 28), line, "sans", 20)
    text(d, (660, 200), "SHIP TO", "sans_bold", 16, GREY)
    for i, line in enumerate(["Meridian Bonded Warehouse 4", "Lot 88, Tuen Mun Area 38",
                              "New Territories", "Hong Kong SAR"]):
        text(d, (660, 226 + i * 28), line, "sans", 20)

    y = table(d, 80, 370, [
        ["Description", "Qty", "Unit", "Amount (HKD)"],
        ["Ocean freight, 40ft HC, SHA-HKG", "3", "8,400.00", "25,200.00"],
        ["Terminal handling, origin", "3", "1,150.00", "3,450.00"],
        ["Customs entry, bonded", "1", "2,700.00", "2,700.00"],
        ["Documentation fee", "1", "480.00", "480.00"],
    ], [620, 100, 180, 220])
    y = table(d, 780, y + 20, [["Subtotal", "31,830.00"], ["Levy 0.5%", "159.15"],
                               ["Total due", "31,989.15"]], [220, 200], header=False)

    callout(d, 80, y + 40, 620, "Payment terms",
            "Net 30 days from the date of issue. Late settlement carries interest at 1% per "
            "month. Quote the invoice number on every remittance.")
    barcode(d, 80, 1180, "MS 2026 0417 3")

    signature_block(d, 80, 1330, "K. W. Ladipo", "Finance Director, Meridian Shipping", seed=4)
    signature_block(d, 660, 1330, "V. Thorbecke", "For Harbourline Freight Limited", seed=9)
    chop(im, 830, 1180, [("PAID", 34), ("14 MAR 2026", 17)], seed=6)
    text(d, (80, 1660), "Meridian Shipping & Logistics Limited · BR 61234567-000 · "
                        "meridian.example · +852 2555 0100", "sans", 15, GREY)
    return im


# ---------------------------------------------------------------------------- Chinese
def chinese_notice() -> Image.Image:
    """Traditional Chinese, a table, and a chop — the writing system the model is weakest on."""
    im, d = page()
    centred(d, 86, "貨 物 放 行 通 知 書", "cjk", 40)
    d.line([(80, 158), (1161, 158)], fill=(190, 190, 186), width=2)
    text(d, (80, 188), "編號：HKG-2026-0417", "cjk", 22)
    text(d, (80, 226), "日期：二〇二六年三月十四日", "cjk", 22)
    text(d, (760, 188), "受文者：滙誠報關行有限公司", "cjk", 22)

    paragraph(d, 80, 288,
              "茲通知 貴公司下列貨物已完成報關手續，並已於本通知書所載日期放行。"
              "請於七個工作天內安排提貨，逾期將按倉租表計收費用。如對溫控紀錄有異議，"
              "應於提貨前以書面提出。", width=1080, size=23, kind="cjk", leading=40)

    y = table(d, 80, 440, [
        ["提單號碼", "櫃號", "件數", "重量（公斤）"],
        ["MSHK2260417", "MSKU 738201-4", "1,240", "18,430"],
        ["MSHK2260418", "TGHU 559013-0", "880", "12,905"],
    ], [300, 320, 200, 260], size=21)

    text(d, (80, y + 40), "備註：", "cjk", 22)
    paragraph(d, 80, y + 80, "第二櫃之溫控紀錄已隨附。本通知書不構成貨物狀況之保證。",
              width=1000, size=22, kind="cjk", leading=36)

    text(d, (80, 1300), "負責人簽署", "cjk", 20, GREY)
    signature(d, 80, 1342, seed=33)
    d.line([(80, 1396), (390, 1396)], fill=GREY, width=1)
    text(d, (80, 1406), "戴逸軒　營運經理", "cjk", 22)
    text(d, (80, 1442), "潤德國際物流（香港）有限公司", "cjk", 20, GREY)
    chop(im, 820, 1210, [("已放行", 30), ("2026年3月14日", 15)], seed=2)
    return im


# ---------------------------------------------------------------------------- lighting
def as_scan(im: Image.Image, seed=7) -> Image.Image:
    """The same page as a phone photo: lit from one side, slightly crooked, noisy, soft.

    Real documents arrive like this far more often than they arrive as clean renders, and this is
    where an OCR model either holds up or does not.
    """
    rng = random.Random(seed)
    out = im.rotate(rng.uniform(-1.6, 1.6), resample=Image.BICUBIC, fillcolor="white")

    light = Image.new("L", out.size)
    d = ImageDraw.Draw(light)
    for x in range(0, out.width, 8):                      # a soft gradient across the page
        d.rectangle([x, 0, x + 8, out.height], fill=int(150 + 105 * (x / out.width) ** 1.4))
    light = light.filter(ImageFilter.GaussianBlur(60))
    out = Image.composite(out, Image.new("RGB", out.size, (238, 232, 220)), light)

    out = out.filter(ImageFilter.GaussianBlur(0.6))
    pixels = out.load()
    for _ in range(int(out.width * out.height * 0.012)):  # sensor noise
        x, y = rng.randrange(out.width), rng.randrange(out.height)
        r, g, b = pixels[x, y]
        n = rng.randint(-34, 34)
        pixels[x, y] = (max(0, min(255, r + n)), max(0, min(255, g + n)), max(0, min(255, b + n)))
    return out


SAMPLES = [
    ("legal_deed.pdf", [deed_page_one, deed_page_two, deed_divider],
     "Legal — deed, numbered clauses, schedule table, three signatures, "
     "an inked and an embossed seal, then a near-blank exhibit divider"),
    ("gov_permit.pdf", [permit],
     "Government — premises licence, crest, hours table, conditions box, "
     "figure, barcode, council seal"),
    ("school_transcript.pdf", [transcript],
     "School — transcript, grade table, tutor's remarks, two signatures, embossed seal"),
    ("commercial_invoice.pdf", [invoice],
     "Commercial — invoice, logo, line-item table, totals, barcode, PAID chop"),
    ("notice_zh.pdf", [chinese_notice],
     "Traditional Chinese — release notice, table, signature, chop"),
    ("scan_permit.png", [lambda: as_scan(permit(), seed=7)],
     "Poor lighting — the permit as a phone photo: skewed, noisy, lit from one side"),
    ("scan_notice_zh.png", [lambda: as_scan(chinese_notice(), seed=15)],
     "Poor lighting — the Chinese notice as a phone photo"),
]


def main() -> int:
    TRUTH_DIR.mkdir(exist_ok=True)
    index = {}
    for name, builders, label in SAMPLES:
        pages, truth = [], []
        for build in builders:
            TRUTH.clear()
            pages.append(build())
            truth.append(list(TRUTH))          # the answer key for the page just drawn
        path = HERE / name
        if path.suffix == ".pdf":
            pages[0].save(path, "PDF", resolution=150.0, save_all=True, append_images=pages[1:])
        else:
            pages[0].save(path, quality=88)
        (TRUTH_DIR / (path.stem + ".json")).write_text(
            json.dumps({"file": name, "pages": truth}, indent=1, ensure_ascii=False) + "\n")
        index[name] = label
        items = sum(len(t) for t in truth)
        print(f"wrote {path.relative_to(HERE.parent)}  "
              f"({len(pages)} page(s), {items} things in the answer key)")
    (HERE / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {(HERE / 'index.json').relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
