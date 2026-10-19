"""DAT-340: place the two honest results charts into deck slides 14 and 15.

Slide 14 "GRPO results - the honest read"  -> cpl_sft_vs_rl.png
Slide 15 "So did RL teach anything new?"    -> behaviour_vs_headline.png

The body text on those slides becomes a one-line caption above the chart, so
the slide stays readable at the back of the room. Saves a NEW file so the
draft is preserved: talk/pmlg_chess_rl_talk_final.pptx
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches

TALK = Path(__file__).resolve().parent
SRC = TALK / "pmlg_chess_rl_talk_draft.pptx"
DST = TALK / "pmlg_chess_rl_talk_final.pptx"
CHARTS = TALK / "charts"

prs = Presentation(SRC)
SLIDE_W, SLIDE_H = prs.slide_width, prs.slide_height

def clear_body(slide):
    """Remove body text boxes (keep title + anything named DAT340_*)."""
    for sh in list(slide.shapes):
        if sh == slide.shapes.title:
            continue
        if sh.name.startswith("DAT340_"):
            continue
        if sh.has_text_frame and not sh.has_table:
            sh._element.getparent().remove(sh._element)

def add_chart(slide, img, caption, top=Inches(1.55), max_h=Inches(4.6), max_w=Inches(11.4)):
    from PIL import Image
    with Image.open(img) as im:
        w, h = im.size
    aspect = h / w
    width = max_w
    height = int(width * aspect)
    if height > max_h:
        height = max_h
        width = int(height / aspect)
    left = int((SLIDE_W - width) / 2)
    pic = slide.shapes.add_picture(str(img), left, top, width=width, height=height)
    pic.name = f"DAT340_CHART_{img.stem}"
    # caption under the chart
    tb = slide.shapes.add_textbox(Inches(0.5), top + height + Inches(0.12), Inches(12.3), Inches(0.6))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.text = caption
    for p in tf.paragraphs:
        for r in p.runs:
            r.font.size = None  # inherit template
    tb.name = f"DAT340_CAPTION_{img.stem}"

# slide 14 — CPL
s = prs.slides[13]
assert "GRPO results" in s.shapes.title.text, s.shapes.title.text
clear_body(s)
add_chart(
    s, CHARTS / "cpl_sft_vs_rl.png",
    "Mean CPL 1045.7 -> 1022.7 (-2.2%); median 81 -> 90. Both heavy-tailed "
    "(std ~2,900): a few blunders dominate the mean. Source: comparison.json, 1,000 positions.",
)

# slide 15 — behaviour vs headline
s = prs.slides[14]
assert "teach anything new" in s.shapes.title.text, s.shapes.title.text
clear_body(s)
add_chart(
    s, CHARTS / "behaviour_vs_headline.png",
    "Games got longer (24.3 -> 27.7 plies) and the move distribution changed, but headline "
    "strength did not: 0W-0D-20L vs Stockfish for both checkpoints.",
)

# sanity: nothing outside the slide
for i, s in enumerate(prs.slides, 1):
    for sh in s.shapes:
        if sh.left is None:
            continue
        if sh.left < 0 or sh.top < 0 or sh.left + sh.width > SLIDE_W or sh.top + sh.height > SLIDE_H:
            print(f"WARN slide {i} shape '{sh.name}' outside bounds")

prs.save(DST)
print("saved:", DST)
print("slides:", len(prs.slides))
