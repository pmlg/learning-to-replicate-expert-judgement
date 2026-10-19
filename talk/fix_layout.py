"""DAT-340: fix render overflow found during LibreOffice render verification.

Issues fixed (slides are 13.33in wide, content boxes were authored 9.0in wide):
- Content/title placeholders widened 9.0in -> 12.33in on every slide (removes
  excessive wrapping that pushed text past the slide bottom on slides
  2, 6, 11, 12, 13, 22, 23).
- Body placeholders get "shrink text on overflow" (normAutofit) so PowerPoint
  also self-corrects if a slide is ever edited to overflow again.
- Slide 21: chart moved below the source-caption placeholder (it previously
  overlapped the caption text).
- Slide 26: QR caption darkened (was very light gray, low contrast on a projector).

Usage:
    uv run --with python-pptx python talk/fix_layout.py
"""
from pptx import Presentation
from pptx.util import Inches
from pptx.enum.text import MSO_AUTO_SIZE
from pptx.dml.color import RGBColor

import os

HERE = os.path.dirname(os.path.abspath(__file__))
PPTX = os.path.join(HERE, 'pmlg_chess_rl_talk_draft.pptx')

SLIDE_W = 13.333
LEFT = 0.5
FULL_W = SLIDE_W - 2 * LEFT  # 12.33in

prs = Presentation(PPTX)
for idx, slide in enumerate(prs.slides, 1):
    for sh in slide.shapes:
        if sh.name == 'Chart 3' and idx == 21:
            sh.top = Inches(2.60)
            sh.height = Inches(3.85)
            print(f'slide {idx}: moved chart below caption')
            continue
        if sh.name == 'Table 3' and idx == 14:
            sh.top = Inches(2.20)
            print(f'slide {idx}: moved table below the sources line')
            continue
        if not sh.is_placeholder:
            continue
        # IMPORTANT: set ALL FOUR coords explicitly. Placeholder geometry may be
        # inherited from the layout; touching only left/width makes python-pptx
        # materialise an xfrm with top=0/height=0 (breaks rendering).
        if sh.name.startswith('Title'):
            sh.left, sh.top = Inches(LEFT), Inches(0.30)
            sh.width, sh.height = Inches(FULL_W), Inches(1.25)
        else:
            sh.left, sh.top = Inches(LEFT), Inches(1.75)
            sh.width, sh.height = Inches(FULL_W), Inches(5.20)
            tf = sh.text_frame
            tf.word_wrap = True
            tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    # slide 14: the canonical-file decision was made - update the stale note
    if idx == 14:
        for sh in slide.shapes:
            if sh.has_text_frame and 'comparison.json' in sh.text_frame.text:
                for p in sh.text_frame.paragraphs:
                    for r in p.runs:
                        if 'comparison.json' in r.text:
                            r.text = ('Sources: evaluation_results/comparison.json '
                                      '(canonical, frozen); benchmark_results.json is a '
                                      'stale second rl/final run - not used')
                            print('slide 14: updated sources note to frozen decision')
    # darken QR caption on the closing slide
    if idx == 26:
        for sh in slide.shapes:
            if sh.has_text_frame and 'Scan for slides' in sh.text_frame.text:
                for p in sh.text_frame.paragraphs:
                    for r in p.runs:
                        r.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
                print('slide 26: darkened QR caption')

prs.save(PPTX)
print('saved', PPTX)
