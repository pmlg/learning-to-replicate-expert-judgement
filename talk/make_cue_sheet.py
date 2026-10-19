"""DAT-340: build a dry-run cue sheet from the deck's timed speaker notes.

Parses [M:SS-M:SS] time budgets in each slide's speaker notes and writes
talk/cue_sheet.md (slide #, title, start-end, duration, cumulative % of talk).
"""
from pptx import Presentation
from pathlib import Path
import re

deck = Path("talk/pmlg_chess_rl_talk_draft.pptx")
out = Path("talk/cue_sheet.md")
prs = Presentation(deck)


def parse_ts(s):
    m = re.match(r"^(\d+):(\d\d)$", s)
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


rows, unparsed = [], []
for idx, slide in enumerate(prs.slides, start=1):
    title = ""
    for sh in slide.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip():
            title = sh.text_frame.text.strip().splitlines()[0]
            break
    notes = slide.notes_slide.notes_text_frame.text if slide.has_notes_slide else ""
    m = re.search(r"\[(\d+:\d\d)-(\d+:\d\d)(\+?)\]", notes)
    if m:
        t0, t1 = parse_ts(m.group(1)), parse_ts(m.group(2))
        rows.append({"n": idx, "title": title, "start": t0, "end": t1, "dur": t1 - t0,
                     "open_ended": bool(m.group(3))})
    else:
        unparsed.append(idx)

total = sum(r["dur"] for r in rows)


def fmt(sec):
    return f"{sec // 60}:{sec % 60:02d}"


lines = [
    "# PMLG dry-run cue sheet",
    "",
    f"Generated from `{deck.name}` speaker notes. **Total speaking time: {fmt(total)}** "
    f"({total / 60:.0f} min) across {len(rows)} timed slides.",
    "",
    "| # | Start | End | Dur | Slide |",
    "|---|-------|-----|-----|-------|",
]
qa_start = next((r["start"] for r in rows if r.get("open_ended")), None)
for r in rows:
    end_label = fmt(r["end"]) + ("+" if r.get("open_ended") else "")
    lines.append(f'| {r["n"]} | {fmt(r["start"])} | {end_label} | {r["dur"]/60:.1f}m | {r["title"]} |')

if unparsed:
    lines += ["", f"**Slides with NO timed notes (fix before dry run): {unparsed}**"]

# pacing warnings
warnings = []
talk_len = 45 * 60  # PMLG slot
if qa_start is not None:
    qa = talk_len - qa_start
    warnings.append(f"Q&A opens at {fmt(qa_start)} — {fmt(qa)} left of a 45-min slot.")
if talk_len - total < 0:
    warnings.append(f"Speaking time ({fmt(total)}) OVERFLOWS a 45-min slot by {fmt(total - talk_len)}.")
long_slides = [r for r in rows if r["dur"] > 180]
if long_slides:
    warnings.append("Slides over 3 min (rehearse these hardest): " + ", ".join(f'#{r["n"]} {fmt(r["dur"])}' for r in long_slides))
transitions = [r for r in rows if r["dur"] <= 30]
if transitions:
    warnings.append("Sub-30s transition slides: " + ", ".join(f'#{r["n"]}' for r in transitions))

lines += ["", "## Pacing notes", ""]
lines += [f"- {w}" for w in warnings] or ["- Timing fits the slot."]
out.write_text("\n".join(lines) + "\n")
print(f"total={fmt(total)} slides={len(rows)} unparsed={unparsed}")
print(f"wrote {out}")
