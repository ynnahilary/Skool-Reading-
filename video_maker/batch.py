#!/usr/bin/env python3
"""Render the 30-day series from series/scripts.csv.

Cards live in series/cards/dayNN_card.(png|webp|jpg); videos go to
series/out/dayNN_V.mp4. Existing videos and days without a card are skipped.

Usage: python3 batch.py            # everything that has a card
       python3 batch.py 4 5        # only these days
"""
import csv
import glob
import json
import os
import sys

from make_video import render_video, split_sentences

HERE = os.path.dirname(os.path.abspath(__file__))
SERIES = os.path.join(HERE, "series")


def main(days):
    settings = json.load(open(os.path.join(SERIES, "settings.json")))
    rows = list(csv.DictReader(open(os.path.join(SERIES, "scripts.csv"), encoding="utf-8-sig")))
    done, skipped = [], set()
    for r in rows:
        day, vid = int(r["Day"]), int(r["Video"])
        if days and day not in days:
            continue
        cards = glob.glob(os.path.join(SERIES, "cards", f"day{day:02d}_card.*"))
        if not cards:
            skipped.add(day)
            continue
        out = os.path.join(SERIES, "out", f"day{day:02d}_{vid}.mp4")
        if os.path.exists(out):
            continue
        render_video(r["Title"], cards[0], split_sentences(r["Script"]), out, **settings)
        done.append(out)
    print(f"rendered {len(done)} videos")
    if skipped:
        print("no card yet for days:", ", ".join(map(str, sorted(skipped))))


if __name__ == "__main__":
    main({int(d) for d in sys.argv[1:]})
