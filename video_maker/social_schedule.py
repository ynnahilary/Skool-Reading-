#!/usr/bin/env python3
"""Build the social posting schedule for the 90-video series.

Reads series/production_details.csv and writes series/social_schedule.csv:
one row per video with date, time, caption, hashtags and YouTube title,
ready to paste into Metricool's CSV import template.

Usage: python3 social_schedule.py 2026-10-19   # date of series day 1
"""
import csv
import datetime as dt
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERIES = os.path.join(HERE, "series")

BRAND = "Midnight Oracle"
JOIN_LINE = f"✨ Join the {BRAND} circle: link in bio."
TIMES = ["09:00", "13:00", "19:00"]   # video 1, 2, 3 of each day
BASE_TAGS = ["#tarot", "#tarotreading", "#spirituality", "#intuition"]
FORMAT_TAGS = {
    "school": "#tarotforbeginners", "lesson": "#tarotforbeginners",
    "tutorial": "#tarotforbeginners", "beginner": "#tarotforbeginners",
    "quiz": "#tarotquiz", "pick": "#pickacard", "interactive": "#pickacard",
    "journal": "#journalprompts", "reflection": "#selfreflection",
    "self-care": "#selfcare", "ritual": "#ritual",
}


def theme_tag(theme):
    name = theme.split("—")[0].strip()
    name = re.sub(r"^the\s+", "", name, flags=re.I)
    tag = re.sub(r"[^A-Za-z0-9]", "", name.title())
    return "#" + tag.lower() if tag and len(tag) <= 24 else None


def caption(row):
    clean = lambda s: s.replace("’", "'").strip()
    tags = list(BASE_TAGS)
    t = theme_tag(row["Theme"])
    if t and t not in tags:
        tags.insert(1, t)
    fmt = row["Format"].lower()
    for key, tag in FORMAT_TAGS.items():
        if key in fmt and tag not in tags:
            tags.append(tag)
            break
    return "\n\n".join([
        f"{clean(row['Hook'])} {clean(row['Social Caption'])}",
        clean(row["CTA"]),
        JOIN_LINE,
        " ".join(tags[:6]),
    ])


def main(start):
    rows = list(csv.DictReader(open(os.path.join(SERIES, "production_details.csv"),
                                    encoding="utf-8-sig")))
    out_path = os.path.join(SERIES, "social_schedule.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "time", "video_id", "video_file", "video_ready",
                    "platforms", "youtube_title", "caption"])
        for r in rows:
            day, vid = int(r["Day"]), int(r["Video"])
            date = start + dt.timedelta(days=day - 1)
            vid_file = f"day{day:02d}_{vid}.mp4"
            ready = os.path.exists(os.path.join(SERIES, "out", vid_file))
            w.writerow([date.isoformat(), TIMES[vid - 1], r["Video ID"], vid_file,
                        "yes" if ready else "NO - needs card",
                        "TikTok, Instagram Reels, YouTube Shorts, Facebook Reels",
                        r["Title"].replace("’", "'")[:95] + " #shorts",
                        caption(r)])
    print("wrote", out_path)


if __name__ == "__main__":
    main(dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.date(2026, 10, 19))
