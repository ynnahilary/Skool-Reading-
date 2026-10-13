"""Render every galaxy colour loop in parallel (one-time cache warm-up)."""
import json
import os
import sys
from multiprocessing import Pool

from make_video import CACHE, galaxy_loop

HERE = os.path.dirname(os.path.abspath(__file__))


def one(p):
    galaxy_loop(os.path.join(CACHE, f"galaxy_{p}.mp4"), palette=p)
    return p


if __name__ == "__main__":
    palettes = json.load(open(os.path.join(HERE, "series", "settings.json")))["palettes"]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as pool:
        for p in pool.imap_unordered(one, palettes):
            print("done", p, flush=True)
