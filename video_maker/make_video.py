#!/usr/bin/env python3
"""Build a vertical tarot-style video from a script.

Pipeline: script -> Kokoro TTS voiceover -> Whisper word timings ->
word-by-word glowing captions, over a galaxy background with a title and
a card image.

Usage: python3 make_video.py videos/four_suits.json
"""
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import wave
from difflib import SequenceMatcher

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 30
SR = 24000
HERE = os.path.dirname(os.path.abspath(__file__))

TITLE_FONT = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
TITLE_COLOR = (255, 243, 200)
GLOW_COLOR = (255, 222, 140)
CAPTION_FONT = "Open Sans"


# ---------------------------------------------------------------- voiceover

def kokoro():
    from kokoro_onnx import Kokoro
    snaps = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--fastrtc--kokoro-onnx/snapshots/*/"))
    if not snaps:
        from huggingface_hub import hf_hub_download
        for f in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
            hf_hub_download("fastrtc/kokoro-onnx", f)
        return kokoro()
    d = snaps[0]
    return Kokoro(d + "kokoro-v1.0.onnx", d + "voices-v1.0.bin")


def norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower().replace("’", "'"))


def align(script_words, heard):
    """Give each script word a (start, end) using Whisper's word list."""
    a = [norm(w) for w in script_words]
    b = [norm(h[0]) for h in heard]
    times = [None] * len(a)
    for i, j, n in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(n):
            times[i + k] = heard[j + k][1:]
    # Fill gaps by spreading unmatched words between known neighbours.
    i = 0
    while i < len(times):
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < len(times) and times[j] is None:
            j += 1
        lo = times[i - 1][1] if i > 0 else (heard[0][1] if heard else 0.0)
        hi = times[j][0] if j < len(times) else (heard[-1][2] if heard else lo + 0.4 * (j - i))
        step = max(hi - lo, 0.01) / (j - i)
        for k in range(i, j):
            times[k] = (lo + step * (k - i), lo + step * (k - i + 1))
        i = j
    return times


def make_voiceover(paragraphs, voice, speed, out_wav):
    from faster_whisper import WhisperModel
    tts = kokoro()
    stt = WhisperModel("small.en", compute_type="int8")
    lang = "en-gb" if voice.startswith("b") else "en-us"
    lead, gap, tail = 0.4, 0.45, 1.2

    pieces = [np.zeros(int(lead * SR), np.float32)]
    t = lead
    words = []  # (text, start, end, para_index)
    for p, text in enumerate(paragraphs):
        audio, sr = tts.create(text, voice=voice, speed=speed, lang=lang)
        assert sr == SR
        audio = audio.astype(np.float32)
        a16 = np.interp(np.arange(0, len(audio), SR / 16000), np.arange(len(audio)), audio)
        segs, _ = stt.transcribe(a16.astype(np.float32), word_timestamps=True, language="en")
        heard = [(w.word.strip(), w.start, w.end) for s in segs for w in s.words]
        script_words = text.split()
        for w, (s, e) in zip(script_words, align(script_words, heard)):
            words.append((w, t + s, t + e, p))
        pieces.append(audio)
        t += len(audio) / SR
        pieces.append(np.zeros(int(gap * SR), np.float32))
        t += gap
    pieces.append(np.zeros(int(tail * SR), np.float32))
    audio = np.concatenate(pieces)
    with wave.open(out_wav, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return words, len(audio) / SR


# ---------------------------------------------------------------- captions

def clean(w):
    w = w.replace("’", "'")
    return re.sub(r"[,.;:]+$", "", w)


def chunk_words(words, max_chars=22):
    chunks, cur = [], []
    for w in words:
        if cur and (w[3] != cur[-1][3]
                    or re.search(r"[.,:;!?]$", cur[-1][0])
                    or len(" ".join(clean(x[0]) for x in cur + [w])) > max_chars):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def ass_time(t):
    t = max(t, 0)
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def write_ass(words, path, y=1440, size=84):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{CAPTION_FONT},{size},&H00C2F4FF,&H00C2F4FF,&H008CDEFF,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,8,80,80,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    chunks = chunk_words(words)
    for c, chunk in enumerate(chunks):
        nxt = chunks[c + 1][0][1] if c + 1 < len(chunks) else chunk[-1][2] + 1.0
        end = min(nxt, chunk[-1][2] + 0.7)
        for i, w in enumerate(chunk):
            start = w[1]
            stop = chunk[i + 1][1] if i + 1 < len(chunk) else end
            glow, crisp = [], []
            for k, x in enumerate(chunk):
                txt = clean(x[0]).replace("{", "(").replace("}", ")")
                on = k <= i
                glow.append(("{\\1a&HFF&\\3a&H40&}" if on else "{\\1a&HFF&\\3a&HFF&}") + txt)
                crisp.append(("{\\alpha&H00&}" if on else "{\\alpha&HFF&}") + txt)
            pos = f"{{\\an8\\pos({W // 2},{y})}}"
            lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(stop)},Cap,,0,0,0,,"
                         f"{pos}{{\\bord9\\blur16}}" + " ".join(glow))
            lines.append(f"Dialogue: 1,{ass_time(start)},{ass_time(stop)},Cap,,0,0,0,,"
                         f"{pos}{{\\bord0\\blur0.6}}" + " ".join(crisp))
    with open(path, "w") as f:
        f.write(head + "\n".join(lines) + "\n")


# ---------------------------------------------------------------- visuals

def fbm(h, w, rng, scales, weights):
    out = np.zeros((h, w), np.float32)
    for s, wt in zip(scales, weights):
        small = rng.random((max(2, h // s), max(2, w // s))).astype(np.float32)
        img = Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        out += wt * np.asarray(img, np.float32) / 255
    out -= out.min()
    return out / out.max()


def galaxy(h, w, seed=7):
    """A purple/pink/blue nebula still, plus two twinkle star layers."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yn, xn = yy / h, xx / w
    sc = [480, 240, 120, 60, 30, 15, 7, 3]
    wt = [1, .7, .5, .35, .25, .15, .1, .06]

    def cloud(lo, gamma):
        return np.clip((fbm(h, w, rng, sc, wt) - lo) / (1 - lo), 0, 1) ** gamma

    # Diagonal band where the nebula is densest, pinkest on the right.
    band = np.exp(-((xn * .6 + yn * .4 - .62) / .2) ** 2)
    pink = cloud(.4, 1.3) * (.2 + 1.2 * band) * np.clip(xn * 1.5, .15, 1.3)
    violet = cloud(.35, 1.15)
    blue = cloud(.32, 1.2) * np.clip(1.25 - xn * 1.3, 0, 1)
    filament = (1 - np.abs(2 * fbm(h, w, rng, sc, wt) - 1)) ** 3 * (.3 + band)

    img = np.zeros((h, w, 3), np.float32) + np.array([6, 4, 22]) / 255
    img += violet[..., None] * np.array([110, 50, 215]) / 255 * 1.1
    img += blue[..., None] * np.array([50, 60, 235]) / 255 * .9
    img += pink[..., None] * np.array([240, 85, 220]) / 255 * 1.1
    img += (filament * violet)[..., None] * np.array([200, 140, 255]) / 255 * .5

    density = .15 + violet + pink + blue

    def stars(n, blur, bright, power):
        layer = np.zeros((h, w), np.float32)
        ys = rng.integers(0, h, n)
        xs = rng.integers(0, w, n)
        keep = rng.random(n) < density[ys, xs] / density.max()
        ys, xs = ys[keep], xs[keep]
        np.add.at(layer, (ys, xs), rng.random(len(ys)) ** power * bright)
        if blur:
            g = Image.fromarray(np.clip(layer * 255, 0, 255).astype(np.uint8))
            g = np.asarray(g.filter(ImageFilter.GaussianBlur(blur)), np.float32) / 255
            layer = layer + g * blur * 7
        return layer

    img += np.clip(stars(h * w // 25, 0, 1.0, 4) + stars(h * w // 600, .8, 1.0, 2),
                   0, 1.6)[..., None]
    tw_a = stars(h * w // 9000, 2.2, 1.6, 2)
    tw_b = stars(h * w // 9000, 2.2, 1.6, 2)
    return np.clip(img, 0, 1), tw_a, tw_b


def render_title(lines, max_w=920):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    size = 74
    font = ImageFont.truetype(TITLE_FONT, size)
    while max(font.getlength(l) for l in lines) > max_w:
        size -= 2
        font = ImageFont.truetype(TITLE_FONT, size)
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd, td = ImageDraw.Draw(glow), ImageDraw.Draw(layer)
    y = 120
    for l in lines:
        x = (W - font.getlength(l)) / 2
        gd.text((x, y), l, font=font, fill=GLOW_COLOR + (255,), stroke_width=3,
                stroke_fill=GLOW_COLOR + (255,))
        td.text((x, y), l, font=font, fill=TITLE_COLOR + (255,))
        y += int(size * 1.08)
    soft = glow.filter(ImageFilter.GaussianBlur(16))
    soft = Image.alpha_composite(soft, glow.filter(ImageFilter.GaussianBlur(6)))
    return Image.alpha_composite(soft, layer)


def render_card(path, box_w=720, box_h=1040, top=320):
    card = Image.open(path).convert("RGBA")
    scale = min(box_w / card.width, box_h / card.height)
    card = card.resize((round(card.width * scale), round(card.height * scale)), Image.LANCZOS)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    x = (W - card.width) // 2
    halo = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    halo.paste((150, 90, 255, 255), (x, top), card.getchannel("A"))
    halo = halo.filter(ImageFilter.GaussianBlur(28))
    halo.putalpha(halo.getchannel("A").point(lambda a: int(a * .55)))
    layer = Image.alpha_composite(layer, halo)
    layer.alpha_composite(card, (x, top))
    return layer, top + card.height


def bg_video_frames(path, n):
    """Yield n frames from a background clip, cropped to fill and looped."""
    vf = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS}"
    while True:
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-f", "rawvideo",
                              "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        while True:
            buf = p.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32) / 255
            n -= 1
            if n <= 0:
                p.kill()
                return
        p.wait()


# ---------------------------------------------------------------- main

def main(cfg_path):
    cfg = json.load(open(cfg_path))
    base = os.path.dirname(os.path.abspath(cfg_path))
    rel = lambda p: p if os.path.isabs(p) else os.path.join(base, p)
    out = rel(cfg.get("output", os.path.splitext(os.path.basename(cfg_path))[0] + ".mp4"))
    tmp = tempfile.mkdtemp()

    paragraphs = [p.strip() for p in cfg["script"] if p.strip()]
    wav = os.path.join(tmp, "voice.wav")
    print("voiceover...")
    words, dur = make_voiceover(paragraphs, cfg.get("voice", "af_heart"),
                                cfg.get("speed", 1.0), wav)
    ass = os.path.join(tmp, "captions.ass")
    title = render_title(cfg["title"])
    card, card_bottom = render_card(rel(cfg["card"]))
    write_ass(words, ass, y=card_bottom + 70)
    json.dump(words, open(os.path.splitext(out)[0] + "_words.json", "w"), indent=1)

    fg = Image.alpha_composite(card, title)
    fg = np.asarray(fg, np.float32) / 255
    fg_rgb, fg_a = fg[..., :3], fg[..., 3:]
    n = int(dur * FPS)

    if cfg.get("background"):
        frames = bg_video_frames(rel(cfg["background"]), n)
    else:
        print("galaxy...")
        gh, gw = int(H * 1.12), int(W * 1.12)
        neb, tw_a, tw_b = galaxy(gh, gw, cfg.get("seed", 7))
        neb_i = Image.fromarray((neb * 255).astype(np.uint8))
        ta_i = Image.fromarray((np.clip(tw_a, 0, 1) * 255).astype(np.uint8))
        tb_i = Image.fromarray((np.clip(tw_b, 0, 1) * 255).astype(np.uint8))

        def gen():
            for f in range(n):
                t = f / FPS
                k = f / max(n - 1, 1)
                z = 1.0 + 0.06 * k                     # slow push in
                ox = (gw - W / z) / 2 + 20 * np.sin(k * np.pi)
                oy = (gh - H / z) / 2 - 30 * (k - .5)
                m = (1 / z, 0, ox, 0, 1 / z, oy)
                bg = np.asarray(neb_i.transform((W, H), Image.AFFINE, m, Image.BILINEAR),
                                np.float32) / 255
                a = np.asarray(ta_i.transform((W, H), Image.AFFINE, m, Image.BILINEAR),
                               np.float32)[..., None] / 255
                b = np.asarray(tb_i.transform((W, H), Image.AFFINE, m, Image.BILINEAR),
                               np.float32)[..., None] / 255
                wa = .55 + .45 * np.sin(t * 2.1)
                wb = .55 + .45 * np.sin(t * 1.7 + 2)
                yield np.minimum(bg + a * wa + b * wb, 1)
        frames = gen()

    print(f"rendering {n} frames ({dur:.1f}s)...")
    ass_esc = ass.replace(":", "\\:")
    ff = subprocess.Popen([
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", wav,
        "-vf", f"ass={ass_esc}:fontsdir={os.path.expanduser('~/.fonts')}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out,
    ], stdin=subprocess.PIPE)
    for frame in frames:
        comp = frame * (1 - fg_a) + fg_rgb * fg_a
        ff.stdin.write((comp * 255 + .5).astype(np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
