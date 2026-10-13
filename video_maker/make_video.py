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


_models = {}


def voice_style(tts, voice):
    """'af_heart' or a blend like 'af_heart:0.7,af_nicole:0.3'."""
    if ":" not in voice:
        return voice
    parts = [v.split(":") for v in voice.split(",")]
    return sum(tts.get_voice_style(name) * float(w) for name, w in parts)


def make_voiceover(paragraphs, voice, speed, out_wav, gap=0.45):
    from faster_whisper import WhisperModel
    if not _models:
        _models["tts"] = kokoro()
        _models["stt"] = WhisperModel("small.en", compute_type="int8")
    tts, stt = _models["tts"], _models["stt"]
    lang = "en-gb" if voice.startswith("b") else "en-us"
    style = voice_style(tts, voice)
    lead, tail = 0.8, 2.0

    pieces = [np.zeros(int(lead * SR), np.float32)]
    t = lead
    words = []  # (text, start, end, para_index)
    for p, text in enumerate(paragraphs):
        audio, sr = tts.create(text, voice=style, speed=speed, lang=lang)
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


# Nebula colour themes: base, main cloud, left cloud, bright band, filaments.
PALETTES = {
    "amethyst": [(6, 4, 22), (110, 50, 215), (50, 60, 235), (240, 85, 220), (200, 140, 255)],
    "ocean":    [(2, 8, 22), (20, 90, 170), (10, 150, 170), (60, 200, 230), (150, 230, 255)],
    "emerald":  [(2, 12, 14), (15, 110, 95), (20, 70, 140), (90, 210, 150), (170, 255, 210)],
    "rose":     [(16, 4, 14), (170, 40, 100), (90, 40, 160), (255, 120, 150), (255, 190, 210)],
    "sunset":   [(16, 5, 10), (170, 50, 40), (110, 30, 110), (255, 150, 60), (255, 210, 140)],
    "midnight": [(2, 4, 16), (35, 50, 150), (20, 25, 90), (110, 130, 255), (190, 200, 255)],
    "aurora":   [(3, 8, 18), (20, 120, 120), (70, 40, 170), (120, 230, 120), (210, 170, 255)],
}


def galaxy(h, w, seed=7, palette="amethyst"):
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

    base, c_main, c_left, c_band, c_fil = (np.array(c) / 255 for c in PALETTES[palette])
    img = np.zeros((h, w, 3), np.float32) + base
    img += violet[..., None] * c_main * 1.1
    img += blue[..., None] * c_left * .9
    img += pink[..., None] * c_band * 1.1
    img += (filament * violet)[..., None] * c_fil * .5

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


# ---------------------------------------------------------------- render

CACHE = os.path.join(HERE, "cache")
LOOP_SECONDS = 60


def galaxy_loop(path, seed=7, palette="amethyst"):
    """Render a seamlessly looping galaxy clip once; every video reuses it."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("rendering galaxy loop (one time)...")
    gh, gw = int(H * 1.12), int(W * 1.12)
    neb, tw_a, tw_b = galaxy(gh, gw, seed, palette)
    to_img = lambda a: Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    neb_i, ta_i, tb_i = to_img(neb), to_img(tw_a), to_img(tw_b)
    n = LOOP_SECONDS * FPS
    ff = subprocess.Popen([
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", path,
    ], stdin=subprocess.PIPE)
    for f in range(n):
        p = 2 * np.pi * f / n                       # one full cycle -> loops cleanly
        z = 1.03 - 0.03 * np.cos(p)                 # gentle breathe in and out
        ox = (gw - W / z) / 2 + 22 * np.sin(p)
        oy = (gh - H / z) / 2 + 30 * np.sin(p + 1)
        m = (1 / z, 0, ox, 0, 1 / z, oy)
        warp = lambda im: np.asarray(im.transform((W, H), Image.AFFINE, m, Image.BILINEAR),
                                     np.float32) / 255
        wa = .55 + .45 * np.sin(p * 19)
        wb = .55 + .45 * np.sin(p * 15 + 2)
        frame = np.minimum(warp(neb_i) + warp(ta_i)[..., None] * wa
                           + warp(tb_i)[..., None] * wb, 1)
        ff.stdin.write((frame * 255 + .5).astype(np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()
    return path


def music_track(path, seconds=240):
    if not os.path.exists(path):
        from music import make_music
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("composing music (one time)...")
        make_music(path, seconds)
    return path


def split_sentences(text):
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    parts = re.split(r"(?<=[.?!])\s+|(?<=[.?!]['\"])\s+", text.strip())
    return [p for p in parts if p]


def title_lines(title, max_w=920):
    title = title.replace("\u2019", "'").upper()
    if ":" in title:
        a, b = title.split(":", 1)
        return [a.strip() + ":", b.strip()]
    font = ImageFont.truetype(TITLE_FONT, 74)
    if font.getlength(title) <= max_w:
        return [title]
    words = title.split()
    best = min(range(1, len(words)), key=lambda i: abs(
        font.getlength(" ".join(words[:i])) - font.getlength(" ".join(words[i:]))))
    return [" ".join(words[:best]), " ".join(words[best:])]


def render_video(title, card_path, sentences, out, voice="af_heart", speed=1.0,
                 gap=0.45, background=None, music=True, music_volume=0.16,
                 palette="amethyst"):
    tmp = tempfile.mkdtemp()
    wav = os.path.join(tmp, "voice.wav")
    print(f"[{os.path.basename(out)}] voiceover...")
    words, dur = make_voiceover(sentences, voice, speed, wav, gap=gap)
    ass = os.path.join(tmp, "captions.ass")
    lines = title if isinstance(title, list) else title_lines(title)
    card, card_bottom = render_card(card_path)
    write_ass(words, ass, y=card_bottom + 70)
    fg_png = os.path.join(tmp, "fg.png")
    Image.alpha_composite(card, render_title(lines)).save(fg_png)
    json.dump(words, open(os.path.splitext(out)[0] + "_words.json", "w"), indent=1)

    bg = background or galaxy_loop(os.path.join(CACHE, f"galaxy_{palette}.mp4"),
                                   palette=palette)
    cmd = ["ffmpeg", "-v", "error", "-y", "-stream_loop", "-1", "-i", bg,
           "-loop", "1", "-i", fg_png, "-i", wav]
    ass_esc = ass.replace(":", "\\:")
    vf = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
          f"fps={FPS},setsar=1[bg];[bg][1:v]overlay=0:0:format=auto,"
          f"ass={ass_esc}:fontsdir={os.path.expanduser('~/.fonts')}[v]")
    voice_chain = "[2:a]aresample=48000,highshelf=f=6500:g=-4,lowshelf=f=180:g=1.5,"\
                  "dynaudnorm=f=250:g=15:p=0.85,aformat=channel_layouts=stereo[vo]"
    if music:
        cmd += ["-i", music_track(os.path.join(CACHE, "ambient.wav"))]
        af = (f"{voice_chain};[3:a]atrim=0:{dur:.2f},afade=t=out:st={dur - 2.5:.2f}:d=2.5,"
              f"volume={music_volume}[mu];[vo][mu]amix=inputs=2:duration=first:normalize=0[a]")
    else:
        af = voice_chain.replace("[vo]", "[a]")
    cmd += ["-filter_complex", vf + ";" + af, "-map", "[v]", "-map", "[a]", "-t", f"{dur:.2f}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out]
    print(f"[{os.path.basename(out)}] encoding {dur:.1f}s...")
    subprocess.run(cmd, check=True)
    return out


def main(cfg_path):
    cfg = json.load(open(cfg_path))
    base = os.path.dirname(os.path.abspath(cfg_path))
    rel = lambda p: p if os.path.isabs(p) else os.path.join(base, p)
    out = rel(cfg.get("output", os.path.splitext(os.path.basename(cfg_path))[0] + ".mp4"))
    script = cfg["script"]
    sentences = split_sentences(script) if isinstance(script, str) else script
    render_video(cfg["title"], rel(cfg["card"]), sentences, out,
                 voice=cfg.get("voice", "af_heart"), speed=cfg.get("speed", 1.0),
                 gap=cfg.get("gap", 0.45),
                 background=rel(cfg["background"]) if cfg.get("background") else None,
                 music=cfg.get("music", True))
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
