"""Calm ambient pad music, generated so there are no licensing worries."""
import wave

import numpy as np

SR = 48000

# Slow progression in D: Dmaj9, Bm9, Gmaj9, Asus2 (MIDI note numbers).
CHORDS = [
    [50, 57, 61, 64, 66],
    [47, 54, 57, 61, 62],
    [43, 50, 54, 57, 59],
    [45, 52, 57, 59, 64],
]
BELLS = [62, 64, 66, 69, 71, 74, 76, 78]  # D major pentatonic, high


def hz(m):
    return 440 * 2 ** ((m - 69) / 12)


def pad_note(freq, n, rng):
    t = np.arange(n) / SR
    out = np.zeros(n, np.float32)
    for detune in (-0.0025, 0, 0.0025):
        ph = rng.random() * 2 * np.pi
        out += np.sin(2 * np.pi * freq * (1 + detune) * t + ph)
    out += .25 * np.sin(2 * np.pi * freq * 2 * t)          # soft upper partial
    trem = 1 + .15 * np.sin(2 * np.pi * (.08 + rng.random() * .05) * t)
    return out * trem / 4


def reverb(x, rng):
    """A small Schroeder-style reverb: parallel combs then allpasses."""
    out = np.zeros_like(x)
    for d_ms, g in ((50.3, .78), (56.7, .77), (61.1, .76), (68.9, .75)):
        d = int(SR * d_ms / 1000)
        y = x.copy()
        for i in range(d, len(y), d):
            y[i:i + d] += g * y[i - d:i][:len(y[i:i + d])]
        out += y
    out /= 4
    for d_ms, g in ((12.7, .7), (4.1, .7)):
        d = int(SR * d_ms / 1000)
        y = np.zeros_like(out)
        y[:d] = out[:d]
        for i in range(d, len(out), d):
            seg = out[i:i + d]
            y[i:i + len(seg)] = -g * seg + out[i - d:i - d + len(seg)] + g * y[i - d:i - d + len(seg)]
        out = y
    return out


def make_music(path, seconds, seed=3, chord_len=8.0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    mix = np.zeros(n, np.float32)
    step = int(chord_len * SR)
    fade = int(3.0 * SR)
    for c, start in enumerate(range(0, n, step)):
        length = min(step + fade, n - start)
        env = np.ones(length, np.float32)
        a = min(fade, length)
        env[:a] = np.sin(np.linspace(0, np.pi / 2, a)) ** 2
        r = min(fade, length)
        env[-r:] *= np.cos(np.linspace(0, np.pi / 2, r)) ** 2
        for m in CHORDS[c % len(CHORDS)]:
            mix[start:start + length] += pad_note(hz(m), length, rng) * env * .22

    # Sparse soft bells.
    t = 1.5
    while t < seconds - 3:
        i = int(t * SR)
        length = min(int(4 * SR), n - i)
        tt = np.arange(length) / SR
        f = hz(rng.choice(BELLS))
        bell = (np.sin(2 * np.pi * f * tt) + .3 * np.sin(2 * np.pi * f * 2.01 * tt))
        mix[i:i + length] += bell * np.exp(-tt * 1.4) * .05 * (1 - np.exp(-tt * 60))
        t += rng.uniform(2.0, 4.5)

    mix = .65 * mix + .35 * reverb(mix, rng)
    # Gentle low-pass (one-pole) to keep it warm.
    alpha = 1 - np.exp(-2 * np.pi * 3500 / SR)
    lp = np.empty_like(mix)
    acc = 0.0
    for i in range(0, n, 4096):
        seg = mix[i:i + 4096]
        out = np.empty_like(seg)
        for k, v in enumerate(seg):
            acc += alpha * (v - acc)
            out[k] = acc
        lp[i:i + 4096] = out
    mix = lp
    fin, fout = int(2 * SR), int(3 * SR)
    mix[:fin] *= np.linspace(0, 1, fin)
    mix[-fout:] *= np.linspace(1, 0, fout)
    mix *= .5 / np.abs(mix).max()
    stereo = np.stack([mix, np.roll(mix, int(.012 * SR))], 1)
    with wave.open(path, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((stereo * 32767).astype(np.int16).tobytes())
