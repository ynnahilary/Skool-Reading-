# Video maker

Builds a vertical (1080x1920) tarot video: Kokoro text-to-speech voiceover,
Whisper word timings, glowing word-by-word captions, a title, a card image
and a galaxy background (generated, or your own clip via `"background"`).

```
pip install kokoro-onnx soundfile faster-whisper
python3 make_video.py videos/four_suits.json
```

Each video is a JSON file in `videos/` — copy `four_suits.json` and change
`title`, `card`, `script` (one line per sentence), `voice` and `output`.
