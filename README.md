# podcast_tools

Three script-first media utilities live here:

- `pan_mix_truncate.py` pans two source tracks apart, mixes them into one stereo file, and optionally truncates silence after the combined mix.
- `transcribe_and_cut_fillers.py` creates a timestamped JSON transcript with faster-whisper word timings and can cut matched filler words such as `uh` from the audio.
- `yt_shuffle_scenes.py` downloads or reuses a source video, formats it to the same silent Vegas-ready H.264 MP4 profile as `yt_to_vegas.py`, cuts scene clips, shuffles them, and renders a final master.

## Prerequisites

- `ffmpeg` and `ffprobe` must be on `PATH`.
- Use the local virtual environment in this folder: `source .venv/bin/activate`
- Install Python dependencies with `pip install -r requirements.txt`

## Examples

Mix two mono-style tracks, pan them apart, and truncate silence on the combined output:

```bash
./.venv/bin/python pan_mix_truncate.py host.wav guest.wav merged.wav --overwrite
```

Transcribe a merged file and cut detected `uh` fillers while saving both the transcript JSON and cleaned audio:

```bash
./.venv/bin/python transcribe_and_cut_fillers.py merged.wav --transcript-output transcript.json --cleaned-output cleaned.wav --language en --overwrite
```

Shuffle scene clips from a YouTube video into a new silent Vegas-ready master while keeping intermediates for inspection:

```bash
./.venv/bin/python yt_shuffle_scenes.py 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' --shuffle-seed 7 --keep-intermediate --overwrite
```

## Notes

- `transcribe_and_cut_fillers.py` defaults to the `large-v3` Whisper model for accuracy. The first real run will download model files unless you point `--model` at a local model path or use `--local-files-only`.
- If startup time or hardware is a concern, choose a smaller model with `--model`, but expect lower transcript quality and less reliable filler cuts.
- `pan_mix_truncate.py` is tuned for mono-like input tracks. Stereo inputs currently trigger a warning because the balance-shift approach is not a true positional pan.
- `yt_shuffle_scenes.py` now defaults to a more sensitive scene threshold (`0.20`) and shorter minimum kept scene duration (`0.75s`) so smaller cuts survive detection more often.
- `yt_shuffle_scenes.py` applies a conservative text-only/title-card filter before clip cutting. It now combines low motion, flat-background coverage, edge density, and edge-row concentration so obvious static cards are caught more reliably without treating moving footage with subtitles as text-only. Disable it with `--no-filter-text-only-scenes` or tune it with the `--text-filter-*` flags if your source includes borderline static graphics.
- `yt_shuffle_scenes.py` also runs a separate trailing end-card pass by default. It samples the remaining late-timeline scenes even when the text-only filter is disabled, and only removes a final suffix of likely static branded outro/end-screen scenes. If that pass would remove every remaining segment, the script warns, preserves them all, and records the suppression in the summary/report instead of emitting an empty concat. Disable it with `--no-remove-end-cards` or tighten the late-window bounds with `--end-card-window-seconds`, `--end-card-min-start-ratio`, `--end-card-min-duration`, and `--end-card-max-duration`.
- `yt_shuffle_scenes.py` also accepts a local video file path instead of a YouTube URL so the formatting, scene-cutting, and shuffle pipeline can be tested offline.