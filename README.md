# podcast_tools

Tools for producing a two-person podcast: an **audio chain** that turns two raw speaker
recordings into a cleaned, filler-free episode file, and a **video chain** that turns
YouTube videos into silent, scene-shuffled Vegas-ready background visuals.

`pipeline.py` runs each chain end to end. Every stage script also works standalone.

```
AUDIO CHAIN   host.wav + guest.wav
                │  pan_mix_truncate.py        pan apart, mix, truncate silence
                ▼
              <name>_mixed.wav                (kept — the "raw" mix)
                │  transcribe_and_cut_fillers.py   whisper transcript, cut "um"/"uh"
                ▼
              <name>_final.mp3                (kept — the episode)
                │  fit_media_under_cap.py     optional, only with --max-size
                ▼
              <name>_final.mp3 under the cap

VIDEO CHAIN   YouTube URL / local file / links.txt
                │  yt_shuffle_scenes.py       download, encode silent H.264,
                │                             detect scenes, drop junk scenes,
                ▼                             shuffle, concat
              <source>_shuffled.mp4           (kept)

              (--no-shuffle instead runs yt_to_vegas.py: plain batch
               download + silent Vegas-ready encode, no scene work)
```

## Setup

- `ffmpeg` and `ffprobe` on `PATH` (NVENC is used automatically when available).
- `source .venv/bin/activate` then `pip install -r requirements.txt`.
- The `deno` binary in this folder is used by yt-dlp for YouTube signature solving.

## Usage

Full audio chain — mixes, cuts fillers, writes `ep12_mixed.wav` + `ep12_final.mp3`:

```bash
./pipeline.py audio host.wav guest.wav --name ep12 --out-dir episodes/
```

Add a size cap and keep the transcript JSON:

```bash
./pipeline.py audio host.wav guest.wav --name ep12 --max-size 500M --keep-intermediate
```

Full video chain — one shuffled MP4 per URL in the links file, written to `vegas_output/`:

```bash
./pipeline.py video links.txt
./pipeline.py video 'https://www.youtube.com/watch?v=...' --shuffle-seed 7
./pipeline.py video links.txt --no-shuffle     # plain Vegas encodes, no shuffling
```

By default only the end products (plus the mixed raw audio) are kept; per-scene clips,
manifests, filter reports, and transcripts are deleted. Pass `--keep-intermediate` to
either subcommand to keep them for inspection.

## Stage scripts

| Script | Role | Standalone example |
| --- | --- | --- |
| `pan_mix_truncate.py` | Pan two mono-like tracks apart, mix to stereo, truncate silence Audacity-style (removed from the middle of each silent region). | `./pan_mix_truncate.py host.wav guest.wav merged.wav --overwrite` |
| `transcribe_and_cut_fillers.py` | faster-whisper transcript with word timings; cuts matched `um`/`uh` (and elongated `umm`/`uhh`) intervals. | `./transcribe_and_cut_fillers.py merged.wav --transcript-output t.json --cleaned-output clean.mp3 --language en` |
| `fit_media_under_cap.py` | Two-pass re-encode to the best quality that fits a size cap; leaves files already under the cap alone. | `./fit_media_under_cap.py episode.mp4 --max-size 10GiB` |
| `yt_shuffle_scenes.py` | Download/reuse a video, encode silent Vegas H.264, detect scenes, drop junk scenes, shuffle, concat. | `./yt_shuffle_scenes.py 'https://youtube.com/watch?v=...' --shuffle-seed 7 --overwrite` |
| `yt_to_vegas.py` | Parallel batch download + silent Vegas-ready encode of every URL in a links file. **Wipes its download/output dirs on every run.** | `./yt_to_vegas.py --input-file links.txt` |

## Filler-word cutting: how accuracy was fixed

Whisper models are trained on transcripts that omit disfluencies, so by default they
simply don't write most "um"s — no matching can recover them afterwards. The script now:

- feeds filler-heavy **hotwords** and an **initial prompt** to the decoder so it
  transcribes disfluencies verbatim,
- disables Whisper's default token suppression list,
- accepts filler matches at **any confidence** (Whisper assigns fillers a median
  probability of ~0.08 even when they are real — the old 0.60 gate rejected nearly all
  true matches),
- matches elongated variants (`ummm`, `uhh`) by collapsing repeated letters.

Add more tokens with `--filler` (repeatable/comma-separated). If it ever cuts real words,
raise `--min-word-probability` or lower `--max-filler-duration`.

## Scene filters: what gets dropped and why

Three filters run before clips are cut, in this order:

1. **Text-only filter** — static scenes with a flat background and either a compact band
   of text-like edges (title cards, quote screens) or almost no edges at all (blank
   cards). Motion scoring ignores the single largest frame-to-frame delta, so cards with
   fade-in/out animations still register as static (this was the main source of leaked
   title cards before).
2. **Black-scene filter** — scenes where ≥90% of pixels are dark; catches fade-through
   transitions and dark company logo cards. Disable with `--no-filter-black-scenes`.
3. **End-card filter** — static, branded-layout scenes in the trailing timeline window
   (outro/end screens). Tune with the `--end-card-*` flags.

Every filter is suppression-safe: if it would drop *all* remaining scenes it keeps them
instead and records the suppression in the run report
(`<work-dir>/<source>/manifests/<source>_segment_filter_report.json`, kept only with
`--keep-intermediate`).

## Notes

- `transcribe_and_cut_fillers.py` defaults to the `large-v3` model; the first run
  downloads weights unless you use `--local-files-only` or point `--model` elsewhere.
  On a CUDA GPU, `--device cuda --compute-type float16` is much faster.
- `yt_shuffle_scenes.py` accepts a local video file instead of a URL, which is handy for
  offline testing of the scene/filter/shuffle pipeline.
- `pan_mix_truncate.py` expects mono-like inputs; stereo sources are balance-shifted
  rather than true-panned and produce a warning.
