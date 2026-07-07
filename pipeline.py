#!/usr/bin/env python3
"""Run the podcast_tools workflows end to end.

Two independent chains live in this repo; this orchestrator wires each one together so a
single command replaces the old copy-paste-between-scripts workflow. Every stage script
remains runnable standalone.

Audio chain (`pipeline.py audio`):
    1. pan_mix_truncate.py         pan two source tracks apart, mix, truncate silence
    2. transcribe_and_cut_fillers.py  whisper transcript + cut "um"/"uh" fillers
    3. fit_media_under_cap.py      (optional, --max-size) shrink under an upload cap

    Kept by default: the mixed "raw" file (<name>_mixed.wav) and the final cleaned
    audio (<name>_final.mp3). The transcript JSON is discarded unless
    --keep-intermediate is passed.

Video chain (`pipeline.py video`):
    Each source (YouTube URL, local file, or a links .txt file with one URL per line)
    is run through yt_shuffle_scenes.py, producing a silent Vegas-ready shuffled MP4
    per source. Pass --no-shuffle to instead batch-encode straight Vegas-ready MP4s
    with yt_to_vegas.py (URLs only).

Examples:
    ./pipeline.py audio host.wav guest.wav --name ep12 --out-dir episodes/
    ./pipeline.py audio host.wav guest.wav --name ep12 --max-size 500M
    ./pipeline.py video links.txt
    ./pipeline.py video 'https://www.youtube.com/watch?v=...' --shuffle-seed 7
    ./pipeline.py video local_clip.mp4 --keep-intermediate
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run_stage(label, command):
    print(f"\n=== [{label}] {' '.join(str(part) for part in command[1:])}", flush=True)
    result = subprocess.run([str(part) for part in command])
    if result.returncode != 0:
        raise SystemExit(f"Stage '{label}' failed with exit code {result.returncode}")


def script(name):
    return [sys.executable, SCRIPT_DIR / name]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Orchestrate the podcast audio chain and the scene-shuffle video chain.",
        epilog="Run 'pipeline.py audio --help' or 'pipeline.py video --help' for stage options.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audio = subparsers.add_parser(
        "audio",
        help="Mix two tracks, cut filler words, optionally fit under a size cap.",
        description=(
            "Audio chain: pan/mix/truncate the two input tracks, transcribe and cut filler "
            "words, and optionally re-encode the result under a size cap. Keeps the mixed "
            "raw file and the final cleaned audio."
        ),
    )
    audio.add_argument("left_input", help="First track (panned left).")
    audio.add_argument("right_input", help="Second track (panned right).")
    audio.add_argument(
        "--name",
        help="Base name for output files. Default: the first input's stem.",
    )
    audio.add_argument(
        "--out-dir",
        default=".",
        help="Directory for the kept outputs. Default: current directory.",
    )
    audio.add_argument(
        "--final-format",
        default="mp3",
        help="Extension of the final cleaned audio (mp3, wav, m4a, ...). Default: mp3.",
    )
    audio.add_argument(
        "--max-size",
        help=(
            "Optional size cap such as 500M or 2GiB. When given, the final file is "
            "re-encoded under the cap with fit_media_under_cap.py."
        ),
    )
    audio.add_argument("--model", default="large-v3", help="Whisper model for transcription. Default: large-v3.")
    audio.add_argument("--language", default="en", help="Transcription language code. Default: en.")
    audio.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Also keep the transcript JSON next to the outputs.",
    )
    audio.add_argument("--overwrite", action="store_true", help="Replace existing output files.")

    video = subparsers.add_parser(
        "video",
        help="Shuffle scenes of each source video (or batch-encode with --no-shuffle).",
        description=(
            "Video chain: run yt_shuffle_scenes.py for each source. Sources may be YouTube "
            "URLs, local video files, or .txt files containing one URL per line."
        ),
    )
    video.add_argument("sources", nargs="+", help="YouTube URLs, local video files, or links .txt files.")
    video.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle scenes (default). --no-shuffle batch-encodes plain Vegas-ready MP4s instead.",
    )
    video.add_argument("--output-dir", help="Directory for final outputs. Default: ./vegas_output.")
    video.add_argument("--shuffle-seed", type=int, default=0, help="Deterministic shuffle seed. Default: 0.")
    video.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep per-scene clips, manifests, and filter reports in the work directory.",
    )
    video.add_argument("--overwrite", action="store_true", help="Replace existing outputs and job directories.")
    video.add_argument(
        "--skip-yt-dlp-update",
        action="store_true",
        help="Skip the yt-dlp self-update before downloading.",
    )

    return parser.parse_args()


def run_audio_chain(args):
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or Path(args.left_input).stem
    final_extension = "." + args.final_format.lstrip(".")

    mixed_path = out_dir / f"{name}_mixed.wav"
    final_path = out_dir / f"{name}_final{final_extension}"
    kept_transcript_path = out_dir / f"{name}_transcript.json"

    for path in (mixed_path, final_path):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Output already exists. Use --overwrite to replace it: {path}")

    mix_command = script("pan_mix_truncate.py") + [args.left_input, args.right_input, mixed_path]
    if args.overwrite:
        mix_command.append("--overwrite")
    run_stage("mix", mix_command)

    with tempfile.TemporaryDirectory(prefix="pipeline_audio_") as temp_dir:
        transcript_path = kept_transcript_path if args.keep_intermediate else Path(temp_dir) / "transcript.json"
        filler_command = script("transcribe_and_cut_fillers.py") + [
            mixed_path,
            "--transcript-output", transcript_path,
            "--cleaned-output", final_path,
            "--model", args.model,
            "--language", args.language,
            "--overwrite",
        ]
        run_stage("cut-fillers", filler_command)

    if args.max_size:
        cap_command = script("fit_media_under_cap.py") + [final_path, "--max-size", args.max_size, "--overwrite"]
        run_stage("size-cap", cap_command)

    print("\nAudio chain complete.")
    print(f"  Mixed raw:   {mixed_path}")
    print(f"  Final audio: {final_path}")
    if args.keep_intermediate:
        print(f"  Transcript:  {kept_transcript_path}")


def expand_video_sources(raw_sources):
    sources = []
    for raw in raw_sources:
        path = Path(raw).expanduser()
        if path.is_file() and path.suffix.lower() == ".txt":
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            sources.extend(lines)
        else:
            sources.append(raw)
    return list(dict.fromkeys(sources))


def run_video_chain(args):
    sources = expand_video_sources(args.sources)
    if not sources:
        raise SystemExit("No video sources found.")

    if not args.shuffle:
        local_files = [s for s in sources if Path(s).expanduser().is_file()]
        if local_files:
            raise SystemExit(
                "--no-shuffle uses yt_to_vegas.py, which only accepts URLs. Local files: "
                + ", ".join(local_files)
            )
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("\n".join(sources) + "\n")
            links_path = handle.name
        command = script("yt_to_vegas.py") + ["--input-file", links_path]
        if args.output_dir:
            command += ["--output-dir", args.output_dir]
        run_stage("batch-encode", command)
        Path(links_path).unlink(missing_ok=True)
        return

    failures = []
    for index, source in enumerate(sources, 1):
        command = script("yt_shuffle_scenes.py") + [source, "--shuffle-seed", str(args.shuffle_seed)]
        if args.output_dir:
            command += ["--output-dir", args.output_dir]
        if args.keep_intermediate:
            command.append("--keep-intermediate")
        if args.overwrite:
            command.append("--overwrite")
        if args.skip_yt_dlp_update or index > 1:
            command.append("--skip-yt-dlp-update")
        try:
            run_stage(f"shuffle {index}/{len(sources)}", command)
        except SystemExit as exc:
            print(f"Warning: {exc}; continuing with remaining sources.", file=sys.stderr)
            failures.append(source)

    print("\nVideo chain complete.")
    if failures:
        print(f"  {len(failures)} source(s) failed:", file=sys.stderr)
        for source in failures:
            print(f"    {source}", file=sys.stderr)
        raise SystemExit(1)


def main():
    args = parse_args()
    if args.command == "audio":
        run_audio_chain(args)
    else:
        run_video_chain(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
