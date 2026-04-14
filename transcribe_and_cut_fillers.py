#!/usr/bin/env python3

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_MODEL = "large-v3"
DEFAULT_FILLERS = ["uh"]
DEFAULT_WORD_PROBABILITY = 0.60
DEFAULT_MAX_FILLER_DURATION = 0.80
DEFAULT_PADDING_SECONDS = 0.02
DEFAULT_AUDIO_BITRATE_BY_CODEC = {
    "aac": "192k",
    "libmp3lame": "192k",
    "libopus": "160k",
}
OUTPUT_CODEC_BY_SUFFIX = {
    ".wav": "pcm_s16le",
    ".mp3": "libmp3lame",
    ".m4a": "aac",
    ".mp4": "aac",
    ".aac": "aac",
    ".ogg": "libopus",
    ".opus": "libopus",
    ".flac": "flac",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a timestamped JSON transcript with faster-whisper word timings, then optionally cut "
            "matched filler-word intervals from the audio using ffmpeg trim and concat operations."
        ),
        epilog=(
            "Requires faster-whisper in the active environment. ffmpeg is required when cuts are applied. "
            "The default model is large-v3 for accuracy, which will download model weights on first use unless "
            "you point --model at a local path or use --local-files-only."
        ),
    )
    parser.add_argument("input", help="Merged input audio file.")
    parser.add_argument(
        "--transcript-output",
        required=True,
        help="Path for the JSON transcript artifact with segment and word timestamps.",
    )
    parser.add_argument(
        "--cleaned-output",
        required=True,
        help="Path for the cleaned audio artifact. If no cuts are applied, the input is copied here.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Whisper model name or local model path. Default: large-v3.",
    )
    parser.add_argument(
        "--download-root",
        help="Optional directory for Whisper model downloads and cache.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse network model downloads and only use locally available model files.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Transcription device: auto, cuda, or cpu. Default: auto.",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        help="ctranslate2 compute type such as default, float16, int8, or int8_float16. Default: default.",
    )
    parser.add_argument("--language", help="Optional language code, for example en.")
    parser.add_argument(
        "--beam-size",
        type=positive_int,
        default=5,
        help="Beam size for decoding. Default: 5.",
    )
    parser.add_argument(
        "--best-of",
        type=positive_int,
        default=5,
        help="Best-of count when sampling is used. Default: 5.",
    )
    parser.add_argument(
        "--vad-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable faster-whisper VAD before transcription. Default: enabled.",
    )
    parser.add_argument(
        "--vad-min-silence-ms",
        type=non_negative_int,
        default=500,
        help="Minimum silence for VAD segmentation, in milliseconds. Default: 500.",
    )
    parser.add_argument(
        "--vad-speech-pad-ms",
        type=non_negative_int,
        default=200,
        help="Speech padding used by VAD, in milliseconds. Default: 200.",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preserve previous-text conditioning between segments. Default: disabled.",
    )
    parser.add_argument(
        "--cut-fillers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove matched filler-word intervals from the output audio. Default: enabled.",
    )
    parser.add_argument(
        "--filler",
        action="append",
        default=[],
        help="Filler token to remove. Repeat or pass comma-separated values. Default: uh.",
    )
    parser.add_argument(
        "--min-word-probability",
        type=bounded_probability,
        default=DEFAULT_WORD_PROBABILITY,
        help="Minimum word probability required before a filler match is eligible for cutting. Default: 0.60.",
    )
    parser.add_argument(
        "--max-filler-duration",
        type=positive_float,
        default=DEFAULT_MAX_FILLER_DURATION,
        help="Only cut matches this long or shorter, in seconds. Default: 0.80.",
    )
    parser.add_argument(
        "--padding-seconds",
        type=non_negative_float,
        default=DEFAULT_PADDING_SECONDS,
        help="Padding added around each cut interval, in seconds. Default: 0.02.",
    )
    parser.add_argument(
        "--audio-codec",
        help="Optional ffmpeg audio codec override for the cleaned output.",
    )
    parser.add_argument(
        "--audio-bitrate",
        help="Optional ffmpeg audio bitrate override such as 192k.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable to use for audio cutting. Default: ffmpeg.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing transcript or cleaned output artifacts.",
    )
    return parser.parse_args()


def positive_int(value):
    amount = int(value)
    if amount <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return amount


def non_negative_int(value):
    amount = int(value)
    if amount < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return amount


def positive_float(value):
    amount = float(value)
    if amount <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return amount


def non_negative_float(value):
    amount = float(value)
    if amount < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return amount


def bounded_probability(value):
    amount = float(value)
    if not 0.0 <= amount <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0.0 and 1.0")
    return amount


def parse_fillers(raw_values):
    tokens = []
    for raw_value in raw_values:
        for piece in raw_value.split(","):
            normalized = normalize_token(piece)
            if normalized:
                tokens.append(normalized)
    if not tokens:
        tokens = list(DEFAULT_FILLERS)
    unique_tokens = []
    seen = set()
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)
    return unique_tokens


def normalize_token(text):
    cleaned = re.sub(r"[^a-z0-9']+", "", text.lower())
    return cleaned.strip("'")


def require_binary(binary_name):
    resolved = shutil.which(binary_name)
    if not resolved:
        raise SystemExit(f"Required executable not found on PATH: {binary_name}")
    return resolved


def serialize_info(info):
    language_probs = getattr(info, "all_language_probs", None)
    return {
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "duration_after_vad": getattr(info, "duration_after_vad", None),
        "all_language_probs": language_probs,
    }


def serialize_segments(segments):
    serialized = []
    for segment in segments:
        words = []
        for word in segment.words or []:
            words.append(
                {
                    "start": word.start,
                    "end": word.end,
                    "word": word.word,
                    "probability": getattr(word, "probability", None),
                }
            )
        serialized.append(
            {
                "id": segment.id,
                "seek": segment.seek,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "avg_logprob": getattr(segment, "avg_logprob", None),
                "compression_ratio": getattr(segment, "compression_ratio", None),
                "no_speech_prob": getattr(segment, "no_speech_prob", None),
                "words": words,
            }
        )
    return serialized


def collect_filler_matches(segments, filler_tokens, min_probability, max_duration):
    matches = []
    for segment in segments:
        for word in segment.words or []:
            if word.start is None or word.end is None:
                continue
            normalized_word = normalize_token(word.word)
            if normalized_word not in filler_tokens:
                continue
            duration = float(word.end) - float(word.start)
            probability = getattr(word, "probability", 0.0) or 0.0
            if duration <= 0 or duration > max_duration:
                continue
            if probability < min_probability:
                continue
            matches.append(
                {
                    "word": word.word,
                    "normalized": normalized_word,
                    "start": float(word.start),
                    "end": float(word.end),
                    "duration": duration,
                    "probability": probability,
                    "segment_id": segment.id,
                }
            )
    return matches


def merge_intervals(intervals):
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        current = merged[-1]
        if start <= current[1]:
            current[1] = max(current[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def build_cut_intervals(matches, total_duration, padding_seconds):
    expanded = []
    for match in matches:
        start = max(0.0, match["start"] - padding_seconds)
        end = min(total_duration, match["end"] + padding_seconds)
        if end > start:
            expanded.append((start, end))
    expanded.sort(key=lambda item: item[0])
    return merge_intervals(expanded)


def build_keep_intervals(cut_intervals, total_duration):
    if not cut_intervals:
        return [(0.0, total_duration)]
    keep = []
    cursor = 0.0
    for start, end in cut_intervals:
        if start > cursor:
            keep.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < total_duration:
        keep.append((cursor, total_duration))
    return [(start, end) for start, end in keep if end - start > 0.001]


def choose_audio_codec(output_path, requested_codec):
    if requested_codec:
        return requested_codec
    return OUTPUT_CODEC_BY_SUFFIX.get(output_path.suffix.lower(), "pcm_s16le")


def choose_audio_bitrate(codec, requested_bitrate):
    if requested_bitrate:
        return requested_bitrate
    return DEFAULT_AUDIO_BITRATE_BY_CODEC.get(codec)


def run_command(command):
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def write_cleaned_audio(ffmpeg_bin, input_path, output_path, keep_intervals, audio_codec, audio_bitrate, overwrite):
    if not keep_intervals:
        raise SystemExit("No non-filler audio remains after applying the requested cut intervals")

    with tempfile.TemporaryDirectory(prefix="filler_cut_") as temp_dir:
        script_path = Path(temp_dir) / "filtergraph.txt"
        parts = []
        labels = []
        for index, (start, end) in enumerate(keep_intervals):
            label = f"a{index}"
            labels.append(f"[{label}]")
            parts.append(
                f"[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[{label}]"
            )
        if len(labels) == 1:
            parts.append(f"{labels[0]}anull[outa]")
        else:
            parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[outa]")
        script_path.write_text(";\n".join(parts), encoding="ascii")

        command = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y" if overwrite else "-n",
            "-i",
            str(input_path),
            "-filter_complex_script",
            str(script_path),
            "-map",
            "[outa]",
            "-vn",
            "-c:a",
            audio_codec,
        ]
        if audio_bitrate:
            command.extend(["-b:a", audio_bitrate])
        if audio_codec == "aac" and output_path.suffix.lower() in {".m4a", ".mp4"}:
            command.extend(["-movflags", "+faststart"])
        command.append(str(output_path))
        run_command(command)


def main():
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    transcript_output = Path(args.transcript_output).expanduser().resolve()
    cleaned_output = Path(args.cleaned_output).expanduser().resolve()

    if not input_path.is_file():
        raise SystemExit(f"Input file does not exist: {input_path}")
    if cleaned_output == input_path:
        raise SystemExit("Cleaned output path must be different from the input path")
    for artifact_path in (transcript_output, cleaned_output):
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if artifact_path.exists() and not args.overwrite:
            raise SystemExit(f"Artifact already exists. Use --overwrite to replace it: {artifact_path}")

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit(
            "faster-whisper is not installed in this environment. Install it with the local .venv pip first."
        ) from exc

    filler_tokens = parse_fillers(args.filler)
    ffmpeg_bin = None
    if args.cut_fillers:
        ffmpeg_bin = require_binary(args.ffmpeg_bin)

    print(f"Loading model: {args.model} (device={args.device}, compute={args.compute_type})")
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        download_root=args.download_root,
        local_files_only=args.local_files_only,
    )
    print(f"Transcribing: {input_path.name}")
    segments_iterable, info = model.transcribe(
        str(input_path),
        language=args.language,
        beam_size=args.beam_size,
        best_of=args.best_of,
        word_timestamps=True,
        vad_filter=args.vad_filter,
        vad_parameters={
            "min_silence_duration_ms": args.vad_min_silence_ms,
            "speech_pad_ms": args.vad_speech_pad_ms,
        },
        condition_on_previous_text=args.condition_on_previous_text,
    )
    segments = list(segments_iterable)
    total_duration = float(getattr(info, "duration", 0.0) or 0.0)
    detected_lang = getattr(info, "language", "unknown")
    print(f"Transcribed {len(segments)} segment(s), language={detected_lang}, duration={total_duration:.1f}s")

    print(f"Scanning for fillers: {', '.join(filler_tokens)}")
    matches = collect_filler_matches(
        segments,
        set(filler_tokens),
        args.min_word_probability,
        args.max_filler_duration,
    )
    print(f"Found {len(matches)} filler match(es)")
    cut_intervals = []
    keep_intervals = [(0.0, total_duration)] if total_duration > 0 else []
    if args.cut_fillers and total_duration > 0:
        cut_intervals = build_cut_intervals(matches, total_duration, args.padding_seconds)
        keep_intervals = build_keep_intervals(cut_intervals, total_duration)

    transcript_payload = {
        "input": str(input_path),
        "transcription": {
            "model": args.model,
            "device": args.device,
            "compute_type": args.compute_type,
            "language_requested": args.language,
            "beam_size": args.beam_size,
            "best_of": args.best_of,
            "vad_filter": args.vad_filter,
            "vad_min_silence_ms": args.vad_min_silence_ms,
            "vad_speech_pad_ms": args.vad_speech_pad_ms,
            "condition_on_previous_text": args.condition_on_previous_text,
        },
        "info": serialize_info(info),
        "fillers": {
            "cut_fillers": args.cut_fillers,
            "tokens": filler_tokens,
            "min_word_probability": args.min_word_probability,
            "max_filler_duration": args.max_filler_duration,
            "padding_seconds": args.padding_seconds,
            "matches": matches,
            "cut_intervals": [
                {"start": start, "end": end, "duration": end - start}
                for start, end in cut_intervals
            ],
        },
        "segments": serialize_segments(segments),
    }
    transcript_output.write_text(json.dumps(transcript_payload, indent=2), encoding="utf-8")
    print(f"Wrote transcript → {transcript_output}")

    if not args.cut_fillers or not cut_intervals:
        if cleaned_output.exists() and not args.overwrite:
            raise SystemExit(f"Cleaned output already exists. Use --overwrite to replace it: {cleaned_output}")
        shutil.copy2(input_path, cleaned_output)
        print(f"No fillers to cut; copied input → {cleaned_output}")
    else:
        cut_total = sum(e - s for s, e in cut_intervals)
        print(f"Cutting {len(cut_intervals)} filler interval(s) ({cut_total:.2f}s removed)")
        audio_codec = choose_audio_codec(cleaned_output, args.audio_codec)
        audio_bitrate = choose_audio_bitrate(audio_codec, args.audio_bitrate)
        write_cleaned_audio(
            ffmpeg_bin,
            input_path,
            cleaned_output,
            keep_intervals,
            audio_codec,
            audio_bitrate,
            args.overwrite,
        )
        print(f"Wrote cleaned audio → {cleaned_output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)