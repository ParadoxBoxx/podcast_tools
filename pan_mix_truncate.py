#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_LEFT_PAN = 0.30
DEFAULT_RIGHT_PAN = 0.30
DEFAULT_SILENCE_THRESHOLD = "-35dB"
DEFAULT_SILENCE_DURATION = 0.25
DEFAULT_RETAIN_SILENCE = 0.50
DEFAULT_DROPOUT_TRANSITION = 0.50

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

DEFAULT_BITRATE_BY_CODEC = {
    "aac": "192k",
    "libmp3lame": "192k",
    "libopus": "160k",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Pan two input tracks apart, merge them into a stereo output, and then truncate silence on "
            "the combined result. This follows an ffmpeg-first workflow and approximates Audacity's "
            "Truncate Silence behavior by shortening silent regions after the mix instead of trimming each "
            "source independently."
        ),
        epilog=(
            "Requires ffmpeg on PATH. The default silence settings mirror the requested Audacity intent: "
            "threshold -35 dB, detect silence after 0.25 s, and retain up to 0.5 s of each detected silent "
            "region in the combined output. Panning assumes mono-like source tracks; if you pass stereo "
            "inputs, the script warns because ffmpeg balance shifting does not produce a true positional pan."
        ),
    )
    parser.add_argument("left_input", help="First input audio file. This track is panned left by default.")
    parser.add_argument("right_input", help="Second input audio file. This track is panned right by default.")
    parser.add_argument("output", help="Output audio path.")
    parser.add_argument(
        "--left-pan",
        type=bounded_pan,
        default=DEFAULT_LEFT_PAN,
        help="Pan amount for the first input, expressed as 0.0 to 1.0 toward the left. Default: 0.30.",
    )
    parser.add_argument(
        "--right-pan",
        type=bounded_pan,
        default=DEFAULT_RIGHT_PAN,
        help="Pan amount for the second input, expressed as 0.0 to 1.0 toward the right. Default: 0.30.",
    )
    parser.add_argument(
        "--silence-threshold",
        default=DEFAULT_SILENCE_THRESHOLD,
        help="Silence threshold passed to ffmpeg silenceremove. Default: -35dB.",
    )
    parser.add_argument(
        "--silence-duration",
        type=positive_float,
        default=DEFAULT_SILENCE_DURATION,
        help="Silence must last at least this many seconds before truncation applies. Default: 0.25.",
    )
    parser.add_argument(
        "--retain-silence",
        type=non_negative_float,
        default=DEFAULT_RETAIN_SILENCE,
        help="Amount of each detected silent region to retain after truncation. Default: 0.50.",
    )
    parser.add_argument(
        "--dropout-transition",
        type=non_negative_float,
        default=DEFAULT_DROPOUT_TRANSITION,
        help=(
            "Seconds for ffmpeg amix to smooth level changes when one input ends before the other. "
            "Default: 0.50."
        ),
    )
    parser.add_argument(
        "--audio-codec",
        help="Optional ffmpeg audio codec override. Defaults from the output extension.",
    )
    parser.add_argument(
        "--audio-bitrate",
        help="Optional audio bitrate override such as 192k. Defaults by codec when relevant.",
    )
    parser.add_argument(
        "--mix-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ffmpeg amix normalization to reduce clipping risk when summing both tracks. Default: enabled.",
    )
    parser.add_argument(
        "--truncate-silence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply post-mix silence truncation. Disable to inspect the merged stereo mix without truncation.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable to use. Default: ffmpeg.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser.parse_args()


def bounded_pan(value):
    amount = float(value)
    if not 0.0 <= amount <= 1.0:
        raise argparse.ArgumentTypeError("pan amount must be between 0.0 and 1.0")
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


def require_binary(binary_name):
    resolved = shutil.which(binary_name)
    if not resolved:
        raise SystemExit(f"Required executable not found on PATH: {binary_name}")
    return resolved


def choose_audio_codec(output_path, requested_codec):
    if requested_codec:
        return requested_codec
    return OUTPUT_CODEC_BY_SUFFIX.get(output_path.suffix.lower(), "pcm_s16le")


def choose_audio_bitrate(codec, requested_bitrate):
    if requested_bitrate:
        return requested_bitrate
    return DEFAULT_BITRATE_BY_CODEC.get(codec)


def probe_channel_count(input_path):
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return None

    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    output = result.stdout.strip()
    if not output:
        return None

    try:
        return int(output)
    except ValueError:
        return None


def warn_on_non_mono_inputs(left_input, right_input):
    warned = False
    for label, input_path in (("left", left_input), ("right", right_input)):
        channel_count = probe_channel_count(input_path)
        if channel_count and channel_count > 1:
            print(
                (
                    f"Warning: {label} input has {channel_count} channels. This script's panning path is "
                    "tuned for mono-like sources; stereo inputs are balance-shifted rather than true-panned."
                ),
                file=sys.stderr,
            )
            warned = True
    if warned:
        print(
            "Warning: Downmix stereo sources to mono before running if you need predictable voice placement.",
            file=sys.stderr,
        )


def build_filtergraph(args):
    normalize_flag = 1 if args.mix_normalize else 0
    left_balance = -abs(args.left_pan)
    right_balance = abs(args.right_pan)

    mix_chain = (
        "[left][right]amix="
        f"inputs=2:normalize={normalize_flag}:dropout_transition={args.dropout_transition:.3f}"
    )
    if args.truncate_silence:
        mix_chain += (
            ",silenceremove="
            f"start_periods=1:start_duration={args.silence_duration:.3f}:"
            f"start_threshold={args.silence_threshold}:start_silence={args.retain_silence:.3f}:"
            f"stop_periods=-1:stop_duration={args.silence_duration:.3f}:"
            f"stop_threshold={args.silence_threshold}:stop_silence={args.retain_silence:.3f}:"
            "detection=rms"
        )
    mix_chain += "[outa]"

    return ";".join(
        [
            f"[0:a]aformat=channel_layouts=stereo,stereotools=balance_in={left_balance:.3f}[left]",
            f"[1:a]aformat=channel_layouts=stereo,stereotools=balance_in={right_balance:.3f}[right]",
            mix_chain,
        ]
    )


def run_command(command):
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    args = parse_args()

    ffmpeg_bin = require_binary(args.ffmpeg_bin)
    left_input = Path(args.left_input).expanduser().resolve()
    right_input = Path(args.right_input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    for path in (left_input, right_input):
        if not path.is_file():
            raise SystemExit(f"Input file does not exist: {path}")

    if left_input == output_path or right_input == output_path:
        raise SystemExit("Output path must be different from both input files")

    warn_on_non_mono_inputs(left_input, right_input)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists. Use --overwrite to replace it: {output_path}")

    audio_codec = choose_audio_codec(output_path, args.audio_codec)
    audio_bitrate = choose_audio_bitrate(audio_codec, args.audio_bitrate)

    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y" if args.overwrite else "-n",
        "-i",
        str(left_input),
        "-i",
        str(right_input),
        "-filter_complex",
        build_filtergraph(args),
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
    print(f"Wrote merged output: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)