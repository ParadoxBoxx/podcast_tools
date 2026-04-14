#!/usr/bin/env python3

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
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
        help="Silence threshold for detection. Default: -35dB.",
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
        help="Total silence to retain per detected region, split evenly around the cut point. Default: 0.50.",
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


def build_mix_filtergraph(args):
    normalize_flag = 1 if args.mix_normalize else 0
    left_balance = -abs(args.left_pan)
    right_balance = abs(args.right_pan)

    mix_chain = (
        "[left][right]amix="
        f"inputs=2:normalize={normalize_flag}:dropout_transition={args.dropout_transition:.3f}"
        "[outa]"
    )

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


def format_duration(seconds):
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}:{secs:05.2f}"


def probe_duration(input_path):
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return None
    command = [
        ffprobe_bin, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def detect_silence_regions(ffmpeg_bin, audio_path, threshold, duration):
    """Detect silence regions using ffmpeg silencedetect filter."""
    command = [
        ffmpeg_bin, "-hide_banner", "-i", str(audio_path),
        "-af", f"silencedetect=noise={threshold}:duration={duration:.3f}:mono=false",
        "-f", "null", "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    regions = []
    start = None
    for line in result.stderr.splitlines():
        m_start = re.search(r"silence_start:\s*([\d.]+)", line)
        m_end = re.search(r"silence_end:\s*([\d.]+)", line)
        if m_start:
            start = float(m_start.group(1))
        elif m_end and start is not None:
            end = float(m_end.group(1))
            regions.append((start, end))
            start = None
    return regions


def compute_keep_intervals(silence_regions, retain_seconds, total_duration):
    """Compute time intervals to keep, removing from the CENTER of each silence region.

    This matches Audacity's Truncate Silence behavior: half the retain time is
    preserved on each side of the cut, keeping word tails and onsets intact.
    """
    half = retain_seconds / 2.0
    remove_ranges = []
    for s_start, s_end in silence_regions:
        if s_end - s_start <= retain_seconds:
            continue
        cut_start = s_start + half
        cut_end = s_end - half
        if cut_end > cut_start:
            remove_ranges.append((cut_start, cut_end))

    if not remove_ranges:
        return None

    keep = []
    pos = 0.0
    for cut_start, cut_end in remove_ranges:
        if cut_start > pos:
            keep.append((pos, cut_start))
        pos = cut_end
    if pos < total_duration:
        keep.append((pos, total_duration))
    return [(s, e) for s, e in keep if e - s > 0.001]


def write_trimmed_audio(ffmpeg_bin, input_path, output_path, keep_intervals,
                        audio_codec, audio_bitrate, overwrite):
    """Write output audio keeping only the specified intervals, using atrim+concat."""
    with tempfile.TemporaryDirectory(prefix="silence_trim_") as temp_dir:
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
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-y" if overwrite else "-n",
            "-i", str(input_path),
            "-filter_complex_script", str(script_path),
            "-map", "[outa]", "-vn", "-c:a", audio_codec,
        ]
        if audio_bitrate:
            command.extend(["-b:a", audio_bitrate])
        if audio_codec == "aac" and output_path.suffix.lower() in {".m4a", ".mp4"}:
            command.extend(["-movflags", "+faststart"])
        command.append(str(output_path))
        run_command(command)


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

    if not args.truncate_silence:
        print(f"Mixing {left_input.name} + {right_input.name} (no silence truncation)")
        command = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-y" if args.overwrite else "-n",
            "-i", str(left_input), "-i", str(right_input),
            "-filter_complex", build_mix_filtergraph(args),
            "-map", "[outa]", "-vn", "-c:a", audio_codec,
        ]
        if audio_bitrate:
            command.extend(["-b:a", audio_bitrate])
        if audio_codec == "aac" and output_path.suffix.lower() in {".m4a", ".mp4"}:
            command.extend(["-movflags", "+faststart"])
        command.append(str(output_path))
        run_command(command)
        print(f"Done → {output_path}")
        return

    # Two-pass approach matching Audacity Truncate Silence:
    #   1. Pan + mix → temporary WAV
    #   2. Detect silence → trim from middle of each region → encode output
    with tempfile.TemporaryDirectory(prefix="pan_mix_trunc_") as temp_dir:
        temp_path = Path(temp_dir) / "mixed.wav"

        print(f"Mixing {left_input.name} + {right_input.name}")
        mix_cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(left_input), "-i", str(right_input),
            "-filter_complex", build_mix_filtergraph(args),
            "-map", "[outa]", "-vn", "-c:a", "pcm_s16le",
            str(temp_path),
        ]
        run_command(mix_cmd)

        total_duration = probe_duration(temp_path)
        if total_duration:
            print(f"Mixed duration: {format_duration(total_duration)}")

        print(f"Detecting silence (threshold {args.silence_threshold}, min duration {args.silence_duration}s)")
        regions = detect_silence_regions(
            ffmpeg_bin, temp_path, args.silence_threshold, args.silence_duration
        )
        print(f"Found {len(regions)} silence region(s)")

        keep_intervals = None
        if regions and total_duration:
            keep_intervals = compute_keep_intervals(regions, args.retain_silence, total_duration)

        if keep_intervals:
            kept = sum(e - s for s, e in keep_intervals)
            removed = total_duration - kept
            print(f"Trimming {format_duration(removed)} of silence → {format_duration(kept)} final")
            print("Encoding output")
            write_trimmed_audio(
                ffmpeg_bin, temp_path, output_path, keep_intervals,
                audio_codec, audio_bitrate, args.overwrite,
            )
        else:
            print("No silence long enough to truncate; encoding as-is")
            encode_cmd = [
                ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",
                "-y" if args.overwrite else "-n",
                "-i", str(temp_path),
                "-vn", "-c:a", audio_codec,
            ]
            if audio_bitrate:
                encode_cmd.extend(["-b:a", audio_bitrate])
            if audio_codec == "aac" and output_path.suffix.lower() in {".m4a", ".mp4"}:
                encode_cmd.extend(["-movflags", "+faststart"])
            encode_cmd.append(str(output_path))
            run_command(encode_cmd)

    print(f"Done → {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)