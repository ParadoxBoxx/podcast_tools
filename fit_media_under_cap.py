#!/usr/bin/env python3

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_MAX_SIZE_BYTES = 10 * 1024 ** 3
DEFAULT_SAFETY_BYTES = max(16 * 1024 ** 2, DEFAULT_MAX_SIZE_BYTES // 100)
MIN_VIDEO_BITRATE = 150_000
MIN_AUDIO_BITRATE = 32_000


VIDEO_CONTAINER_PLANS = {
    ".mp4": {"video_codec": "libx264", "audio_codec": "aac", "extra_args": ["-movflags", "+faststart"]},
    ".m4v": {"video_codec": "libx264", "audio_codec": "aac", "extra_args": ["-movflags", "+faststart"]},
    ".mov": {"video_codec": "libx264", "audio_codec": "aac", "extra_args": []},
    ".mkv": {"video_codec": "libx264", "audio_codec": "libopus", "extra_args": []},
    ".webm": {"video_codec": "libvpx-vp9", "audio_codec": "libopus", "extra_args": []},
    ".flv": {"video_codec": "libx264", "audio_codec": "aac", "extra_args": []},
    ".ts": {"video_codec": "libx264", "audio_codec": "aac", "extra_args": []},
    ".m2ts": {"video_codec": "libx264", "audio_codec": "aac", "extra_args": []},
}

VIDEO_FALLBACK_PLAN = {
    "extension": ".mkv",
    "video_codec": "libx264",
    "audio_codec": "libopus",
    "extra_args": [],
    "reason": "original container is not a safe bitrate-targeted output for this media; using .mkv instead",
}

AUDIO_CONTAINER_PLANS = {
    ".mp3": {"audio_codec": "libmp3lame", "extra_args": []},
    ".m4a": {"audio_codec": "aac", "extra_args": ["-movflags", "+faststart"]},
    ".aac": {"audio_codec": "aac", "extra_args": []},
    ".ogg": {"audio_codec": "libopus", "extra_args": []},
    ".opus": {"audio_codec": "libopus", "extra_args": []},
    ".webm": {"audio_codec": "libopus", "extra_args": []},
    ".mkv": {"audio_codec": "libopus", "extra_args": []},
}

AUDIO_FALLBACK_PLAN = {
    "extension": ".m4a",
    "audio_codec": "aac",
    "extra_args": ["-movflags", "+faststart"],
    "reason": "original audio container is not a reliable bitrate-targeted output; using .m4a instead",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Shrink a media file to the highest practical quality that stays under a size cap. "
            "The script preserves the original file type when the detected media and container "
            "support a safe ffmpeg target; otherwise it falls back to a compatible container. "
            "By default, the original file is kept."
        ),
        epilog=(
            "Default behavior: keep the original file. If re-encoding is needed and --output is omitted, "
            "the script writes a sibling file with an '_under_cap' suffix. Use --overwrite only when you "
            "want replacement behavior."
        ),
    )
    parser.add_argument("input", help="Path to the input media file.")
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Optional output path. If omitted, the default is to keep the source file and write a sibling "
            "file with an '_under_cap' suffix when re-encoding is needed."
        ),
    )
    parser.add_argument(
        "--max-size",
        default="10GiB",
        help="Maximum output size. Accepts raw bytes or units like 10G, 9500MB, 9.5GiB, or 10GiB. Default: 10GiB.",
    )
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--keep-original",
        dest="overwrite",
        action="store_false",
        help=(
            "Keep the source file. Without --output, a re-encode is written to a separate '_under_cap' "
            "file. With --output, the requested destination is used but existing files are not replaced. "
            "This is the default behavior."
        ),
    )
    overwrite_group.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help=(
            "Allow replacement behavior. Without --output, replace the original file in place after a "
            "successful re-encode. With --output, allow replacing an existing destination file."
        ),
    )
    parser.set_defaults(overwrite=False)
    return parser.parse_args()


def parse_size_bytes(value):
    text = str(value).strip()
    if not text:
        raise ValueError("size value is empty")

    units = {
        "b": 1,
        "k": 1024,
        "m": 1024 ** 2,
        "g": 1024 ** 3,
        "t": 1024 ** 4,
        "kb": 1000,
        "mb": 1000 ** 2,
        "gb": 1000 ** 3,
        "tb": 1000 ** 4,
        "kib": 1024,
        "mib": 1024 ** 2,
        "gib": 1024 ** 3,
        "tib": 1024 ** 4,
    }

    number = []
    suffix = []
    for char in text:
        if char.isdigit() or char == ".":
            number.append(char)
        elif not char.isspace():
            suffix.append(char)

    if not number:
        raise ValueError(f"invalid size: {value}")

    numeric_value = float("".join(number))
    unit = "".join(suffix).lower() or "b"
    if unit not in units:
        raise ValueError(f"unsupported size unit: {unit}")

    return int(numeric_value * units[unit])


def format_bytes(num_bytes):
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def run_command(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "(no stderr)"
        raise RuntimeError(f"command failed: {' '.join(command)}\n{stderr}")
    return result


def probe_size_bytes(format_info, input_path):
    raw_size = format_info.get("size")
    try:
        size = int(raw_size or 0)
    except (TypeError, ValueError):
        size = 0
    return size or input_path.stat().st_size


def probe_media(input_path):
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(input_path),
        ]
    )
    payload = json.loads(result.stdout)
    format_info = payload.get("format", {})
    streams = payload.get("streams", [])

    duration = float(format_info.get("duration") or 0)
    size = probe_size_bytes(format_info, input_path)
    has_video = any(stream.get("codec_type") == "video" for stream in streams)
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)

    if duration <= 0:
        raise RuntimeError("ffprobe did not return a usable duration; cannot calculate a size-targeted bitrate")

    if not has_video and not has_audio:
        raise RuntimeError("input does not contain audio or video streams")

    return {
        "format": format_info,
        "streams": streams,
        "duration": duration,
        "size": size,
        "has_video": has_video,
        "has_audio": has_audio,
    }


def pick_output_plan(extension, has_video):
    extension = extension.lower()
    if has_video:
        plan = VIDEO_CONTAINER_PLANS.get(extension)
        if plan:
            chosen = dict(plan)
            chosen["extension"] = extension
            chosen["preserved_container"] = True
            chosen["fallback_reason"] = ""
            return chosen
        chosen = dict(VIDEO_FALLBACK_PLAN)
        chosen["preserved_container"] = False
        chosen["fallback_reason"] = chosen.pop("reason")
        return chosen

    plan = AUDIO_CONTAINER_PLANS.get(extension)
    if plan:
        chosen = dict(plan)
        chosen["extension"] = extension
        chosen["preserved_container"] = True
        chosen["fallback_reason"] = ""
        return chosen
    chosen = dict(AUDIO_FALLBACK_PLAN)
    chosen["preserved_container"] = False
    chosen["fallback_reason"] = chosen.pop("reason")
    return chosen


def choose_audio_bitrate(stream, audio_only):
    channels = int(stream.get("channels") or 2)
    bitrate = int(stream.get("bit_rate") or 0)
    if channels <= 1:
        target = 64_000
    elif channels <= 2:
        target = 128_000
    elif channels <= 6:
        target = 192_000
    else:
        target = 256_000

    if audio_only:
        if channels <= 2:
            target = min(max(target, 160_000), 320_000)
        else:
            target = min(max(target, 256_000), 512_000)

    if bitrate > 0:
        return min(max(bitrate, MIN_AUDIO_BITRATE), target) if audio_only else min(bitrate, target)
    return target


def distribute_audio_bitrates(streams, total_audio_budget):
    if not streams:
        return []

    minimum_total = MIN_AUDIO_BITRATE * len(streams)
    if total_audio_budget <= minimum_total:
        return [MIN_AUDIO_BITRATE for _ in streams]

    desired = [max(MIN_AUDIO_BITRATE, bitrate) for bitrate in streams]
    desired_total = sum(desired)
    if desired_total <= total_audio_budget:
        return desired

    scale = total_audio_budget / desired_total
    reduced = [max(MIN_AUDIO_BITRATE, int(bitrate * scale)) for bitrate in desired]

    while sum(reduced) > total_audio_budget:
        largest_index = max(range(len(reduced)), key=lambda index: reduced[index])
        if reduced[largest_index] <= MIN_AUDIO_BITRATE:
            break
        reduced[largest_index] -= 1_000

    return reduced


def calculate_bitrates(probe_data, max_size_bytes):
    duration = probe_data["duration"]
    has_video = probe_data["has_video"]
    audio_streams = [stream for stream in probe_data["streams"] if stream.get("codec_type") == "audio"]

    safety_bytes = max(16 * 1024 ** 2, max_size_bytes // 100)
    usable_bytes = max(max_size_bytes - safety_bytes, max_size_bytes // 2)
    target_total_bitrate = int((usable_bytes * 8) / duration)
    mux_overhead_bitrate = max(32_000, target_total_bitrate // 100)

    desired_audio = [choose_audio_bitrate(stream, audio_only=not has_video) for stream in audio_streams]

    if has_video:
        max_audio_budget = max(target_total_bitrate - mux_overhead_bitrate - MIN_VIDEO_BITRATE, MIN_AUDIO_BITRATE * len(audio_streams))
    else:
        max_audio_budget = max(target_total_bitrate - mux_overhead_bitrate, MIN_AUDIO_BITRATE * len(audio_streams))

    audio_bitrates = distribute_audio_bitrates(desired_audio, max_audio_budget) if audio_streams else []
    audio_total_bitrate = sum(audio_bitrates)

    if has_video:
        video_bitrate = max(target_total_bitrate - mux_overhead_bitrate - audio_total_bitrate, MIN_VIDEO_BITRATE)
    else:
        video_bitrate = 0

    return {
        "target_total_bitrate": target_total_bitrate,
        "mux_overhead_bitrate": mux_overhead_bitrate,
        "audio_bitrates": audio_bitrates,
        "audio_total_bitrate": audio_total_bitrate,
        "video_bitrate": video_bitrate,
    }


def make_temporary_output_path(final_output_path):
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{final_output_path.stem}_under_cap_",
        suffix=final_output_path.suffix,
        dir=str(final_output_path.parent),
    )
    os.close(file_descriptor)
    Path(temp_name).unlink(missing_ok=True)
    return Path(temp_name)


def resolve_output_paths(input_path, requested_output, output_extension, overwrite):
    replace_original = False

    if requested_output:
        final_output_path = Path(requested_output).expanduser().resolve()
        if final_output_path == input_path:
            if not overwrite:
                raise RuntimeError("refusing to replace the input file without --overwrite")
            replace_original = True
    elif overwrite:
        replace_original = True
        final_output_path = input_path
    else:
        final_output_path = input_path.with_name(f"{input_path.stem}_under_cap{output_extension}")

    if replace_original and final_output_path.suffix.lower() != output_extension.lower():
        final_output_path = input_path.with_suffix(output_extension)
    elif final_output_path.suffix.lower() != output_extension.lower():
        final_output_path = final_output_path.with_suffix(output_extension)

    if final_output_path.exists() and final_output_path != input_path and not overwrite:
        raise RuntimeError(f"output file already exists: {final_output_path}. Use --overwrite to replace it.")

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    working_output_path = make_temporary_output_path(final_output_path) if replace_original else final_output_path
    return working_output_path, final_output_path, replace_original


def finalize_output_path(working_output_path, final_output_path, input_path, replace_original):
    if not replace_original:
        return final_output_path

    os.replace(working_output_path, final_output_path)
    if final_output_path != input_path and input_path.exists():
        input_path.unlink()
    return final_output_path


def cleanup_temporary_output(path):
    if path.exists():
        path.unlink()


def build_audio_args(plan, audio_bitrates):
    if not audio_bitrates:
        return ["-an"]

    args = ["-c:a", plan["audio_codec"]]
    for index, bitrate in enumerate(audio_bitrates):
        args.extend([f"-b:a:{index}", str(bitrate)])
    return args


def encode_once(input_path, output_path, plan, bitrates, attempt_label):
    with tempfile.TemporaryDirectory(prefix="fit_media_cap_") as temp_dir:
        passlog = Path(temp_dir) / "ffmpeg_pass"
        common_args = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-map",
            "0:v:0?",
            "-map",
            "0:a?",
            "-dn",
            "-sn",
        ]

        if bitrates["video_bitrate"] > 0:
            first_pass = common_args + [
                "-c:v",
                plan["video_codec"],
                "-b:v",
                str(bitrates["video_bitrate"]),
                "-pass",
                "1",
                "-passlogfile",
                str(passlog),
                "-an",
                "-f",
                "null",
                os.devnull,
            ]

            second_pass = common_args + [
                "-c:v",
                plan["video_codec"],
                "-b:v",
                str(bitrates["video_bitrate"]),
                "-pass",
                "2",
                "-passlogfile",
                str(passlog),
            ]
            second_pass.extend(build_audio_args(plan, bitrates["audio_bitrates"]))
            second_pass.extend(plan["extra_args"])
            second_pass.append(str(output_path))

            print(f"[{attempt_label}] First pass at video bitrate {bitrates['video_bitrate'] / 1000:.0f} kbps")
            run_command(first_pass)
            print(f"[{attempt_label}] Second pass writing {output_path}")
            run_command(second_pass)
            return

        audio_only_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-map",
            "0:a?",
            "-vn",
            "-dn",
            "-sn",
            "-c:a",
            plan["audio_codec"],
        ]
        for index, bitrate in enumerate(bitrates["audio_bitrates"]):
            audio_only_cmd.extend([f"-b:a:{index}", str(bitrate)])
        audio_only_cmd.extend(plan["extra_args"])
        audio_only_cmd.append(str(output_path))

        print(f"[{attempt_label}] Audio-only encode writing {output_path}")
        run_command(audio_only_cmd)


def reencode_to_cap(input_path, output_path, plan, max_size_bytes, bitrates):
    attempts = [bitrates]
    encode_once(input_path, output_path, plan, attempts[0], "attempt 1")
    actual_size = output_path.stat().st_size
    if actual_size <= max_size_bytes:
        return actual_size, 1

    if attempts[0]["video_bitrate"] <= 0:
        return actual_size, 1

    overshoot_ratio = max_size_bytes / actual_size
    adjusted_video_bitrate = max(int(attempts[0]["video_bitrate"] * overshoot_ratio * 0.98), MIN_VIDEO_BITRATE)
    adjusted = dict(attempts[0])
    adjusted["video_bitrate"] = adjusted_video_bitrate

    print(
        f"[attempt 2] First output was {format_bytes(actual_size)}; lowering video bitrate to "
        f"{adjusted_video_bitrate / 1000:.0f} kbps and retrying"
    )
    encode_once(input_path, output_path, plan, adjusted, "attempt 2")
    return output_path.stat().st_size, 2


def copy_without_reencoding(input_path, output_path, overwrite):
    if output_path.exists():
        if not overwrite:
            raise RuntimeError(f"output file already exists: {output_path}. Use --overwrite to replace it.")
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists() or not input_path.is_file():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    max_size_bytes = parse_size_bytes(args.max_size)
    if max_size_bytes <= 0:
        print("--max-size must be greater than zero", file=sys.stderr)
        return 1

    try:
        probe_data = probe_media(input_path)
        source_extension = input_path.suffix.lower()
        requested_extension = Path(args.output).suffix.lower() if args.output and Path(args.output).suffix else ""
        preferred_extension = requested_extension or source_extension
        plan = pick_output_plan(preferred_extension, probe_data["has_video"])

        print(f"Input: {input_path}")
        print(f"Current size: {format_bytes(probe_data['size'])}")
        print(f"Target max size: {format_bytes(max_size_bytes)}")
        print(f"Duration: {probe_data['duration']:.2f} seconds")
        if args.output:
            requested_output_path = Path(args.output).expanduser().resolve()
            print(f"Output mode: explicit output path requested at {requested_output_path}")
            if requested_output_path == input_path:
                replacement_note = "enabled" if args.overwrite else "disabled"
                print(f"Source replacement for that path is {replacement_note}.")
        elif args.overwrite:
            print("Output mode: replace the original file in place when re-encoding is needed (--overwrite).")
        else:
            print(
                "Output mode: keep the original file and write a separate '_under_cap' output only if "
                "re-encoding is needed (default)."
            )

        if requested_extension and requested_extension != plan["extension"]:
            print(
                f"Requested output extension {requested_extension} is not safe for this encode target; "
                f"using {plan['extension']} instead"
            )
        if not plan["preserved_container"]:
            print(f"Container fallback: {plan['fallback_reason']}")

        if probe_data["size"] <= max_size_bytes:
            requested_output_path = Path(args.output).expanduser().resolve() if args.output else None
            if requested_output_path and requested_output_path != input_path:
                output_path, _, _ = resolve_output_paths(input_path, args.output, source_extension, args.overwrite)
                copy_without_reencoding(input_path, output_path, args.overwrite)
                print("Source is already under the requested limit; copied unchanged to the requested output path.")
                print("Original file kept.")
                print(f"Output: {output_path}")
            else:
                print("Source is already under the requested limit; leaving the original file unchanged.")
                print(f"Output: {input_path}")
            return 0

        working_output_path, final_output_path, replace_original = resolve_output_paths(
            input_path,
            args.output,
            plan["extension"],
            args.overwrite,
        )
        bitrates = calculate_bitrates(probe_data, max_size_bytes)

        print(f"Reserved audio bitrate: {bitrates['audio_total_bitrate'] / 1000:.0f} kbps")
        if bitrates["video_bitrate"] > 0:
            print(f"Target video bitrate: {bitrates['video_bitrate'] / 1000:.0f} kbps")
        print(f"Output container: {final_output_path.suffix.lower()}")
        if replace_original:
            if final_output_path == input_path:
                print("Final output path: original file will be replaced after a successful encode.")
            else:
                print(f"Final output path: source file will be replaced with {final_output_path}")
        else:
            print(f"Final output path: original file kept; writing {final_output_path}")

        try:
            actual_size, attempts = reencode_to_cap(input_path, working_output_path, plan, max_size_bytes, bitrates)
            output_path = finalize_output_path(working_output_path, final_output_path, input_path, replace_original)
        except Exception:
            if replace_original:
                cleanup_temporary_output(working_output_path)
            raise

        print(f"Output: {output_path}")
        print(f"Output size: {format_bytes(actual_size)}")

        if actual_size > max_size_bytes:
            retry_clause = " after retry" if attempts > 1 else ""
            print(
                f"Warning: ffmpeg completed, but the output is still above the requested size cap{retry_clause}.",
                file=sys.stderr,
            )
            return 2

        print(f"Completed in {attempts} pass cycle(s).")
        return 0
    except ValueError as exc:
        print(f"Invalid argument: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())