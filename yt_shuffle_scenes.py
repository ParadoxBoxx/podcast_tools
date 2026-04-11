#!/usr/bin/env python3

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from yt_to_vegas import DEFAULT_DOWNLOAD_DIR, DEFAULT_OUTPUT_DIR, normalize_url, sanitize_filename


DEFAULT_SCENE_THRESHOLD = 0.20
DEFAULT_MIN_SCENE_DURATION = 0.75
DEFAULT_WORK_DIR = DEFAULT_DOWNLOAD_DIR / "scene_shuffle"
DEFAULT_TEXT_FILTER_SAMPLE_COUNT = 3
DEFAULT_TEXT_FILTER_FRAME_WIDTH = 160
DEFAULT_TEXT_FILTER_FRAME_HEIGHT = 90
DEFAULT_TEXT_FILTER_MAX_MOTION = 4.0
DEFAULT_TEXT_FILTER_MIN_DOMINANT_COVERAGE = 0.68
DEFAULT_TEXT_FILTER_MIN_EDGE_DENSITY = 0.01
DEFAULT_TEXT_FILTER_MAX_EDGE_DENSITY = 0.16
DEFAULT_TEXT_FILTER_EDGE_THRESHOLD = 18
DEFAULT_TEXT_FILTER_MAX_EDGE_ROW_COVERAGE = 0.60
DEFAULT_END_CARD_WINDOW_SECONDS = 18.0
DEFAULT_END_CARD_MIN_START_RATIO = 0.82
DEFAULT_END_CARD_MIN_DURATION = 1.0
DEFAULT_END_CARD_MAX_DURATION = 20.0
DEFAULT_END_CARD_MAX_MOTION = 2.2
DEFAULT_END_CARD_MIN_DOMINANT_COVERAGE = 0.42
DEFAULT_END_CARD_MIN_EDGE_DENSITY = 0.008
DEFAULT_END_CARD_MAX_EDGE_DENSITY = 0.26
DEFAULT_END_CARD_MAX_EDGE_ROW_COVERAGE = 0.82


@dataclass(frozen=True)
class TextOnlyFilterSettings:
    enabled: bool
    sample_count: int
    max_motion: float
    min_dominant_coverage: float
    min_edge_density: float
    max_edge_density: float
    max_edge_row_coverage: float
    frame_width: int = DEFAULT_TEXT_FILTER_FRAME_WIDTH
    frame_height: int = DEFAULT_TEXT_FILTER_FRAME_HEIGHT
    edge_threshold: int = DEFAULT_TEXT_FILTER_EDGE_THRESHOLD


@dataclass(frozen=True)
class TextOnlyFilterSuppression:
    fired: bool
    reason: str = ""
    suppressed_drop_count: int = 0
    message: str = ""


@dataclass(frozen=True)
class EndCardFilterSettings:
    enabled: bool
    window_seconds: float
    min_start_ratio: float
    min_duration: float
    max_duration: float
    max_motion: float
    min_dominant_coverage: float
    min_edge_density: float
    max_edge_density: float
    max_edge_row_coverage: float


@dataclass(frozen=True)
class EndCardFilterSuppression:
    fired: bool
    reason: str = ""
    suppressed_drop_count: int = 0
    message: str = ""


VISUAL_ANALYSIS_METRIC_KEYS = (
    "motion_score",
    "average_dominant_coverage",
    "average_edge_density",
    "average_edge_row_coverage",
)


def positive_float(value: str) -> float:
    amount = float(value)
    if amount <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return amount


def positive_int(value: str) -> int:
    amount = int(value)
    if amount <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return amount


def unit_interval_float(value: str) -> float:
    amount = float(value)
    if amount < 0 or amount > 1:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return amount


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download or reuse a source video, transcode it into the repo's Vegas-ready silent MP4 profile, "
            "detect scene boundaries, cut per-scene clips, shuffle them, and render a shuffled master."
        ),
        epilog=(
            "The source may be a YouTube URL or an existing local video file for offline testing. Scene clips and "
            "metadata are kept under the work directory when --keep-intermediate is enabled."
        ),
    )
    parser.add_argument(
        "source",
        help="YouTube video URL or a local video file path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for final shuffled outputs and manifests. Default: ./vegas_output.",
    )
    parser.add_argument(
        "--work-dir",
        default=str(DEFAULT_WORK_DIR),
        help="Directory for raw downloads, formatted intermediates, and scene clips. Default: ./downloads/scene_shuffle.",
    )
    parser.add_argument(
        "--scene-threshold",
        type=positive_float,
        default=DEFAULT_SCENE_THRESHOLD,
        help="Scene detection threshold passed to ffmpeg's scene score comparison. Lower values detect more cuts. Default: 0.20.",
    )
    parser.add_argument(
        "--min-scene-duration",
        type=positive_float,
        default=DEFAULT_MIN_SCENE_DURATION,
        help="Minimum kept scene duration in seconds. Shorter detected splits are merged forward/backward. Default: 0.75.",
    )
    parser.add_argument(
        "--filter-text-only-scenes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Sample detected scenes before clip cutting and drop clear static text-only/title-card segments. "
            "Ambiguous scenes are kept. Default: enabled."
        ),
    )
    parser.add_argument(
        "--text-filter-sample-count",
        type=positive_int,
        default=DEFAULT_TEXT_FILTER_SAMPLE_COUNT,
        help="Number of evenly spaced frames to sample per scene for the text-only filter. Default: 3.",
    )
    parser.add_argument(
        "--text-filter-max-motion",
        type=positive_float,
        default=DEFAULT_TEXT_FILTER_MAX_MOTION,
        help=(
            "Maximum mean grayscale frame-to-frame delta for a scene to still count as static in the text-only "
            "filter. Lower values keep more scenes. Default: 4.0."
        ),
    )
    parser.add_argument(
        "--text-filter-min-dominant-coverage",
        type=unit_interval_float,
        default=DEFAULT_TEXT_FILTER_MIN_DOMINANT_COVERAGE,
        help=(
            "Minimum share of pixels covered by the two most common grayscale buckets for a scene to count as "
            "flat-background in the text-only filter. Default: 0.68."
        ),
    )
    parser.add_argument(
        "--text-filter-min-edge-density",
        type=unit_interval_float,
        default=DEFAULT_TEXT_FILTER_MIN_EDGE_DENSITY,
        help=(
            "Minimum average edge density required before a static flat scene is treated as text-like instead of "
            "blank/ambiguous. Default: 0.01."
        ),
    )
    parser.add_argument(
        "--text-filter-max-edge-density",
        type=unit_interval_float,
        default=DEFAULT_TEXT_FILTER_MAX_EDGE_DENSITY,
        help=(
            "Maximum average edge density allowed before a scene is treated as real footage instead of a text card. "
            "Default: 0.16."
        ),
    )
    parser.add_argument(
        "--text-filter-max-edge-row-coverage",
        type=unit_interval_float,
        default=DEFAULT_TEXT_FILTER_MAX_EDGE_ROW_COVERAGE,
        help=(
            "Maximum share of sampled rows allowed to contain strong edges before a static scene is treated as "
            "layout-heavy footage instead of a text card. Default: 0.60."
        ),
    )
    parser.add_argument(
        "--remove-end-cards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Scan the trailing portion of the timeline for static branded outro/end-card scenes and remove only the "
            "final matching suffix before clip extraction. If every remaining segment matches, the script warns and "
            "preserves them all instead of producing an empty output. Default: enabled."
        ),
    )
    parser.add_argument(
        "--end-card-window-seconds",
        type=positive_float,
        default=DEFAULT_END_CARD_WINDOW_SECONDS,
        help="How much of the trailing timeline to consider for end-card removal. Default: 18.0.",
    )
    parser.add_argument(
        "--end-card-min-start-ratio",
        type=unit_interval_float,
        default=DEFAULT_END_CARD_MIN_START_RATIO,
        help=(
            "Earliest normalized start position that still qualifies as late-timeline for end-card removal. "
            "Default: 0.82."
        ),
    )
    parser.add_argument(
        "--end-card-min-duration",
        type=positive_float,
        default=DEFAULT_END_CARD_MIN_DURATION,
        help="Minimum scene duration eligible for end-card removal. Default: 1.0.",
    )
    parser.add_argument(
        "--end-card-max-duration",
        type=positive_float,
        default=DEFAULT_END_CARD_MAX_DURATION,
        help="Maximum scene duration eligible for end-card removal. Default: 20.0.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Seed for deterministic clip shuffling. Default: 0.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep raw downloads, formatted video, scene clips, and concat manifests in the work directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace any existing job directory and final output for the same source name.",
    )
    parser.add_argument(
        "--skip-yt-dlp-update",
        action="store_true",
        help="Do not run the best-effort yt-dlp self-update before downloading from YouTube.",
    )
    args = parser.parse_args()
    if args.text_filter_sample_count < 2:
        parser.error("--text-filter-sample-count must be at least 2 so the filter can compare sampled frames")
    return args


def run_command(
    command: list[str],
    *,
    capture_output: bool = False,
    env: dict[str, str] | None = None,
    text: bool = True,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        check=True,
        text=text,
        capture_output=capture_output,
        env=env,
    )


def build_text_only_filter_settings(args: argparse.Namespace) -> TextOnlyFilterSettings:
    if args.text_filter_min_edge_density > args.text_filter_max_edge_density:
        raise RuntimeError("--text-filter-min-edge-density cannot be greater than --text-filter-max-edge-density")
    return TextOnlyFilterSettings(
        enabled=args.filter_text_only_scenes,
        sample_count=args.text_filter_sample_count,
        max_motion=args.text_filter_max_motion,
        min_dominant_coverage=args.text_filter_min_dominant_coverage,
        min_edge_density=args.text_filter_min_edge_density,
        max_edge_density=args.text_filter_max_edge_density,
        max_edge_row_coverage=args.text_filter_max_edge_row_coverage,
    )


def build_end_card_filter_settings(args: argparse.Namespace) -> EndCardFilterSettings:
    if args.end_card_min_duration > args.end_card_max_duration:
        raise RuntimeError("--end-card-min-duration cannot be greater than --end-card-max-duration")
    return EndCardFilterSettings(
        enabled=args.remove_end_cards,
        window_seconds=args.end_card_window_seconds,
        min_start_ratio=args.end_card_min_start_ratio,
        min_duration=args.end_card_min_duration,
        max_duration=args.end_card_max_duration,
        max_motion=DEFAULT_END_CARD_MAX_MOTION,
        min_dominant_coverage=DEFAULT_END_CARD_MIN_DOMINANT_COVERAGE,
        min_edge_density=DEFAULT_END_CARD_MIN_EDGE_DENSITY,
        max_edge_density=DEFAULT_END_CARD_MAX_EDGE_DENSITY,
        max_edge_row_coverage=DEFAULT_END_CARD_MAX_EDGE_ROW_COVERAGE,
    )


def build_end_card_visual_analysis_settings(
    sampling_settings: TextOnlyFilterSettings,
    end_card_settings: EndCardFilterSettings,
) -> TextOnlyFilterSettings:
    return TextOnlyFilterSettings(
        enabled=True,
        sample_count=sampling_settings.sample_count,
        max_motion=end_card_settings.max_motion,
        min_dominant_coverage=end_card_settings.min_dominant_coverage,
        min_edge_density=end_card_settings.min_edge_density,
        max_edge_density=end_card_settings.max_edge_density,
        max_edge_row_coverage=end_card_settings.max_edge_row_coverage,
        frame_width=sampling_settings.frame_width,
        frame_height=sampling_settings.frame_height,
        edge_threshold=sampling_settings.edge_threshold,
    )


def check_required_binaries() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"Required binary not found on PATH: {', '.join(missing)}")


def update_yt_dlp() -> None:
    print(">>> Syncing yt-dlp to latest nightly...")
    try:
        run_command([sys.executable, "-m", "pip", "install", "-U", "--pre", "yt-dlp[default]"])
    except subprocess.CalledProcessError:
        print(">>> Warning: yt-dlp update failed; continuing with the currently installed version.")


@lru_cache(maxsize=1)
def ffmpeg_supports_nvenc() -> bool:
    try:
        result = run_command(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True)
    except subprocess.CalledProcessError:
        return False
    return "h264_nvenc" in result.stdout


def infer_source_name(source: str, normalized_url: str | None = None) -> str:
    path = Path(source).expanduser()
    if path.exists() and path.is_file():
        return sanitize_filename(path.stem) or "source_video"
    candidate = normalized_url or source
    stripped = candidate.rstrip("/").split("/")[-1]
    stripped = stripped.split("?")[-1].replace("=", "_")
    return sanitize_filename(stripped) or "youtube_video"


def prepare_job_directories(job_dir: Path, overwrite: bool) -> None:
    if job_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Job directory already exists: {job_dir}. Use --overwrite or choose a different --work-dir."
            )
        shutil.rmtree(job_dir)
    for child in (job_dir / "raw", job_dir / "formatted", job_dir / "clips", job_dir / "manifests"):
        child.mkdir(parents=True, exist_ok=True)


def build_yt_dlp_env() -> dict[str, str]:
    current_dir = str(Path(__file__).resolve().parent)
    env = os.environ.copy()
    env["PATH"] = current_dir + os.pathsep + env.get("PATH", "")
    return env


def find_downloaded_video(raw_dir: Path) -> Path:
    candidates = sorted(
        [
            path
            for path in raw_dir.iterdir()
            if path.is_file() and not path.name.endswith((".part", ".ytdl", ".json", ".description", ".jpg", ".png"))
        ],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"yt-dlp did not produce a usable media file in {raw_dir}")
    return candidates[0]


def download_source_video(url: str, raw_dir: Path) -> Path:
    clean_url = normalize_url(url)
    env = build_yt_dlp_env()
    print(f"Downloading: {clean_url}")
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--cookies-from-browser",
        "firefox",
        "--impersonate",
        "Firefox-135",
        "--sleep-requests",
        "1",
        "--remote-components",
        "ejs:github",
        "--ffmpeg-location",
        str(Path(__file__).resolve().parent),
        "-f",
        "bestvideo/best",
        "-P",
        str(raw_dir),
        "--print",
        "after_move:filepath",
        clean_url,
    ]
    try:
        result = run_command(command, capture_output=True, env=env)
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if lines:
            downloaded_path = Path(lines[-1]).expanduser().resolve()
            if downloaded_path.exists():
                return downloaded_path
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() if exc.stderr else str(exc)
        raise RuntimeError(f"yt-dlp download failed: {message}") from exc
    return find_downloaded_video(raw_dir)


def copy_local_source(source_path: Path, raw_dir: Path) -> Path:
    destination = raw_dir / source_path.name
    shutil.copy2(source_path, destination)
    return destination


def encode_silent_h264(input_path: Path, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nvenc_available = ffmpeg_supports_nvenc()
    if nvenc_available:
        gpu_command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "cuda",
            "-i",
            str(input_path),
            "-c:v",
            "h264_nvenc",
            "-preset",
            "fast",
            "-cq",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output_path),
        ]
        try:
            run_command(gpu_command)
            return "h264_nvenc"
        except subprocess.CalledProcessError:
            pass

    cpu_command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_path),
    ]
    run_command(cpu_command)
    return "libx264"


def get_duration_seconds(video_path: Path) -> float:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
    )
    return float(result.stdout.strip())


def detect_scene_times(video_path: Path, threshold: float) -> list[float]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(video_path),
        "-filter_complex",
        f"[0:v]select='gt(scene,{threshold})',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    try:
        result = run_command(command, capture_output=True)
        stderr_text = result.stderr
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() if exc.stderr else str(exc)
        raise RuntimeError(f"ffmpeg scene detection failed: {message}") from exc

    times: list[float] = []
    marker = "pts_time:"
    for line in stderr_text.splitlines():
        if marker not in line:
            continue
        start = line.find(marker) + len(marker)
        tail = line[start:]
        token = tail.split()[0]
        try:
            value = float(token)
        except ValueError:
            continue
        if not times or abs(value - times[-1]) > 0.05:
            times.append(value)
    return times


def build_segments(scene_times: Iterable[float], total_duration: float, min_scene_duration: float) -> list[dict[str, float | int]]:
    if total_duration <= 0:
        raise RuntimeError("Input video duration is zero; cannot build scene segments")

    filtered_boundaries = [0.0]
    for time_point in sorted(scene_times):
        clamped = max(0.0, min(time_point, total_duration))
        if clamped - filtered_boundaries[-1] >= min_scene_duration:
            filtered_boundaries.append(clamped)

    segments: list[dict[str, float | int]] = []
    for index, start_time in enumerate(filtered_boundaries):
        end_time = filtered_boundaries[index + 1] if index + 1 < len(filtered_boundaries) else total_duration
        if end_time <= start_time:
            continue
        duration = end_time - start_time
        if duration < min_scene_duration and segments:
            segments[-1]["end"] = end_time
            segments[-1]["duration"] = round(float(segments[-1]["end"]) - float(segments[-1]["start"]), 6)
            continue
        segments.append(
            {
                "scene_index": len(segments),
                "start": round(start_time, 6),
                "end": round(end_time, 6),
                "duration": round(duration, 6),
            }
        )

    if not segments:
        segments.append(
            {
                "scene_index": 0,
                "start": 0.0,
                "end": round(total_duration, 6),
                "duration": round(total_duration, 6),
            }
        )
    return segments


def sample_segment_frames(
    video_path: Path,
    *,
    start_time: float,
    duration: float,
    settings: TextOnlyFilterSettings,
) -> list[bytes]:
    if duration <= 0:
        return []

    sampling_window = max(duration, 0.25)
    sample_rate = max(settings.sample_count / sampling_window, 1.0)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_time:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(video_path),
        "-vf",
        (
            f"fps={sample_rate:.6f},scale={settings.frame_width}:{settings.frame_height}:flags=area,"
            "format=gray"
        ),
        "-frames:v",
        str(settings.sample_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    result = run_command(command, capture_output=True, text=False)
    frame_size = settings.frame_width * settings.frame_height
    if frame_size <= 0:
        return []
    payload = result.stdout
    frame_count = len(payload) // frame_size
    return [payload[index * frame_size : (index + 1) * frame_size] for index in range(frame_count)]


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_dominant_coverage(frame: bytes, bucket_count: int = 16) -> float:
    histogram = [0] * bucket_count
    for value in frame:
        histogram[(value * bucket_count) // 256] += 1
    dominant = sorted(histogram, reverse=True)[:2]
    return sum(dominant) / len(frame)


def compute_edge_profile(frame: bytes, width: int, height: int, threshold: int) -> dict[str, float]:
    if width < 2 or height < 2:
        return {
            "density": 0.0,
            "row_coverage": 0.0,
        }

    pixels = memoryview(frame)
    edge_pixels = 0
    comparisons = 0
    edge_rows = [False] * max(height - 1, 1)
    for y in range(height - 1):
        row_offset = y * width
        next_row_offset = (y + 1) * width
        for x in range(width - 1):
            comparisons += 1
            index = row_offset + x
            if abs(pixels[index] - pixels[index + 1]) >= threshold or abs(pixels[index] - pixels[next_row_offset + x]) >= threshold:
                edge_pixels += 1
                edge_rows[y] = True
    return {
        "density": edge_pixels / comparisons if comparisons else 0.0,
        "row_coverage": sum(1 for present in edge_rows if present) / len(edge_rows) if edge_rows else 0.0,
    }


def compute_motion_score(frames: list[bytes]) -> float:
    if len(frames) < 2:
        return 0.0

    scores: list[float] = []
    for previous, current in zip(frames, frames[1:]):
        diff_total = 0
        for left, right in zip(previous, current):
            diff_total += abs(left - right)
        scores.append(diff_total / len(previous))
    return mean(scores)


def analyze_sampled_frames(
    frames: list[bytes],
    *,
    settings: TextOnlyFilterSettings,
) -> dict[str, bool | float | int | str]:
    if len(frames) < 2:
        return {
            "drop": False,
            "reason": "insufficient_samples",
            "sampled_frames": len(frames),
        }

    dominant_coverages = [compute_dominant_coverage(frame) for frame in frames]
    edge_profiles = [compute_edge_profile(frame, settings.frame_width, settings.frame_height, settings.edge_threshold) for frame in frames]
    edge_densities = [profile["density"] for profile in edge_profiles]
    edge_row_coverages = [profile["row_coverage"] for profile in edge_profiles]
    average_dominant_coverage = mean(dominant_coverages)
    average_edge_density = mean(edge_densities)
    average_edge_row_coverage = mean(edge_row_coverages)
    motion_score = compute_motion_score(frames)

    is_static = motion_score <= settings.max_motion
    has_flat_background = average_dominant_coverage >= settings.min_dominant_coverage
    has_text_like_edges = settings.min_edge_density <= average_edge_density <= settings.max_edge_density
    has_compact_edge_rows = average_edge_row_coverage <= settings.max_edge_row_coverage
    drop = is_static and has_flat_background and has_text_like_edges and has_compact_edge_rows

    if drop:
        reason = "text_only_heuristic_matched"
    else:
        failures: list[str] = []
        if not is_static:
            failures.append("moving_background")
        if not has_flat_background:
            failures.append("background_not_flat")
        if not has_text_like_edges:
            if average_edge_density < settings.min_edge_density:
                failures.append("too_few_text_like_edges")
            else:
                failures.append("too_many_edges_for_text_card")
        if not has_compact_edge_rows:
            failures.append("edges_spread_across_too_many_rows")
        reason = ",".join(failures) if failures else "kept_ambiguous"

    return {
        "drop": drop,
        "reason": reason,
        "sampled_frames": len(frames),
        "motion_score": round(motion_score, 6),
        "average_dominant_coverage": round(average_dominant_coverage, 6),
        "average_edge_density": round(average_edge_density, 6),
        "average_edge_row_coverage": round(average_edge_row_coverage, 6),
    }


def classify_segment_for_text_only_filter(
    video_path: Path,
    segment: dict[str, float | int],
    settings: TextOnlyFilterSettings,
) -> dict[str, bool | float | int | str]:
    if not settings.enabled:
        return {
            "enabled": False,
            "drop": False,
            "reason": "disabled",
        }

    try:
        frames = sample_segment_frames(
            video_path,
            start_time=float(segment["start"]),
            duration=float(segment["duration"]),
            settings=settings,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace").strip() if exc.stderr else str(exc)
        return {
            "enabled": True,
            "drop": False,
            "reason": "analysis_error",
            "error": message[:240],
        }

    analysis = analyze_sampled_frames(frames, settings=settings)
    analysis["enabled"] = True
    return analysis


def has_visual_analysis_metrics(analysis: object) -> bool:
    return isinstance(analysis, dict) and all(key in analysis for key in VISUAL_ANALYSIS_METRIC_KEYS)


def classify_segment_for_visual_analysis(
    video_path: Path,
    segment: dict[str, float | int | str | bool | dict[str, bool | float | int | str]],
    settings: TextOnlyFilterSettings,
) -> dict[str, bool | float | int | str]:
    try:
        frames = sample_segment_frames(
            video_path,
            start_time=float(segment["start"]),
            duration=float(segment["duration"]),
            settings=settings,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace").strip() if exc.stderr else str(exc)
        return {
            "enabled": True,
            "drop": False,
            "reason": "analysis_error",
            "error": message[:240],
        }

    analysis = analyze_sampled_frames(frames, settings=settings)
    analysis["enabled"] = True
    return analysis


def filter_text_only_segments(
    video_path: Path,
    segments: list[dict[str, float | int]],
    settings: TextOnlyFilterSettings,
) -> tuple[
    list[dict[str, float | int | str | bool | dict[str, bool | float | int | str]]],
    list[dict[str, float | int | str | bool | dict[str, bool | float | int | str]]],
    list[dict[str, float | int | str | bool | dict[str, bool | float | int | str]]],
    TextOnlyFilterSuppression,
]:
    candidates: list[dict[str, float | int | str | dict[str, bool | float | int | str]]] = []
    kept: list[dict[str, float | int | str | dict[str, bool | float | int | str]]] = []
    dropped: list[dict[str, float | int | str | dict[str, bool | float | int | str]]] = []
    suppression = TextOnlyFilterSuppression(fired=False)

    for segment in segments:
        candidate = dict(segment)
        candidate["source_scene_index"] = int(segment["scene_index"])
        candidate["text_only_filter"] = classify_segment_for_text_only_filter(video_path, segment, settings)
        candidate["text_only_filter_effective_drop"] = bool(candidate["text_only_filter"]["drop"])
        candidates.append(candidate)

        if candidate["text_only_filter_effective_drop"]:
            dropped.append(candidate)
            continue

        kept_segment = dict(candidate)
        kept_segment["scene_index"] = len(kept)
        kept.append(kept_segment)

    if settings.enabled and segments and not kept:
        suppressed_drop_count = len(dropped)
        suppression = TextOnlyFilterSuppression(
            fired=True,
            reason="all_segments_flagged",
            suppressed_drop_count=suppressed_drop_count,
            message=(
                "Text-only filter suppression fired because every detected segment was flagged; "
                "preserving all segments for this run."
            ),
        )
        kept = []
        dropped = []
        for candidate in candidates:
            candidate["text_only_filter_effective_drop"] = False
            kept_segment = dict(candidate)
            kept_segment["scene_index"] = len(kept)
            kept.append(kept_segment)

    return kept, dropped, candidates, suppression


def classify_segment_for_end_card_filter(
    video_path: Path,
    segment: dict[str, float | int | str | bool | dict[str, bool | float | int | str]],
    *,
    total_duration: float,
    settings: EndCardFilterSettings,
    visual_analysis_settings: TextOnlyFilterSettings,
    trailing_candidate: bool,
    text_filter_suppression_fired: bool,
) -> dict[str, bool | float | str]:
    if not settings.enabled:
        return {
            "enabled": False,
            "drop": False,
            "reason": "disabled",
        }

    if text_filter_suppression_fired:
        return {
            "enabled": True,
            "drop": False,
            "reason": "skipped_text_filter_suppressed",
            "trailing_candidate": trailing_candidate,
        }

    analysis = segment.get("text_only_filter")
    analysis_source = "text_only_filter"
    if not has_visual_analysis_metrics(analysis):
        analysis = classify_segment_for_visual_analysis(video_path, segment, visual_analysis_settings)
        analysis_source = "independent_end_card_analysis"

    if not has_visual_analysis_metrics(analysis):
        return {
            "enabled": True,
            "drop": False,
            "reason": str(analysis.get("reason", "missing_visual_analysis")) if isinstance(analysis, dict) else "missing_visual_analysis",
            "analysis_source": analysis_source,
            "trailing_candidate": trailing_candidate,
        }

    duration = float(segment["duration"])
    start_time = float(segment["start"])
    end_time = float(segment["end"])
    late_window_start = max(0.0, total_duration - settings.window_seconds)
    start_ratio = (start_time / total_duration) if total_duration > 0 else 0.0
    late_timeline = end_time >= late_window_start or start_ratio >= settings.min_start_ratio
    within_duration_range = settings.min_duration <= duration <= settings.max_duration
    motion_score = float(analysis.get("motion_score", 0.0))
    dominant_coverage = float(analysis.get("average_dominant_coverage", 0.0))
    edge_density = float(analysis.get("average_edge_density", 0.0))
    edge_row_coverage = float(analysis.get("average_edge_row_coverage", 1.0))
    is_static = motion_score <= settings.max_motion
    has_branded_layout = (
        dominant_coverage >= settings.min_dominant_coverage
        and settings.min_edge_density <= edge_density <= settings.max_edge_density
        and edge_row_coverage <= settings.max_edge_row_coverage
    )
    text_like_scene = bool(analysis.get("drop"))
    drop = trailing_candidate and late_timeline and within_duration_range and is_static and (text_like_scene or has_branded_layout)

    if drop:
        reason = "late_trailing_text_card" if text_like_scene else "late_trailing_branded_end_card"
    else:
        failures: list[str] = []
        if not trailing_candidate:
            failures.append("not_trailing_suffix")
        if not late_timeline:
            failures.append("outside_late_timeline_window")
        if not within_duration_range:
            failures.append("duration_out_of_end_card_range")
        if not is_static:
            failures.append("moving_scene")
        if not text_like_scene and not has_branded_layout:
            failures.append("layout_not_end_card_like")
        reason = ",".join(failures) if failures else "kept_ambiguous"

    return {
        "enabled": True,
        "drop": drop,
        "reason": reason,
        "analysis_source": analysis_source,
        "trailing_candidate": trailing_candidate,
        "late_timeline": late_timeline,
        "start_ratio": round(start_ratio, 6),
        "late_window_start": round(late_window_start, 6),
        "static_scene": is_static,
        "text_like_scene": text_like_scene,
        "branded_layout": has_branded_layout,
    }


def filter_end_card_segments(
    video_path: Path,
    segments: list[dict[str, float | int | str | bool | dict[str, bool | float | int | str]]],
    *,
    total_duration: float,
    settings: EndCardFilterSettings,
    visual_analysis_settings: TextOnlyFilterSettings,
    text_filter_suppression: TextOnlyFilterSuppression,
) -> tuple[
    list[dict[str, float | int | str | bool | list[str] | dict[str, bool | float | int | str]]],
    list[dict[str, float | int | str | bool | list[str] | dict[str, bool | float | int | str]]],
    EndCardFilterSuppression,
]:
    annotated_segments: list[dict[str, float | int | str | bool | list[str] | dict[str, bool | float | int | str]]] = []
    trailing_candidate = True
    for segment in reversed(segments):
        candidate = dict(segment)
        candidate["removal_reasons"] = list(candidate.get("removal_reasons", []))
        candidate["effective_removed"] = bool(candidate.get("effective_removed", False))
        candidate["end_card_filter"] = classify_segment_for_end_card_filter(
            video_path,
            candidate,
            total_duration=total_duration,
            settings=settings,
            visual_analysis_settings=visual_analysis_settings,
            trailing_candidate=trailing_candidate,
            text_filter_suppression_fired=text_filter_suppression.fired,
        )
        candidate["end_card_filter_effective_drop"] = bool(candidate["end_card_filter"]["drop"])
        if candidate["end_card_filter_effective_drop"]:
            candidate["removal_reasons"].append("end_card")
            candidate["effective_removed"] = True
        else:
            trailing_candidate = False
        annotated_segments.append(candidate)

    annotated_segments.reverse()
    kept: list[dict[str, float | int | str | bool | list[str] | dict[str, bool | float | int | str]]] = []
    dropped: list[dict[str, float | int | str | bool | list[str] | dict[str, bool | float | int | str]]] = []
    suppression = EndCardFilterSuppression(fired=False)
    for candidate in annotated_segments:
        if candidate["end_card_filter_effective_drop"]:
            dropped.append(candidate)
            continue
        kept_segment = dict(candidate)
        kept_segment["scene_index"] = len(kept)
        kept.append(kept_segment)

    if settings.enabled and segments and not kept:
        suppressed_drop_count = len(dropped)
        suppression = EndCardFilterSuppression(
            fired=True,
            reason="all_segments_flagged",
            suppressed_drop_count=suppressed_drop_count,
            message=(
                "End-card removal suppression fired because it would have removed every remaining segment; "
                "preserving all segments for this run."
            ),
        )
        kept = []
        dropped = []
        for candidate in annotated_segments:
            candidate["end_card_filter_effective_drop"] = False
            candidate["removal_reasons"] = [reason for reason in candidate["removal_reasons"] if reason != "end_card"]
            candidate["effective_removed"] = bool(candidate["removal_reasons"])
            kept_segment = dict(candidate)
            kept_segment["scene_index"] = len(kept)
            kept.append(kept_segment)

    return kept, dropped, suppression


def cut_scene_clip(formatted_path: Path, clip_path: Path, start_time: float, duration: float) -> str:
    if duration <= 0:
        raise RuntimeError(f"Cannot cut a non-positive duration clip from {formatted_path}")

    nvenc_available = ffmpeg_supports_nvenc()
    if nvenc_available:
        gpu_command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "cuda",
            "-i",
            str(formatted_path),
            "-ss",
            f"{start_time:.6f}",
            "-t",
            f"{duration:.6f}",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "fast",
            "-cq",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(clip_path),
        ]
        try:
            run_command(gpu_command)
            return "h264_nvenc"
        except subprocess.CalledProcessError:
            pass

    cpu_command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(formatted_path),
        "-ss",
        f"{start_time:.6f}",
        "-t",
        f"{duration:.6f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(clip_path),
    ]
    run_command(cpu_command)
    return "libx264"


def build_shuffle_order(scene_count: int, shuffle_seed: int) -> list[int]:
    order = list(range(scene_count))
    if scene_count <= 1:
        return order

    generator = random.Random(shuffle_seed)
    generator.shuffle(order)
    if order == list(range(scene_count)):
        order = order[1:] + order[:1]
    return order


def write_concat_manifest(manifest_path: Path, clip_paths: list[Path]) -> None:
    lines = [f"file '{path.as_posix().replace("'", "'\\''")}'" for path in clip_paths]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def concat_scene_clips(manifest_path: Path, output_path: Path) -> str:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest_path),
        "-c",
        "copy",
        str(output_path),
    ]
    run_command(command)
    return "copy"


def write_summary(
    summary_path: Path,
    *,
    source: str,
    formatted_path: Path,
    final_output_path: Path,
    filter_manifest_path: Path,
    scene_times: list[float],
    segments: list[dict[str, float | int | str | dict[str, bool | float | int | str]]],
    dropped_segments: list[dict[str, float | int | str | dict[str, bool | float | int | str]]],
    flagged_segments: list[dict[str, float | int | str | dict[str, bool | float | int | str]]],
    candidate_segment_count: int,
    shuffled_order: list[int],
    shuffle_seed: int,
    encoder_used: str,
    text_only_filter_settings: TextOnlyFilterSettings,
    text_only_filter_suppression: TextOnlyFilterSuppression,
    end_card_filter_settings: EndCardFilterSettings,
    end_card_filter_suppression: EndCardFilterSuppression,
) -> None:
    text_only_dropped_count = sum(1 for segment in flagged_segments if segment.get("text_only_filter_effective_drop"))
    end_card_flagged_count = sum(1 for segment in segments + dropped_segments if segment.get("end_card_filter", {}).get("drop"))
    end_card_dropped_count = sum(1 for segment in segments + dropped_segments if segment.get("end_card_filter_effective_drop"))
    payload = {
        "source": source,
        "formatted_video": str(formatted_path),
        "final_output": str(final_output_path),
        "filter_manifest": str(filter_manifest_path),
        "shuffle_seed": shuffle_seed,
        "encoder_used": encoder_used,
        "detected_scene_times": [round(value, 6) for value in scene_times],
        "text_only_filter": {
            "enabled": text_only_filter_settings.enabled,
            "settings": asdict(text_only_filter_settings),
            "candidate_segment_count": candidate_segment_count,
            "flagged_segment_count": len(flagged_segments),
            "kept_segment_count": len(segments),
            "dropped_segment_count": text_only_dropped_count,
            "suppression": asdict(text_only_filter_suppression),
        },
        "end_card_filter": {
            "enabled": end_card_filter_settings.enabled,
            "settings": asdict(end_card_filter_settings),
            "flagged_segment_count": end_card_flagged_count,
            "dropped_segment_count": end_card_dropped_count,
            "suppression": asdict(end_card_filter_suppression),
        },
        "segments": segments,
        "dropped_segments": dropped_segments,
        "flagged_segments": flagged_segments,
        "shuffled_order": shuffled_order,
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_segment_filter_manifest(
    manifest_path: Path,
    *,
    text_only_filter_settings: TextOnlyFilterSettings,
    end_card_filter_settings: EndCardFilterSettings,
    segments: list[dict[str, float | int | str | dict[str, bool | float | int | str]]],
    text_only_filter_suppression: TextOnlyFilterSuppression,
    end_card_filter_suppression: EndCardFilterSuppression,
) -> None:
    flagged_segment_count = sum(1 for segment in segments if segment["text_only_filter"]["drop"])
    effective_dropped_segment_count = sum(1 for segment in segments if segment["text_only_filter_effective_drop"])
    end_card_flagged_segment_count = sum(1 for segment in segments if segment.get("end_card_filter", {}).get("drop"))
    end_card_dropped_segment_count = sum(1 for segment in segments if segment.get("end_card_filter_effective_drop"))
    payload = {
        "text_only_filter": {
            "enabled": text_only_filter_settings.enabled,
            "settings": asdict(text_only_filter_settings),
            "segment_count": len(segments),
            "flagged_segment_count": flagged_segment_count,
            "effective_dropped_segment_count": effective_dropped_segment_count,
            "suppression": asdict(text_only_filter_suppression),
        },
        "end_card_filter": {
            "enabled": end_card_filter_settings.enabled,
            "settings": asdict(end_card_filter_settings),
            "flagged_segment_count": end_card_flagged_segment_count,
            "effective_dropped_segment_count": end_card_dropped_segment_count,
            "suppression": asdict(end_card_filter_suppression),
        },
        "segments": segments,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    check_required_binaries()
    text_only_filter_settings = build_text_only_filter_settings(args)
    end_card_filter_settings = build_end_card_filter_settings(args)
    end_card_visual_analysis_settings = build_end_card_visual_analysis_settings(
        text_only_filter_settings,
        end_card_filter_settings,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    work_root = Path(args.work_dir).expanduser().resolve()
    source_path = Path(args.source).expanduser()
    normalized_url = None if source_path.exists() else normalize_url(args.source)
    source_name = infer_source_name(args.source, normalized_url)
    job_dir = work_root / source_name
    raw_dir = job_dir / "raw"
    formatted_dir = job_dir / "formatted"
    clips_dir = job_dir / "clips"
    manifests_dir = job_dir / "manifests"
    formatted_path = formatted_dir / f"{source_name}_vegas_source.mp4"
    final_output_path = output_dir / f"{source_name}_shuffled.mp4"
    summary_path = output_dir / f"{source_name}_shuffle_summary.json"
    filter_manifest_path = output_dir / f"{source_name}_segment_filter_report.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    if final_output_path.exists():
        if args.overwrite:
            final_output_path.unlink()
        else:
            raise FileExistsError(
                f"Final output already exists: {final_output_path}. Use --overwrite or choose a different --output-dir."
            )
    if summary_path.exists() and args.overwrite:
        summary_path.unlink()
    if filter_manifest_path.exists() and args.overwrite:
        filter_manifest_path.unlink()

    prepare_job_directories(job_dir, args.overwrite)

    if source_path.exists() and source_path.is_file():
        acquired_source = copy_local_source(source_path.resolve(), raw_dir)
    else:
        if not args.skip_yt_dlp_update:
            update_yt_dlp()
        acquired_source = download_source_video(args.source, raw_dir)

    print(f"Formatting source video for Vegas-ready output: {acquired_source.name}")
    source_encoder = encode_silent_h264(acquired_source, formatted_path)
    total_duration = get_duration_seconds(formatted_path)

    print(
        f"Detecting scene boundaries with threshold {args.scene_threshold:.3f} and minimum duration {args.min_scene_duration:.3f}s"
    )
    scene_times = detect_scene_times(formatted_path, args.scene_threshold)
    detected_segments = build_segments(scene_times, total_duration, args.min_scene_duration)
    text_kept_segments, text_dropped_segments, candidate_segments, text_only_filter_suppression = filter_text_only_segments(
        formatted_path,
        detected_segments,
        text_only_filter_settings,
    )
    for candidate_segment in candidate_segments:
        candidate_segment["removal_reasons"] = []
        candidate_segment["effective_removed"] = False
        candidate_segment["end_card_filter"] = {
            "enabled": end_card_filter_settings.enabled,
            "drop": False,
            "reason": "not_evaluated",
        }
        candidate_segment["end_card_filter_effective_drop"] = False

    for candidate_segment in text_dropped_segments:
        candidate_segment["removal_reasons"] = ["text_only"]
        candidate_segment["effective_removed"] = True
        candidate_segment["end_card_filter"] = {
            "enabled": end_card_filter_settings.enabled,
            "drop": False,
            "reason": "skipped_text_only_removed",
        }
        candidate_segment["end_card_filter_effective_drop"] = False

    segments, end_card_dropped_segments, end_card_filter_suppression = filter_end_card_segments(
        formatted_path,
        text_kept_segments,
        total_duration=total_duration,
        settings=end_card_filter_settings,
        visual_analysis_settings=end_card_visual_analysis_settings,
        text_filter_suppression=text_only_filter_suppression,
    )
    end_card_updates = {
        int(segment["source_scene_index"]): segment
        for segment in segments + end_card_dropped_segments
    }
    for candidate_segment in candidate_segments:
        update = end_card_updates.get(int(candidate_segment["source_scene_index"]))
        if update is None:
            continue
        candidate_segment["end_card_filter"] = update["end_card_filter"]
        candidate_segment["end_card_filter_effective_drop"] = update["end_card_filter_effective_drop"]
        candidate_segment["removal_reasons"] = update["removal_reasons"]
        candidate_segment["effective_removed"] = update["effective_removed"]
    dropped_segments = text_dropped_segments + end_card_dropped_segments
    flagged_segments = [segment for segment in candidate_segments if segment["text_only_filter"]["drop"]]
    write_segment_filter_manifest(
        filter_manifest_path,
        text_only_filter_settings=text_only_filter_settings,
        end_card_filter_settings=end_card_filter_settings,
        segments=candidate_segments,
        text_only_filter_suppression=text_only_filter_suppression,
        end_card_filter_suppression=end_card_filter_suppression,
    )

    if text_only_filter_settings.enabled:
        if text_only_filter_suppression.fired:
            print(f"Warning: {text_only_filter_suppression.message}", file=sys.stderr)
            print(
                "Text-only scene filter flagged all detected segments; suppression preserved all segments and the report "
                f"records {text_only_filter_suppression.suppressed_drop_count} originally flagged candidate(s)."
            )
        elif text_dropped_segments:
            print(
                f"Dropped {len(text_dropped_segments)} text-only scene candidate(s); {len(segments)} segment(s) remain after filtering."
            )
        else:
            print("Text-only scene filter kept all detected segments.")

    if end_card_filter_settings.enabled:
        if end_card_filter_suppression.fired:
            print(f"Warning: {end_card_filter_suppression.message}", file=sys.stderr)
            print(
                "End-card removal flagged every remaining segment; suppression preserved all segments and the report "
                f"records {end_card_filter_suppression.suppressed_drop_count} originally flagged candidate(s)."
            )
        elif end_card_dropped_segments:
            print(
                f"Removed {len(end_card_dropped_segments)} likely trailing end-card segment(s) before clip cutting."
            )
        else:
            print("End-card removal kept the trailing scene suffix unchanged.")

    if len(segments) == 1:
        print("Scene processing produced a single kept segment; final output will match that segment's formatted profile.")
    else:
        print(f"Cutting {len(segments)} scene clips...")

    clip_paths: list[Path] = []
    clip_encoder = source_encoder
    for segment in segments:
        clip_path = clips_dir / f"scene_{int(segment['scene_index']):03d}.mp4"
        clip_encoder = cut_scene_clip(
            formatted_path,
            clip_path,
            float(segment["start"]),
            float(segment["duration"]),
        )
        clip_paths.append(clip_path)

    shuffled_order = build_shuffle_order(len(clip_paths), args.shuffle_seed)
    shuffled_clip_paths = [clip_paths[index] for index in shuffled_order]
    write_concat_manifest(manifests_dir / "original_order.txt", clip_paths)
    write_concat_manifest(manifests_dir / "shuffled_order.txt", shuffled_clip_paths)

    if len(shuffled_clip_paths) == 1:
        shutil.copy2(shuffled_clip_paths[0], final_output_path)
        final_encoder = clip_encoder
    else:
        final_encoder = concat_scene_clips(manifests_dir / "shuffled_order.txt", final_output_path)

    write_summary(
        summary_path,
        source=args.source,
        formatted_path=formatted_path,
        final_output_path=final_output_path,
        filter_manifest_path=filter_manifest_path,
        scene_times=scene_times,
        segments=segments,
        dropped_segments=dropped_segments,
        flagged_segments=flagged_segments,
        candidate_segment_count=len(candidate_segments),
        shuffled_order=shuffled_order,
        shuffle_seed=args.shuffle_seed,
        encoder_used=final_encoder,
        text_only_filter_settings=text_only_filter_settings,
        text_only_filter_suppression=text_only_filter_suppression,
        end_card_filter_settings=end_card_filter_settings,
        end_card_filter_suppression=end_card_filter_suppression,
    )

    if not args.keep_intermediate:
        shutil.rmtree(job_dir)

    print(f"Wrote final shuffled output: {final_output_path}")
    print(f"Wrote shuffle summary: {summary_path}")
    print(f"Wrote segment filter report: {filter_manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)