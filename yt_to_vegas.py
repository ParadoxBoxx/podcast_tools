import argparse
import subprocess
import concurrent.futures
import os
import sys
import shutil
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlparse, parse_qs

# --- Configuration ---
DEFAULT_INPUT_FILE = "links.txt"
DEFAULT_NUM_DOWNLOAD_WORKERS = 4
DEFAULT_NUM_ENCODE_WORKERS = 4

# Output directory for Vegas-ready files
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "vegas_output"
# Temporary directory for raw YouTube downloads
DEFAULT_DOWNLOAD_DIR = Path(__file__).resolve().parent / "downloads"

INPUT_FILE = DEFAULT_INPUT_FILE
NUM_DOWNLOAD_WORKERS = DEFAULT_NUM_DOWNLOAD_WORKERS
NUM_ENCODE_WORKERS = DEFAULT_NUM_ENCODE_WORKERS
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
DOWNLOAD_DIR = DEFAULT_DOWNLOAD_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download videos listed in a links file with yt-dlp, then transcode them into Vegas-ready "
            "H.264 MP4 files without audio tracks."
        ),
        epilog=(
            "Default behavior: read URLs from links.txt in the script directory, clear the working "
            "download/output folders, update yt-dlp inside the active Python environment, then download "
            "and encode with four workers per stage."
        ),
    )
    parser.add_argument(
        "--input-file",
        default=DEFAULT_INPUT_FILE,
        help="Path to the text file containing one YouTube URL per line. Default: links.txt.",
    )
    parser.add_argument(
        "--download-workers",
        type=positive_int,
        default=DEFAULT_NUM_DOWNLOAD_WORKERS,
        help="Maximum concurrent yt-dlp download jobs. Default: 4.",
    )
    parser.add_argument(
        "--encode-workers",
        type=positive_int,
        default=DEFAULT_NUM_ENCODE_WORKERS,
        help="Maximum concurrent ffmpeg encode jobs. Default: 4.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for Vegas-ready MP4 outputs. Default: ./vegas_output.",
    )
    parser.add_argument(
        "--download-dir",
        default=str(DEFAULT_DOWNLOAD_DIR),
        help="Directory for raw yt-dlp downloads. Default: ./downloads.",
    )
    return parser.parse_args()


def positive_int(value):
    amount = int(value)
    if amount <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return amount

def read_links(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc in ["www.youtube.com", "youtube.com", "m.youtube.com"]:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
            if video_id:
                return f"https://www.youtube.com/watch?v={video_id}"
    elif parsed.netloc in ["youtu.be"]:
        video_id = parsed.path.lstrip("/")
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    return url

def sanitize_filename(name: str) -> str:
    forbidden = r'[/\x00:<>"|?*\u2013\u2014\u2015\u2500\uFF5C\u2016\u2223\u2758\u23D0\uFE31\uFE32\uFE58\uFFE8]'
    name = re.sub(forbidden, "_", name)
    name = ''.join(ch for ch in name if not unicodedata.category(ch).startswith("So"))
    return name.strip()[:200]

def update_yt_dlp():
    """Updates yt-dlp to latest nightly inside the venv."""
    print(">>> Syncing yt-dlp to latest nightly...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", "--pre", "yt-dlp[default]"], check=True)
    except subprocess.CalledProcessError:
        print(">>> Warning: Update failed.")

def download_video(url):
    """Downloads best video using Deno (local) and GitHub logic."""
    clean_url = normalize_url(url)
    current_dir = str(Path(__file__).resolve().parent)
    
    # Add the current folder to the temporary PATH for this process
    # so yt-dlp can find the 'deno' binary automatically
    env = os.environ.copy()
    env["PATH"] = current_dir + os.pathsep + env["PATH"]

    print(f"Downloading: {clean_url}")
    try:
        subprocess.run([
            sys.executable, "-m", "yt_dlp",
            "--cookies-from-browser", "firefox",
            "--impersonate", "Firefox-135",
            "--sleep-requests", "1",
            "--remote-components", "ejs:github",
            "--ffmpeg-location", current_dir, # Search here for binaries
            "-f", "bestvideo/best", 
            "-P", str(DOWNLOAD_DIR),
            clean_url
        ], check=True, env=env) # Pass the updated PATH
    except subprocess.CalledProcessError:
        print(f"!!! Download failed: {clean_url}")

def encode_video(video_path):
    """Converts video to Vegas-compatible H.264 and strips all audio tracks."""
    input_path = Path(video_path)
    output_path = OUTPUT_DIR / f"{sanitize_filename(input_path.stem)}_vegas.mp4"
    print(f"Processing for Vegas (Stripping Audio): {input_path.name}")

    # Try NVENC GPU acceleration first
    ffmpeg_gpu_cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-i", str(input_path),
        "-c:v", "h264_nvenc",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",
        str(output_path)
    ]

    try:
        subprocess.run(ffmpeg_gpu_cmd, check=True)
        print(f"  Used: GPU NVENC acceleration")
    except subprocess.CalledProcessError:
        # Fall back to software x264 encoding
        print(f"  Note: Falling back to software CPU encoding (GPU not available or nvcodec missing)")
        ffmpeg_cpu_cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-an",
            str(output_path)
        ]
        subprocess.run(ffmpeg_cpu_cmd, check=True)
        print(f"  Used: CPU software encoding")

def main():
    global INPUT_FILE, NUM_DOWNLOAD_WORKERS, NUM_ENCODE_WORKERS, OUTPUT_DIR, DOWNLOAD_DIR

    args = parse_args()
    INPUT_FILE = args.input_file
    NUM_DOWNLOAD_WORKERS = args.download_workers
    NUM_ENCODE_WORKERS = args.encode_workers
    OUTPUT_DIR = Path(args.output_dir).expanduser().resolve()
    DOWNLOAD_DIR = Path(args.download_dir).expanduser().resolve()

    # Clean up directories before processing
    print(f"Cleaning directories...")
    if DOWNLOAD_DIR.exists():
        shutil.rmtree(DOWNLOAD_DIR)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    update_yt_dlp()

    # Check if CUDA-enabled ffmpeg is needed
    try:
        import subprocess
        test_result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        if "h264_nvenc" not in test_result.stdout:
            print(f"\ufffd NVENC encoder not found.")
            print(f"To enable GPU acceleration, install with:")
            print(f"  sudo apt update && sudo apt install nvidia-cuda-toolkit")
        else:
            print(f"  NVENC encoder available")
    except FileNotFoundError:
        # ffmpeg not found in PATH - will use yt-dlp's internal ffmpeg
        print(f"\ufffd Standard ffmpeg not found. Using yt-dlp's internal encoder.")

    links = read_links(INPUT_FILE)
    if not links:
        print(f"No links in {INPUT_FILE}.")
        return

    unique_links = list(dict.fromkeys([normalize_url(l) for l in links]))
    
    # Download Phase
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_DOWNLOAD_WORKERS) as executor:
        executor.map(download_video, unique_links)

    # Encode Phase
    downloaded_files = [str(p) for p in DOWNLOAD_DIR.iterdir() if p.is_file() and not p.name.endswith(".part")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_ENCODE_WORKERS) as executor:
        executor.map(encode_video, downloaded_files)

    print(f"\n✅ All set! Vegas-ready files are in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
