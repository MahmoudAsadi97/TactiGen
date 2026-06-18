"""
TactiGen Dataset Download Script
Run: python data/download_datasets.py
"""
import os
import subprocess
from pathlib import Path
from loguru import logger

BASE_DIR = Path(__file__).parent


def download_statsbomb():
    """Download StatsBomb open data (free, no authentication needed)."""
    target = BASE_DIR / "statsbomb" / "open-data"
    if target.exists():
        logger.info("StatsBomb open-data already exists. Skipping.")
        return
    logger.info("Cloning StatsBomb open-data repository...")
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/statsbomb/open-data.git",
        str(target)
    ], check=True)
    logger.success(f"StatsBomb data saved to {target}")


def download_soccernet_labels():
    """Download SoccerNet action spotting labels (free, no password)."""
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
        target = str(BASE_DIR / "soccernet")
        os.makedirs(target, exist_ok=True)
        downloader = SoccerNetDownloader(LocalDirectory=target)
        downloader.downloadGames(
            files=["Labels-v2.json"],
            split=["train", "valid", "test"]
        )
        logger.success("SoccerNet Labels-v2.json downloaded.")
    except Exception as e:
        logger.warning(f"SoccerNet label download failed: {e}")
        logger.info(
            "To download SoccerNet labels manually:\n"
            "  1. Register at https://www.soccer-net.org/data\n"
            "  2. Set SOCCERNET_PASSWORD in your .env file\n"
            "  3. Re-run this script"
        )


def generate_synthetic_sample_clip():
    """
    Generate a synthetic sample clip with known player positions for pipeline testing.
    This replaces real video when SoccerNet video files are not available.
    """
    import json
    import numpy as np

    logger.info("Generating synthetic sample clip metadata for pipeline testing...")
    sample_dir = BASE_DIR / "sample_clips"
    os.makedirs(sample_dir, exist_ok=True)

    # Simulate 25 frames of player positions (2 teams, 5 visible players each)
    rng = np.random.default_rng(42)
    frames = []
    for frame_id in range(25):
        timestamp = frame_id * 0.2  # 5 FPS sampling
        players = []
        # Team A (attacking, positions shifting right)
        for p in range(5):
            players.append({
                "team": "A",
                "player_id": p,
                "x_world": float(rng.uniform(40 + frame_id * 0.3, 80)),
                "y_world": float(rng.uniform(10 + p * 10, 20 + p * 10)),
                "confidence": float(rng.uniform(0.75, 0.97))
            })
        # Team B (defending, compact block)
        for p in range(5):
            players.append({
                "team": "B",
                "player_id": 10 + p,
                "x_world": float(rng.uniform(60, 75)),
                "y_world": float(rng.uniform(15 + p * 8, 25 + p * 8)),
                "confidence": float(rng.uniform(0.70, 0.95))
            })
        frames.append({"frame_id": frame_id, "timestamp": timestamp, "players": players})

    metadata = {
        "clip_id": "synthetic_clip_001",
        "source": "synthetic",
        "fps": 5,
        "duration_seconds": 5.0,
        "pitch_dims": {"length": 105, "width": 68},
        "frames": frames
    }
    out_path = sample_dir / "synthetic_clip_001.json"
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.success(f"Synthetic sample clip saved to {out_path}")


if __name__ == "__main__":
    logger.info("=== TactiGen Dataset Download ===")
    download_statsbomb()
    download_soccernet_labels()
    generate_synthetic_sample_clip()
    logger.success("Dataset download complete.")
