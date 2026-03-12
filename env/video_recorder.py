"""
Video recorder for MuJoCo rollouts.
Collects rendered RGB frames and saves to mp4 via imageio/ffmpeg.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np
import imageio


class VideoRecorder:
    """
    Accumulates rendered frames and writes to a video file.

    Usage:
        recorder = VideoRecorder("videos/rollout.mp4", fps=30)
        with recorder:
            while not done:
                frame = env.render()       # (H, W, 3) uint8
                recorder.add_frame(frame)
    """

    def __init__(self, path: str | Path, fps: int = 30):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._frames: list[np.ndarray] = []

    def add_frame(self, frame: np.ndarray) -> None:
        """Append a single RGB frame (H, W, 3) uint8."""
        if frame is None:
            return
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        self._frames.append(frame)

    def save(self) -> Path:
        """Write accumulated frames to mp4. Returns the output path."""
        if not self._frames:
            raise RuntimeError("No frames to save.")
        # Use imageio v3 writer with explicit ffmpeg plugin for mp4
        with imageio.get_writer(str(self.path), fps=self.fps, codec="libx264",
                                 quality=None, pixelformat="yuv420p",
                                 output_params=["-crf", "23"]) as writer:
            for frame in self._frames:
                writer.append_data(frame)
        print(f"[VideoRecorder] Saved {len(self._frames)} frames → {self.path}")
        return self.path

    def clear(self) -> None:
        self._frames.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self._frames:
            self.save()


# ─────────────────────────────────────────────────────────────────────────── #
# Utility                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def make_grid(frames: Sequence[np.ndarray], ncols: int = 4) -> list[np.ndarray]:
    """
    Stack a list of per-episode frame sequences into a grid video.
    Each sequence has the same number of frames (or is padded with last frame).
    """
    if not frames:
        return []
    max_len = max(len(seq) for seq in frames)
    padded = [
        list(seq) + [seq[-1]] * (max_len - len(seq)) for seq in frames
    ]
    H, W, C = padded[0][0].shape
    nrows = (len(frames) + ncols - 1) // ncols
    grid_frames = []
    for t in range(max_len):
        row_imgs = []
        for r in range(nrows):
            col_imgs = []
            for c in range(ncols):
                idx = r * ncols + c
                if idx < len(padded):
                    col_imgs.append(padded[idx][t])
                else:
                    col_imgs.append(np.zeros((H, W, C), dtype=np.uint8))
            row_imgs.append(np.concatenate(col_imgs, axis=1))
        grid_frames.append(np.concatenate(row_imgs, axis=0))
    return grid_frames
