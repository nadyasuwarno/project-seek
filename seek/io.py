"""Read a video (.mp4/.avi/.mov or .tif stack) into grayscale frames, write MP4s."""
import hashlib
import os
import tempfile

import cv2
import numpy as np
import tifffile


def file_hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:12]


def _to_uint8(stack: np.ndarray) -> np.ndarray:
    if stack.dtype == np.uint8:
        return stack
    lo, hi = np.percentile(stack, [0.1, 99.9])
    return np.clip((stack - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)


def read_video(data: bytes, name: str, max_width: int = 800, max_frames: int = 1200):
    """Return (frames[T,H,W] uint8, info). info['scale'] is resize factor applied (≤1);
    divide the user's µm/px by it to get the µm/px of the returned frames."""
    suffix = os.path.splitext(name)[1].lower() or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        fps = None
        if suffix in (".tif", ".tiff"):
            stack = tifffile.imread(path)
            if stack.ndim == 4:                      # T,H,W,C → gray
                stack = stack[..., :3].mean(axis=-1)
            if stack.ndim == 2:
                stack = stack[None]
            frames = list(_to_uint8(stack[:max_frames]))
            total = len(stack)
        else:
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS) or None
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = []
            while len(frames) < max_frames:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(fr if fr.ndim == 2 else cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
            cap.release()
    finally:
        os.unlink(path)
    if not frames:
        raise ValueError("Could not read any frames from this file.")

    h, w = frames[0].shape
    scale = min(1.0, max_width / w)
    if scale < 1:
        size = (round(w * scale), round(h * scale))
        frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]
    info = {"fps": round(fps, 3) if fps and fps < 1000 else None,
            "n_frames": len(frames), "n_frames_in_file": max(total, len(frames)),
            "width": w, "height": h, "scale": scale}
    return np.stack(frames), info


def write_mp4(frames_rgb, fps: float) -> bytes:
    """Encode RGB frames to a browser-playable H.264 MP4 (via bundled ffmpeg)."""
    import imageio_ffmpeg
    h, w = frames_rgb[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        path = tmp.name
    writer = imageio_ffmpeg.write_frames(path, (w, h), fps=fps, codec="libx264",
                                         pix_fmt_out="yuv420p", macro_block_size=2,
                                         output_params=["-crf", "20"])
    writer.send(None)
    for f in frames_rgb:
        writer.send(np.ascontiguousarray(f))
    writer.close()
    with open(path, "rb") as fh:
        out = fh.read()
    os.unlink(path)
    return out
