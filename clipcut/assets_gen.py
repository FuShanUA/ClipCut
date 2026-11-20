from __future__ import annotations

import wave
from pathlib import Path
import numpy as np

from .config import ASSETS_DIR


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 44100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return path


def ensure_default_sfx() -> Path:
    """Generate a subtle, warm low-mid transition whoosh + soft sub hit (~0.55s)."""
    out = ASSETS_DIR / "sfx" / "transition_whoosh.wav"
    if out.exists() and out.stat().st_size > 1024:
        return out

    sr = 44100
    dur = 0.55
    n = int(sr * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)

    # Filtered noise sweep for smooth air whoosh
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 1.0, n)
    # Moving average low-pass smoothing (3 stages)
    kernel = np.ones(45) / 45.0
    smooth = np.convolve(noise, kernel, mode="same")
    smooth = np.convolve(smooth, kernel, mode="same")

    # Envelope peaking around 0.22s
    env = np.sin(np.pi * (t / dur) ** 0.75) ** 2.2
    # Gentle tonal sweep 140Hz -> 72Hz
    freq = 140.0 * np.exp(-2.0 * t) + 55.0
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    sub = 0.45 * np.sin(phase) * np.exp(-6.5 * np.maximum(0.0, t - 0.14)) * (t >= 0.08)

    mono = (smooth * 2.2 * env + sub) * 0.55
    stereo = np.stack([mono, mono], axis=1)
    return _write_wav(out, stereo, sr)


def ensure_default_bgm() -> Path:
    """Generate a clean, understated 32-second loopable corporate-tech ambient bed."""
    out = ASSETS_DIR / "bgm" / "ambient_pulse.wav"
    if out.exists() and out.stat().st_size > 4096:
        return out

    sr = 44100
    dur = 32.0
    n = int(sr * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)

    # 4-bar chord progression (8s per chord): Dm9 -> Bbmaj7 -> Fmaj7 -> Csus2
    chords = [
        [146.83, 220.00, 261.63, 329.63],  # Dm9
        [116.54, 174.61, 220.00, 293.66],  # Bbmaj7
        [174.61, 220.00, 261.63, 329.63],  # Fmaj7
        [130.81, 196.00, 220.00, 293.66],  # C6/9
    ]

    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)
    seg_len = n // len(chords)

    for idx, freqs in enumerate(chords):
        start = idx * seg_len
        end = n if idx == len(chords) - 1 else (idx + 1) * seg_len
        seg_t = t[start:end] - t[start]
        seg_dur = (end - start) / sr
        # Smooth crossfade window inside each chord
        fade = np.minimum(1.0, seg_t / 0.8) * np.minimum(1.0, (seg_dur - seg_t) / 0.8)
        fade = np.clip(fade, 0.0, 1.0)
        # Subtle warm pulse at 100 BPM (0.6s beat)
        pulse = 0.85 + 0.15 * np.sin(2.0 * np.pi * (seg_t / 0.6))

        pad_l = np.zeros_like(seg_t)
        pad_r = np.zeros_like(seg_t)
        for fi, f in enumerate(freqs):
            detune = 1.0 + (0.002 if fi % 2 == 0 else -0.002)
            weight = 0.22 / (1.0 + 0.25 * fi)
            pad_l += weight * np.sin(2.0 * np.pi * f * seg_t)
            pad_r += weight * np.sin(2.0 * np.pi * f * detune * seg_t + 0.4)
            # Soft second harmonic
            pad_l += (weight * 0.18) * np.sin(2.0 * np.pi * (f * 2.0) * seg_t)
            pad_r += (weight * 0.18) * np.sin(2.0 * np.pi * (f * 2.0 * detune) * seg_t)

        # Gentle deep bass root
        bass_f = freqs[0] / 2.0
        bass = 0.18 * np.sin(2.0 * np.pi * bass_f * seg_t) * pulse
        left[start:end] = (pad_l * pulse + bass) * fade
        right[start:end] = (pad_r * pulse + bass) * fade

    stereo = np.stack([left, right], axis=1) * 0.48
    return _write_wav(out, stereo, sr)


def ensure_default_assets() -> dict[str, Path]:
    return {
        "sfx": ensure_default_sfx(),
        "bgm": ensure_default_bgm(),
    }
