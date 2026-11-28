from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..cancel import check_cancelled, run_cancellable_subprocess
from ..models import ProjectManifest
from ..receipts import ProjectStore


def _zscore(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std < 1e-9:
        return np.zeros_like(arr)
    return (arr - mean) / std


def _extract_audio_envelope(video_path: Path, sample_rate: int = 16000, hop_ms: int = 20) -> tuple[np.ndarray, float]:
    """Extract 16kHz mono PCM via ffmpeg and compute 20ms frame RMS envelope."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    res = run_cancellable_subprocess(cmd, capture_output=True)
    if res.returncode != 0 or not res.stdout:
        return np.zeros(0, dtype=np.float32), hop_ms / 1000.0

    pcm = np.frombuffer(res.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    hop_samples = int(sample_rate * hop_ms / 1000)
    n_frames = len(pcm) // hop_samples
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float32), hop_ms / 1000.0
    frames = pcm[: n_frames * hop_samples].reshape(n_frames, hop_samples)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12).astype(np.float32)
    return rms, hop_ms / 1000.0


def _extract_visual_motion(video_path: Path, fps: float = 1.0, width: int = 160, height: int = 90) -> np.ndarray:
    """Extract 1fps low-res grayscale frames and compute sustained motion score (suppressing single-frame hard cuts)."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        f"fps={fps},scale={width}:{height},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    res = run_cancellable_subprocess(cmd, capture_output=True)
    if res.returncode != 0 or not res.stdout:
        return np.zeros(0, dtype=np.float32)

    frame_bytes = width * height
    n_frames = len(res.stdout) // frame_bytes
    if n_frames < 3:
        return np.zeros(max(n_frames, 0), dtype=np.float32)

    raw = np.frombuffer(res.stdout[: n_frames * frame_bytes], dtype=np.uint8).reshape(n_frames, frame_bytes).astype(np.float32)
    diffs = np.mean(np.abs(np.diff(raw, axis=0)), axis=1)
    # Pad to n_frames
    diffs = np.concatenate([[0.0], diffs])

    # Hard cut suppression: a single-frame scene cut creates one spike while neighbors stay calm.
    # Taking min(diff[t], max(diff[t-1], diff[t+1])) preserves sustained gestures/laughter (>=2s) while dropping 1-frame cuts.
    sustained = np.zeros_like(diffs)
    for i in range(1, len(diffs) - 1):
        neighbor_max = max(diffs[i - 1], diffs[i + 1])
        sustained[i] = min(diffs[i], neighbor_max)
    sustained[-1] = min(diffs[-1], diffs[-2])
    return sustained.astype(np.float32)


def _count_units(en: str, zh: str) -> float:
    en_words = len(re.findall(r"[A-Za-z0-9']+", en))
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", zh))
    return float(max(en_words, int(zh_chars * 0.65)))


def run_signals_stage(
    manifest: ProjectManifest,
    store: ProjectStore,
    run_id: str,
) -> ProjectManifest:
    t0 = time.monotonic()
    analysis_video = Path(manifest.source.analysis_video)
    utts = manifest.utterances
    if not utts:
        raise ValueError(
            "未找到字幕文件，无法执行声画信号与 AI 切片分析；若只需剪切画幅，请直接使用「裁剪画面」功能导出。"
        )

    store.log(f"[signals] 开始提取声画信号: {analysis_video.name}")
    check_cancelled()
    rms, hop_s = _extract_audio_envelope(analysis_video)
    check_cancelled()
    motion = _extract_visual_motion(analysis_video, fps=1.0)
    check_cancelled()

    # Save audio envelope for accurate acoustic pause snapping in verify_and_plan
    env_path = store.work_dir / "audio_envelope.npz"
    np.savez_compressed(env_path, rms=rms, hop_s=np.array([hop_s], dtype=np.float32), motion=motion)

    n = len(utts)
    utt_rms = np.zeros(n, dtype=np.float32)
    utt_peak = np.zeros(n, dtype=np.float32)
    utt_rate = np.zeros(n, dtype=np.float32)
    utt_motion = np.zeros(n, dtype=np.float32)

    for i, u in enumerate(utts):
        dur = max(0.2, u.end_s - u.start_s)
        utt_rate[i] = _count_units(u.en, u.zh) / dur

        if rms.size > 0:
            i0 = max(0, int(u.start_s / hop_s))
            i1 = min(rms.size, max(i0 + 1, int(u.end_s / hop_s)))
            seg = rms[i0:i1]
            utt_rms[i] = float(np.mean(seg))
            utt_peak[i] = float(np.percentile(seg, 95))

        if motion.size > 0:
            m0 = max(0, int(u.start_s))
            m1 = min(motion.size, max(m0 + 1, int(u.end_s) + 1))
            utt_motion[i] = float(np.max(motion[m0:m1]))

    rms_z = _zscore(utt_rms)
    peak_z = _zscore(utt_peak)
    rate_z = _zscore(utt_rate)
    motion_z = _zscore(utt_motion)

    hot_audio_count = 0
    hot_motion_count = 0
    rapid_turn_count = 0

    for i, u in enumerate(utts):
        u.audio_rms_z = round(float(rms_z[i]), 2)
        u.audio_peak_z = round(float(peak_z[i]), 2)
        u.speech_rate_z = round(float(rate_z[i]), 2)
        u.motion_z = round(float(motion_z[i]), 2)

        # Rapid turn-taking / overlap heuristic: very short gap AND short utterance burst
        short_turn = (u.end_s - u.start_s) < 3.2
        tight_gap = (i > 0 and u.silence_before_ms < 150) and (i + 1 < n and u.silence_after_ms < 150)
        u.rapid_turn = bool(short_turn and tight_gap and (u.audio_rms_z > 0.5 or u.speech_rate_z > 0.8))

        tags: list[str] = []
        if u.audio_rms_z >= 1.5 or u.audio_peak_z >= 1.8:
            tags.append(f"音量峰值+{max(u.audio_rms_z, u.audio_peak_z):.1f}σ")
            hot_audio_count += 1
        if u.speech_rate_z >= 1.6:
            tags.append(f"语速激增+{u.speech_rate_z:.1f}σ")
        if u.motion_z >= 1.8:
            tags.append(f"大幅动作+{u.motion_z:.1f}σ")
            hot_motion_count += 1
        if u.rapid_turn:
            tags.append("高频交锋")
            rapid_turn_count += 1
        u.signal_tags = tags

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    top_peaks: list[dict[str, Any]] = []
    ranked = sorted(utts, key=lambda x: x.audio_peak_z + x.motion_z + (1.0 if x.rapid_turn else 0.0), reverse=True)
    for u in ranked[:12]:
        if not u.signal_tags:
            continue
        top_peaks.append(
            {
                "utt_id": u.id,
                "time_s": u.start_s,
                "tags": u.signal_tags,
                "text": (u.zh or u.en)[:80],
            }
        )

    check_cancelled()
    store.save(manifest)
    store.append_receipt(
        run_id=run_id,
        stage="signals",
        actor="system",
        action="extract_av_signals",
        elapsed_ms=elapsed_ms,
        summary=f"声画信号提取完成（音频帧 {rms.size}，动作帧 {motion.size}）：标记高音量 {hot_audio_count} 句、大幅动作 {hot_motion_count} 句、高频交锋 {rapid_turn_count} 句",
        details={
            "audio_frames": int(rms.size),
            "motion_frames": int(motion.size),
            "hot_audio_count": hot_audio_count,
            "hot_motion_count": hot_motion_count,
            "rapid_turn_count": rapid_turn_count,
            "top_signal_peaks": top_peaks,
        },
    )
    return manifest
