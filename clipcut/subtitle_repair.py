from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import srt

from .adapters.ingest import parse_subtitle_file, sha256_file
from .crop import crop_vf_expr, resolve_effective_crop
from .models import CropRect, ProjectManifest, SubtitleRepairItem
from .pipeline.content_filter import get_active_cut_spans
from .pipeline.verify_and_plan import compute_clip_render_hash
from .receipts import ProjectStore
from .render.renderer import _has_videotoolbox, _probe_video_wh, _run_ffmpeg_encode

_AUTOSUB_SRT_TO_ASS = (
    Path.home() / "cc" / "Library" / "Tools" / "autosub" / "Library" / "Tools" / "hardsubber" / "srt_to_ass.py"
)


@dataclass
class SubtitleRepairOutcome:
    changed_utterance_ids: list[int]
    bi_srt_path: Path
    cn_srt_path: Path
    en_srt_path: Path
    ass_path: Path
    clip_output_path: Path | None
    full_hardsub_path: Path | None


def _parse_srt(path: Path) -> list[srt.Subtitle]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    return list(srt.parse(raw))


def _write_srt(path: Path, subs: list[srt.Subtitle]) -> None:
    path.write_text(srt.compose(subs, reindex=False), encoding="utf-8")


def _language_lines(content: str, language: Literal["zh", "en"]) -> str:
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if language == "zh":
        selected = [ln for ln in lines if any("\u4e00" <= ch <= "\u9fff" for ch in ln)]
    else:
        selected = [ln for ln in lines if not any("\u4e00" <= ch <= "\u9fff" for ch in ln)]
    return "\n".join(selected)


def _generate_ass(bi_srt: Path, ass_path: Path, width: int, height: int) -> None:
    if not _AUTOSUB_SRT_TO_ASS.exists():
        raise FileNotFoundError(f"未找到 AutoSub ASS 生成脚本: {_AUTOSUB_SRT_TO_ASS}")
    cmd = [
        sys.executable,
        str(_AUTOSUB_SRT_TO_ASS),
        str(bi_srt),
        str(ass_path),
        "--layout",
        "bilingual",
        "--main-lang",
        "cn",
        "--cn-font",
        "STKaiti",
        "--en-font",
        "Calibri",
        "--cn-size",
        "60",
        "--en-size",
        "36",
        "--cn-color",
        "&H0000D7FF",
        "--en-color",
        "&H00FFFFFF",
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not ass_path.exists():
        raise RuntimeError(f"ASS 生成失败: {(res.stderr or res.stdout)[-800:]}")


def _escape_filter_path(path: Path) -> str:
    value = str(path)
    value = value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return value


def _subtitles_filter(ass_path: Path) -> str:
    return f"subtitles=filename='{_escape_filter_path(ass_path)}'"


def _render_full_hardsub(manifest: ProjectManifest, ass_path: Path, output_path: Path) -> None:
    source = Path(manifest.source.analysis_video)
    if not source.exists():
        source = Path(manifest.source.render_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.stem}.repair.tmp.mp4")
    use_vt = _has_videotoolbox()
    audio_codec = (manifest.source.audio_codec or "").lower()
    audio_args = ["-c:a", "copy"] if audio_codec == "aac" else ["-c:a", "aac", "-b:a", "192k"]
    base = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-vf",
        _subtitles_filter(ass_path),
        *audio_args,
        "-movflags",
        "+faststart",
    ]
    cmd_vt = base + ["-c:v", "h264_videotoolbox", "-b:v", "3M", str(tmp_path)]
    cmd_x264 = base + ["-c:v", "libx264", "-preset", "fast", "-crf", "18", str(tmp_path)]
    ok, _, err = _run_ffmpeg_encode(cmd_vt, cmd_x264, use_vt)
    if not ok or not tmp_path.exists() or tmp_path.stat().st_size < 1024:
        raise RuntimeError(f"全片硬字幕重烧失败: {err}")
    tmp_path.replace(output_path)


def _render_clip_with_subtitles(
    manifest: ProjectManifest,
    clip_start_s: float,
    clip_end_s: float,
    ass_path: Path,
    output_path: Path,
    *,
    crop_rect: CropRect | None = None,
) -> None:
    source = Path(manifest.source.analysis_video)
    if not source.exists():
        source = Path(manifest.source.render_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.stem}.repair.tmp.mp4")
    duration = max(0.5, clip_end_s - clip_start_s)
    use_vt = _has_videotoolbox()
    if crop_rect is None:
        matched_clip = next(
            (
                c
                for c in manifest.clips
                if abs(c.start_s - clip_start_s) < 0.01 and abs(c.end_s - clip_end_s) < 0.01
            ),
            None,
        )
        crop_rect = resolve_effective_crop(manifest, matched_clip)
    src_w, src_h = _probe_video_wh(str(source), manifest.source.width or 1920, manifest.source.height or 1080)
    crop_vf = crop_vf_expr(crop_rect, src_w, src_h)
    vf_parts = ([crop_vf] if crop_vf else []) + [_subtitles_filter(ass_path), "setpts=PTS-STARTPTS"]
    base = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{clip_start_s:.3f}",
        "-copyts",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-vf",
        ",".join(vf_parts),
        "-af",
        "asetpts=PTS-STARTPTS",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
    ]
    cmd_vt = base + ["-c:v", "h264_videotoolbox", "-b:v", "2M", str(tmp_path)]
    cmd_x264 = base + ["-c:v", "libx264", "-preset", "fast", "-crf", "18", str(tmp_path)]
    ok, _, err = _run_ffmpeg_encode(cmd_vt, cmd_x264, use_vt)
    if not ok or not tmp_path.exists() or tmp_path.stat().st_size < 1024:
        raise RuntimeError(f"切片硬字幕重烧失败: {err}")
    tmp_path.replace(output_path)


def _affected_clip_ids(manifest: ProjectManifest, changed_ids: list[int]) -> set[str]:
    changed = set(changed_ids)
    return {
        c.clip_id
        for c in manifest.clips
        if any(c.start_utt <= uid <= c.end_utt for uid in changed)
    }


def apply_subtitle_repairs(
    manifest: ProjectManifest,
    store: ProjectStore,
    clip_id: str,
    repairs: list[SubtitleRepairItem],
    *,
    rebuild_full_subtitle: bool = True,
    render_clip: bool = True,
    render_full_video: bool = False,
) -> SubtitleRepairOutcome:
    """Repair subtitle text for one clip and optionally reburn clip/full video."""
    if not repairs:
        raise ValueError("没有可修复的字幕条目")
    target = next((c for c in manifest.clips if c.clip_id == clip_id), None)
    if target is None:
        raise ValueError(f"未找到切片: {clip_id}")

    sub_path = Path(manifest.source.subtitle_path or "").expanduser()
    if sub_path.suffix.lower() != ".srt":
        raise ValueError(f"当前仅支持 SRT 字幕修复，收到: {sub_path.suffix or '(无)'}")
    if not sub_path.exists():
        raise FileNotFoundError(f"字幕文件不存在: {sub_path}")

    repair_map = {r.utterance_id: r for r in repairs}
    invalid_ids = [uid for uid in repair_map if uid < target.start_utt or uid > target.end_utt]
    if invalid_ids:
        raise ValueError(f"字幕条目不在切片范围内: {invalid_ids}")

    store.save_undo_snapshot(manifest, f"修复切片字幕 {clip_id}")
    subs = _parse_srt(sub_path)
    if len(subs) != len(manifest.utterances):
        raise ValueError(f"字幕块数与项目句数不一致: {len(subs)} != {len(manifest.utterances)}")

    changed_ids: list[int] = []
    for uid, repair in repair_map.items():
        sub = subs[uid]
        old_lines = [ln.strip() for ln in sub.content.splitlines() if ln.strip()]
        zh = next((ln for ln in old_lines if any("\u4e00" <= ch <= "\u9fff" for ch in ln)), "")
        en = next((ln for ln in old_lines if not any("\u4e00" <= ch <= "\u9fff" for ch in ln)), "")
        new_zh = zh if repair.zh is None else repair.zh.strip()
        new_en = en if repair.en is None else repair.en.strip()
        if new_zh == zh and new_en == en:
            continue
        sub.content = f"{new_zh}\n{new_en}" if new_zh and new_en else (new_zh or new_en)
        changed_ids.append(uid)

    if not changed_ids:
        raise ValueError("没有实际变更的字幕内容")

    bi_srt = sub_path
    cn_srt = bi_srt.with_name(bi_srt.stem.replace(".bi", "") + ".cn.srt")
    en_srt = bi_srt.with_name(bi_srt.stem.replace(".bi", "") + ".srt")
    ass_path = bi_srt.with_suffix(".ass")

    _write_srt(bi_srt, subs)
    if rebuild_full_subtitle:
        cn_subs = [
            srt.Subtitle(index=i, start=s.start, end=s.end, content=_language_lines(s.content, "zh"))
            for i, s in enumerate(subs, 1)
        ]
        en_subs = [
            srt.Subtitle(index=i, start=s.start, end=s.end, content=_language_lines(s.content, "en"))
            for i, s in enumerate(subs, 1)
        ]
        _write_srt(cn_srt, cn_subs)
        _write_srt(en_srt, en_subs)
    _generate_ass(bi_srt, ass_path, manifest.source.width, manifest.source.height)

    manifest.utterances = parse_subtitle_file(bi_srt)
    manifest.source.subtitle_path = str(bi_srt)
    manifest.source.subtitle_format = "srt"
    manifest.source.subtitle_sha256 = sha256_file(bi_srt)

    affected = _affected_clip_ids(manifest, changed_ids)
    for c in manifest.clips:
        active_cuts = get_active_cut_spans(manifest) if manifest.clean_cut.apply_to_clips else []
        c.render_hash = compute_clip_render_hash(
            c,
            manifest.source.render_video,
            active_cuts,
            manifest.source.subtitle_sha256,
            crop_rect=resolve_effective_crop(manifest, c),
        )
        if c.clip_id in affected or render_full_video:
            c.last_rendered_hash = ""
    if affected or render_full_video:
        manifest.remix.last_rendered_hash = ""
        manifest.compilation.last_rendered_hash = ""

    clip_output: Path | None = None
    if render_clip:
        clip_output = Path(target.output_file or "").expanduser() if target.output_file else None
        if clip_output is None:
            clip_output = Path(manifest.source.output_dir) / f"{target.index:02d}_{target.slug}.mp4"
        _render_clip_with_subtitles(manifest, target.start_s, target.end_s, ass_path, clip_output)
        target.output_file = str(clip_output)
        target.last_rendered_hash = target.render_hash

    full_output: Path | None = None
    if render_full_video:
        analysis = Path(manifest.source.analysis_video)
        full_output = analysis.with_name(analysis.stem + "_hardsub.mp4")
        _render_full_hardsub(manifest, ass_path, full_output)
        manifest.source.render_video = str(full_output)
        active_cuts = get_active_cut_spans(manifest) if manifest.clean_cut.apply_to_clips else []
        for c in manifest.clips:
            c.render_hash = compute_clip_render_hash(
                c,
                manifest.source.render_video,
                active_cuts,
                manifest.source.subtitle_sha256,
                crop_rect=resolve_effective_crop(manifest, c),
            )
            c.last_rendered_hash = ""
        manifest.remix.last_rendered_hash = ""
        manifest.compilation.last_rendered_hash = ""

    store.save(manifest)
    store.append_receipt(
        run_id=f"repair_{clip_id}_{len(changed_ids):03d}",
        stage="subtitle_repair",
        actor="human",
        action="repair_clip_subtitles",
        summary=f"修复切片 {clip_id} 字幕 {len(changed_ids)} 句；重渲切片={'是' if render_clip else '否'}，全片重烧={'是' if render_full_video else '否'}",
        details={
            "clip_id": clip_id,
            "changed_utterance_ids": changed_ids,
            "bi_srt": str(bi_srt),
            "cn_srt": str(cn_srt) if rebuild_full_subtitle else "",
            "en_srt": str(en_srt) if rebuild_full_subtitle else "",
            "ass": str(ass_path),
            "clip_output": str(clip_output) if clip_output else "",
            "full_hardsub": str(full_output) if full_output else "",
        },
    )
    return SubtitleRepairOutcome(
        changed_utterance_ids=changed_ids,
        bi_srt_path=bi_srt,
        cn_srt_path=cn_srt,
        en_srt_path=en_srt,
        ass_path=ass_path,
        clip_output_path=clip_output,
        full_hardsub_path=full_output,
    )
