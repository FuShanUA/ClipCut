from __future__ import annotations

from datetime import datetime
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from ..adapters.ingest import ingest_source, resolve_work_dir, sha256_file
from ..cancel import JobCancelToken, JobCancelledError, check_cancelled, use_cancel_token
from ..config import load_editorial_config
from ..llm import parse_provider_model
from ..models import ProjectManifest, StageStatus, utc_now_iso
from ..receipts import ProjectStore
from ..render.renderer import run_render_stage
from .analyze import run_analyze_stage
from .content_filter import scan_cut_segments
from .signals import run_signals_stage
from .verify_and_plan import plan_remix_from_clips, run_verify_and_plan_stage

ALL_STAGES = ["ingest", "signals", "scan_cuts", "analyze", "verify_and_plan", "render"]

DELIVERABLE_DOC_NAMES = {
    "clips_index.md",
    "clips_detail.md",
    "remix_script.md",
    "baocut_export_plan.json",
}
RENDERED_VIDEO_EXTS = {".mp4", ".mov", ".mkv"}
RENDERED_COVER_EXTS = {".jpg", ".jpeg", ".png"}


def find_existing_plan(
    video_path: str | Path,
    *,
    subtitle_path: str | Path | None = None,
    expected_title: str | None = None,
) -> ProjectManifest | None:
    """Return a completed planning manifest for this exact media and subtitle."""
    from ..receipts import ProjectStore

    vpath = Path(video_path).expanduser().resolve()
    work_dir = resolve_work_dir(vpath)
    manifest_path = work_dir / "project.json"
    if not manifest_path.exists():
        return None
    store = ProjectStore(work_dir)
    try:
        manifest = store.load()
    except Exception:
        return None

    if Path(manifest.source.render_video).expanduser().resolve() != vpath:
        return None
    if expected_title and manifest.metadata.original_title != expected_title.strip():
        return None

    required_stages = ("signals", "analyze", "verify_and_plan")
    for stage_name in required_stages:
        status = manifest.stages.get(stage_name)
        if not status or status.status != "completed":
            return None
    if not manifest.topic_map:
        return None
    if not (manifest.clips or manifest.remix.items):
        return None

    manifest_subtitle = Path(manifest.source.subtitle_path or "").expanduser().resolve()
    if not manifest_subtitle.exists():
        return None

    if not manifest.source.subtitle_sha256:
        # Projects created before the hash field was introduced can be migrated
        # only when the subtitle was not modified after analyze completed.
        planned_at = manifest.stages["analyze"].finished_at or manifest.updated_at
        try:
            planned_timestamp = datetime.fromisoformat(planned_at).timestamp()
        except (TypeError, ValueError):
            return None
        if manifest_subtitle.stat().st_mtime > planned_timestamp:
            return None
        manifest.source.subtitle_sha256 = sha256_file(manifest_subtitle)
        store.save(manifest)
    elif sha256_file(manifest_subtitle) != manifest.source.subtitle_sha256:
        return None

    if subtitle_path:
        current_subtitle = Path(subtitle_path).expanduser().resolve()
        if manifest_subtitle != current_subtitle:
            return None
    return manifest


def _apply_model_overrides(
    cfg: dict[str, Any],
    text_model: str | None = None,
    multimodal_model: str | None = None,
) -> dict[str, Any]:
    m_cfg = cfg.setdefault("models", {})
    if text_model:
        prov, mod = parse_provider_model(text_model, m_cfg.get("text_provider", "dashscope"))
        m_cfg["text_provider"] = prov
        m_cfg["text_model"] = mod
        m_cfg["remix_model"] = mod
    if multimodal_model:
        prov, mod = parse_provider_model(multimodal_model, m_cfg.get("multimodal_provider", "dashscope"))
        m_cfg["multimodal_provider"] = prov
        m_cfg["multimodal_model"] = mod
    return cfg


def _mark_stage(manifest: ProjectManifest, store: ProjectStore, stage: str, status: str, message: str = "", elapsed_ms: int = 0, error: str | None = None) -> None:
    st = manifest.stages.get(stage, StageStatus())
    st.status = status  # type: ignore[arg-type]
    if status == "running":
        st.started_at = utc_now_iso()
        st.error = None
    elif status in ("completed", "error"):
        st.finished_at = utc_now_iso()
        st.elapsed_ms = elapsed_ms
        st.error = error
    st.message = message
    manifest.stages[stage] = st
    store.save(manifest)


def run_pipeline(
    *,
    video_path: str | Path,
    subtitle_path: str | Path | None = None,
    info_json_path: str | Path | None = None,
    source_url: str | None = None,
    manual_title: str | None = None,
    manual_description: str | None = None,
    stages: list[str] | None = None,
    config_path: Path | None = None,
    force_render: bool = False,
    fetch_online: bool = True,
    text_model: str | None = None,
    multimodal_model: str | None = None,
) -> tuple[ProjectManifest, ProjectStore]:
    cfg = _apply_model_overrides(load_editorial_config(config_path), text_model, multimodal_model)
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    target_stages = stages or ALL_STAGES

    # Stage 1: ingest (always needed to locate/load manifest)
    t_ing = time.monotonic()
    manifest, store = ingest_source(
        video_path=video_path,
        subtitle_path=subtitle_path,
        info_json_path=info_json_path,
        source_url=source_url,
        manual_title=manual_title,
        manual_description=manual_description,
        fetch_online=fetch_online,
        run_id=run_id,
    )
    _mark_stage(
        manifest,
        store,
        "ingest",
        "completed",
        f"字幕 {len(manifest.utterances)} 句 / 章节 {len(manifest.metadata.chapters)} 个",
        int((time.monotonic() - t_ing) * 1000),
    )

    stage_runners: dict[str, Any] = {
        "signals": lambda m: run_signals_stage(m, store, run_id),
        "scan_cuts": lambda m: scan_cut_segments(m, store, cfg, run_id, scan_ads=True, scan_political=True),
        "analyze": lambda m: run_analyze_stage(m, store, cfg, run_id),
        "verify_and_plan": lambda m: run_verify_and_plan_stage(m, store, cfg, run_id),
        "render": lambda m: run_render_stage(m, store, cfg, run_id, force=force_render),
    }

    for st_name in ALL_STAGES[1:]:
        if st_name not in target_stages:
            continue
        _mark_stage(manifest, store, st_name, "running", "执行中...")
        t0 = time.monotonic()
        try:
            next_manifest = stage_runners[st_name](manifest)
            if next_manifest is not None:
                manifest = next_manifest
            elapsed = int((time.monotonic() - t0) * 1000)
            _mark_stage(manifest, store, st_name, "completed", "完成", elapsed)
        except Exception as exc:
            elapsed = int((time.monotonic() - t0) * 1000)
            _mark_stage(manifest, store, st_name, "error", str(exc)[:180], elapsed, error=str(exc))
            store.append_receipt(
                run_id=run_id,
                stage=st_name,
                actor="system",
                action="stage_failed",
                elapsed_ms=elapsed,
                summary=f"阶段 {st_name} 执行失败",
                error=str(exc),
            )
            raise

    return manifest, store


def run_existing_project_stages(
    work_dir: Path,
    stages: list[str],
    *,
    force_render: bool = False,
    render_targets: list[str] | None = None,
    config_path: Path | None = None,
    text_model: str | None = None,
    multimodal_model: str | None = None,
    cancel_token: JobCancelToken | None = None,
) -> ProjectManifest:
    cfg = _apply_model_overrides(load_editorial_config(config_path), text_model, multimodal_model)
    store = ProjectStore(work_dir)
    manifest = store.load()
    run_id = f"run_{uuid.uuid4().hex[:8]}"

    with use_cancel_token(cancel_token):
        check_cancelled()
        if "ingest" in stages:
            manifest, store = ingest_source(
                video_path=manifest.source.render_video,
                subtitle_path=manifest.source.subtitle_path,
                source_url=manifest.metadata.webpage_url,
                run_id=run_id,
            )
            _mark_stage(manifest, store, "ingest", "completed", "重新载入完成")

        stage_runners: dict[str, Any] = {
            "signals": lambda m: run_signals_stage(m, store, run_id),
            "analyze": lambda m: run_analyze_stage(m, store, cfg, run_id),
            "verify_and_plan": lambda m: run_verify_and_plan_stage(m, store, cfg, run_id),
            "replan_remix": lambda m: plan_remix_from_clips(m, store, cfg, run_id),
            "scan_cuts": lambda m: scan_cut_segments(m, store, cfg, run_id, scan_ads=True, scan_political=True),
            "render": lambda m: run_render_stage(
                m, store, cfg, run_id, force=force_render, render_targets=render_targets
            ),
        }

        for st_name in stages:
            check_cancelled()
            if st_name == "ingest" or st_name not in stage_runners:
                continue
            stage_key = "verify_and_plan" if st_name == "replan_remix" else st_name
            _mark_stage(manifest, store, stage_key, "running", "执行中...")
            t0 = time.monotonic()
            try:
                next_manifest = stage_runners[st_name](manifest)
                if next_manifest is not None:
                    manifest = next_manifest
                check_cancelled()
                elapsed = int((time.monotonic() - t0) * 1000)
                _mark_stage(manifest, store, stage_key, "completed", "完成", elapsed)
            except JobCancelledError:
                elapsed = int((time.monotonic() - t0) * 1000)
                _mark_stage(manifest, store, stage_key, "idle", "已手动中止", elapsed, error=None)
                store.log(f"[pipeline] 阶段 {st_name} 已手动中止")
                store.append_receipt(
                    run_id=run_id,
                    stage=stage_key,
                    actor="human",
                    action="stage_cancelled",
                    elapsed_ms=elapsed,
                    summary=f"阶段 {st_name} 已手动中止",
                )
                raise
            except Exception as exc:
                elapsed = int((time.monotonic() - t0) * 1000)
                _mark_stage(manifest, store, stage_key, "error", str(exc)[:180], elapsed, error=str(exc))
                store.append_receipt(
                    run_id=run_id,
                    stage=stage_key,
                    actor="system",
                    action="stage_failed",
                    elapsed_ms=elapsed,
                    summary=f"阶段 {st_name} 执行失败",
                    error=str(exc),
                )
                raise
    return manifest


def _resolve_safe_output_dir(manifest: ProjectManifest) -> tuple[bool, Path, set[Path]]:
    """Verify output_dir is a dedicated directory distinct from the source video parent."""
    raw_out = (manifest.source.output_dir or "").strip()
    if not raw_out:
        return False, Path("."), set()
    out_dir = Path(raw_out).expanduser().resolve()
    protected: set[Path] = set()
    for raw_p in (
        manifest.source.render_video,
        manifest.source.analysis_video,
        manifest.source.subtitle_path or "",
    ):
        if raw_p:
            try:
                protected.add(Path(raw_p).expanduser().resolve())
            except Exception:
                pass
    try:
        render_parent = Path(manifest.source.render_video).expanduser().resolve().parent
    except Exception:
        render_parent = None
    safe = render_parent is None or out_dir != render_parent
    return safe, out_dir, protected


def inspect_rendered_assets(manifest: ProjectManifest) -> dict[str, Any]:
    """Inspect rendered videos, covers, and deliverable docs before re-planning strategy."""
    safe, out_dir, protected = _resolve_safe_output_dir(manifest)
    video_files: list[str] = []
    cover_files: list[str] = []
    doc_files: list[str] = []
    total_size_bytes = 0

    if safe and out_dir.exists() and out_dir.is_dir():
        for entry in sorted(out_dir.iterdir(), key=lambda p: p.name):
            if not entry.is_file():
                continue
            try:
                resolved = entry.resolve()
            except Exception:
                resolved = entry
            if resolved in protected:
                continue
            ext = entry.suffix.lower()
            if ext in RENDERED_VIDEO_EXTS:
                video_files.append(entry.name)
                total_size_bytes += entry.stat().st_size
            elif ext in RENDERED_COVER_EXTS:
                cover_files.append(entry.name)
                total_size_bytes += entry.stat().st_size
            elif entry.name in DELIVERABLE_DOC_NAMES:
                doc_files.append(entry.name)
                total_size_bytes += entry.stat().st_size

    rendered_clips_count = sum(1 for c in manifest.clips if c.output_file)
    remix_rendered = bool(manifest.remix.output_file)
    compilation_rendered = bool(manifest.compilation.output_file)
    clean_cut_rendered = bool(manifest.clean_cut.output_file)
    has_prior_stage = any(
        st.status != "idle" or bool(st.finished_at) or bool(st.message)
        for name, st in manifest.stages.items()
        if name != "ingest"
    )
    has_prior_analysis = bool(
        manifest.clips
        or manifest.topic_map
        or manifest.candidates
        or manifest.remix.items
        or manifest.cut_segments
        or manifest.clean_cut.scanned_at
        or has_prior_stage
    )
    total_files = len(video_files) + len(cover_files) + len(doc_files)
    has_rendered = bool(
        video_files
        or cover_files
        or rendered_clips_count > 0
        or remix_rendered
        or compilation_rendered
        or clean_cut_rendered
    )
    has_strategy = bool(has_prior_analysis or has_rendered)
    return {
        "has_strategy": has_strategy,
        "has_prior_analysis": has_prior_analysis,
        "has_rendered": has_rendered,
        "output_dir": str(out_dir) if manifest.source.output_dir else "",
        "video_files": video_files,
        "cover_files": cover_files,
        "doc_files": doc_files,
        "total_files": total_files,
        "total_size_bytes": total_size_bytes,
        "rendered_clips_count": rendered_clips_count,
        "remix_rendered": remix_rendered,
        "compilation_rendered": compilation_rendered,
        "clean_cut_rendered": clean_cut_rendered,
    }


def reset_project_strategy(
    manifest: ProjectManifest,
    store: ProjectStore,
    *,
    delete_rendered_files: bool = False,
    force_signals: bool = False,
    run_id: str = "",
) -> dict[str, Any]:
    """Snapshot manifest for undo, optionally delete rendered outputs, and reset planning fields."""
    store.save_undo_snapshot(manifest, "重新制定切片策略前备份")

    deleted_files: list[str] = []
    freed_bytes = 0
    if delete_rendered_files:
        safe, out_dir, protected = _resolve_safe_output_dir(manifest)
        if safe and out_dir.exists() and out_dir.is_dir():
            allowed_exts = RENDERED_VIDEO_EXTS | RENDERED_COVER_EXTS
            for entry in sorted(out_dir.iterdir(), key=lambda p: p.name):
                if not entry.is_file():
                    continue
                try:
                    resolved = entry.resolve()
                except Exception:
                    resolved = entry
                if resolved in protected:
                    continue
                if entry.suffix.lower() in allowed_exts or entry.name in DELIVERABLE_DOC_NAMES:
                    size_b = entry.stat().st_size
                    entry.unlink(missing_ok=True)
                    deleted_files.append(entry.name)
                    freed_bytes += size_b

        for tmp_name in ("cover_candidates", "remix_tmp", "compilation_tmp", "verify_previews"):
            tmp_dir = store.work_dir / tmp_name
            if tmp_dir.exists() and tmp_dir.is_dir():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    manifest.primary_topics = []
    manifest.metadata.chapter_audits = []
    manifest.topic_map = []
    manifest.suppressed_hot_topics = []
    manifest.candidates = []
    manifest.clips = []
    manifest.remix_only_pool = []

    manifest.remix.items = []
    manifest.remix.source_clip_ids = None
    manifest.remix.manual_script_locked = False
    manifest.remix.title = "高光混剪集锦"
    manifest.remix.cover_title = ""
    manifest.remix.editorial_thesis = ""
    manifest.remix.planned_source_signature = ""
    manifest.remix.render_hash = ""
    manifest.remix.last_rendered_hash = ""
    manifest.remix.output_file = None
    manifest.remix.cover_file = None
    manifest.remix.cover_hash = ""
    manifest.remix.rendered_filter_tag = ""
    manifest.remix.publish_records = []

    manifest.compilation.clip_ids = None
    manifest.compilation.render_hash = ""
    manifest.compilation.last_rendered_hash = ""
    manifest.compilation.output_file = None
    manifest.compilation.cover_file = None
    manifest.compilation.cover_hash = ""
    manifest.compilation.rendered_filter_tag = ""
    manifest.compilation.publish_records = []

    manifest.cut_segments = []
    manifest.clean_cut.enabled = False
    manifest.clean_cut.title = ""
    manifest.clean_cut.cover_title = ""
    manifest.clean_cut.cut_ads = False
    manifest.clean_cut.cut_political_sensitive = False
    manifest.clean_cut.scanned_at = None
    manifest.clean_cut.render_hash = ""
    if delete_rendered_files:
        manifest.clean_cut.output_file = None
        manifest.clean_cut.cover_file = None
        manifest.clean_cut.cover_hash = ""
        manifest.clean_cut.last_rendered_hash = ""
        manifest.clean_cut.rendered_filter_tag = ""
        manifest.clean_cut.publish_records = []
        manifest.stages["render"] = StageStatus()

    if force_signals:
        (store.work_dir / "audio_envelope.npz").unlink(missing_ok=True)
        manifest.stages["signals"] = StageStatus()

    for st_name in ("scan_cuts", "analyze", "verify_and_plan"):
        manifest.stages[st_name] = StageStatus()

    store.save(manifest)
    summary = (
        f"重做切片策略：已清理 {len(deleted_files)} 个已渲染文件（释放 {freed_bytes / (1024 * 1024):.1f} MB）并清空旧策略"
        if delete_rendered_files
        else "重做切片策略：保留磁盘已有成片文件，已清空旧策略并准备重新分析"
    )
    store.append_receipt(
        run_id=run_id or f"replan_{uuid.uuid4().hex[:6]}",
        stage="replan",
        actor="human",
        action="replan_strategy",
        summary=summary,
        details={
            "delete_rendered_files": delete_rendered_files,
            "deleted_files": deleted_files,
            "deleted_count": len(deleted_files),
            "freed_bytes": freed_bytes,
        },
    )
    return {
        "deleted_files": deleted_files,
        "deleted_count": len(deleted_files),
        "freed_bytes": freed_bytes,
    }
