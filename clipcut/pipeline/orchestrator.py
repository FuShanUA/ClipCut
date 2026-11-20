from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from ..adapters.ingest import ingest_source
from ..config import load_editorial_config
from ..llm import parse_provider_model
from ..models import ProjectManifest, StageStatus, utc_now_iso
from ..receipts import ProjectStore
from ..render.renderer import run_render_stage
from .analyze import run_analyze_stage
from .signals import run_signals_stage
from .verify_and_plan import plan_remix_from_clips, run_verify_and_plan_stage

ALL_STAGES = ["ingest", "signals", "analyze", "verify_and_plan", "render"]


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
            manifest = stage_runners[st_name](manifest)
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
    config_path: Path | None = None,
    text_model: str | None = None,
    multimodal_model: str | None = None,
) -> ProjectManifest:
    cfg = _apply_model_overrides(load_editorial_config(config_path), text_model, multimodal_model)
    store = ProjectStore(work_dir)
    manifest = store.load()
    run_id = f"run_{uuid.uuid4().hex[:8]}"

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
        "render": lambda m: run_render_stage(m, store, cfg, run_id, force=force_render),
    }

    for st_name in stages:
        if st_name == "ingest" or st_name not in stage_runners:
            continue
        stage_key = "verify_and_plan" if st_name == "replan_remix" else st_name
        _mark_stage(manifest, store, stage_key, "running", "执行中...")
        t0 = time.monotonic()
        try:
            manifest = stage_runners[st_name](manifest)
            elapsed = int((time.monotonic() - t0) * 1000)
            _mark_stage(manifest, store, stage_key, "completed", "完成", elapsed)
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
