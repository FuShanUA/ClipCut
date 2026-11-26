from __future__ import annotations

from datetime import datetime, timezone
import copy
import json
import mimetypes
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..adapters.ingest import ingest_source, resolve_work_dir, scan_input_path
from ..cancel import JobCancelToken, JobCancelledError, use_cancel_token
from ..config import PROJECT_ROOT, load_editorial_config, load_editorial_raw, save_editorial_config, save_editorial_raw
from ..llm import MODEL_CATALOG, parse_provider_model
from ..models import CutSegment, ProjectManifest, PublishRecord, RemixItem, SubtitleRepairItem, utc_now_iso
from ..pipeline.content_filter import (
    DEFAULT_POLITICAL_PROFILES,
    VALID_POLITICAL_ACTIONS,
    VALID_SEGMENT_ACTIONS,
    VALID_SEVERITIES,
    compute_clean_cut_render_hash,
    extract_precise_terms_for_segments,
    get_active_cut_segments,
    get_active_cut_spans,
    get_active_delete_spans,
    get_overlapping_mask_payload,
    get_political_profiles,
    has_clean_source_video,
    scan_cut_segments,
    segment_matches_sensitivity,
    sync_all_render_hashes,
)
from ..pipeline.orchestrator import (
    ALL_STAGES,
    inspect_rendered_assets,
    reset_project_strategy,
    run_existing_project_stages,
)
from ..pipeline.verify_and_plan import (
    GENERIC_COMPILATION_TITLES,
    _has_cjk,
    _load_audio_envelope,
    build_default_clean_cut_title,
    build_default_compilation_title,
    compute_clip_render_hash,
    compute_compilation_render_hash,
    compute_remix_render_hash,
    compute_remix_source_signature,
    distill_clean_cut_title,
    distill_compilation_title,
    distill_target_publish_description,
    get_compilation_clips,
    is_untranslated_clean_cut_title,
    select_curated_top_clips,
    snap_bounds_with_acoustic_guard,
)
from ..publish import (
    get_platforms_status,
    launch_platform_login,
    publish_batch_targets,
    publish_to_platforms,
    reset_publish_records,
)
from ..publish.publishers import _default_target_metadata
from ..receipts import ProjectStore
from ..render.renderer import (
    refresh_single_cover,
    render_clean_cut_and_cover,
    render_compilation_and_cover,
    render_remix_and_cover,
    render_single_clip_and_cover,
)
from ..subtitle_repair import apply_subtitle_repairs

REGISTRY_PATH = PROJECT_ROOT / ".clipcut_registry.json"

app = FastAPI(title="ClipCut Studio API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _no_cache_html(request: Request, call_next):
    response = await call_next(request)
    if request.url.path in ("/", "/index.html"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response

# Background job state per project_id
_ACTIVE_JOBS: dict[str, dict[str, Any]] = {}
_JOB_CANCEL_TOKENS: dict[str, JobCancelToken] = {}
_ACTIVE_PUBLISH_JOBS: dict[str, dict[str, dict[str, Any]]] = {}
_PUBLISH_CANCEL_TOKENS: dict[str, dict[str, JobCancelToken]] = {}
_JOBS_LOCK = threading.Lock()
_SERVER_STARTED_AT = utc_now_iso()


def _list_running_publish_jobs_unlocked(project_id: str) -> list[dict[str, Any]]:
    proj_jobs = _ACTIVE_PUBLISH_JOBS.get(project_id) or {}
    return [copy.deepcopy(entry) for entry in proj_jobs.values() if entry.get("running")]


def _list_running_publish_jobs(project_id: str) -> list[dict[str, Any]]:
    with _JOBS_LOCK:
        return _list_running_publish_jobs_unlocked(project_id)


def _has_running_publish_for_target(project_id: str, target: str) -> bool:
    with _JOBS_LOCK:
        for entry in (_ACTIVE_PUBLISH_JOBS.get(project_id) or {}).values():
            if entry.get("running") and target in (entry.get("targets") or []):
                return True
    return False


def _start_job_entry(
    project_id: str,
    job_id: str,
    stages: list[str],
    *,
    cancellable: bool = True,
) -> JobCancelToken | None:
    token = JobCancelToken() if cancellable else None
    with _JOBS_LOCK:
        if _ACTIVE_JOBS.get(project_id, {}).get("running"):
            raise HTTPException(status_code=409, detail="该项目已有流水线任务正在运行中")
        if _list_running_publish_jobs_unlocked(project_id):
            raise HTTPException(status_code=409, detail="该项目有发布任务正在运行中，请先等待完成或中止后再执行规划/渲染")
        _ACTIVE_JOBS[project_id] = {
            "job_id": job_id,
            "running": True,
            "cancelling": False,
            "cancelled": False,
            "stages": list(stages),
            "error": None,
        }
        if token is not None:
            _JOB_CANCEL_TOKENS[project_id] = token
        else:
            _JOB_CANCEL_TOKENS.pop(project_id, None)
    return token


def _finish_job_entry(
    project_id: str,
    job_id: str,
    token: JobCancelToken | None,
    *,
    cancelled: bool = False,
    error: str | None = None,
) -> None:
    with _JOBS_LOCK:
        cur = _ACTIVE_JOBS.get(project_id)
        if cur and cur.get("job_id") == job_id:
            cur["running"] = False
            cur["cancelling"] = False
            cur["cancelled"] = cancelled
            cur["error"] = error
        if token is not None and _JOB_CANCEL_TOKENS.get(project_id) is token:
            _JOB_CANCEL_TOKENS.pop(project_id, None)


def _start_publish_job_entry(
    project_id: str,
    job_id: str,
    *,
    kind: str,
    targets: list[str],
    platforms: list[str],
    collection: str = "",
) -> JobCancelToken:
    token = JobCancelToken()
    target_set = set(targets)
    with _JOBS_LOCK:
        if _ACTIVE_JOBS.get(project_id, {}).get("running"):
            raise HTTPException(status_code=409, detail="该项目有渲染或规划任务正在运行，请等待完成或先中止")
        for existing in (_ACTIVE_PUBLISH_JOBS.get(project_id) or {}).values():
            if existing.get("running") and target_set.intersection(existing.get("targets") or []):
                raise HTTPException(status_code=409, detail="目标条目正在发布中，请等待完成或先中止")
        _ACTIVE_PUBLISH_JOBS.setdefault(project_id, {})[job_id] = {
            "job_id": job_id,
            "kind": kind,
            "running": True,
            "cancelling": False,
            "cancelled": False,
            "targets": list(targets),
            "current_target": targets[0] if targets else None,
            "completed_targets": [],
            "failed_targets": [],
            "skipped_targets": [],
            "platforms": list(platforms),
            "collection": collection,
            "error": None,
            "started_at": utc_now_iso(),
            "finished_at": None,
        }
        _PUBLISH_CANCEL_TOKENS.setdefault(project_id, {})[job_id] = token
    return token


def _update_publish_job_entry(project_id: str, job_id: str, **fields: Any) -> None:
    with _JOBS_LOCK:
        cur = (_ACTIVE_PUBLISH_JOBS.get(project_id) or {}).get(job_id)
        if cur is None:
            return
        cur.update(fields)


def _finish_publish_job_entry(
    project_id: str,
    job_id: str,
    token: JobCancelToken | None,
    *,
    cancelled: bool = False,
    error: str | None = None,
) -> None:
    with _JOBS_LOCK:
        proj_jobs = _ACTIVE_PUBLISH_JOBS.get(project_id)
        if proj_jobs and job_id in proj_jobs:
            cur = proj_jobs[job_id]
            cur["running"] = False
            cur["cancelling"] = False
            cur["cancelled"] = cancelled
            cur["current_target"] = None
            cur["error"] = error
            cur["finished_at"] = utc_now_iso()
            # Prune finished entries while keeping any still-running jobs
            for k in [k for k, v in proj_jobs.items() if not v.get("running")]:
                proj_jobs.pop(k, None)
            if not proj_jobs:
                _ACTIVE_PUBLISH_JOBS.pop(project_id, None)
        proj_tokens = _PUBLISH_CANCEL_TOKENS.get(project_id)
        if proj_tokens and token is not None and proj_tokens.get(job_id) is token:
            proj_tokens.pop(job_id, None)
            if not proj_tokens:
                _PUBLISH_CANCEL_TOKENS.pop(project_id, None)


def _is_before_server_start(ts: str) -> bool:
    if not ts:
        return True
    try:
        t_rec = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        t_srv = datetime.fromisoformat(_SERVER_STARTED_AT.replace("Z", "+00:00"))
        if t_rec.tzinfo is None:
            t_rec = t_rec.replace(tzinfo=timezone.utc)
        if t_srv.tzinfo is None:
            t_srv = t_srv.replace(tzinfo=timezone.utc)
        return t_rec < t_srv
    except Exception:
        return ts < _SERVER_STARTED_AT


def _has_stale_running_record(manifest: ProjectManifest) -> bool:
    all_lists = [
        manifest.remix.publish_records,
        manifest.compilation.publish_records,
        manifest.clean_cut.publish_records,
        *[c.publish_records for c in manifest.clips],
    ]
    has_stale_pub = any(
        rec.status == "running" and _is_before_server_start(rec.updated_at)
        for recs in all_lists
        for rec in recs
    )
    if has_stale_pub:
        return True
    return any(
        st.status == "running" and _is_before_server_start(st.started_at or manifest.updated_at)
        for st in manifest.stages.values()
    )


def _cleanup_stale_running_publish_records(store: ProjectStore, job_running: bool) -> ProjectManifest:
    manifest = store.load()
    if job_running or not _has_stale_running_record(manifest):
        return manifest

    now_iso = utc_now_iso()

    def _mutate(m: ProjectManifest) -> None:
        all_lists = [
            m.remix.publish_records,
            m.compilation.publish_records,
            m.clean_cut.publish_records,
            *[c.publish_records for c in m.clips],
        ]
        for recs in all_lists:
            for rec in recs:
                if rec.status == "running" and _is_before_server_start(rec.updated_at):
                    rec.status = "error"
                    rec.progress_pct = None
                    rec.progress_stage = ""
                    rec.message = "上传因服务重启中断"
                    rec.error = "上次上传未完成（服务已重启），请重新发布"
                    rec.error_code = "interrupted"
                    rec.solution = "点击「复位状态」清除旧状态后重新发起上传。"
                    rec.updated_at = now_iso
        for st in m.stages.values():
            if st.status == "running" and _is_before_server_start(st.started_at or m.updated_at):
                st.status = "idle"
                st.message = "上次执行因服务重启中断"
                st.error = None

    return store.update_manifest(_mutate)


def _load_registry() -> dict[str, dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_registry(reg: dict[str, dict[str, Any]]) -> None:
    REGISTRY_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def _register_project(project_id: str, work_dir: Path, title: str, video_path: str) -> None:
    reg = _load_registry()
    reg[project_id] = {
        "project_id": project_id,
        "work_dir": str(work_dir),
        "title": title,
        "video_path": video_path,
    }
    _save_registry(reg)


def _get_store(project_id: str) -> ProjectStore:
    reg = _load_registry()
    entry = reg.get(project_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"未找到项目: {project_id}")
    work_dir = Path(entry["work_dir"])
    store = ProjectStore(work_dir)
    if not store.exists():
        raise HTTPException(status_code=404, detail=f"项目状态文件缺失: {store.manifest_path}")
    return store


# ── Request Schemas ──────────────────────────────────────────────────────────

class ScanRequest(BaseModel):
    path: str


class PickPathRequest(BaseModel):
    kind: str = "file"  # file | folder


class OpenProjectRequest(BaseModel):
    video_path: str
    subtitle_path: str | None = None
    info_json_path: str | None = None
    source_url: str | None = None
    manual_title: str | None = None
    manual_description: str | None = None
    fetch_online: bool = True


class RunStagesRequest(BaseModel):
    stages: list[str] = Field(default_factory=lambda: ["signals", "scan_cuts", "analyze", "verify_and_plan"])
    force_render: bool = False
    render_targets: list[str] | None = None
    text_model: str | None = None
    multimodal_model: str | None = None


class ReplanStrategyRequest(BaseModel):
    delete_rendered_files: bool = False
    force_signals: bool = False
    text_model: str | None = None
    multimodal_model: str | None = None


class BatchClipsRequest(BaseModel):
    action: str  # "enable_all" | "disable_all" | "top_n"
    n: int = 5


class UpdateModelsRequest(BaseModel):
    text_provider: str | None = None
    text_model: str | None = None
    remix_model: str | None = None
    multimodal_provider: str | None = None
    multimodal_model: str | None = None


class PatchClipRequest(BaseModel):
    enabled: bool | None = None
    title: str | None = None
    cover_title: str | None = None
    title_en: str | None = None
    start_utt: int | None = None
    end_utt: int | None = None


class PatchRemixItem(BaseModel):
    remix_id: str
    order: int | None = None
    enabled: bool | None = None
    group_label: str | None = None
    card_title: str | None = None
    start_utt: int | None = None
    end_utt: int | None = None


class PatchRemixRequest(BaseModel):
    enabled: bool | None = None
    title: str | None = None
    cover_title: str | None = None
    card_mode: str | None = None
    card_duration_s: float | None = None
    enable_sfx: bool | None = None
    enable_bgm: bool | None = None
    bgm_volume: float | None = None
    sfx_volume: float | None = None
    source_mode: str | None = None
    custom_prompt: str | None = None
    source_clip_ids: list[str] | None = None
    clear_source_clip_ids: bool = False
    manual_script_locked: bool | None = None
    add_clip_id: str | None = None
    remove_remix_id: str | None = None
    items: list[PatchRemixItem] | None = None


class PatchCompilationRequest(BaseModel):
    enabled: bool | None = None
    title: str | None = None
    cover_title: str | None = None
    clip_ids: list[str] | None = None
    clear_clip_ids: bool = False
    card_duration_s: float | None = None
    card_fade_s: float | None = None


class ScanCleanCutRequest(BaseModel):
    cut_ads: bool | None = None
    cut_political_sensitive: bool | None = None
    sensitivity: str | None = None


class CutSegmentPatch(BaseModel):
    cut_id: str
    enabled: bool | None = None
    kind: str | None = None
    severity: str | None = None
    action: str | None = None
    sensitive_terms_zh: list[str] | None = None
    sensitive_terms_en: list[str] | None = None
    precise_terms_zh: list[str] | None = None
    precise_terms_en: list[str] | None = None
    start_utt: int | None = None
    end_utt: int | None = None
    start_s: float | None = None
    end_s: float | None = None
    reason: str | None = None


class ExtractPreciseTermsRequest(BaseModel):
    force: bool = False
    cut_ids: list[str] | None = None


class AddCutSegmentPayload(BaseModel):
    kind: str = "political_sensitive"
    start_s: float
    end_s: float
    reason: str = "人工手动屏蔽区间"


class PatchCleanCutRequest(BaseModel):
    enabled: bool | None = None
    title: str | None = None
    cover_title: str | None = None
    cut_ads: bool | None = None
    cut_political_sensitive: bool | None = None
    sensitivity: str | None = None
    political_action: str | None = None
    apply_to_clips: bool | None = None
    cut_mode: str | None = None
    segments: list[CutSegmentPatch] | None = None
    add_segment: AddCutSegmentPayload | None = None
    delete_ids: list[str] | None = None


class PatchMetadataRequest(BaseModel):
    channel_zh: str | None = None
    guest_zh: str | None = None
    published_date: str | None = None


class RefreshCoverRequest(BaseModel):
    target: str = "remix"


class SubtitleRepairRequest(BaseModel):
    repairs: list[SubtitleRepairItem]
    rebuild_full_subtitle: bool = True
    render_clip: bool = True
    render_full_video: bool = False


class ReorderClipsRequest(BaseModel):
    clip_ids: list[str]


class CancelJobRequest(BaseModel):
    job_id: str | None = None
    target: str | None = None


class PublishRequest(BaseModel):
    target: str
    platforms: list[str]
    title: str | None = None
    cover_title: str | None = None
    desc: str | None = None
    tags: list[str] | None = None
    is_public: bool = False
    collection: str | None = None
    tid: int | None = None
    ui_schema: int | None = None


class SuggestPublishDescRequest(BaseModel):
    target: str


class BatchPublishRequest(BaseModel):
    targets: list[str] | None = None
    platforms: list[str]
    is_public: bool = False
    collection: str | None = None
    skip_already_published: bool = True
    tid: int | None = None
    extra_tags: list[str] | None = None


class ResetPublishRequest(BaseModel):
    target: str
    platform: str | None = None
    only_errors: bool | None = None


class UpdateEditorialRequest(BaseModel):
    raw_yaml: str


class PoliticalRuleSchema(BaseModel):
    id: str = ""
    name: str
    severity: str = "medium"
    description: str = ""


class PoliticalProfileSchema(BaseModel):
    label: str
    description: str = ""
    trigger_severities: list[str]
    rules: list[PoliticalRuleSchema] = Field(default_factory=list)


class UpdateContentFilterConfigRequest(BaseModel):
    default_sensitivity: str | None = None
    default_political_action: str | None = None
    political_profiles: dict[str, PoliticalProfileSchema] | None = None


class ResetContentFilterConfigRequest(BaseModel):
    profile: str | None = None


class UpdateNavigationConfigRequest(BaseModel):
    style: str | None = None
    remix_style: str | None = None
    compilation_style: str | None = None
    screen_layout: str | None = None
    color_scheme: str | None = None
    custom_accent: str | None = None


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "app": "clipcut", "version": "0.1.0"}


@app.post("/api/pick-path")
def pick_local_path(req: PickPathRequest):
    """macOS native Finder picker for selecting a video file or folder."""
    if req.kind == "folder":
        script = 'POSIX path of (choose folder with prompt "选择包含视频的文件夹")'
    else:
        script = 'POSIX path of (choose file with prompt "选择要切片的视频文件")'
    try:
        res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=120)
        if res.returncode != 0:
            return {"cancelled": True, "path": None}
        return {"cancelled": False, "path": res.stdout.strip()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"无法唤起系统选择器: {exc}")


@app.post("/api/scan")
def api_scan(req: ScanRequest):
    try:
        res = scan_input_path(req.path)
        data = res.model_dump()
        data["need_selection"] = len(res.videos) > 1
        return data
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/projects")
def list_projects():
    reg = _load_registry()
    items = []
    for pid, entry in reg.items():
        wd = Path(entry["work_dir"])
        store = ProjectStore(wd)
        if store.exists():
            try:
                m = store.load()
                items.append(
                    {
                        "project_id": pid,
                        "title": m.metadata.original_theme_zh or m.metadata.original_title,
                        "video_path": m.source.render_video,
                        "duration_s": m.source.duration_s,
                        "clip_count": len(m.clips),
                        "updated_at": m.updated_at,
                    }
                )
            except Exception:
                continue
    items.sort(key=lambda x: x["updated_at"], reverse=True)
    return {"projects": items}


@app.post("/api/projects/open")
def open_project(req: OpenProjectRequest):
    try:
        vpath = Path(req.video_path).expanduser().resolve()
        wd = resolve_work_dir(vpath)
        store = ProjectStore(wd)
        if store.exists() and not any([req.subtitle_path, req.info_json_path, req.source_url, req.manual_title, req.manual_description]):
            manifest = store.load()
        else:
            manifest, store = ingest_source(
                video_path=vpath,
                subtitle_path=req.subtitle_path,
                info_json_path=req.info_json_path,
                source_url=req.source_url,
                manual_title=req.manual_title,
                manual_description=req.manual_description,
                fetch_online=req.fetch_online,
                run_id=f"open_{uuid.uuid4().hex[:6]}",
            )
            manifest.stages["ingest"].status = "completed"
            manifest.stages["ingest"].message = f"字幕 {len(manifest.utterances)} 句 / 章节 {len(manifest.metadata.chapters)} 个"
            store.save(manifest)

        _register_project(
            manifest.project_id,
            wd,
            manifest.metadata.original_theme_zh or manifest.metadata.original_title,
            manifest.source.render_video,
        )
        return {"project_id": manifest.project_id, "manifest": manifest.model_dump()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    store = _get_store(project_id)
    with _JOBS_LOCK:
        job = dict(_ACTIVE_JOBS.get(project_id) or {"running": False})
        running_pub_jobs = _list_running_publish_jobs_unlocked(project_id)
    manifest = _cleanup_stale_running_publish_records(
        store,
        bool(job.get("running")) or bool(running_pub_jobs),
    )
    if not bool(job.get("running")) and manifest.clean_cut.enabled:
        cfg = load_editorial_config()
        if not get_active_cut_segments(manifest, cfg):
            manifest.clean_cut.enabled = False
            sync_all_render_hashes(manifest, cfg=cfg)
            store.save(manifest)
    if (
        not bool(job.get("running"))
        and manifest.clips
        and (manifest.compilation.title or "").strip() in GENERIC_COMPILATION_TITLES
    ):
        cand_title = build_default_compilation_title(manifest)
        if cand_title and cand_title not in GENERIC_COMPILATION_TITLES:
            manifest.compilation.title = cand_title
            manifest.compilation.render_hash = compute_compilation_render_hash(manifest)
            store.save(manifest)
    if (
        not bool(job.get("running"))
        and (manifest.cut_segments or manifest.clean_cut.output_file)
        and is_untranslated_clean_cut_title(manifest, manifest.clean_cut.title)
    ):
        cand_cc_title = build_default_clean_cut_title(manifest)
        if cand_cc_title and _has_cjk(cand_cc_title) and cand_cc_title != manifest.clean_cut.title:
            cfg = load_editorial_config()
            cc = manifest.clean_cut
            was_up_to_date = bool(
                cc.output_file and Path(cc.output_file).exists() and cc.render_hash == cc.last_rendered_hash
            )
            cc.title = cand_cc_title
            cc.render_hash = compute_clean_cut_render_hash(manifest, cfg)
            if was_up_to_date:
                cc.last_rendered_hash = cc.render_hash
            store.save(manifest)
            if cc.cover_file and not cc.cover_title:
                refresh_single_cover(manifest, store, cfg, "clean_cut")
                manifest = store.load()
    if not bool(job.get("running")):
        dirty_hashes = False
        if manifest.remix.items:
            exp_rh = compute_remix_render_hash(manifest)
            if exp_rh != manifest.remix.render_hash:
                manifest.remix.render_hash = exp_rh
                dirty_hashes = True
        if manifest.clips:
            exp_ch = compute_compilation_render_hash(manifest)
            if exp_ch != manifest.compilation.render_hash:
                manifest.compilation.render_hash = exp_ch
                dirty_hashes = True
        if dirty_hashes:
            store.save(manifest)
    return {
        "manifest": manifest.model_dump(),
        "job": job,
        "publish_jobs": running_pub_jobs,
        "assets": inspect_rendered_assets(manifest),
        "can_mask": has_clean_source_video(manifest),
    }


@app.post("/api/projects/{project_id}/run")
def run_project_stages(project_id: str, req: RunStagesRequest):
    store = _get_store(project_id)
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    token = _start_job_entry(project_id, job_id, req.stages, cancellable=True)
    assert token is not None

    def _worker():
        try:
            with use_cancel_token(token):
                run_existing_project_stages(
                    store.work_dir,
                    req.stages,
                    force_render=req.force_render,
                    render_targets=req.render_targets,
                    text_model=req.text_model,
                    multimodal_model=req.multimodal_model,
                    cancel_token=token,
                )
            _finish_job_entry(project_id, job_id, token)
        except JobCancelledError:
            _finish_job_entry(project_id, job_id, token, cancelled=True, error=None)
        except Exception as exc:
            _finish_job_entry(project_id, job_id, token, error=str(exc))

    threading.Thread(target=_worker, daemon=True).start()
    return {"started": True, "job_id": job_id, "stages": req.stages}


@app.post("/api/projects/{project_id}/cancel")
def cancel_project_job(project_id: str, req: CancelJobRequest | None = Body(default=None)):
    store = _get_store(project_id)
    _ = store
    req_job_id = (req.job_id or "").strip() if req and req.job_id else ""
    req_target = (req.target or "").strip() if req and req.target else ""

    tokens_to_cancel: list[JobCancelToken] = []
    cancelled_job_ids: list[str] = []

    with _JOBS_LOCK:
        pub_jobs = _ACTIVE_PUBLISH_JOBS.get(project_id) or {}
        pub_tokens = _PUBLISH_CANCEL_TOKENS.get(project_id) or {}

        if req_job_id:
            if req_job_id in pub_jobs and pub_jobs[req_job_id].get("running"):
                tok = pub_tokens.get(req_job_id)
                if tok is None:
                    raise HTTPException(status_code=400, detail="当前发布任务不支持中途取消")
                pub_jobs[req_job_id]["cancelling"] = True
                tokens_to_cancel.append(tok)
                cancelled_job_ids.append(req_job_id)
            else:
                job = _ACTIVE_JOBS.get(project_id)
                if job and job.get("running") and job.get("job_id") == req_job_id:
                    tok = _JOB_CANCEL_TOKENS.get(project_id)
                    if tok is None:
                        raise HTTPException(status_code=400, detail="当前任务类型不支持中途取消")
                    job["cancelling"] = True
                    tokens_to_cancel.append(tok)
                    cancelled_job_ids.append(req_job_id)
        elif req_target:
            for jid, entry in pub_jobs.items():
                if entry.get("running") and req_target in (entry.get("targets") or []):
                    tok = pub_tokens.get(jid)
                    if tok is not None:
                        entry["cancelling"] = True
                        tokens_to_cancel.append(tok)
                        cancelled_job_ids.append(jid)
        else:
            job = _ACTIVE_JOBS.get(project_id)
            if job and job.get("running"):
                tok = _JOB_CANCEL_TOKENS.get(project_id)
                if tok is None:
                    raise HTTPException(status_code=400, detail="当前任务类型不支持中途取消")
                job["cancelling"] = True
                tokens_to_cancel.append(tok)
                cancelled_job_ids.append(str(job.get("job_id") or ""))
            else:
                for jid, entry in pub_jobs.items():
                    if entry.get("running"):
                        tok = pub_tokens.get(jid)
                        if tok is not None:
                            entry["cancelling"] = True
                            tokens_to_cancel.append(tok)
                            cancelled_job_ids.append(jid)

    if not tokens_to_cancel:
        return {"cancelled": False, "message": "当前没有正在运行的对应任务"}

    for tok in tokens_to_cancel:
        tok.cancel()
    return {
        "cancelled": True,
        "job_id": cancelled_job_ids[0],
        "job_ids": cancelled_job_ids,
        "message": "已发送中止信号，正在停止任务...",
    }


@app.get("/api/projects/{project_id}/rendered-assets")
def get_rendered_assets(project_id: str):
    store = _get_store(project_id)
    manifest = store.load()
    return inspect_rendered_assets(manifest)


@app.post("/api/projects/{project_id}/replan-strategy")
def replan_strategy(project_id: str, req: ReplanStrategyRequest):
    store = _get_store(project_id)
    manifest = store.load()
    has_signals = (
        not req.force_signals
        and (store.work_dir / "audio_envelope.npz").exists()
        and manifest.stages.get("signals") is not None
        and manifest.stages["signals"].status == "completed"
    )
    stages = ["scan_cuts", "analyze", "verify_and_plan"] if has_signals else ["signals", "scan_cuts", "analyze", "verify_and_plan"]

    job_id = f"replan_{uuid.uuid4().hex[:8]}"
    token = _start_job_entry(project_id, job_id, stages, cancellable=True)
    assert token is not None

    try:
        cleanup_info = reset_project_strategy(
            manifest,
            store,
            delete_rendered_files=req.delete_rendered_files,
            force_signals=req.force_signals,
            run_id=job_id,
        )
        fresh_manifest = store.load()
    except Exception as exc:
        _finish_job_entry(project_id, job_id, token, error=str(exc))
        raise HTTPException(status_code=500, detail=f"重置策略失败: {exc}")

    def _worker():
        try:
            with use_cancel_token(token):
                run_existing_project_stages(
                    store.work_dir,
                    stages,
                    text_model=req.text_model,
                    multimodal_model=req.multimodal_model,
                    cancel_token=token,
                )
            _finish_job_entry(project_id, job_id, token)
        except JobCancelledError:
            _finish_job_entry(project_id, job_id, token, cancelled=True, error=None)
        except Exception as exc:
            _finish_job_entry(project_id, job_id, token, error=str(exc))

    threading.Thread(target=_worker, daemon=True).start()
    return {
        "started": True,
        "job_id": job_id,
        "stages": stages,
        **cleanup_info,
        "manifest": fresh_manifest.model_dump(),
    }


def _sync_derived_hashes_after_clip_change(manifest: ProjectManifest, *, mark_remix_stale: bool = True) -> None:
    manifest.compilation.render_hash = compute_compilation_render_hash(manifest)
    if mark_remix_stale and not manifest.remix.planned_source_signature:
        manifest.remix.planned_source_signature = "__stale__"
    manifest.remix.render_hash = compute_remix_render_hash(manifest)


def _apply_reordered_clips(manifest: ProjectManifest, ordered_clips: list[Any]) -> None:
    old_sig = compute_remix_source_signature(manifest)
    for idx, c in enumerate(ordered_clips, 1):
        c.index = idx
    manifest.clips = ordered_clips
    new_sig = compute_remix_source_signature(manifest)
    if manifest.remix.items and manifest.remix.planned_source_signature in ("", old_sig):
        clip_pos = {c.clip_id: i for i, c in enumerate(manifest.clips)}
        fallback_pos = len(manifest.clips)
        reordered_items = [
            it
            for _, it in sorted(
                enumerate(manifest.remix.items),
                key=lambda pair: (clip_pos.get(pair[1].source_clip_id or "", fallback_pos), pair[0]),
            )
        ]
        for idx, it in enumerate(reordered_items, 1):
            it.order = idx
        manifest.remix.items = reordered_items
        if manifest.remix.planned_source_signature == old_sig:
            manifest.remix.planned_source_signature = new_sig
    manifest.compilation.render_hash = compute_compilation_render_hash(manifest)
    manifest.remix.render_hash = compute_remix_render_hash(manifest)


@app.post("/api/projects/{project_id}/clips/batch")
def batch_clips(project_id: str, req: BatchClipsRequest):
    store = _get_store(project_id)
    manifest = store.load()
    store.save_undo_snapshot(manifest, f"批量操作切片: {req.action}")

    if req.action == "enable_all":
        for c in manifest.clips:
            c.enabled = True
    elif req.action == "disable_all":
        for c in manifest.clips:
            c.enabled = False
            c.curation_tag = ""
    elif req.action == "top_n":
        curated = select_curated_top_clips(manifest, n=req.n)
        curated_ids = {c.clip_id: tag for c, tag in curated}
        for c in manifest.clips:
            if c.clip_id in curated_ids:
                c.enabled = True
                c.curation_tag = curated_ids[c.clip_id]
            else:
                c.enabled = False
                c.curation_tag = ""
    else:
        raise HTTPException(status_code=400, detail=f"不支持的批量操作: {req.action}")

    _sync_derived_hashes_after_clip_change(manifest, mark_remix_stale=True)
    store.save(manifest)
    enabled_count = sum(1 for c in manifest.clips if c.enabled)
    store.append_receipt(
        run_id=f"human_{uuid.uuid4().hex[:6]}",
        stage="human_edit",
        actor="human",
        action="batch_clips",
        summary=f"批量更新切片状态 [{req.action}]：当前已启用 {enabled_count}/{len(manifest.clips)} 支",
    )
    return {"manifest": manifest.model_dump(), "enabled_count": enabled_count}


@app.post("/api/projects/{project_id}/clips/reorder")
def reorder_clips(project_id: str, req: ReorderClipsRequest):
    store = _get_store(project_id)
    manifest = store.load()
    existing_map = {c.clip_id: c for c in manifest.clips}
    if set(req.clip_ids) != set(existing_map.keys()):
        raise HTTPException(status_code=400, detail="切片列表ID与现有切片不一致")

    store.save_undo_snapshot(manifest, "手动调整切片顺序")
    new_clips = [existing_map[cid] for cid in req.clip_ids]
    _apply_reordered_clips(manifest, new_clips)
    store.save(manifest)
    store.append_receipt(
        run_id=f"human_{uuid.uuid4().hex[:6]}",
        stage="human_edit",
        actor="human",
        action="reorder_clips",
        summary=f"手动重排切片顺序（共 {len(manifest.clips)} 支）",
    )
    return {"manifest": manifest.model_dump()}


@app.post("/api/projects/{project_id}/clips/reset-order")
def reset_clip_order(project_id: str):
    store = _get_store(project_id)
    manifest = store.load()
    store.save_undo_snapshot(manifest, "复原原片时间轴切片顺序")
    sorted_clips = sorted(manifest.clips, key=lambda c: (c.start_s, c.start_utt))
    _apply_reordered_clips(manifest, sorted_clips)
    store.save(manifest)
    store.append_receipt(
        run_id=f"human_{uuid.uuid4().hex[:6]}",
        stage="human_edit",
        actor="human",
        action="reset_clip_order",
        summary="按原片时间轴顺序一键复原切片排序",
    )
    return {"manifest": manifest.model_dump()}


@app.patch("/api/projects/{project_id}/clips/{clip_id}")
def patch_clip(project_id: str, clip_id: str, req: PatchClipRequest):
    store = _get_store(project_id)
    manifest = store.load()
    cfg = load_editorial_config()
    target = next((c for c in manifest.clips if c.clip_id == clip_id), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"未找到切片: {clip_id}")

    before = {
        "enabled": target.enabled,
        "title": target.title,
        "cover_title": target.cover_title,
        "start_utt": target.start_utt,
        "end_utt": target.end_utt,
        "start_s": target.start_s,
        "end_s": target.end_s,
    }
    store.save_undo_snapshot(manifest, f"修改切片 {clip_id}")

    n_utts = len(manifest.utterances)
    if req.enabled is not None:
        target.enabled = req.enabled
    title_changed = False
    if req.title is not None:
        new_t = req.title.strip() or target.title
        if new_t != target.title:
            target.title = new_t
            title_changed = True
    cover_title_changed = False
    if req.cover_title is not None:
        new_ct = req.cover_title.strip()
        if new_ct != target.cover_title:
            target.cover_title = new_ct
            cover_title_changed = True
    if req.title_en is not None:
        target.title_en = req.title_en.strip()

    bounds_changed = False
    if req.start_utt is not None:
        target.start_utt = max(0, min(req.start_utt, n_utts - 1))
        bounds_changed = True
    if req.end_utt is not None:
        target.end_utt = max(target.start_utt, min(req.end_utt, n_utts - 1))
        bounds_changed = True

    if bounds_changed:
        if target.end_utt < target.start_utt:
            target.end_utt = target.start_utt
        rms, hop_s = _load_audio_envelope(store)
        s_t, e_t, ds, de, _ = snap_bounds_with_acoustic_guard(
            start_utt=target.start_utt,
            end_utt=target.end_utt,
            utterances=manifest.utterances,
            rms=rms,
            hop_s=hop_s,
            video_duration=manifest.source.duration_s,
            cfg=cfg,
        )
        target.start_s = s_t
        target.end_s = e_t
        target.snap_delta_start_ms = ds
        target.snap_delta_end_ms = de
        target.hook = manifest.utterances[target.start_utt].zh or manifest.utterances[target.start_utt].en

    active_cuts = get_active_delete_spans(manifest, cfg) if manifest.clean_cut.apply_to_clips else []
    clip_masks = get_overlapping_mask_payload(target.start_s, target.end_s, manifest, cfg) if manifest.clean_cut.apply_to_clips else []
    target.render_hash = compute_clip_render_hash(
        target,
        manifest.source.render_video,
        active_cuts,
        manifest.source.subtitle_sha256,
        mask_payload=clip_masks,
    )
    enabled_changed = req.enabled is not None and target.enabled != before["enabled"]
    _sync_derived_hashes_after_clip_change(manifest, mark_remix_stale=(enabled_changed or bounds_changed))
    store.save(manifest)
    if (cover_title_changed or (title_changed and not target.cover_title)) and target.cover_file:
        refresh_single_cover(manifest, store, cfg, clip_id)
        manifest = store.load()
        target = next((c for c in manifest.clips if c.clip_id == clip_id), target)

    after = {
        "enabled": target.enabled,
        "title": target.title,
        "cover_title": target.cover_title,
        "start_utt": target.start_utt,
        "end_utt": target.end_utt,
        "start_s": target.start_s,
        "end_s": target.end_s,
    }
    store.append_receipt(
        run_id=f"human_{uuid.uuid4().hex[:6]}",
        stage="human_edit",
        actor="human",
        action="edit_clip",
        summary=f"人工微调切片 {clip_id} ({target.title})：u{target.start_utt:04d}-u{target.end_utt:04d} ({target.start_s:.2f}s–{target.end_s:.2f}s)",
        diff={"clip_id": clip_id, "before": before, "after": after},
    )
    return {"manifest": manifest.model_dump()}


@app.post("/api/projects/{project_id}/clips/{clip_id}/render")
def render_single_clip(project_id: str, clip_id: str, force: bool = False):
    store = _get_store(project_id)
    manifest = store.load()
    cfg = load_editorial_config()
    ok, res = render_single_clip_and_cover(manifest, store, clip_id, cfg, force=force)
    if not ok:
        raise HTTPException(status_code=500, detail=f"渲染切片失败: {res}")
    manifest = store.load()
    return {"ok": True, "output_file": res, "manifest": manifest.model_dump()}


@app.post("/api/projects/{project_id}/clips/{clip_id}/subtitle-repair")
def repair_clip_subtitles(project_id: str, clip_id: str, req: SubtitleRepairRequest):
    store = _get_store(project_id)
    manifest = store.load()
    try:
        outcome = apply_subtitle_repairs(
            manifest,
            store,
            clip_id,
            req.repairs,
            rebuild_full_subtitle=req.rebuild_full_subtitle,
            render_clip=req.render_clip,
            render_full_video=req.render_full_video,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    manifest = store.load()
    return {
        "ok": True,
        "changed_utterance_ids": outcome.changed_utterance_ids,
        "bi_srt_path": str(outcome.bi_srt_path),
        "cn_srt_path": str(outcome.cn_srt_path),
        "en_srt_path": str(outcome.en_srt_path),
        "ass_path": str(outcome.ass_path),
        "clip_output_path": str(outcome.clip_output_path) if outcome.clip_output_path else None,
        "full_hardsub_path": str(outcome.full_hardsub_path) if outcome.full_hardsub_path else None,
        "manifest": manifest.model_dump(),
    }


@app.patch("/api/projects/{project_id}/remix")
def patch_remix(project_id: str, req: PatchRemixRequest):
    store = _get_store(project_id)
    manifest = store.load()
    cfg = load_editorial_config()
    store.save_undo_snapshot(manifest, "修改 Remix 编排")

    r = manifest.remix
    if req.enabled is not None:
        r.enabled = req.enabled
    title_changed = False
    if req.title is not None:
        new_t = req.title.strip() or r.title
        if new_t != r.title:
            r.title = new_t
            title_changed = True
    cover_title_changed = False
    if req.cover_title is not None:
        new_ct = req.cover_title.strip()
        if new_ct != r.cover_title:
            r.cover_title = new_ct
            cover_title_changed = True
    if req.card_mode in ("group", "every", "none"):
        r.card_mode = req.card_mode  # type: ignore[assignment]
    if req.card_duration_s is not None:
        r.card_duration_s = max(0.3, min(3.0, float(req.card_duration_s)))
    if req.enable_sfx is not None:
        r.enable_sfx = req.enable_sfx
    if req.enable_bgm is not None:
        r.enable_bgm = req.enable_bgm
    if req.bgm_volume is not None:
        r.bgm_volume = max(0.0, min(1.0, float(req.bgm_volume)))
    if req.sfx_volume is not None:
        r.sfx_volume = max(0.0, min(1.0, float(req.sfx_volume)))

    clip_by_id = {c.clip_id: c for c in manifest.clips}
    if req.source_mode is not None:
        sm = str(req.source_mode).strip().lower()
        if sm not in ("compilation", "manual_clips", "prompt", "ai_auto"):
            raise HTTPException(status_code=400, detail=f"未知混剪素材模式: {req.source_mode}")
        r.source_mode = sm  # type: ignore[assignment]
        r.manual_script_locked = False
        if sm != "manual_clips" and req.source_clip_ids is None:
            r.source_clip_ids = None
    if req.custom_prompt is not None:
        r.custom_prompt = str(req.custom_prompt).strip()
        if r.source_mode == "prompt":
            r.manual_script_locked = False
    if req.clear_source_clip_ids:
        r.source_clip_ids = None
        r.manual_script_locked = False
    elif req.source_clip_ids is not None:
        deduped_src: list[str] = []
        seen_src: set[str] = set()
        for cid in req.source_clip_ids:
            if cid in clip_by_id and cid not in seen_src:
                deduped_src.append(cid)
                seen_src.add(cid)
        r.source_clip_ids = deduped_src
        if req.source_mode is None:
            r.source_mode = "manual_clips" if deduped_src else "ai_auto"
        r.manual_script_locked = False

    if req.add_clip_id:
        add_c = clip_by_id.get(req.add_clip_id)
        if not add_c:
            raise HTTPException(status_code=404, detail=f"未找到切片: {req.add_clip_id}")
        next_order = max((it.order for it in r.items), default=0) + 1
        r.items.append(
            RemixItem(
                remix_id=f"rmx_{uuid.uuid4().hex[:6]}",
                order=next_order,
                enabled=True,
                source_clip_id=add_c.clip_id,
                role="core",
                beat_type="case_demo",
                theme_id=f"M{next_order:02d}",
                group_label=(add_c.chapter_title or add_c.title)[:10],
                card_title=(add_c.cover_title or add_c.title)[:24],
                start_utt=add_c.start_utt,
                end_utt=add_c.end_utt,
                start_s=add_c.start_s,
                end_s=add_c.end_s,
                quote=(add_c.hook or add_c.key_quote or add_c.title)[:160],
                reason="人工手动加入混剪",
            )
        )
        r.publish_desc = ""
        r.manual_script_locked = True

    if req.remove_remix_id:
        r.items = [it for it in r.items if it.remix_id != req.remove_remix_id]
        for idx, it in enumerate(sorted(r.items, key=lambda x: x.order), 1):
            it.order = idx
        r.publish_desc = ""
        r.manual_script_locked = True

    if req.items is not None:
        rms, hop_s = _load_audio_envelope(store)
        n_utts = len(manifest.utterances)
        by_id = {it.remix_id: it for it in r.items}
        for upd in req.items:
            it = by_id.get(upd.remix_id)
            if not it:
                continue
            if upd.order is not None:
                it.order = upd.order
            if upd.enabled is not None:
                it.enabled = upd.enabled
            if upd.group_label is not None:
                it.group_label = upd.group_label.strip()
            if upd.card_title is not None:
                it.card_title = upd.card_title.strip()
            bounds_changed = False
            if upd.start_utt is not None:
                it.start_utt = max(0, min(upd.start_utt, n_utts - 1))
                bounds_changed = True
            if upd.end_utt is not None:
                it.end_utt = max(it.start_utt, min(upd.end_utt, n_utts - 1))
                bounds_changed = True
            if bounds_changed:
                s_t, e_t, _, _, _ = snap_bounds_with_acoustic_guard(
                    start_utt=it.start_utt,
                    end_utt=it.end_utt,
                    utterances=manifest.utterances,
                    rms=rms,
                    hop_s=hop_s,
                    video_duration=manifest.source.duration_s,
                    cfg=cfg,
                )
                it.start_s = s_t
                it.end_s = e_t
        r.items.sort(key=lambda x: x.order)
        r.publish_desc = ""
        r.manual_script_locked = True

    if req.manual_script_locked is not None:
        r.manual_script_locked = bool(req.manual_script_locked)
    if r.manual_script_locked:
        r.planned_source_signature = compute_remix_source_signature(manifest)

    r.render_hash = compute_remix_render_hash(manifest)
    store.save(manifest)
    if (cover_title_changed or (title_changed and not r.cover_title)) and r.cover_file:
        refresh_single_cover(manifest, store, cfg, "remix")
        manifest = store.load()
    store.append_receipt(
        run_id=f"human_{uuid.uuid4().hex[:6]}",
        stage="human_edit",
        actor="human",
        action="edit_remix",
        summary=f"人工更新 Remix 编排（卡片:{r.card_mode}，BGM:{'开' if r.enable_bgm else '关'}，启用段落 {sum(1 for x in r.items if x.enabled)}/{len(r.items)}）",
    )
    return {"manifest": manifest.model_dump()}


def _set_render_stage_state(
    store: ProjectStore,
    *,
    status: str,
    message: str,
    elapsed_ms: int | None = None,
    error: str | None = None,
) -> None:
    def _mut(m: ProjectManifest) -> None:
        st = m.stages.get("render")
        if st is None:
            return
        st.status = status  # type: ignore[assignment]
        st.message = message
        if status == "running":
            st.started_at = utc_now_iso()
            st.finished_at = None
            st.error = None
        else:
            st.finished_at = utc_now_iso()
            st.error = error
        if elapsed_ms is not None:
            st.elapsed_ms = elapsed_ms

    store.update_manifest(_mut)


def _launch_target_render_job(
    project_id: str,
    store: ProjectStore,
    prefix: str,
    label: str,
    runner_fn: Any,
) -> dict[str, Any]:
    job_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
    token = _start_job_entry(project_id, job_id, ["render"], cancellable=True)
    assert token is not None
    _set_render_stage_state(store, status="running", message=f"正在渲染{label}...")

    def _worker():
        t0 = time.monotonic()
        try:
            manifest = store.load()
            with use_cancel_token(token):
                ok, res = runner_fn(manifest)
                if not ok:
                    raise RuntimeError(res)
            elapsed = int((time.monotonic() - t0) * 1000)
            _set_render_stage_state(store, status="completed", message=f"{label}渲染完成", elapsed_ms=elapsed)
            _finish_job_entry(project_id, job_id, token)
        except JobCancelledError:
            elapsed = int((time.monotonic() - t0) * 1000)
            _set_render_stage_state(store, status="idle", message="已手动中止", elapsed_ms=elapsed)
            _finish_job_entry(project_id, job_id, token, cancelled=True, error=None)
        except Exception as exc:
            elapsed = int((time.monotonic() - t0) * 1000)
            _set_render_stage_state(
                store, status="error", message=str(exc)[:180], elapsed_ms=elapsed, error=str(exc)
            )
            _finish_job_entry(project_id, job_id, token, error=str(exc))

    threading.Thread(target=_worker, daemon=True).start()
    return {"started": True, "job_id": job_id}


@app.post("/api/projects/{project_id}/remix/render")
def render_remix_endpoint(project_id: str, force: bool = False):
    store = _get_store(project_id)
    cfg = load_editorial_config()
    manifest_chk = store.load()
    if manifest_chk.remix.items and not any(it.enabled for it in manifest_chk.remix.items):
        raise HTTPException(status_code=400, detail="未勾选任何混剪段落，请至少勾选 1 段后再渲染")
    if not manifest_chk.remix.items and not (manifest_chk.clips or manifest_chk.remix_only_pool):
        raise HTTPException(status_code=400, detail="没有可用于混剪的切片素材")
    return _launch_target_render_job(
        project_id,
        store,
        "remix",
        "AI 混剪",
        lambda m: render_remix_and_cover(m, store, cfg, force=force),
    )


@app.patch("/api/projects/{project_id}/compilation")
def patch_compilation(project_id: str, req: PatchCompilationRequest):
    store = _get_store(project_id)
    manifest = store.load()
    cfg = load_editorial_config()
    store.save_undo_snapshot(manifest, "修改切片精选集设置")

    c = manifest.compilation
    if req.enabled is not None:
        c.enabled = req.enabled
    title_changed = False
    if req.title is not None:
        new_t = req.title.strip() or c.title
        if new_t != c.title:
            c.title = new_t
            title_changed = True
    cover_title_changed = False
    if req.cover_title is not None:
        new_ct = req.cover_title.strip()
        if new_ct != c.cover_title:
            c.cover_title = new_ct
            cover_title_changed = True
    if req.clear_clip_ids:
        c.clip_ids = []
        c.publish_desc = ""
    elif req.clip_ids is not None:
        valid_ids = {cl.clip_id for cl in manifest.clips}
        deduped_cids: list[str] = []
        seen_cids: set[str] = set()
        for cid in req.clip_ids:
            if cid in valid_ids and cid not in seen_cids:
                deduped_cids.append(cid)
                seen_cids.add(cid)
        c.clip_ids = deduped_cids
        c.publish_desc = ""
    if req.card_duration_s is not None:
        c.card_duration_s = max(0.5, min(5.0, float(req.card_duration_s)))
    if req.card_fade_s is not None:
        c.card_fade_s = max(0.05, min(1.0, float(req.card_fade_s)))

    c.render_hash = compute_compilation_render_hash(manifest)
    store.save(manifest)
    if (cover_title_changed or (title_changed and not c.cover_title)) and c.cover_file:
        refresh_single_cover(manifest, store, cfg, "compilation")
        manifest = store.load()
    store.append_receipt(
        run_id=f"human_{uuid.uuid4().hex[:6]}",
        stage="human_edit",
        actor="human",
        action="edit_compilation",
        summary=f"人工更新切片精选集设置（启用:{c.enabled}，片头卡片停留:{c.card_duration_s:.1f}s，已选 {len(get_compilation_clips(manifest))} 支切片）",
    )
    return {"manifest": manifest.model_dump()}


@app.post("/api/projects/{project_id}/compilation/suggest-title")
def suggest_compilation_title_endpoint(project_id: str):
    store = _get_store(project_id)
    cfg = load_editorial_config()
    manifest = store.load()
    if not manifest.clips:
        raise HTTPException(status_code=400, detail="该项目尚无切片可用于提炼精选集标题")
    store.save_undo_snapshot(manifest, "AI 提炼精选集标题")
    distilled = distill_compilation_title(
        manifest,
        store,
        cfg,
        run_id=f"comptitle_{uuid.uuid4().hex[:6]}",
    )
    if manifest.compilation.cover_file and not manifest.compilation.cover_title:
        refresh_single_cover(manifest, store, cfg, "compilation")
        manifest = store.load()
    return {"manifest": manifest.model_dump(), "title": distilled}


@app.post("/api/projects/{project_id}/compilation/render")
def render_compilation_endpoint(project_id: str, force: bool = False):
    store = _get_store(project_id)
    cfg = load_editorial_config()
    manifest_chk = store.load()
    if not get_compilation_clips(manifest_chk):
        raise HTTPException(status_code=400, detail="精选集未选择任何切片，请先在精选集内勾选切片")
    return _launch_target_render_job(
        project_id,
        store,
        "comp",
        "切片精选集",
        lambda m: render_compilation_and_cover(m, store, cfg, force=force),
    )


@app.post("/api/projects/{project_id}/clean-cut/scan")
def scan_clean_cut_endpoint(project_id: str, req: ScanCleanCutRequest):
    store = _get_store(project_id)
    cfg = load_editorial_config()
    job_id = f"scan_{uuid.uuid4().hex[:8]}"
    token = _start_job_entry(project_id, job_id, ["clean_cut_scan"], cancellable=True)
    assert token is not None

    def _worker():
        try:
            manifest = store.load()
            with use_cancel_token(token):
                scan_cut_segments(
                    manifest,
                    store,
                    cfg,
                    job_id,
                    scan_ads=req.cut_ads,
                    scan_political=req.cut_political_sensitive,
                    sensitivity=req.sensitivity,
                )
            _finish_job_entry(project_id, job_id, token)
        except JobCancelledError:
            store.log("[clean_cut] 合规扫描已手动中止")
            store.append_receipt(
                run_id=job_id,
                stage="clean_cut",
                actor="human",
                action="scan_cancelled",
                summary="广告/敏感内容扫描已手动中止",
            )
            _finish_job_entry(project_id, job_id, token, cancelled=True, error=None)
        except Exception as exc:
            _finish_job_entry(project_id, job_id, token, error=str(exc))

    threading.Thread(target=_worker, daemon=True).start()
    return {"started": True, "job_id": job_id}


@app.patch("/api/projects/{project_id}/metadata")
def patch_metadata(project_id: str, req: PatchMetadataRequest):
    store = _get_store(project_id)
    manifest = store.load()
    if req.channel_zh is not None:
        manifest.metadata.channel_zh = req.channel_zh.strip()
    if req.guest_zh is not None:
        manifest.metadata.guest_zh = req.guest_zh.strip()
    if req.published_date is not None:
        manifest.metadata.published_date = req.published_date.strip()
    manifest.remix.render_hash = compute_remix_render_hash(manifest)
    manifest.compilation.render_hash = compute_compilation_render_hash(manifest)
    store.save(manifest)
    return {"manifest": manifest.model_dump()}


@app.patch("/api/projects/{project_id}/clean-cut")
def patch_clean_cut(project_id: str, req: PatchCleanCutRequest):
    store = _get_store(project_id)
    manifest = store.load()
    cfg = load_editorial_config()
    store.save_undo_snapshot(manifest, "修改全片净版设置与剪除区间")

    cc = manifest.clean_cut
    was_up_to_date = bool(
        cc.output_file and Path(cc.output_file).exists() and cc.render_hash == cc.last_rendered_hash
    )
    only_title_fields = (
        req.enabled is None
        and req.cut_ads is None
        and req.cut_political_sensitive is None
        and req.sensitivity is None
        and req.political_action is None
        and req.apply_to_clips is None
        and req.cut_mode is None
        and req.segments is None
        and req.add_segment is None
        and req.delete_ids is None
    )
    prev_sens = str(cc.sensitivity or "strict")
    prev_has_pol = any(
        s.kind == "political_sensitive" and segment_matches_sensitivity(s, prev_sens, cfg)
        for s in manifest.cut_segments
    )
    if req.enabled is not None:
        cc.enabled = req.enabled
    title_changed = False
    if req.title is not None:
        new_t = req.title.strip()
        if new_t != cc.title:
            cc.title = new_t
            title_changed = True
    cover_title_changed = False
    if req.cover_title is not None:
        new_ct = req.cover_title.strip()
        if new_ct != cc.cover_title:
            cc.cover_title = new_ct
            cover_title_changed = True
    if req.sensitivity in ("strict", "balanced", "loose"):
        cc.sensitivity = req.sensitivity  # type: ignore[assignment]
    if req.political_action in VALID_POLITICAL_ACTIONS:
        cc.political_action = req.political_action  # type: ignore[assignment]
    if req.apply_to_clips is not None:
        cc.apply_to_clips = req.apply_to_clips
    if req.cut_mode in ("auto", "precise", "reencode"):
        cc.cut_mode = req.cut_mode  # type: ignore[assignment]

    if req.delete_ids:
        del_set = set(req.delete_ids)
        manifest.cut_segments = [s for s in manifest.cut_segments if s.cut_id not in del_set]

    n_utts = len(manifest.utterances)
    dur = float(manifest.source.duration_s or (manifest.utterances[-1].end_s if manifest.utterances else 0.0))
    if req.segments:
        rms, hop_s = _load_audio_envelope(store)
        by_id = {s.cut_id: s for s in manifest.cut_segments}
        for upd in req.segments:
            seg = by_id.get(upd.cut_id)
            if not seg:
                continue
            if upd.enabled is not None:
                seg.enabled = upd.enabled
            if upd.kind in ("ad", "political_sensitive"):
                seg.kind = upd.kind  # type: ignore[assignment]
            if upd.severity in ("high", "medium", "low"):
                seg.severity = upd.severity  # type: ignore[assignment]
            if upd.action in VALID_SEGMENT_ACTIONS:
                seg.action = upd.action  # type: ignore[assignment]
            if upd.sensitive_terms_zh is not None:
                seg.sensitive_terms_zh = [str(x).strip() for x in upd.sensitive_terms_zh if str(x).strip()]
            if upd.sensitive_terms_en is not None:
                seg.sensitive_terms_en = [str(x).strip() for x in upd.sensitive_terms_en if str(x).strip()]
            if upd.precise_terms_zh is not None:
                seg.precise_terms_zh = [str(x).strip() for x in upd.precise_terms_zh if str(x).strip()]
                seg.precise_terms_extracted = True
            if upd.precise_terms_en is not None:
                seg.precise_terms_en = [str(x).strip() for x in upd.precise_terms_en if str(x).strip()]
                seg.precise_terms_extracted = True
            if upd.reason is not None:
                seg.reason = upd.reason.strip()
            utt_bounds_changed = False
            if n_utts > 0 and upd.start_utt is not None:
                seg.start_utt = max(0, min(upd.start_utt, n_utts - 1))
                utt_bounds_changed = True
            if n_utts > 0 and upd.end_utt is not None:
                base_su = seg.start_utt if seg.start_utt is not None else 0
                seg.end_utt = max(base_su, min(upd.end_utt, n_utts - 1))
                utt_bounds_changed = True
            if utt_bounds_changed and seg.start_utt is not None and seg.end_utt is not None:
                s_t, e_t, _, _, _ = snap_bounds_with_acoustic_guard(
                    start_utt=seg.start_utt,
                    end_utt=seg.end_utt,
                    utterances=manifest.utterances,
                    rms=rms,
                    hop_s=hop_s,
                    video_duration=dur,
                    cfg=cfg,
                )
                seg.start_s = s_t
                seg.end_s = e_t
                seg.quote = " ".join(
                    (manifest.utterances[i].zh or manifest.utterances[i].en).strip()
                    for i in range(seg.start_utt, seg.end_utt + 1)
                )[:180]
            else:
                if upd.start_s is not None:
                    seg.start_s = round(max(0.0, float(upd.start_s)), 3)
                if upd.end_s is not None:
                    seg.end_s = round(min(dur if dur > 0 else float(upd.end_s), max(seg.start_s + 0.2, float(upd.end_s))), 3)

    added_kind: str | None = None
    if req.add_segment is not None:
        ns = max(0.0, float(req.add_segment.start_s))
        ne = min(dur if dur > 0 else float(req.add_segment.end_s), max(ns + 0.5, float(req.add_segment.end_s)))
        su = next((u.id for u in manifest.utterances if u.end_s > ns), None)
        eu = next((u.id for u in reversed(manifest.utterances) if u.start_s < ne), None)
        if su is not None and eu is not None and eu < su:
            su, eu = None, None
        quote = ""
        if su is not None and eu is not None:
            quote = " ".join(
                (manifest.utterances[i].zh or manifest.utterances[i].en).strip()
                for i in range(su, eu + 1)
            )[:180]
        kind_val = req.add_segment.kind if req.add_segment.kind in ("ad", "political_sensitive") else "political_sensitive"
        added_kind = kind_val
        manifest.cut_segments.append(
            CutSegment(
                cut_id=f"cut_m_{uuid.uuid4().hex[:6]}",
                kind=kind_val,  # type: ignore[arg-type]
                subtype="manual",
                enabled=True,
                detector="manual",
                start_utt=su,
                end_utt=eu,
                start_s=round(ns, 3),
                end_s=round(ne, 3),
                quote=quote,
                reason=req.add_segment.reason.strip() or "人工手动屏蔽区间",
                severity="high",
            )
        )
        manifest.cut_segments.sort(key=lambda x: (x.start_s, x.end_s))

    has_ads = any(s.kind == "ad" for s in manifest.cut_segments)
    has_pol = any(
        s.kind == "political_sensitive" and segment_matches_sensitivity(s, cc.sensitivity, cfg)
        for s in manifest.cut_segments
    )
    if added_kind == "ad":
        cc.cut_ads = True
    elif added_kind == "political_sensitive":
        cc.cut_political_sensitive = True
    if req.cut_ads is not None:
        cc.cut_ads = bool(req.cut_ads) and has_ads
    if req.cut_political_sensitive is not None:
        cc.cut_political_sensitive = bool(req.cut_political_sensitive) and has_pol
    elif cc.sensitivity != prev_sens:
        if not has_pol:
            cc.cut_political_sensitive = False
        elif not prev_has_pol and has_pol:
            cc.cut_political_sensitive = True
    if not has_ads:
        cc.cut_ads = False
    if not has_pol:
        cc.cut_political_sensitive = False
    if not get_active_cut_segments(manifest, cfg):
        cc.enabled = False

    sync_all_render_hashes(manifest, cfg=cfg)
    if only_title_fields and was_up_to_date:
        cc.last_rendered_hash = cc.render_hash
    store.save(manifest)
    if (cover_title_changed or (title_changed and not cc.cover_title)) and cc.cover_file:
        refresh_single_cover(manifest, store, cfg, "clean_cut")
        manifest = store.load()
    store.append_receipt(
        run_id=f"human_{uuid.uuid4().hex[:6]}",
        stage="human_edit",
        actor="human",
        action="edit_clean_cut",
        summary=f"人工更新全片净版设置（启用:{cc.enabled}，广告:{cc.cut_ads}，政敏:{cc.cut_political_sensitive}，同步切片:{cc.apply_to_clips}，分段 {len(manifest.cut_segments)} 处）",
    )
    return {"manifest": manifest.model_dump()}


@app.post("/api/projects/{project_id}/clean-cut/suggest-title")
def suggest_clean_cut_title_endpoint(project_id: str):
    store = _get_store(project_id)
    cfg = load_editorial_config()
    manifest = store.load()
    store.save_undo_snapshot(manifest, "AI 提炼全片净版标题")
    distilled = distill_clean_cut_title(
        manifest,
        store,
        cfg,
        run_id=f"cctitle_{uuid.uuid4().hex[:6]}",
    )
    if manifest.clean_cut.cover_file and not manifest.clean_cut.cover_title:
        refresh_single_cover(manifest, store, cfg, "clean_cut")
        manifest = store.load()
    return {"manifest": manifest.model_dump(), "title": distilled}


@app.post("/api/projects/{project_id}/clean-cut/render")
def render_clean_cut_endpoint(project_id: str, force: bool = False):
    store = _get_store(project_id)
    cfg = load_editorial_config()
    manifest_chk = store.load()
    if not get_active_cut_segments(manifest_chk, cfg):
        if manifest_chk.clean_cut.enabled:
            manifest_chk.clean_cut.enabled = False
            sync_all_render_hashes(manifest_chk, cfg=cfg)
            store.save(manifest_chk)
        raise HTTPException(status_code=400, detail="当前没有任何需要剪切或脱敏的内容，无需输出全片净版")
    return _launch_target_render_job(
        project_id,
        store,
        "clean",
        "全片净版",
        lambda m: render_clean_cut_and_cover(m, store, cfg, force=force),
    )


@app.post("/api/projects/{project_id}/clean-cut/extract-precise-terms")
def extract_precise_terms_endpoint(project_id: str, req: ExtractPreciseTermsRequest):
    store = _get_store(project_id)
    cfg = load_editorial_config()
    manifest = store.load()
    store.save_undo_snapshot(manifest, "AI 提取精准脱敏关键词")
    try:
        updated_manifest, updated_count = extract_precise_terms_for_segments(
            manifest,
            store,
            cfg,
            run_id=f"precise_{uuid.uuid4().hex[:6]}",
            cut_ids=req.cut_ids,
            force=req.force,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"提取精准关键词失败: {exc}")
    return {"manifest": updated_manifest.model_dump(), "updated_count": updated_count}



@app.post("/api/projects/{project_id}/covers/refresh")
def api_refresh_cover(project_id: str, req: RefreshCoverRequest):
    store = _get_store(project_id)
    manifest = store.load()
    cfg = load_editorial_config()
    out_p = refresh_single_cover(manifest, store, cfg, req.target)
    if out_p is None:
        raise HTTPException(status_code=400, detail=f"无法刷新封面: {req.target}")
    manifest = store.load()
    return {"refreshed": str(out_p), "manifest": manifest.model_dump()}


@app.get("/api/publish/status")
def api_publish_status():
    return {"platforms": get_platforms_status()}


@app.post("/api/publish/login/{platform}")
def api_publish_login(platform: str):
    try:
        return launch_platform_login(platform)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/projects/{project_id}/publish")
def api_publish_project(project_id: str, req: PublishRequest):
    store = _get_store(project_id)
    if not req.platforms:
        raise HTTPException(status_code=400, detail="请至少选择一个发布平台")
    manifest_chk = store.load()
    try:
        vpath, _, _, _, _, _ = _default_target_metadata(manifest_chk, req.target)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if vpath is None or not vpath.exists():
        raise HTTPException(status_code=400, detail="目标视频尚未渲染，请先执行渲染后再发布")

    job_id = f"pub_{uuid.uuid4().hex[:8]}"
    col_str = (req.collection or "").strip()
    token = _start_publish_job_entry(
        project_id,
        job_id,
        kind="single",
        targets=[req.target],
        platforms=req.platforms,
        collection=col_str,
    )

    def _worker():
        try:
            publish_to_platforms(
                store=store,
                target=req.target,
                platforms=req.platforms,
                title=req.title,
                cover_title=req.cover_title,
                desc=req.desc,
                tags=req.tags,
                is_public=req.is_public,
                collection=req.collection,
                tid=req.tid,
                ui_schema=req.ui_schema,
                cancel_token=token,
            )
            _finish_publish_job_entry(project_id, job_id, token)
        except JobCancelledError:
            _finish_publish_job_entry(project_id, job_id, token, cancelled=True, error=None)
        except Exception as exc:
            _finish_publish_job_entry(project_id, job_id, token, error=str(exc))

    threading.Thread(target=_worker, daemon=True).start()
    return {"started": True, "job_id": job_id, "target": req.target, "platforms": req.platforms}


@app.post("/api/projects/{project_id}/publish/suggest-desc")
def suggest_publish_desc_endpoint(project_id: str, req: SuggestPublishDescRequest):
    store = _get_store(project_id)
    cfg = load_editorial_config()
    manifest = store.load()
    target = str(req.target or "").strip()
    if target not in ("clean_cut", "remix", "compilation") and not any(c.clip_id == target for c in manifest.clips):
        raise HTTPException(status_code=404, detail=f"未找到发布目标: {req.target}")
    store.save_undo_snapshot(manifest, f"AI 提炼 {target} 发布简介")
    distilled = distill_target_publish_description(
        manifest,
        store,
        cfg,
        target,
        run_id=f"pubdesc_{uuid.uuid4().hex[:6]}",
    )
    return {"manifest": manifest.model_dump(), "desc": distilled}


@app.post("/api/projects/{project_id}/publish/batch")
def api_publish_batch_project(project_id: str, req: BatchPublishRequest):
    store = _get_store(project_id)
    if not req.platforms:
        raise HTTPException(status_code=400, detail="请至少选择一个发布平台")

    manifest_chk = store.load()
    clip_by_id = {c.clip_id: c for c in manifest_chk.clips}
    if req.targets is not None:
        raw_targets = list(dict.fromkeys(str(t).strip() for t in req.targets if str(t).strip()))
        unknown = [t for t in raw_targets if t not in clip_by_id]
        if unknown:
            raise HTTPException(status_code=400, detail=f"包含未知或非独立切片目标: {', '.join(unknown)}")
    else:
        raw_targets = [c.clip_id for c in manifest_chk.clips if c.enabled]

    if not raw_targets:
        raise HTTPException(status_code=400, detail="当前没有选中的独立切片可发布")

    skipped_unrendered: list[str] = []
    skipped_already_published: list[str] = []
    runnable_targets: list[str] = []
    for cid in raw_targets:
        clip = clip_by_id[cid]
        vpath = Path(clip.output_file) if clip.output_file else None
        if vpath is None or not vpath.exists():
            skipped_unrendered.append(cid)
            continue
        if req.skip_already_published:
            done_plats = {r.platform for r in clip.publish_records if r.status in ("published", "draft")}
            if all(p in done_plats for p in req.platforms):
                skipped_already_published.append(cid)
                continue
        runnable_targets.append(cid)

    if not runnable_targets:
        if skipped_unrendered and not skipped_already_published:
            raise HTTPException(status_code=400, detail="所选切片均尚未渲染，请先点击「渲染选中」生成成片")
        if skipped_already_published:
            raise HTTPException(status_code=400, detail="所选已渲染切片在目标平台均已存草稿或已发布（如需重传请取消「跳过已上传」）")
        raise HTTPException(status_code=400, detail="没有符合批量上传条件的已渲染切片")

    job_id = f"pubbatch_{uuid.uuid4().hex[:8]}"
    col_str = (req.collection or "").strip()
    token = _start_publish_job_entry(
        project_id,
        job_id,
        kind="batch",
        targets=runnable_targets,
        platforms=req.platforms,
        collection=col_str,
    )

    def _on_progress(prog: dict[str, Any]) -> None:
        _update_publish_job_entry(
            project_id,
            job_id,
            current_target=prog.get("current_target"),
            completed_targets=prog.get("completed_targets", []),
            failed_targets=prog.get("failed_targets", []),
            skipped_targets=prog.get("skipped_targets", []),
        )

    def _worker():
        try:
            res = publish_batch_targets(
                store=store,
                targets=runnable_targets,
                platforms=req.platforms,
                is_public=req.is_public,
                collection=col_str,
                skip_already_published=req.skip_already_published,
                tid=req.tid,
                extra_tags=req.extra_tags,
                run_id=job_id,
                cancel_token=token,
                on_progress=_on_progress,
            )
            _update_publish_job_entry(
                project_id,
                job_id,
                completed_targets=res.get("completed_targets", []),
                failed_targets=res.get("failed_targets", []),
                skipped_targets=res.get("skipped_targets", []),
            )
            _finish_publish_job_entry(project_id, job_id, token)
        except JobCancelledError:
            _finish_publish_job_entry(project_id, job_id, token, cancelled=True, error=None)
        except Exception as exc:
            _finish_publish_job_entry(project_id, job_id, token, error=str(exc))

    threading.Thread(target=_worker, daemon=True).start()
    return {
        "started": True,
        "job_id": job_id,
        "targets": runnable_targets,
        "platforms": req.platforms,
        "collection": col_str,
        "skipped_unrendered": skipped_unrendered,
        "skipped_already_published": skipped_already_published,
    }


@app.post("/api/projects/{project_id}/publish/reset")
def api_reset_publish_records(project_id: str, req: ResetPublishRequest):
    store = _get_store(project_id)
    with _JOBS_LOCK:
        if _ACTIVE_JOBS.get(project_id, {}).get("running"):
            raise HTTPException(status_code=409, detail="当前有规划或渲染任务正在运行，请先中止后再复位发布状态")
    if _has_running_publish_for_target(project_id, req.target):
        raise HTTPException(status_code=409, detail="该条目正在发布中，请先中止后再复位发布状态")
    eff_only_errors = req.only_errors if req.only_errors is not None else (req.platform is None)
    try:
        manifest, removed = reset_publish_records(
            store,
            req.target,
            platform=req.platform,
            only_errors=eff_only_errors,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"manifest": manifest.model_dump(), "removed": removed}


@app.post("/api/projects/{project_id}/open-output")
def open_output_folder(project_id: str):
    store = _get_store(project_id)
    manifest = store.load()
    out_dir = Path(manifest.source.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["open", str(out_dir)], check=False)
    return {"opened": str(out_dir)}


@app.post("/api/projects/{project_id}/undo")
def undo_edit(project_id: str):
    store = _get_store(project_id)
    res = store.undo_last(f"undo_{uuid.uuid4().hex[:6]}")
    if res is None:
        raise HTTPException(status_code=400, detail="没有可撤销的人工修改记录")
    manifest, snap = res
    return {"restored": snap.get("label"), "manifest": manifest.model_dump()}


@app.get("/api/projects/{project_id}/receipts")
def get_receipts(project_id: str, stage: str | None = Query(None), limit: int = Query(200)):
    store = _get_store(project_id)
    receipts = store.list_receipts(stage=stage, limit=limit)
    logs = store.read_logs(tail=150)
    return {
        "receipts": [r.model_dump() for r in reversed(receipts)],
        "pipeline_log": logs,
    }


def _stream_file_range(file_path: Path, request: Request):
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"媒体文件不存在: {file_path.name}")
    file_size = file_path.stat().st_size
    content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    range_header = request.headers.get("range")

    if not range_header:
        return FileResponse(str(file_path), media_type=content_type)

    try:
        units, rng = range_header.split("=", 1)
        start_s, end_s = rng.split("-", 1)
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except Exception:
        start, end = 0, file_size - 1

    start = max(0, min(start, file_size - 1))
    end = max(start, min(end, file_size - 1))
    chunk_size = end - start + 1

    def iterfile():
        with file_path.open("rb") as f:
            f.seek(start)
            remaining = chunk_size
            while remaining > 0:
                buf = f.read(min(1024 * 256, remaining))
                if not buf:
                    break
                remaining -= len(buf)
                yield buf

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
    }
    return StreamingResponse(iterfile(), status_code=206, headers=headers, media_type=content_type)


@app.get("/api/projects/{project_id}/media/{kind}")
def stream_media(project_id: str, kind: str, request: Request, clip_id: str | None = Query(None)):
    store = _get_store(project_id)
    manifest = store.load()
    if kind == "source":
        return _stream_file_range(Path(manifest.source.render_video), request)
    if kind == "remix":
        if not manifest.remix.output_file:
            raise HTTPException(status_code=404, detail="Remix 尚未渲染")
        return _stream_file_range(Path(manifest.remix.output_file), request)
    if kind == "clip":
        c = next((x for x in manifest.clips if x.clip_id == clip_id), None)
        if not c or not c.output_file:
            raise HTTPException(status_code=404, detail="该切片尚未渲染")
        return _stream_file_range(Path(c.output_file), request)
    if kind == "cover":
        c = next((x for x in manifest.clips if x.clip_id == clip_id), None)
        if not c or not c.cover_file:
            raise HTTPException(status_code=404, detail="封面不存在")
        return FileResponse(c.cover_file, media_type="image/jpeg")
    if kind == "remix_cover":
        if not manifest.remix.cover_file:
            raise HTTPException(status_code=404, detail="混剪封面不存在")
        return FileResponse(manifest.remix.cover_file, media_type="image/jpeg")
    if kind == "compilation":
        if not manifest.compilation.output_file:
            raise HTTPException(status_code=404, detail="精选集尚未渲染")
        return _stream_file_range(Path(manifest.compilation.output_file), request)
    if kind == "compilation_cover":
        if not manifest.compilation.cover_file:
            raise HTTPException(status_code=404, detail="精选集封面不存在")
        return FileResponse(manifest.compilation.cover_file, media_type="image/jpeg")
    if kind == "clean_cut":
        if not manifest.clean_cut.output_file:
            raise HTTPException(status_code=404, detail="全片净版尚未渲染")
        return _stream_file_range(Path(manifest.clean_cut.output_file), request)
    if kind == "clean_cut_cover":
        if not manifest.clean_cut.cover_file:
            raise HTTPException(status_code=404, detail="全片净版封面不存在")
        return FileResponse(manifest.clean_cut.cover_file, media_type="image/jpeg")
    raise HTTPException(status_code=400, detail=f"未知媒体类型: {kind}")


@app.get("/api/models")
def get_models():
    cfg = load_editorial_config()
    m = cfg.get("models", {})
    return {
        "catalog": MODEL_CATALOG,
        "selected": {
            "text_provider": m.get("text_provider", "dashscope"),
            "text_model": m.get("text_model", "qwen3.8-max"),
            "remix_model": m.get("remix_model") or m.get("text_model", "qwen3.8-max"),
            "multimodal_provider": m.get("multimodal_provider", "dashscope"),
            "multimodal_model": m.get("multimodal_model", "qwen3.5-omni-plus"),
        },
    }


@app.put("/api/models")
def update_models(req: UpdateModelsRequest):
    cfg = load_editorial_config()
    m = cfg.setdefault("models", {})
    if req.text_model:
        prov, mod = parse_provider_model(req.text_model, req.text_provider or m.get("text_provider", "dashscope"))
        m["text_provider"] = prov
        m["text_model"] = mod
        if req.remix_model is None:
            m["remix_model"] = ""
    elif req.text_provider:
        m["text_provider"] = req.text_provider
    if req.remix_model is not None:
        m["remix_model"] = req.remix_model.strip() or ""
    if req.multimodal_model:
        prov, mod = parse_provider_model(
            req.multimodal_model, req.multimodal_provider or m.get("multimodal_provider", "dashscope")
        )
        m["multimodal_provider"] = prov
        m["multimodal_model"] = mod
    elif req.multimodal_provider:
        m["multimodal_provider"] = req.multimodal_provider
    save_editorial_config(cfg)
    return {"saved": True, "selected": get_models()["selected"]}


@app.get("/api/config/editorial")
def get_editorial_config():
    return {"raw_yaml": load_editorial_raw()}


@app.put("/api/config/editorial")
def put_editorial_config(req: UpdateEditorialRequest):
    try:
        parsed = save_editorial_raw(req.raw_yaml)
        return {"saved": True, "categories": len(parsed.get("hot_topics", {}).get("categories", []))}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"YAML 格式错误: {exc}")


@app.get("/api/config/content-filter")
def get_content_filter_config():
    cfg = load_editorial_config()
    cf = cfg.get("content_filter", {})
    default_sens = str(cf.get("sensitivity") or "strict").strip().lower()
    if default_sens not in ("strict", "balanced", "loose"):
        default_sens = "strict"
    default_action = str(cf.get("political_action") or "auto").strip().lower()
    if default_action not in VALID_POLITICAL_ACTIONS:
        default_action = "auto"
    return {
        "default_sensitivity": default_sens,
        "default_political_action": default_action,
        "political_profiles": get_political_profiles(cfg),
    }


@app.put("/api/config/content-filter")
def put_content_filter_config(req: UpdateContentFilterConfigRequest):
    cfg = load_editorial_config()
    cf = cfg.setdefault("content_filter", {})

    if req.default_sensitivity is not None:
        def_sens = str(req.default_sensitivity or "strict").strip().lower()
        if def_sens not in ("strict", "balanced", "loose"):
            raise HTTPException(status_code=400, detail=f"未知政审默认标准: {req.default_sensitivity}")
        cf["sensitivity"] = def_sens

    if req.default_political_action is not None:
        def_action = str(req.default_political_action or "").strip().lower()
        if def_action not in VALID_POLITICAL_ACTIONS:
            raise HTTPException(status_code=400, detail=f"未知默认脱敏策略: {req.default_political_action}")
        cf["political_action"] = def_action

    if req.political_profiles is not None:
        normalized: dict[str, dict[str, Any]] = {}
        for key in ("strict", "balanced", "loose"):
            prof = req.political_profiles.get(key)
            if prof is None:
                raise HTTPException(status_code=400, detail=f"缺少政审档位配置: {key}")
            trig = [s for s in VALID_SEVERITIES if s in {x.strip().lower() for x in prof.trigger_severities}]
            if not trig:
                raise HTTPException(status_code=400, detail=f"档位 {key} 至少须包含一个生效风险等级")
            clean_rules: list[dict[str, str]] = []
            seen_ids: set[str] = set()
            for idx, r in enumerate(prof.rules, 1):
                name = r.name.strip()
                desc = r.description.strip()
                if not name and not desc:
                    continue
                rid = (r.id or "").strip() or f"rule_{idx}"
                if rid in seen_ids:
                    rid = f"{rid}_{idx}"
                seen_ids.add(rid)
                sev = str(r.severity or "medium").strip().lower()
                if sev not in VALID_SEVERITIES:
                    raise HTTPException(status_code=400, detail=f"规则 {name or rid} 风险等级无效: {r.severity}")
                clean_rules.append(
                    {
                        "id": rid,
                        "name": name or rid,
                        "severity": sev,
                        "description": desc,
                    }
                )
            normalized[key] = {
                "label": prof.label.strip() or DEFAULT_POLITICAL_PROFILES[key]["label"],
                "description": prof.description.strip(),
                "trigger_severities": trig,
                "rules": clean_rules,
            }
        cf["political_profiles"] = normalized

    save_editorial_config(cfg)
    return {
        "saved": True,
        **get_content_filter_config(),
    }


@app.post("/api/config/content-filter/reset")
def reset_content_filter_config(req: ResetContentFilterConfigRequest):
    cfg = load_editorial_config()
    cf = cfg.setdefault("content_filter", {})
    profiles = get_political_profiles(cfg)
    if req.profile is not None:
        if req.profile not in ("strict", "balanced", "loose"):
            raise HTTPException(status_code=400, detail=f"未知政审档位: {req.profile}")
        profiles[req.profile] = copy.deepcopy(DEFAULT_POLITICAL_PROFILES[req.profile])
    else:
        profiles = copy.deepcopy(DEFAULT_POLITICAL_PROFILES)
        cf["sensitivity"] = "strict"
        cf["political_action"] = "auto"
    cf["political_profiles"] = profiles
    save_editorial_config(cfg)
    return {
        "reset": True,
        **get_content_filter_config(),
    }


@app.get("/api/config/navigation")
def get_navigation_config():
    from ..render.renderer import NAV_COLOR_PRESETS, resolve_navigation_settings

    cfg = load_editorial_config()
    settings = resolve_navigation_settings(cfg)
    return {
        **settings,
        "presets": [
            {"id": k, "label": v["label"], "hex": v["hex"]}
            for k, v in NAV_COLOR_PRESETS.items()
        ],
    }


@app.put("/api/config/navigation")
def put_navigation_config(req: UpdateNavigationConfigRequest):
    from ..render.renderer import NAV_COLOR_PRESETS, _normalize_hex_color

    cfg = load_editorial_config()
    nav = cfg.setdefault("navigation", {})
    valid_styles = ("top_nav", "both", "title_card")
    if req.style is not None:
        st = str(req.style).strip().lower()
        if st not in valid_styles:
            raise HTTPException(status_code=400, detail=f"未知导航模式: {req.style}")
        nav["style"] = st
        if req.remix_style is None:
            nav["remix_style"] = st
        if req.compilation_style is None:
            nav["compilation_style"] = st
    if req.remix_style is not None:
        rst = str(req.remix_style).strip().lower()
        if rst not in valid_styles:
            raise HTTPException(status_code=400, detail=f"未知混剪导航模式: {req.remix_style}")
        nav["remix_style"] = rst
    if req.compilation_style is not None:
        cst = str(req.compilation_style).strip().lower()
        if cst not in valid_styles:
            raise HTTPException(status_code=400, detail=f"未知精选集导航模式: {req.compilation_style}")
        nav["compilation_style"] = cst
    if req.screen_layout is not None:
        sl = str(req.screen_layout).strip().lower()
        if sl not in ("portrait", "landscape"):
            raise HTTPException(status_code=400, detail=f"未知屏幕布局: {req.screen_layout}")
        nav["screen_layout"] = sl
    if req.color_scheme is not None:
        sc = str(req.color_scheme).strip().lower()
        if sc not in NAV_COLOR_PRESETS and sc != "custom":
            raise HTTPException(status_code=400, detail=f"未知导航色系: {req.color_scheme}")
        nav["color_scheme"] = sc
    if req.custom_accent is not None:
        norm = _normalize_hex_color(req.custom_accent)
        if not norm:
            raise HTTPException(status_code=400, detail=f"自定义颜色格式无效（需 #RRGGBB）: {req.custom_accent}")
        nav["custom_accent"] = norm
    save_editorial_config(cfg)
    return {
        "saved": True,
        **get_navigation_config(),
    }


# Serve compiled React SPA if present
WEB_DIST = PROJECT_ROOT / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
