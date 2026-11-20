from __future__ import annotations

import json
import mimetypes
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..adapters.ingest import ingest_source, resolve_work_dir, scan_input_path
from ..config import PROJECT_ROOT, load_editorial_config, load_editorial_raw, save_editorial_config, save_editorial_raw
from ..llm import MODEL_CATALOG, parse_provider_model
from ..pipeline.orchestrator import ALL_STAGES, run_existing_project_stages
from ..pipeline.verify_and_plan import (
    _load_audio_envelope,
    compute_clip_render_hash,
    compute_remix_render_hash,
    snap_bounds_with_acoustic_guard,
)
from ..publish import get_platforms_status, launch_platform_login, publish_to_platforms
from ..receipts import ProjectStore
from ..render.renderer import refresh_single_cover, render_single_clip_and_cover

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
_JOBS_LOCK = threading.Lock()


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
    stages: list[str] = Field(default_factory=lambda: ["signals", "analyze", "verify_and_plan"])
    force_render: bool = False
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
    items: list[PatchRemixItem] | None = None


class RefreshCoverRequest(BaseModel):
    target: str = "remix"


class PublishRequest(BaseModel):
    target: str
    platforms: list[str]
    title: str | None = None
    cover_title: str | None = None
    desc: str | None = None
    tags: list[str] | None = None
    is_public: bool = False
    tid: int | None = None
    ui_schema: int | None = None


class UpdateEditorialRequest(BaseModel):
    raw_yaml: str


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
    manifest = store.load()
    with _JOBS_LOCK:
        job = _ACTIVE_JOBS.get(project_id, {"running": False})
    return {
        "manifest": manifest.model_dump(),
        "job": job,
    }


@app.post("/api/projects/{project_id}/run")
def run_project_stages(project_id: str, req: RunStagesRequest):
    store = _get_store(project_id)
    with _JOBS_LOCK:
        if _ACTIVE_JOBS.get(project_id, {}).get("running"):
            raise HTTPException(status_code=409, detail="该项目已有流水线任务正在运行中")
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        _ACTIVE_JOBS[project_id] = {
            "job_id": job_id,
            "running": True,
            "stages": req.stages,
            "error": None,
        }

    def _worker():
        try:
            run_existing_project_stages(
                store.work_dir,
                req.stages,
                force_render=req.force_render,
                text_model=req.text_model,
                multimodal_model=req.multimodal_model,
            )
            with _JOBS_LOCK:
                _ACTIVE_JOBS[project_id]["running"] = False
        except Exception as exc:
            with _JOBS_LOCK:
                _ACTIVE_JOBS[project_id]["running"] = False
                _ACTIVE_JOBS[project_id]["error"] = str(exc)

    threading.Thread(target=_worker, daemon=True).start()
    return {"started": True, "job_id": job_id, "stages": req.stages}


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
    elif req.action == "top_n":
        sorted_indices = sorted(
            range(len(manifest.clips)),
            key=lambda i: getattr(manifest.clips[i], "score", 7.5),
            reverse=True,
        )
        top_set = set(sorted_indices[: req.n])
        for idx, c in enumerate(manifest.clips):
            c.enabled = idx in top_set
    else:
        raise HTTPException(status_code=400, detail=f"不支持的批量操作: {req.action}")

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

    target.render_hash = compute_clip_render_hash(target, manifest.source.render_video)
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
    with _JOBS_LOCK:
        if _ACTIVE_JOBS.get(project_id, {}).get("running"):
            raise HTTPException(status_code=409, detail="该项目已有后台任务正在运行中")
        job_id = f"pub_{uuid.uuid4().hex[:8]}"
        _ACTIVE_JOBS[project_id] = {
            "job_id": job_id,
            "running": True,
            "stages": ["publish"],
            "error": None,
        }

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
                tid=req.tid,
                ui_schema=req.ui_schema,
            )
            with _JOBS_LOCK:
                _ACTIVE_JOBS[project_id]["running"] = False
        except Exception as exc:
            with _JOBS_LOCK:
                _ACTIVE_JOBS[project_id]["running"] = False
                _ACTIVE_JOBS[project_id]["error"] = str(exc)

    threading.Thread(target=_worker, daemon=True).start()
    return {"started": True, "job_id": job_id, "target": req.target, "platforms": req.platforms}


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


# Serve compiled React SPA if present
WEB_DIST = PROJECT_ROOT / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
