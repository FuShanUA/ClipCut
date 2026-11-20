from __future__ import annotations

import base64
import io
import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..config import ASSETS_DIR

SWIFT_FACE_SRC = Path(__file__).resolve().parent.parent / "assets" / "face_detect.swift"
FACE_BIN_PATH = ASSETS_DIR / "clipcut_face_detect"
HERO_POOL_SCHEMA_VERSION = "v2-guest-id"


@dataclass
class FaceBox:
    x: float
    y: float
    w: float
    h: float
    quality: float = 0.5
    yaw: float = 0.0
    embedding: list[float] | None = None
    is_guest: bool | None = None

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w * 0.5

    @property
    def cy(self) -> float:
        return self.y + self.h * 0.5


@dataclass
class SelectedFrame:
    image_path: Path
    timestamp_s: float
    faces: list[FaceBox] = field(default_factory=list)
    is_two_shot: bool = False
    score: float = 0.0
    from_fallback: bool = False


@dataclass
class GuestProfile:
    version: str = "v1"
    source: str = ""
    guest_name: str = ""
    guest_embedding: list[float] = field(default_factory=list)
    host_embeddings: list[list[float]] = field(default_factory=list)
    verified_by: str = "screen_time_fallback"
    reason: str = ""


def _decode_emb_b64(b64_str: str | None) -> list[float] | None:
    if not b64_str or not isinstance(b64_str, str):
        return None
    try:
        raw = base64.b64decode(b64_str)
        if not raw or len(raw) % 4 != 0:
            return None
        arr = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
        norm = float(np.linalg.norm(arr))
        if norm > 1e-6:
            arr = arr / norm
        return arr.tolist()
    except Exception:
        return None


def _encode_emb_b64(emb: list[float] | None) -> str | None:
    if not emb:
        return None
    try:
        arr = np.asarray(emb, dtype=np.float32)
        return base64.b64encode(arr.tobytes()).decode("ascii")
    except Exception:
        return None


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-8 or nb <= 1e-8:
        return -1.0
    return float(dot / (na * nb))


def _normalize_vec(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-8:
        arr = arr / norm
    return arr.tolist()


def load_guest_profile(work_dir: Path) -> GuestProfile | None:
    prof_path = work_dir / "hero_frames" / "guest_profile.json"
    if not prof_path.exists():
        return None
    try:
        data = json.loads(prof_path.read_text(encoding="utf-8"))
        if data.get("version") != "v1":
            return None
        g_emb = _decode_emb_b64(data.get("guest_emb_b64")) or data.get("guest_embedding")
        if not g_emb or not isinstance(g_emb, list):
            return None
        h_embs: list[list[float]] = []
        for hb in data.get("host_embs_b64") or []:
            dec = _decode_emb_b64(hb)
            if dec:
                h_embs.append(dec)
        if not h_embs:
            for raw_h in data.get("host_embeddings") or []:
                if isinstance(raw_h, list) and raw_h:
                    h_embs.append([float(v) for v in raw_h])
        return GuestProfile(
            version="v1",
            source=str(data.get("source") or ""),
            guest_name=str(data.get("guest_name") or ""),
            guest_embedding=[float(v) for v in g_emb],
            host_embeddings=h_embs,
            verified_by=str(data.get("verified_by") or "screen_time_fallback"),
            reason=str(data.get("reason") or ""),
        )
    except Exception:
        return None


def _save_guest_profile(work_dir: Path, profile: GuestProfile) -> None:
    prof_dir = work_dir / "hero_frames"
    prof_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": profile.version,
        "source": profile.source,
        "guest_name": profile.guest_name,
        "guest_emb_b64": _encode_emb_b64(profile.guest_embedding),
        "host_embs_b64": [b for b in (_encode_emb_b64(h) for h in profile.host_embeddings) if b],
        "verified_by": profile.verified_by,
        "reason": profile.reason,
    }
    (prof_dir / "guest_profile.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def annotate_faces_with_guest(
    faces: list[FaceBox],
    guest_profile: GuestProfile | None,
) -> list[FaceBox]:
    if not faces:
        return []
    out: list[FaceBox] = []
    for f in faces:
        is_g = f.is_guest
        if guest_profile is not None and guest_profile.guest_embedding and f.embedding:
            sim_g = _cosine_sim(f.embedding, guest_profile.guest_embedding)
            sim_h = max((_cosine_sim(f.embedding, h) for h in guest_profile.host_embeddings), default=-1.0)
            if sim_g >= sim_h + 0.04 and (guest_profile.host_embeddings or sim_g >= 0.72):
                is_g = True
            elif sim_h >= sim_g + 0.04:
                is_g = False
            elif not guest_profile.host_embeddings and sim_g >= 0.72:
                is_g = True
            else:
                is_g = None
        out.append(
            FaceBox(
                x=f.x,
                y=f.y,
                w=f.w,
                h=f.h,
                quality=f.quality,
                yaw=f.yaw,
                embedding=f.embedding,
                is_guest=is_g,
            )
        )
    return out


def _build_cluster_contact_sheet(
    clusters: list[dict[str, Any]],
    out_path: Path | None = None,
) -> bytes | None:
    if not clusters:
        return None
    tile_w, tile_h = 320, 360
    header_h = 44
    sheet = Image.new("RGB", (tile_w * len(clusters), tile_h), (18, 22, 30))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except Exception:
        font = ImageFont.load_default()

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for idx, cl in enumerate(clusters):
        _, fp, fb = cl["best_rep"]
        x0 = idx * tile_w
        try:
            with Image.open(fp) as raw:
                im = raw.convert("RGB")
                iw, ih = im.size
                cx0 = max(0.0, fb.x - 0.45 * fb.w)
                cy0 = max(0.0, fb.y - 0.45 * fb.h)
                cx1 = min(1.0, fb.x + fb.w + 0.45 * fb.w)
                cy1 = min(1.0, fb.y + fb.h + 0.50 * fb.h)
                crop = im.crop((int(cx0 * iw), int(cy0 * ih), max(int(cx0 * iw) + 8, int(cx1 * iw)), max(int(cy0 * ih) + 8, int(cy1 * ih))))
                crop = crop.resize((tile_w, tile_h - header_h), Image.Resampling.LANCZOS)
                sheet.paste(crop, (x0, header_h))
        except Exception:
            pass
        draw.rectangle([x0, 0, x0 + tile_w - 1, header_h], fill=(245, 158, 11))
        lbl = f"Speaker {labels[idx]}"
        draw.text((x0 + 16, 8), lbl, font=font, fill=(15, 23, 42))
        if idx > 0:
            draw.line([(x0, 0), (x0, tile_h)], fill=(255, 255, 255), width=3)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out_path, format="JPEG", quality=88)
    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def identify_guest_profile(
    *,
    sampled_frames: list[tuple[float, Path]],
    faces_map: dict[str, list[FaceBox]],
    source_video: Path,
    work_dir: Path,
    guest_name: str = "",
    channel: str = "",
    original_title: str = "",
    cfg: dict[str, Any] | None = None,
) -> GuestProfile | None:
    """Cluster speakers across sampled video frames and identify which cluster is the Guest."""
    cached = load_guest_profile(work_dir)
    if (
        cached is not None
        and cached.source == str(source_video)
        and cached.guest_name == guest_name
        and cached.guest_embedding
    ):
        return cached

    clusters: list[dict[str, Any]] = []
    for _, fp in sampled_frames:
        valid = [f for f in (faces_map.get(str(fp)) or []) if f.h >= 0.16 and f.w >= 0.08 and f.embedding]
        solo_frame = len(valid) == 1
        for f in valid:
            emb = f.embedding or []
            best_idx = -1
            best_sim = 0.80
            for idx, cl in enumerate(clusters):
                sim = _cosine_sim(emb, cl["centroid"])
                if sim >= best_sim:
                    best_sim = sim
                    best_idx = idx
            rep_score = f.w * f.h * (0.4 + f.quality) * (1.25 if solo_frame else 1.0)
            if best_idx < 0:
                clusters.append(
                    {
                        "sum": np.asarray(emb, dtype=np.float64).copy(),
                        "centroid": _normalize_vec(emb),
                        "total_count": 1,
                        "solo_count": 1 if solo_frame else 0,
                        "best_rep": (rep_score, fp, f),
                    }
                )
            else:
                cl = clusters[best_idx]
                cl["sum"] = cl["sum"] + np.asarray(emb, dtype=np.float64)
                cl["centroid"] = _normalize_vec(cl["sum"].tolist())
                cl["total_count"] += 1
                if solo_frame:
                    cl["solo_count"] += 1
                if rep_score > cl["best_rep"][0]:
                    cl["best_rep"] = (rep_score, fp, f)

    if not clusters:
        return None

    clusters.sort(key=lambda c: (c["total_count"], c["solo_count"], c["best_rep"][0]), reverse=True)
    clusters = clusters[:4]

    if len(clusters) == 1:
        prof = GuestProfile(
            version="v1",
            source=str(source_video),
            guest_name=guest_name,
            guest_embedding=clusters[0]["centroid"],
            host_embeddings=[],
            verified_by="single_speaker",
            reason="全片仅检测到一位主要人物",
        )
        _save_guest_profile(work_dir, prof)
        return prof

    guest_idx = max(
        range(len(clusters)),
        key=lambda i: (clusters[i]["solo_count"], clusters[i]["total_count"], clusters[i]["best_rep"][0]),
    )
    verified_by = "screen_time_fallback"
    reason = "按单人出镜与总出镜频次确定主角"

    if cfg and (guest_name or original_title or channel):
        sheet_path = work_dir / "hero_frames" / "guest_clusters.jpg"
        sheet_bytes = _build_cluster_contact_sheet(clusters, out_path=sheet_path)
        if sheet_bytes:
            labels = [chr(ord("A") + i) for i in range(len(clusters))]
            label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
            try:
                from ..llm import ClipCutLLM, DEFAULT_MULTIMODAL_FALLBACK_MODELS

                models_cfg = cfg.get("models", {})
                llm = ClipCutLLM(models_cfg)
                mm_model = str(models_cfg.get("multimodal_model") or "qwen3-vl-plus")
                mm_prov = str(models_cfg.get("multimodal_provider") or "dashscope")
                mm_fb = list(models_cfg.get("multimodal_fallback_models") or DEFAULT_MULTIMODAL_FALLBACK_MODELS)
                prompt = (
                    "这是一期访谈/播客节目中提取的几位说话人的画面拼接对比图（顶部标注为 Speaker A, Speaker B, ...）。\n"
                    f"原视频标题：{original_title or '未知'}\n"
                    f"频道/节目名：{channel or '未知'}\n"
                    f"本期受访嘉宾 / 主角姓名：{guest_name or '受访核心嘉宾'}\n\n"
                    "请判断哪一位字母编号对应【本期受访嘉宾 / 主角】（被采访者，而非播客主持人/提问者）。\n"
                    f"可选编号：{', '.join(labels)}。\n"
                    "仅返回严格 JSON（不要加 markdown 代码块）：\n"
                    '{"guest_label": "A", "host_labels": ["B"], "reason": "一句话说明识别依据"}'
                )
                res = llm.generate_json(
                    prompt=prompt,
                    model=mm_model,
                    provider=mm_prov,
                    fallback_models=mm_fb,
                    temperature=0.0,
                    max_output_tokens=512,
                    media_bytes=sheet_bytes,
                    media_mime="image/jpeg",
                )
                parsed = res.parsed if isinstance(res.parsed, dict) else {}
                raw_lbl = str(parsed.get("guest_label") or "").strip().upper()
                m_lbl = raw_lbl[-1:] if raw_lbl else ""
                if m_lbl in label_to_idx:
                    guest_idx = label_to_idx[m_lbl]
                    verified_by = "vlm"
                    reason = str(parsed.get("reason") or reason).strip()
            except Exception:
                pass

    prof = GuestProfile(
        version="v1",
        source=str(source_video),
        guest_name=guest_name,
        guest_embedding=clusters[guest_idx]["centroid"],
        host_embeddings=[c["centroid"] for i, c in enumerate(clusters) if i != guest_idx],
        verified_by=verified_by,
        reason=reason,
    )
    _save_guest_profile(work_dir, prof)
    return prof


def _ensure_face_detector_binary() -> Path | None:
    """Compile and cache macOS Apple Vision face detector binary on first use."""
    if not SWIFT_FACE_SRC.exists():
        return None
    if FACE_BIN_PATH.exists():
        try:
            if FACE_BIN_PATH.stat().st_mtime >= SWIFT_FACE_SRC.stat().st_mtime:
                return FACE_BIN_PATH
        except OSError:
            pass
    swiftc = shutil.which("swiftc")
    if not swiftc:
        return None
    FACE_BIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        [swiftc, "-O", str(SWIFT_FACE_SRC), "-o", str(FACE_BIN_PATH)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if res.returncode == 0 and FACE_BIN_PATH.exists():
        return FACE_BIN_PATH
    return None


def detect_faces_batch(image_paths: list[Path]) -> dict[str, list[FaceBox]]:
    """Run batch Apple Vision face capture quality & embedding extraction on a list of frames."""
    existing = [str(p) for p in image_paths if p.exists()]
    if not existing:
        return {}
    bin_path = _ensure_face_detector_binary()
    if bin_path is None:
        return {p: [] for p in existing}

    try:
        res = subprocess.run(
            [str(bin_path), *existing],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if res.returncode != 0 or not res.stdout.strip():
            return {p: [] for p in existing}
        raw = json.loads(res.stdout)
        out: dict[str, list[FaceBox]] = {}
        for p in existing:
            boxes = []
            for item in raw.get(p) or []:
                boxes.append(
                    FaceBox(
                        x=float(item.get("x", 0.0)),
                        y=float(item.get("y", 0.0)),
                        w=float(item.get("w", 0.0)),
                        h=float(item.get("h", 0.0)),
                        quality=float(item.get("quality", 0.5)),
                        yaw=float(item.get("yaw", 0.0)),
                        embedding=_decode_emb_b64(item.get("emb_b64")),
                    )
                )
            out[p] = boxes
        return out
    except Exception:
        return {p: [] for p in existing}


def _compute_luma_and_sharpness(img_path: Path) -> tuple[float, float]:
    with Image.open(img_path) as im:
        gray = im.convert("L").resize((160, 90), Image.Resampling.BILINEAR)
        arr = np.asarray(gray, dtype=np.float32)
    mean_luma = float(arr.mean())
    gx = float(np.abs(np.diff(arr, axis=1)).mean())
    gy = float(np.abs(np.diff(arr, axis=0)).mean())
    sharpness = min(1.5, (gx + gy) / 18.0)
    return mean_luma, sharpness


def score_candidate_frame(
    img_path: Path,
    faces: list[FaceBox],
    prefer_two_shot: bool = True,
    guest_profile: GuestProfile | None = None,
    prefer_guest_solo: bool = False,
) -> tuple[float, list[FaceBox], bool]:
    """
    Score a candidate frame for cover / PiP aesthetics with guest-protagonist enforcement.
    Rejects very dark frames, tiny screen-share PiP corner faces, and host-only solo shots.
    """
    try:
        mean_luma, sharpness = _compute_luma_and_sharpness(img_path)
    except Exception:
        return 0.0, [], False

    if mean_luma < 24.0:
        return 0.05, [], False

    annotated = annotate_faces_with_guest(faces, guest_profile)
    valid = [f for f in annotated if f.h >= 0.16 and f.w >= 0.08]
    if not valid:
        base = min(0.45, 0.15 + 0.1 * sharpness + min(0.15, mean_luma / 600.0))
        return round(base, 4), [], False

    # Sort descending by area * quality
    ranked = sorted(valid, key=lambda f: f.w * f.h * (0.5 + f.quality), reverse=True)

    # Check for clean two-shot framing (host + guest side by side)
    if len(ranked) >= 2:
        top_pair = sorted(ranked[:3], key=lambda f: f.cx)
        best_two: tuple[FaceBox, FaceBox] | None = None
        best_sep = 0.0
        for i in range(len(top_pair)):
            for j in range(i + 1, len(top_pair)):
                a, b = top_pair[i], top_pair[j]
                sep = abs(b.cx - a.cx)
                if sep >= 0.22 and min(a.h, b.h) >= 0.18 and min(a.w, b.w) >= 0.10 and sep > best_sep:
                    best_two = (a, b)
                    best_sep = sep
        if best_two is not None:
            a, b = best_two
            avg_q = (a.quality + b.quality) * 0.5
            max_bottom = max(a.bottom, b.bottom)
            bottom_safety = 0.6 if max_bottom <= 0.74 else max(0.0, 0.6 - (max_bottom - 0.74) * 2.5)
            yaw_bonus = max(0.0, 0.35 - 0.25 * (abs(a.yaw) + abs(b.yaw)) * 0.5)
            two_shot_bonus = 0.35 if prefer_guest_solo else (1.15 if prefer_two_shot else 0.35)
            guest_adj = 0.0
            if guest_profile is not None:
                if a.is_guest is True or b.is_guest is True:
                    guest_adj = 0.8
                elif a.is_guest is False and b.is_guest is False:
                    guest_adj = -3.0
            score = 2.6 + two_shot_bonus + 1.8 * avg_q + bottom_safety + yaw_bonus + 0.3 * sharpness + guest_adj
            return round(score, 4), [a, b], True

    # Single dominant portrait
    dom = ranked[0]
    if 0.22 <= dom.h <= 0.62:
        size_score = 0.8
    elif dom.h < 0.22:
        size_score = 0.3
    elif dom.h <= 0.72:
        size_score = 0.55
    else:
        size_score = 0.1
    center_bonus = max(0.0, 0.45 * (1.0 - abs(dom.cx - 0.5) * 1.4))
    bottom_safety = 0.55 if dom.bottom <= 0.74 else max(0.0, 0.55 - (dom.bottom - 0.74) * 2.2)
    yaw_bonus = max(0.0, 0.4 - 0.45 * abs(dom.yaw))
    score = 1.6 + 2.2 * dom.quality + size_score + center_bonus + bottom_safety + yaw_bonus + 0.25 * sharpness
    if guest_profile is not None:
        if dom.is_guest is True:
            score += 1.5 + (0.6 if prefer_guest_solo else 0.0)
        elif dom.is_guest is False:
            score = min(0.55, score - 3.8)
    return round(score, 4), [dom], False


def _extract_single_frame(video_path: Path, timestamp_s: float, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{max(0.0, timestamp_s):.2f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1024


def _serialize_frame(sf: SelectedFrame) -> dict[str, Any]:
    return {
        "image_path": str(sf.image_path),
        "timestamp_s": round(sf.timestamp_s, 3),
        "is_two_shot": sf.is_two_shot,
        "score": sf.score,
        "from_fallback": sf.from_fallback,
        "faces": [
            {
                "x": f.x,
                "y": f.y,
                "w": f.w,
                "h": f.h,
                "quality": f.quality,
                "yaw": f.yaw,
                "is_guest": f.is_guest,
                **({"emb_b64": eb} if (eb := _encode_emb_b64(f.embedding)) else {}),
            }
            for f in sf.faces
        ],
    }


def _deserialize_frame(data: dict[str, Any]) -> SelectedFrame | None:
    p = Path(str(data.get("image_path") or ""))
    if not p.exists():
        return None
    faces = [
        FaceBox(
            x=float(f.get("x", 0.0)),
            y=float(f.get("y", 0.0)),
            w=float(f.get("w", 0.0)),
            h=float(f.get("h", 0.0)),
            quality=float(f.get("quality", 0.5)),
            yaw=float(f.get("yaw", 0.0)),
            embedding=_decode_emb_b64(f.get("emb_b64")),
            is_guest=f.get("is_guest"),
        )
        for f in (data.get("faces") or [])
    ]
    return SelectedFrame(
        image_path=p,
        timestamp_s=float(data.get("timestamp_s", 0.0)),
        faces=faces,
        is_two_shot=bool(data.get("is_two_shot", False)),
        score=float(data.get("score", 0.0)),
        from_fallback=bool(data.get("from_fallback", False)),
    )


def build_hero_portrait_pool(
    source_video: Path,
    duration_s: float,
    work_dir: Path,
    prefer_two_shot: bool = True,
    guest_name: str = "",
    channel: str = "",
    original_title: str = "",
    cfg: dict[str, Any] | None = None,
) -> list[SelectedFrame]:
    """Build and cache up to 6 top-scoring guest-verified portrait / two-shot frames across the full video."""
    hero_dir = work_dir / "hero_frames"
    index_path = hero_dir / "index.json"
    if index_path.exists():
        try:
            cached = json.loads(index_path.read_text(encoding="utf-8"))
            if cached.get("schema") == HERO_POOL_SCHEMA_VERSION and cached.get("source") == str(source_video):
                loaded = [x for item in (cached.get("frames") or []) if (x := _deserialize_frame(item)) is not None]
                if loaded:
                    return loaded
        except Exception:
            pass

    if not source_video.exists() or duration_s <= 1.0:
        return []

    hero_dir.mkdir(parents=True, exist_ok=True)
    lo = duration_s * 0.06
    hi = max(lo + 5.0, duration_s * 0.94)
    ratios = np.linspace(0.0, 1.0, 16)
    cands: list[tuple[float, Path]] = []
    for idx, r in enumerate(ratios):
        ts = lo + float(r) * (hi - lo)
        fp = hero_dir / f"cand_{idx:02d}.jpg"
        if _extract_single_frame(source_video, ts, fp):
            cands.append((ts, fp))

    if not cands:
        return []

    faces_map = detect_faces_batch([p for _, p in cands])
    guest_profile = identify_guest_profile(
        sampled_frames=cands,
        faces_map=faces_map,
        source_video=source_video,
        work_dir=work_dir,
        guest_name=guest_name,
        channel=channel,
        original_title=original_title,
        cfg=cfg,
    )
    scored: list[SelectedFrame] = []
    for ts, fp in cands:
        sc, valid_faces, is_two = score_candidate_frame(
            fp,
            faces_map.get(str(fp), []),
            prefer_two_shot=prefer_two_shot,
            guest_profile=guest_profile,
        )
        if valid_faces and sc >= 1.2:
            scored.append(
                SelectedFrame(
                    image_path=fp,
                    timestamp_s=ts,
                    faces=valid_faces,
                    is_two_shot=is_two,
                    score=sc,
                    from_fallback=False,
                )
            )

    scored.sort(key=lambda x: x.score, reverse=True)
    guest_solos = [x for x in scored if (not x.is_two_shot) and any(f.is_guest is True for f in x.faces)]
    two_shots = [x for x in scored if x.is_two_shot]

    curated: list[SelectedFrame] = []
    seen_paths: set[Path] = set()
    for item in two_shots[:4] + guest_solos[:2] + scored:
        if item.image_path in seen_paths:
            continue
        curated.append(item)
        seen_paths.add(item.image_path)
        if len(curated) >= 6:
            break
    for item in guest_solos[:2]:
        if item.image_path not in seen_paths:
            if len(curated) >= 6:
                curated[-1] = item
            else:
                curated.append(item)
            seen_paths.add(item.image_path)
    curated.sort(key=lambda x: x.score, reverse=True)
    kept = curated[:6]
    if not kept:
        # Still keep highest-clarity middle frame if vision detected zero faces
        ts, fp = cands[len(cands) // 2]
        sc, valid_faces, is_two = score_candidate_frame(
            fp,
            faces_map.get(str(fp), []),
            prefer_two_shot=prefer_two_shot,
            guest_profile=guest_profile,
        )
        kept = [SelectedFrame(image_path=fp, timestamp_s=ts, faces=valid_faces, is_two_shot=is_two, score=sc, from_fallback=False)]

    kept_paths = {x.image_path for x in kept}
    for _, fp in cands:
        if fp not in kept_paths:
            fp.unlink(missing_ok=True)

    index_path.write_text(
        json.dumps(
            {
                "schema": HERO_POOL_SCHEMA_VERSION,
                "source": str(source_video),
                "frames": [_serialize_frame(x) for x in kept],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return kept


def _fallback_from_hero_pool(hero_pool: list[SelectedFrame], fallback_index: int) -> SelectedFrame | None:
    if not hero_pool:
        return None
    guest_frames = [sf for sf in hero_pool if any(f.is_guest is True for f in sf.faces)]
    pool = guest_frames or hero_pool
    picked = pool[fallback_index % len(pool)]
    return SelectedFrame(
        image_path=picked.image_path,
        timestamp_s=picked.timestamp_s,
        faces=picked.faces,
        is_two_shot=picked.is_two_shot,
        score=picked.score,
        from_fallback=True,
    )


def select_best_frame_for_interval(
    source_video: Path,
    start_s: float,
    end_s: float,
    work_dir: Path,
    cache_key: str,
    hero_pool: list[SelectedFrame],
    fallback_index: int = 0,
    prefer_two_shot: bool = True,
    guest_profile: GuestProfile | None = None,
    prefer_guest_solo: bool = False,
    sample_count: int = 10,
) -> SelectedFrame | None:
    """Select the sharpest, cleanest guest portrait/two-shot frame in [start_s, end_s], or fallback to hero_pool."""
    if not source_video.exists():
        return _fallback_from_hero_pool(hero_pool, fallback_index)

    dur = max(1.0, end_s - start_s)
    pad = min(2.0, dur * 0.08)
    lo = start_s + pad
    hi = max(lo + 0.5, end_s - pad)
    sample_dir = work_dir / "cover_candidates"
    sample_dir.mkdir(parents=True, exist_ok=True)

    cands: list[tuple[float, Path]] = []
    for i, frac in enumerate(np.linspace(0.08, 0.92, max(2, sample_count))):
        ts = lo + float(frac) * (hi - lo)
        fp = sample_dir / f"{cache_key}_{i}.jpg"
        if _extract_single_frame(source_video, ts, fp):
            cands.append((ts, fp))

    faces_map = detect_faces_batch([p for _, p in cands])
    best_sf: SelectedFrame | None = None
    for ts, fp in cands:
        sc, valid_faces, is_two = score_candidate_frame(
            fp,
            faces_map.get(str(fp), []),
            prefer_two_shot=prefer_two_shot,
            guest_profile=guest_profile,
            prefer_guest_solo=prefer_guest_solo,
        )
        sf = SelectedFrame(
            image_path=fp,
            timestamp_s=ts,
            faces=valid_faces,
            is_two_shot=is_two,
            score=sc,
            from_fallback=False,
        )
        if best_sf is None or sf.score > best_sf.score:
            best_sf = sf

    # Check if we got a valid crisp guest-verified portrait/two-shot inside this interval
    has_only_host = guest_profile is not None and best_sf is not None and bool(best_sf.faces) and all(f.is_guest is False for f in best_sf.faces)
    if best_sf is not None and best_sf.faces and best_sf.score >= 1.2 and not has_only_host:
        chosen_raw = sample_dir / f"{cache_key}_selected.jpg"
        best_sf.image_path.replace(chosen_raw)
        for _, fp in cands:
            fp.unlink(missing_ok=True)
        return SelectedFrame(
            image_path=chosen_raw,
            timestamp_s=best_sf.timestamp_s,
            faces=best_sf.faces,
            is_two_shot=best_sf.is_two_shot,
            score=best_sf.score,
            from_fallback=False,
        )

    # Screen-share or host-only cutaway: clean up samples & rotate cleanly through guest-bearing hero_pool
    for _, fp in cands:
        fp.unlink(missing_ok=True)
    fb = _fallback_from_hero_pool(hero_pool, fallback_index)
    return fb if fb is not None else best_sf


def _pick_primary_face(cands: list[FaceBox]) -> FaceBox:
    guests = [f for f in cands if f.is_guest is True]
    non_hosts = [f for f in cands if f.is_guest is not False]
    pool = guests or non_hosts or cands
    return max(pool, key=lambda f: f.w * f.h * (0.5 + f.quality))


def crop_frame_face_safe(
    img: Image.Image,
    target_w: int,
    target_h: int,
    faces: list[FaceBox],
    safe_bottom_ratio: float = 0.71,
) -> tuple[Image.Image, list[FaceBox]]:
    """
    Aspect-fill resize img to (target_w, target_h).
    On two-shot horizontal->portrait crops, locks onto the Guest face instead of the seam or host.
    If faces extend below safe_bottom_ratio, gently zoom up to 1.22x and anchor downward
    so faces lift clear of the bottom overlay box without slicing heads.
    """
    orig_w, orig_h = img.size
    if orig_w <= 0 or orig_h <= 0:
        return img.resize((target_w, target_h)), faces

    s_ar = orig_w / orig_h
    t_ar = target_w / target_h
    if s_ar >= t_ar:
        base_vh = 1.0
        base_vw = (orig_h * t_ar) / orig_w
    else:
        base_vw = 1.0
        base_vh = (orig_w / t_ar) / orig_h

    active_faces = list(faces or [])
    if len(active_faces) >= 2:
        span_x = max(f.x + f.w for f in active_faces) - min(f.x for f in active_faces)
        if span_x > base_vw * 0.90:
            active_faces = [_pick_primary_face(active_faces)]

    zoom = 1.0
    if active_faces:
        max_b = max(f.bottom for f in active_faces)
        min_t = min(f.y for f in active_faces)
        mapped_b = max_b / base_vh
        if mapped_b > safe_bottom_ratio and max_b < 0.96:
            # New bottom ratio when bottom-anchored is 1 - zoom * (1 - mapped_b)
            denom = max(0.05, 1.0 - mapped_b)
            needed_zoom = (1.0 - safe_bottom_ratio) / denom
            max_headroom_zoom = min(1.22, 1.0 / max(0.65, 1.0 - max(0.0, min_t - 0.04)))
            zoom = max(1.0, min(1.22, max_headroom_zoom, needed_zoom))

    vw = min(1.0, base_vw / zoom)
    vh = min(1.0, base_vh / zoom)

    if len(active_faces) >= 2:
        span_x = max(f.x + f.w for f in active_faces) - min(f.x for f in active_faces)
        if span_x > vw * 0.90:
            active_faces = [_pick_primary_face(active_faces)]

    if active_faces:
        avg_cx = sum(f.cx for f in active_faces) / len(active_faces)
        vx0 = max(0.0, min(1.0 - vw, avg_cx - vw * 0.5))
        max_b = max(f.bottom for f in active_faces)
        min_t = min(f.y for f in active_faces)
        if zoom > 1.001 or base_vh < 0.999:
            # Prefer shifting crop window downward (larger vy0) to lift faces above bottom box
            ideal_vy0 = min(1.0 - vh, max(0.0, max_b - safe_bottom_ratio * vh))
            max_safe_head = max(0.0, min_t - 0.04 * vh)
            vy0 = min(ideal_vy0, max_safe_head, 1.0 - vh)
        else:
            vy0 = 0.0
    else:
        vx0 = (1.0 - vw) * 0.5
        vy0 = (1.0 - vh) * 0.5

    px0 = int(round(vx0 * orig_w))
    py0 = int(round(vy0 * orig_h))
    px1 = max(px0 + 1, int(round((vx0 + vw) * orig_w)))
    py1 = max(py0 + 1, int(round((vy0 + vh) * orig_h)))
    cropped = img.crop((px0, py0, px1, py1)).resize((target_w, target_h), Image.Resampling.LANCZOS)

    new_faces: list[FaceBox] = []
    for f in faces:
        ix0 = max(f.x, vx0)
        iy0 = max(f.y, vy0)
        ix1 = min(f.x + f.w, vx0 + vw)
        iy1 = min(f.y + f.h, vy0 + vh)
        iw = max(0.0, ix1 - ix0)
        ih = max(0.0, iy1 - iy0)
        orig_area = max(1e-9, f.w * f.h)
        if (iw * ih) / orig_area < 0.35:
            continue
        nx = max(0.0, min(1.0, (ix0 - vx0) / vw))
        ny = max(0.0, min(1.0, (iy0 - vy0) / vh))
        nw = max(0.0, min(1.0 - nx, iw / vw))
        nh = max(0.0, min(1.0 - ny, ih / vh))
        new_faces.append(
            FaceBox(
                x=nx,
                y=ny,
                w=nw,
                h=nh,
                quality=f.quality,
                yaw=f.yaw,
                embedding=f.embedding,
                is_guest=f.is_guest,
            )
        )
    return cropped, new_faces


def compose_cinema_fit_width_frame(
    img: Image.Image,
    target_w: int,
    target_h: int,
    faces: list[FaceBox],
) -> tuple[Image.Image, list[FaceBox]]:
    """
    Preserve the complete 16:9 two-shot framing inside a 4:3 (e.g. 1080x810) canvas
    over a darkened ambient-blurred backdrop, leaving top room for badges and bottom room for the title plate.
    """
    orig_w, orig_h = img.size
    if orig_w <= 0 or orig_h <= 0:
        return img.resize((target_w, target_h)), faces

    rgba = img.convert("RGBA")
    s_ar = orig_w / orig_h
    t_ar = target_w / target_h
    if s_ar >= t_ar:
        crop_w = int(round(orig_h * t_ar))
        cx0 = max(0, (orig_w - crop_w) // 2)
        amb = rgba.crop((cx0, 0, min(orig_w, cx0 + crop_w), orig_h))
    else:
        crop_h = int(round(orig_w / t_ar))
        cy0 = max(0, (orig_h - crop_h) // 2)
        amb = rgba.crop((0, cy0, orig_w, min(orig_h, cy0 + crop_h)))

    amb = amb.resize((target_w, target_h), Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(radius=30))
    veil = Image.new("RGBA", (target_w, target_h), (6, 12, 16, 165))
    canvas = Image.alpha_composite(amb, veil)

    fit_w = target_w
    fit_h = max(1, min(target_h, int(round(target_w / s_ar))))
    y_offset = max(0, min(target_h - fit_h, int(target_h * 0.095)))

    fg = rgba.resize((fit_w, fit_h), Image.Resampling.LANCZOS)
    canvas.paste(fg, (0, y_offset))

    new_faces: list[FaceBox] = []
    for f in faces:
        ny = max(0.0, min(1.0, (y_offset + f.y * fit_h) / target_h))
        nh = max(0.0, min(1.0 - ny, (f.h * fit_h) / target_h))
        new_faces.append(
            FaceBox(
                x=max(0.0, min(1.0, f.x)),
                y=ny,
                w=max(0.0, min(1.0 - f.x, f.w)),
                h=nh,
                quality=f.quality,
                yaw=f.yaw,
                embedding=f.embedding,
                is_guest=f.is_guest,
            )
        )
    return canvas, new_faces


def crop_pip_portrait(
    img: Image.Image,
    target_w: int,
    target_h: int,
    faces: list[FaceBox],
) -> Image.Image:
    """Crop an image cleanly to target_w x target_h centered slightly above midpoint around faces."""
    orig_w, orig_h = img.size
    if orig_w <= 0 or orig_h <= 0:
        return img.resize((target_w, target_h))

    s_ar = orig_w / orig_h
    t_ar = target_w / target_h
    if s_ar >= t_ar:
        vh = 1.0
        vw = (orig_h * t_ar) / orig_w
    else:
        vw = 1.0
        vh = (orig_w / t_ar) / orig_h

    if faces:
        avg_cx = sum(f.cx for f in faces) / len(faces)
        avg_cy = sum(f.cy for f in faces) / len(faces)
        vx0 = max(0.0, min(1.0 - vw, avg_cx - vw * 0.5))
        vy0 = max(0.0, min(1.0 - vh, avg_cy - vh * 0.42))
        min_t = min(f.y for f in faces)
        vy0 = min(vy0, max(0.0, min_t - 0.03 * vh))
    else:
        vx0 = (1.0 - vw) * 0.5
        vy0 = (1.0 - vh) * 0.42

    px0 = int(round(vx0 * orig_w))
    py0 = int(round(vy0 * orig_h))
    px1 = max(px0 + 1, int(round((vx0 + vw) * orig_w)))
    py1 = max(py0 + 1, int(round((vy0 + vh) * orig_h)))
    return img.crop((px0, py0, px1, py1)).resize((target_w, target_h), Image.Resampling.LANCZOS)
