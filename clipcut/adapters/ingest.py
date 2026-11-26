from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import srt

from ..models import (
    Chapter,
    MediaSource,
    ProjectManifest,
    ScanResult,
    Utterance,
    VideoFileCandidate,
    VideoMetadata,
)
from ..receipts import ProjectStore

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
SUBTITLE_EXTENSIONS = {".srt", ".vtt"}


def probe_video(video_path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return {"duration_s": 0.0, "width": 1920, "height": 1080, "fps": 30.0, "video_codec": "", "audio_codec": ""}
    data = json.loads(res.stdout or "{}")
    fmt = data.get("format", {})
    duration_s = float(fmt.get("duration") or 0.0)
    width, height, fps = 1920, 1080, 30.0
    v_codec, a_codec = "", ""
    for st in data.get("streams", []):
        ctype = st.get("codec_type")
        if ctype == "video" and not v_codec:
            width = int(st.get("width") or 1920)
            height = int(st.get("height") or 1080)
            v_codec = str(st.get("codec_name") or "")
            rate = str(st.get("avg_frame_rate") or st.get("r_frame_rate") or "30/1")
            if "/" in rate:
                num, den = rate.split("/", 1)
                try:
                    fps = float(num) / float(den) if float(den) > 0 else 30.0
                except Exception:
                    fps = 30.0
            else:
                try:
                    fps = float(rate)
                except Exception:
                    fps = 30.0
        elif ctype == "audio" and not a_codec:
            a_codec = str(st.get("codec_name") or "")
    return {
        "duration_s": duration_s,
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "video_codec": v_codec,
        "audio_codec": a_codec,
    }


def _strip_hardsub_suffix(stem: str) -> str:
    return re.sub(r"_hardsub$", "", stem, flags=re.IGNORECASE)


def _find_best_subtitle(video_path: Path, folder: Path) -> Path | None:
    base = _strip_hardsub_suffix(video_path.stem)
    # Priority 1: exact stem matches
    priority_names = [
        f"{base}.bi.srt",
        f"{video_path.stem}.bi.srt",
        f"{base}.cn.srt",
        f"{base}.en.srt",
        f"{base}.srt",
        f"{video_path.stem}.srt",
        f"{base}.vtt",
    ]
    for name in priority_names:
        cand = folder / name
        if cand.exists():
            return cand
    # Priority 2: any sibling subtitle in folder, preferring .bi.srt
    subs = [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in SUBTITLE_EXTENSIONS
    ]
    if not subs:
        return None
    for s in subs:
        if s.name.lower().endswith(".bi.srt"):
            return s
    for s in subs:
        if s.name.lower().endswith(".cn.srt"):
            return s
    return subs[0]


def _find_best_info_json(video_path: Path, folder: Path) -> Path | None:
    base = _strip_hardsub_suffix(video_path.stem)
    for name in (f"{base}.info.json", f"{video_path.stem}.info.json", "metadata.json"):
        p = folder / name
        if p.exists():
            return p
    infos = [p for p in sorted(folder.iterdir()) if p.is_file() and p.name.endswith(".info.json") and not p.name.startswith(".")]
    return infos[0] if infos else None


def _find_clean_analysis_video(selected_video: Path) -> Path:
    """If user chose *_hardsub.mp4 and raw *.mp4 exists alongside, use raw for motion analysis."""
    if not selected_video.stem.lower().endswith("_hardsub"):
        return selected_video
    base = _strip_hardsub_suffix(selected_video.stem)
    for ext in (".mp4", ".mkv", ".mov", ".webm"):
        cand = selected_video.parent / f"{base}{ext}"
        if cand.exists() and cand.is_file():
            return cand
    return selected_video


def scan_input_path(input_path: str | Path) -> ScanResult:
    p = Path(input_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"路径不存在: {p}")

    if p.is_file():
        if p.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"不支持的视频格式: {p.suffix}")
        folder = p.parent
        info = probe_video(p)
        sub = _find_best_subtitle(p, folder)
        inf = _find_best_info_json(p, folder)
        cand = VideoFileCandidate(
            path=str(p),
            filename=p.name,
            size_bytes=p.stat().st_size,
            duration_s=info["duration_s"],
            width=info["width"],
            height=info["height"],
            fps=info["fps"],
            is_hardsub=p.stem.lower().endswith("_hardsub"),
            suggested_subtitle=str(sub) if sub else None,
            suggested_info_json=str(inf) if inf else None,
        )
        subs = [str(f) for f in sorted(folder.iterdir()) if f.is_file() and f.suffix.lower() in SUBTITLE_EXTENSIONS and not f.name.startswith(".")]
        infos = [str(f) for f in sorted(folder.iterdir()) if f.is_file() and f.name.endswith(".info.json") and not f.name.startswith(".")]
        return ScanResult(input_path=str(p), is_directory=False, videos=[cand], subtitles=subs, info_jsons=infos)

    # Directory scan (top-level only, skip .clipcut and clips)
    videos: list[VideoFileCandidate] = []
    subtitles: list[str] = []
    info_jsons: list[str] = []
    for f in sorted(p.iterdir()):
        if not f.is_file() or f.name.startswith("."):
            continue
        suf = f.suffix.lower()
        if suf in VIDEO_EXTENSIONS:
            info = probe_video(f)
            sub = _find_best_subtitle(f, p)
            inf = _find_best_info_json(f, p)
            videos.append(
                VideoFileCandidate(
                    path=str(f),
                    filename=f.name,
                    size_bytes=f.stat().st_size,
                    duration_s=info["duration_s"],
                    width=info["width"],
                    height=info["height"],
                    fps=info["fps"],
                    is_hardsub=f.stem.lower().endswith("_hardsub"),
                    suggested_subtitle=str(sub) if sub else None,
                    suggested_info_json=str(inf) if inf else None,
                )
            )
        elif suf in SUBTITLE_EXTENSIONS:
            subtitles.append(str(f))
        elif f.name.endswith(".info.json") or f.name == "metadata.json":
            info_jsons.append(str(f))

    # Sort hardsub first for convenience, then by size descending
    videos.sort(key=lambda v: (not v.is_hardsub, -v.size_bytes))
    return ScanResult(
        input_path=str(p),
        is_directory=True,
        videos=videos,
        subtitles=subtitles,
        info_jsons=info_jsons,
    )


def extract_youtube_id(text: str) -> str | None:
    # Matches [11-char ID] typical of yt-dlp filenames or folder names
    m = re.search(r"\[([A-Za-z0-9_-]{11})\]", text)
    if m:
        return m.group(1)
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", text)
    return m.group(1) if m else None


def parse_chapters_from_description(description: str, duration_s: float) -> list[Chapter]:
    if not description:
        return []
    pattern = re.compile(r"^\s*(?:\[)?(\d{1,2}:)?(\d{1,2}):(\d{2})(?:\])?\s*[-–—|:]?\s*(.+?)\s*$", re.MULTILINE)
    raw_points: list[tuple[float, str]] = []
    for m in pattern.finditer(description):
        h_str, m_str, s_str, title = m.groups()
        h = int(h_str.rstrip(":")) if h_str else 0
        sec = h * 3600 + int(m_str) * 60 + int(s_str)
        clean_title = title.strip(" -–—|:")
        if clean_title and sec <= max(duration_s, 86400):
            raw_points.append((float(sec), clean_title))

    # Deduplicate & ensure ascending order
    raw_points.sort(key=lambda x: x[0])
    deduped: list[tuple[float, str]] = []
    for sec, title in raw_points:
        if deduped and abs(sec - deduped[-1][0]) < 1.0:
            continue
        deduped.append((sec, title))
    if len(deduped) < 2:
        return []

    chapters: list[Chapter] = []
    for i, (start_s, title) in enumerate(deduped):
        end_s = deduped[i + 1][0] if i + 1 < len(deduped) else max(duration_s, start_s + 60.0)
        chapters.append(
            Chapter(index=i, title=title, start_s=start_s, end_s=end_s, source="description")
        )
    return chapters


def extract_published_date(raw_meta: dict[str, Any] | None) -> str:
    """Extract normalized YYYY-MM-DD broadcast/upload date from yt-dlp/info_json metadata."""
    if not isinstance(raw_meta, dict):
        return ""
    for key in ("release_date", "upload_date"):
        val = str(raw_meta.get(key) or "").strip()
        if re.fullmatch(r"\d{8}", val):
            return f"{val[:4]}-{val[4:6]}-{val[6:8]}"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", val):
            return val
    for key in ("release_timestamp", "timestamp"):
        ts_val = raw_meta.get(key)
        if ts_val is not None:
            try:
                return datetime.fromtimestamp(float(ts_val), tz=timezone.utc).strftime("%Y-%m-%d")
            except Exception:
                continue
    return ""


def fetch_ytdlp_metadata(url_or_id: str) -> dict[str, Any] | None:
    target = url_or_id if url_or_id.startswith("http") else f"https://www.youtube.com/watch?v={url_or_id}"
    try:
        res = subprocess.run(
            ["yt-dlp", "--dump-single-json", "--skip-download", "--no-warnings", target],
            capture_output=True,
            text=True,
            timeout=25,
        )
        if res.returncode == 0 and res.stdout.strip():
            return json.loads(res.stdout)
    except Exception:
        return None
    return None


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _split_bilingual_block(raw_text: str) -> tuple[str, str]:
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    if not lines:
        return "", ""
    # Strip ASS/HTML tags if present
    clean_lines = [re.sub(r"<[^>]+>|\{[^}]+\}", "", ln).strip() for ln in lines]
    clean_lines = [ln for ln in clean_lines if ln]
    if not clean_lines:
        return "", ""
    zh_lines = [ln for ln in clean_lines if _has_cjk(ln)]
    en_lines = [ln for ln in clean_lines if not _has_cjk(ln)]
    zh = " ".join(zh_lines).strip()
    en = " ".join(en_lines).strip()
    return en, zh


def parse_vtt_to_utterances(text: str) -> list[Utterance]:
    # Lightweight WebVTT parser
    blocks = re.split(r"\r?\n\r?\n+", text.strip())
    utts: list[Utterance] = []
    ts_re = re.compile(r"(\d{1,2}:)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{1,2}:)?(\d{2}):(\d{2})[.,](\d{3})")
    idx = 0
    for blk in blocks:
        lines = blk.splitlines()
        ts_line_idx = -1
        m = None
        for i, ln in enumerate(lines):
            m = ts_re.search(ln)
            if m:
                ts_line_idx = i
                break
        if ts_line_idx < 0 or m is None:
            continue
        h1 = int((m.group(1) or "0:").rstrip(":"))
        m1, s1, ms1 = int(m.group(2)), int(m.group(3)), int(m.group(4))
        h2 = int((m.group(5) or "0:").rstrip(":"))
        m2, s2, ms2 = int(m.group(6)), int(m.group(7)), int(m.group(8))
        start_s = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
        end_s = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
        raw = "\n".join(lines[ts_line_idx + 1 :]).strip()
        if not raw:
            continue
        en, zh = _split_bilingual_block(raw)
        utts.append(Utterance(id=idx, start_s=round(start_s, 3), end_s=round(end_s, 3), en=en, zh=zh, raw=raw))
        idx += 1
    return utts


def parse_subtitle_file(sub_path: Path) -> list[Utterance]:
    raw_content = sub_path.read_text(encoding="utf-8-sig", errors="replace")
    if sub_path.suffix.lower() == ".vtt":
        utts = parse_vtt_to_utterances(raw_content)
    else:
        subs = list(srt.parse(raw_content))
        utts = []
        for i, sub in enumerate(subs):
            raw = sub.content.strip()
            if not raw:
                continue
            en, zh = _split_bilingual_block(raw)
            utts.append(
                Utterance(
                    id=len(utts),
                    start_s=round(sub.start.total_seconds(), 3),
                    end_s=round(sub.end.total_seconds(), 3),
                    en=en,
                    zh=zh,
                    raw=raw,
                )
            )

    # Populate nominal inter-utterance gaps from subtitle timestamps
    for i, u in enumerate(utts):
        if i > 0:
            u.silence_before_ms = int(round((u.start_s - utts[i - 1].end_s) * 1000))
        if i + 1 < len(utts):
            u.silence_after_ms = int(round((utts[i + 1].start_s - u.end_s) * 1000))
    return utts


def make_project_id(video_path: Path) -> str:
    h = hashlib.sha1(str(video_path.resolve()).encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^A-Za-z0-9]+", "-", _strip_hardsub_suffix(video_path.stem)).strip("-").lower()[:28]
    return f"{slug}-{h}" if slug else f"vid-{h}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_work_dir(video_path: Path) -> Path:
    pid = make_project_id(video_path)
    return video_path.parent / ".clipcut" / pid


def ingest_source(
    *,
    video_path: str | Path,
    subtitle_path: str | Path | None = None,
    info_json_path: str | Path | None = None,
    source_url: str | None = None,
    manual_title: str | None = None,
    manual_description: str | None = None,
    fetch_online: bool = True,
    run_id: str = "init",
) -> tuple[ProjectManifest, ProjectStore]:
    vpath = Path(video_path).expanduser().resolve()
    if not vpath.exists() or not vpath.is_file():
        raise FileNotFoundError(f"视频文件不存在: {vpath}")

    folder = vpath.parent
    work_dir = resolve_work_dir(vpath)
    output_dir = folder / "clips"
    store = ProjectStore(work_dir)

    probe = probe_video(vpath)
    analysis_video = _find_clean_analysis_video(vpath)

    sub_p = Path(subtitle_path).expanduser().resolve() if subtitle_path else _find_best_subtitle(vpath, folder)
    if sub_p is None or not sub_p.exists():
        raise FileNotFoundError(
            f"未找到字幕文件（目录: {folder}）。请先提供 .srt/.vtt 字幕文件路径，或先完成字幕转写。"
        )

    utterances = parse_subtitle_file(sub_p)
    if not utterances:
        raise ValueError(f"字幕文件为空或解析失败: {sub_p}")

    # Metadata loading cascade: info_json -> yt-dlp -> manual -> filename fallback
    info_p = Path(info_json_path).expanduser().resolve() if info_json_path else _find_best_info_json(vpath, folder)
    raw_meta: dict[str, Any] | None = None
    meta_source = "filename_fallback"

    if info_p and info_p.exists():
        try:
            raw_meta = json.loads(info_p.read_text(encoding="utf-8"))
            meta_source = "info_json"
        except Exception:
            raw_meta = None

    vid_id = extract_youtube_id(source_url or "") or extract_youtube_id(vpath.name) or extract_youtube_id(folder.name)
    if raw_meta is None and fetch_online and (source_url or vid_id):
        fetched = fetch_ytdlp_metadata(source_url or vid_id or "")
        if fetched:
            raw_meta = fetched
            meta_source = "yt_dlp"
            (work_dir / "fetched_info.json").write_text(json.dumps(fetched, ensure_ascii=False, indent=2), encoding="utf-8")

    chapters: list[Chapter] = []
    if raw_meta:
        orig_title = str(raw_meta.get("fulltitle") or raw_meta.get("title") or vpath.stem)
        desc = str(raw_meta.get("description") or "")
        channel = str(raw_meta.get("channel") or raw_meta.get("uploader") or "")
        uploader = str(raw_meta.get("uploader") or channel)
        webpage_url = raw_meta.get("webpage_url") or source_url
        raw_chapters = raw_meta.get("chapters") or []
        for idx, ch in enumerate(raw_chapters):
            st = float(ch.get("start_time", 0.0))
            en = float(ch.get("end_time", probe["duration_s"]))
            chapters.append(
                Chapter(
                    index=idx,
                    title=str(ch.get("title") or f"Chapter {idx + 1}").strip(),
                    start_s=st,
                    end_s=en,
                    source="info_json",
                )
            )
        if not chapters and desc:
            chapters = parse_chapters_from_description(desc, probe["duration_s"])
    else:
        orig_title = _strip_hardsub_suffix(vpath.stem).replace("_", " ")
        desc = ""
        channel = ""
        uploader = ""
        webpage_url = source_url

    if manual_title:
        orig_title = manual_title
        meta_source = "manual"
    if manual_description:
        desc = manual_description
        if not chapters:
            chapters = parse_chapters_from_description(desc, probe["duration_s"])
        meta_source = "manual"

    metadata = VideoMetadata(
        original_title=orig_title,
        description=desc,
        channel=channel,
        uploader=uploader,
        video_id=vid_id,
        webpage_url=webpage_url,
        metadata_source=meta_source,  # type: ignore[arg-type]
        published_date=extract_published_date(raw_meta),
        chapters=chapters,
    )

    source = MediaSource(
        render_video=str(vpath),
        analysis_video=str(analysis_video),
        subtitle_path=str(sub_p),
        subtitle_format=sub_p.suffix.lower().lstrip("."),
        subtitle_sha256=sha256_file(sub_p),
        work_dir=str(work_dir),
        output_dir=str(output_dir),
        duration_s=probe["duration_s"],
        width=probe["width"],
        height=probe["height"],
        fps=probe["fps"],
        video_codec=probe["video_codec"],
        audio_codec=probe["audio_codec"],
    )

    pid = make_project_id(vpath)
    if store.exists():
        manifest = store.load()
        manifest.source = source
        manifest.metadata = metadata
        manifest.utterances = utterances
    else:
        manifest = ProjectManifest(
            project_id=pid,
            source=source,
            metadata=metadata,
            utterances=utterances,
        )
        try:
            from ..config import load_editorial_config

            cf_cfg = load_editorial_config().get("content_filter", {}) or {}
            def_sens = str(cf_cfg.get("sensitivity") or "").strip().lower()
            if def_sens in ("strict", "balanced", "loose"):
                manifest.clean_cut.sensitivity = def_sens  # type: ignore[assignment]
            def_action = str(cf_cfg.get("political_action") or "").strip().lower()
            if def_action in ("auto", "delete", "mask", "precise_mask"):
                manifest.clean_cut.political_action = def_action  # type: ignore[assignment]
        except Exception:
            pass

    store.save(manifest)
    store.append_receipt(
        run_id=run_id,
        stage="ingest",
        actor="system",
        action="ingest_media",
        summary=f"载入视频 {vpath.name}（{probe['duration_s']:.1f}s, {probe['width']}x{probe['height']}），字幕 {len(utterances)} 句，章节 {len(chapters)} 个（来源: {meta_source}）",
        details={
            "render_video": str(vpath),
            "analysis_video": str(analysis_video),
            "subtitle_path": str(sub_p),
            "metadata_source": meta_source,
            "original_title": orig_title,
            "chapter_count": len(chapters),
            "utterance_count": len(utterances),
        },
    )
    return manifest, store
