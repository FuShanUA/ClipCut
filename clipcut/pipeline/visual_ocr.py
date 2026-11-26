from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import platform
import re
import shutil
import tempfile
from typing import Any

from ..cancel import JobCancelledError, check_cancelled, run_cancellable_subprocess


# Deterministic zero-tolerance hard redline rules shared by subtitle guard & visual OCR.
# Keep patterns strict enough to avoid false positives on benign words (e.g. "学习", "习惯", "Taiwan Semi").
HARD_REDLINE_RULES: list[dict[str, str]] = [
    {
        "rule_id": "cn_leader",
        "label": "国家最高领导人",
        "zh_pattern": r"(?:国家主席习近平|习近平|习主席|习总书记|习大大|(?<![学复实习演练传温])习总(?!结))",
        "en_ci_pattern": (
            r"(?<![a-z])(?:president\s+xi(?:\s+jinping)?|chairman\s+xi(?:\s+jinping)?|"
            r"xi\s+jinping|president\s+of\s+china|chinese\s+president)(?![a-z])"
        ),
        "en_cs_pattern": (
            r"(?<!Part\s)(?<!Chapter\s)(?<!Section\s)(?<!Article\s)(?<!Volume\s)"
            r"(?<!Title\s)(?<!Louis\s)(?<![A-Za-z])Xi(?![A-Za-z])"
        ),
    },
    {
        "rule_id": "ccp_mss",
        "label": "执政党、国安与敏感情报机构",
        "zh_pattern": r"(?:中国共产党|中共中央|中南海|中国国家安全部|国家安全部|国安部|中共)",
        "en_ci_pattern": (
            r"(?<![a-z])(?:chinese\s+communist\s+party|(?<!\w)ccp(?!\w)|"
            r"cpc\s+central|zhongnanhai|ministry\s+of\s+state\s+security)(?![a-z])"
        ),
        "en_cs_pattern": r"(?<![A-Za-z])(?:MSS\s+(?:officer|agent|operative|spy|official)|Chinese\s+MSS)(?![A-Za-z])",
    },
    {
        "rule_id": "sovereignty_redline",
        "label": "分裂国家与不规范主权称谓",
        "zh_pattern": (
            r"(?:台独|港独|藏独|疆独|东突|中华民国|台湾总统|台湾当局总统|"
            r"台湾领导人|台湾国家|台湾国|两个中国|一中一台|一边一国|"
            r"香港独立|西藏独立|新疆独立)"
        ),
        "en_ci_pattern": (
            r"(?<![a-z])(?:taiwan\s+independence|hong\s+kong\s+independence|"
            r"tibet\s+independence|free\s+tibet|east\s+turkestan|"
            r"republic\s+of\s+china|president\s+of\s+taiwan|taiwanese\s+president|"
            r"country\s+of\s+taiwan|independent\s+taiwan|two\s+chinas|one\s+china\s+one\s+taiwan)(?![a-z])"
        ),
        "en_cs_pattern": "",
    },
    {
        "rule_id": "normative_terms_redline",
        "label": "不规范国家/地区/场所表述",
        "zh_pattern": r"(?:(?<!侵华日军)南京大屠杀(?:遇难同胞)?纪念馆|(?<=侵华日军)南京大屠杀纪念馆|新疆维吾尔族自治区)",
        "en_ci_pattern": r"(?<![a-z])mainland\s+china(?![a-z])",
        "en_cs_pattern": "",
    },
    {
        "rule_id": "sensitive_history",
        "label": "重大政治敏感事件",
        "zh_pattern": r"(?:天安门事件|六四事件|法轮功)",
        "en_ci_pattern": r"(?<![a-z])(?:tiananmen\s+(?:square\s+)?(?:massacre|protest|incident)|falun\s+gong)(?![a-z])",
        "en_cs_pattern": "",
    },
]

_COMPILED_HARD_RULES: list[dict[str, Any]] = [
    {
        "rule_id": r["rule_id"],
        "label": r["label"],
        "zh_re": re.compile(r["zh_pattern"]) if r["zh_pattern"] else None,
        "en_ci_re": re.compile(r["en_ci_pattern"], re.IGNORECASE) if r["en_ci_pattern"] else None,
        "en_cs_re": re.compile(r["en_cs_pattern"]) if r["en_cs_pattern"] else None,
    }
    for r in HARD_REDLINE_RULES
]


def match_hard_redline_text(text: str) -> list[dict[str, str]]:
    """Return verbatim deterministic hard-redline matches found in text."""
    raw = str(text or "")
    if not raw.strip():
        return []
    hits: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    covered_spans: list[tuple[int, int]] = []

    for rule in _COMPILED_HARD_RULES:
        for key, lang in (("zh_re", "zh"), ("en_ci_re", "en"), ("en_cs_re", "en")):
            pat = rule.get(key)
            if pat is None:
                continue
            for m in pat.finditer(raw):
                s_idx, e_idx = m.start(), m.end()
                if key == "en_cs_re" and any(cs <= s_idx and e_idx <= ce for cs, ce in covered_spans):
                    continue
                matched = m.group(0).strip()
                if not matched:
                    continue
                if lang == "en":
                    covered_spans.append((s_idx, e_idx))
                sig = (rule["rule_id"], matched.lower())
                if sig in seen:
                    continue
                seen.add(sig)
                hits.append(
                    {
                        "rule_id": rule["rule_id"],
                        "label": rule["label"],
                        "matched": matched,
                        "lang": lang,
                    }
                )
    return hits


_SWIFT_OCR_SOURCE = r'''
import Foundation
import Vision
import AppKit

let args = CommandLine.arguments
if args.count < 3 {
    exit(1)
}
let mode = args[1]
let paths = Array(args[2...])

struct BoxOut: Codable {
    let text: String
    let x: Double
    let y: Double
    let w: Double
    let h: Double
}
struct FrameOut: Codable {
    let path: String
    let boxes: [BoxOut]
}

var results = Array<FrameOut?>(repeating: nil, count: paths.count)
let lock = NSLock()

DispatchQueue.concurrentPerform(iterations: paths.count) { i in
    let path = paths[i]
    var boxes: [BoxOut] = []
    if let img = NSImage(contentsOfFile: path),
       let cgImg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) {
        let req = VNRecognizeTextRequest()
        req.recognitionLevel = (mode == "fast") ? .fast : .accurate
        req.recognitionLanguages = ["en-US", "zh-Hans"]
        req.usesLanguageCorrection = false
        let handler = VNImageRequestHandler(cgImage: cgImg, options: [:])
        if (try? handler.perform([req])) != nil, let obs = req.results {
            for o in obs {
                if let top = o.topCandidates(1).first {
                    let bb = o.boundingBox
                    boxes.append(
                        BoxOut(
                            text: top.string,
                            x: Double(bb.origin.x),
                            y: Double(1.0 - bb.origin.y - bb.size.height),
                            w: Double(bb.size.width),
                            h: Double(bb.size.height)
                        )
                    )
                }
            }
        }
    }
    lock.lock()
    results[i] = FrameOut(path: path, boxes: boxes)
    lock.unlock()
}

let out = results.compactMap { $0 }
if let data = try? JSONEncoder().encode(out), let s = String(data: data, encoding: .utf8) {
    print(s)
}
'''

_VISION_CACHE_DIR = Path.home() / ".cache" / "clipcut"
_VISION_BIN_PATH = _VISION_CACHE_DIR / "vision_ocr_v1"
_VISION_SRC_PATH = _VISION_CACHE_DIR / "vision_ocr_v1.swift"


def _ensure_vision_ocr_binary() -> Path | None:
    if platform.system() != "Darwin":
        return None
    if _VISION_BIN_PATH.is_file():
        return _VISION_BIN_PATH
    swiftc = shutil.which("swiftc")
    if not swiftc:
        return None
    try:
        _VISION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _VISION_SRC_PATH.write_text(_SWIFT_OCR_SOURCE, encoding="utf-8")
        res = run_cancellable_subprocess(
            [swiftc, "-O", str(_VISION_SRC_PATH), "-o", str(_VISION_BIN_PATH)],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if res.returncode == 0 and _VISION_BIN_PATH.is_file():
            return _VISION_BIN_PATH
    except JobCancelledError:
        raise
    except Exception:
        pass
    return None


def _ocr_frames_vision(
    vision_bin: Path,
    frame_paths: list[Path],
    *,
    mode: str = "fast",
    batch_size: int = 100,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for i in range(0, len(frame_paths), max(1, batch_size)):
        check_cancelled()
        batch = frame_paths[i : i + batch_size]
        res = run_cancellable_subprocess(
            [str(vision_bin), mode, *[str(p) for p in batch]],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if res.returncode != 0 or not (res.stdout or "").strip():
            continue
        try:
            items = json.loads(res.stdout)
        except Exception:
            continue
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and it.get("path"):
                    boxes = it.get("boxes")
                    out[str(it["path"])] = list(boxes) if isinstance(boxes, list) else []
    return out


def _ocr_frames_tesseract(
    tesseract_bin: Path,
    frame_paths: list[Path],
) -> dict[str, list[dict[str, Any]]]:
    from PIL import Image

    out: dict[str, list[dict[str, Any]]] = {}
    for fp in frame_paths:
        check_cancelled()
        try:
            with Image.open(fp) as im:
                img_w, img_h = im.size
        except Exception:
            out[str(fp)] = []
            continue
        if img_w <= 0 or img_h <= 0:
            out[str(fp)] = []
            continue

        res = run_cancellable_subprocess(
            [str(tesseract_bin), str(fp), "stdout", "-l", "eng", "--psm", "6", "tsv"],
            capture_output=True,
            text=True,
            timeout=25,
        )
        if res.returncode != 0 or not res.stdout.strip():
            out[str(fp)] = []
            continue

        lines: dict[tuple[str, str, str], list[tuple[int, int, int, int, str]]] = defaultdict(list)
        for idx_row, raw_line in enumerate(res.stdout.splitlines()):
            if idx_row == 0:
                continue
            cols = raw_line.split("\t")
            if len(cols) < 12:
                continue
            txt = cols[11].strip()
            if not txt:
                continue
            try:
                conf = float(cols[10])
                left = int(cols[6])
                top = int(cols[7])
                width = int(cols[8])
                height = int(cols[9])
            except ValueError:
                continue
            if conf < 20 or width <= 0 or height <= 0:
                continue
            lines[(cols[2], cols[3], cols[4])].append((left, top, left + width, top + height, txt))

        boxes: list[dict[str, Any]] = []
        for words in lines.values():
            x0 = min(w[0] for w in words)
            y0 = min(w[1] for w in words)
            x1 = max(w[2] for w in words)
            y1 = max(w[3] for w in words)
            boxes.append(
                {
                    "text": " ".join(w[4] for w in words),
                    "x": max(0.0, min(1.0, x0 / img_w)),
                    "y": max(0.0, min(1.0, y0 / img_h)),
                    "w": max(0.0, min(1.0, (x1 - x0) / img_w)),
                    "h": max(0.0, min(1.0, (y1 - y0) / img_h)),
                }
            )
        out[str(fp)] = boxes
    return out


def _run_ocr_on_frames(
    frame_paths: list[Path],
    *,
    mode: str = "fast",
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    if not frame_paths:
        return {}, "none"
    vision_bin = _ensure_vision_ocr_binary()
    if vision_bin is not None:
        return _ocr_frames_vision(vision_bin, frame_paths, mode=mode), f"vision:{mode}"
    tess = shutil.which("tesseract")
    if tess:
        return _ocr_frames_tesseract(Path(tess), frame_paths), "tesseract"
    return {}, "none"


def _probe_keyframe_pts(video_path: Path) -> list[float]:
    res = run_cancellable_subprocess(
        [
            "ffprobe",
            "-v",
            "error",
            "-skip_frame",
            "nokey",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=pts_time",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=45,
    )
    if res.returncode != 0 or not res.stdout.strip():
        return []
    pts: list[float] = []
    for line in res.stdout.splitlines():
        raw = line.strip()
        if not raw or raw == "N/A":
            continue
        try:
            pts.append(float(raw))
        except ValueError:
            continue
    return pts


def _extract_matched_boxes(
    boxes: list[dict[str, Any]],
    *,
    exclude_bottom_subtitle_band: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    matched: list[dict[str, Any]] = []
    labels: list[str] = []
    seen_labels: set[str] = set()
    for b in boxes:
        try:
            bx = float(b.get("x", 0.0))
            by = float(b.get("y", 0.0))
            bw = float(b.get("w", 0.0))
            bh = float(b.get("h", 0.0))
        except (TypeError, ValueError):
            continue
        if bw <= 0.0 or bh <= 0.0:
            continue
        cy = by + bh * 0.5
        if exclude_bottom_subtitle_band and cy >= 0.845:
            continue
        text = str(b.get("text") or "").strip()
        if not text:
            continue
        hits = match_hard_redline_text(text)
        if not hits:
            continue
        matched.append({"x": bx, "y": by, "w": bw, "h": bh, "text": text})
        for h in hits:
            lb = h["label"]
            if lb not in seen_labels:
                seen_labels.add(lb)
                labels.append(lb)
    return matched, labels


def _cluster_visual_boxes(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cluster matched OCR text boxes by vertical proximity and pad slightly for clean blur."""
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda b: (float(b["y"]) + float(b["h"]) * 0.5, float(b["x"])))
    clusters: list[list[dict[str, Any]]] = []
    for b in ordered:
        cy = float(b["y"]) + float(b["h"]) * 0.5
        if clusters:
            prev_cy = sum(float(p["y"]) + float(p["h"]) * 0.5 for p in clusters[-1]) / len(clusters[-1])
            if abs(cy - prev_cy) < 0.08:
                clusters[-1].append(b)
                continue
        clusters.append([b])

    pad_x = 0.018
    pad_y = 0.014
    out: list[dict[str, Any]] = []
    for cl in clusters:
        x0 = max(0.0, min(float(b["x"]) for b in cl) - pad_x)
        y0 = max(0.0, min(float(b["y"]) for b in cl) - pad_y)
        x1 = min(1.0, max(float(b["x"]) + float(b["w"]) for b in cl) + pad_x)
        y1 = min(1.0, max(float(b["y"]) + float(b["h"]) for b in cl) + pad_y)
        if x1 - x0 < 0.02 or y1 - y0 < 0.015:
            continue
        uniq_texts: list[str] = []
        seen_t: set[str] = set()
        for b in cl:
            t = str(b.get("text") or "").strip()
            if t and t.lower() not in seen_t:
                seen_t.add(t.lower())
                uniq_texts.append(t)
        out.append(
            {
                "x": round(x0, 4),
                "y": round(y0, 4),
                "w": round(x1 - x0, 4),
                "h": round(y1 - y0, 4),
                "text": " / ".join(uniq_texts)[:160],
            }
        )
    return out


def _sample_window_ocr(
    video_path: Path,
    win_start: float,
    win_end: float,
    *,
    exclude_bottom_subtitle_band: bool,
    tmp_dir: Path,
    tag: str,
) -> list[tuple[float, list[dict[str, Any]], list[str]]]:
    """Decode and accurate-OCR 0.5s steps inside a narrow boundary window."""
    dur = max(0.5, float(win_end) - float(win_start))
    sub_dir = tmp_dir / tag
    sub_dir.mkdir(parents=True, exist_ok=True)
    res = run_cancellable_subprocess(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{max(0.0, win_start):.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{dur:.3f}",
            "-vf",
            "fps=2,scale=960:-2",
            "-q:v",
            "4",
            str(sub_dir / "w_%04d.jpg"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if res.returncode != 0:
        return []
    imgs = sorted(sub_dir.glob("w_*.jpg"))
    if not imgs:
        return []
    ocr_map, _ = _run_ocr_on_frames(imgs, mode="accurate")
    rows: list[tuple[float, list[dict[str, Any]], list[str]]] = []
    for idx, fp in enumerate(imgs):
        ts = max(0.0, float(win_start)) + idx * 0.5
        m_boxes, lbs = _extract_matched_boxes(
            ocr_map.get(str(fp), []),
            exclude_bottom_subtitle_band=exclude_bottom_subtitle_band,
        )
        rows.append((ts, m_boxes, lbs))
    return rows


def detect_visual_ocr_redline_segments(
    video_path: Path,
    duration_s: float,
    *,
    exclude_bottom_subtitle_band: bool = False,
    log_fn: Any = None,
) -> list[dict[str, Any]]:
    """
    Two-stage coarse-to-fine OCR scanner detecting on-screen hard-redline banners/titles.
    Returns [{start_s, end_s, quote, matched_labels, visual_boxes}].
    """
    if not video_path.is_file():
        return []
    if _ensure_vision_ocr_binary() is None and shutil.which("tesseract") is None:
        return []

    check_cancelled()
    eff_dur = max(0.0, float(duration_s or 0.0))

    with tempfile.TemporaryDirectory(prefix="clipcut_ocr_") as td:
        td_path = Path(td)
        coarse_dir = td_path / "coarse"
        coarse_dir.mkdir(parents=True, exist_ok=True)

        pts_list = _probe_keyframe_pts(video_path)
        avg_gap = (
            (pts_list[-1] - pts_list[0]) / max(1, len(pts_list) - 1)
            if len(pts_list) >= 3
            else 99.0
        )
        use_nokey = len(pts_list) >= 3 and avg_gap <= 8.0

        if use_nokey:
            cmd = [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-skip_frame",
                "nokey",
                "-i",
                str(video_path),
                "-vsync",
                "0",
                "-vf",
                "scale=960:-2",
                "-q:v",
                "5",
                str(coarse_dir / "c_%05d.jpg"),
            ]
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(video_path),
                "-vf",
                "fps=0.25,scale=960:-2",
                "-q:v",
                "5",
                str(coarse_dir / "c_%05d.jpg"),
            ]

        ext_res = run_cancellable_subprocess(cmd, capture_output=True, text=True, timeout=180)
        if ext_res.returncode != 0:
            return []
        coarse_imgs = sorted(coarse_dir.glob("c_*.jpg"))
        if not coarse_imgs:
            return []

        if use_nokey and len(pts_list) >= len(coarse_imgs):
            coarse_times = [float(pts_list[i]) for i in range(len(coarse_imgs))]
        elif use_nokey and pts_list:
            coarse_times = [
                float(pts_list[i]) if i < len(pts_list) else float(pts_list[-1]) + (i - len(pts_list) + 1) * avg_gap
                for i in range(len(coarse_imgs))
            ]
        else:
            coarse_times = [i * 4.0 for i in range(len(coarse_imgs))]

        if eff_dur <= 0.0 and coarse_times:
            eff_dur = coarse_times[-1] + max(4.0, avg_gap if use_nokey else 4.0)

        ocr_map, engine_name = _run_ocr_on_frames(coarse_imgs, mode="fast")
        if engine_name == "none":
            return []

        hit_indices: list[tuple[int, float, list[dict[str, Any]], list[str]]] = []
        for idx_f, fp in enumerate(coarse_imgs):
            m_boxes, lbs = _extract_matched_boxes(
                ocr_map.get(str(fp), []),
                exclude_bottom_subtitle_band=exclude_bottom_subtitle_band,
            )
            if m_boxes:
                hit_indices.append((idx_f, coarse_times[idx_f], m_boxes, lbs))

        if log_fn:
            log_fn(
                f"[clean_cut] 画面文字 OCR 粗扫 {len(coarse_imgs)} 帧（{engine_name}），"
                f"命中 {len(hit_indices)} 帧"
            )
        if not hit_indices:
            return []

        step_est = avg_gap if use_nokey else 4.0
        max_merge_gap = max(14.0, step_est * 2.5)
        groups: list[dict[str, Any]] = []
        for idx_f, ts, m_boxes, lbs in hit_indices:
            prev_t = coarse_times[idx_f - 1] if idx_f > 0 else max(0.0, ts - step_est)
            next_t = coarse_times[idx_f + 1] if idx_f + 1 < len(coarse_times) else min(eff_dur, ts + step_est)
            if groups and ts - groups[-1]["last_hit_t"] <= max_merge_gap:
                g = groups[-1]
                g["last_hit_t"] = ts
                g["next_kf_t"] = max(g["next_kf_t"], next_t)
                g["boxes"].extend(m_boxes)
                for lb in lbs:
                    if lb not in g["labels"]:
                        g["labels"].append(lb)
            else:
                groups.append(
                    {
                        "prev_kf_t": prev_t,
                        "first_hit_t": ts,
                        "last_hit_t": ts,
                        "next_kf_t": next_t,
                        "boxes": list(m_boxes),
                        "labels": list(lbs),
                    }
                )

        # Stage B: refine leading & trailing boundaries to +-0.25s via 0.5s sampling + accurate OCR
        refined: list[dict[str, Any]] = []
        for gi, g in enumerate(groups):
            check_cancelled()
            all_boxes = list(g["boxes"])
            labels = list(g["labels"])

            lead_s = max(0.0, float(g["prev_kf_t"]))
            lead_e = min(eff_dur, float(g["first_hit_t"]) + 0.5)
            exact_start = float(g["first_hit_t"])
            if lead_e - lead_s >= 0.4:
                lead_rows = _sample_window_ocr(
                    video_path,
                    lead_s,
                    lead_e,
                    exclude_bottom_subtitle_band=exclude_bottom_subtitle_band,
                    tmp_dir=td_path,
                    tag=f"lead_{gi:03d}",
                )
                lead_hits = [r for r in lead_rows if r[1]]
                if lead_hits:
                    exact_start = min(exact_start, min(r[0] for r in lead_hits))
                    for _, mb, lbs in lead_hits:
                        all_boxes.extend(mb)
                        for lb in lbs:
                            if lb not in labels:
                                labels.append(lb)

            trail_s = max(lead_s, float(g["last_hit_t"]) - 0.5)
            trail_e = min(eff_dur, float(g["next_kf_t"]) + 0.2)
            exact_end = float(g["last_hit_t"])
            if trail_e - trail_s >= 0.4:
                trail_rows = _sample_window_ocr(
                    video_path,
                    trail_s,
                    trail_e,
                    exclude_bottom_subtitle_band=exclude_bottom_subtitle_band,
                    tmp_dir=td_path,
                    tag=f"trail_{gi:03d}",
                )
                trail_hits = [r for r in trail_rows if r[1]]
                if trail_hits:
                    exact_end = max(exact_end, max(r[0] for r in trail_hits))
                    for _, mb, lbs in trail_hits:
                        all_boxes.extend(mb)
                        for lb in lbs:
                            if lb not in labels:
                                labels.append(lb)

            start_s = round(max(0.0, exact_start - 0.35), 3)
            end_s = round(min(eff_dur, max(start_s + 0.6, exact_end + 0.75)), 3)
            if not all_boxes:
                continue

            refined.append(
                {
                    "start_s": start_s,
                    "end_s": end_s,
                    "matched_labels": "、".join(labels) or "硬红线画面文字",
                    "raw_boxes": all_boxes,
                }
            )

        # Merge overlapping or near-adjacent refined intervals BEFORE clustering padding
        refined.sort(key=lambda r: (r["start_s"], r["end_s"]))
        merged_raw: list[dict[str, Any]] = []
        for item in refined:
            if merged_raw and item["start_s"] <= merged_raw[-1]["end_s"] + 1.0:
                prev = merged_raw[-1]
                prev["end_s"] = round(max(prev["end_s"], item["end_s"]), 3)
                prev["raw_boxes"].extend(item["raw_boxes"])
                for lb in item["matched_labels"].split("、"):
                    if lb and lb not in prev["matched_labels"]:
                        prev["matched_labels"] = f"{prev['matched_labels']}、{lb}"
            else:
                merged_raw.append(item)

        merged: list[dict[str, Any]] = []
        for item in merged_raw:
            clustered_boxes = _cluster_visual_boxes(item["raw_boxes"])
            if not clustered_boxes:
                continue
            quote = " / ".join(str(b["text"]) for b in clustered_boxes if b.get("text"))[:160]
            merged.append(
                {
                    "start_s": item["start_s"],
                    "end_s": item["end_s"],
                    "quote": quote,
                    "matched_labels": item["matched_labels"],
                    "visual_boxes": clustered_boxes,
                }
            )

        if log_fn and merged:
            log_fn(f"[clean_cut] 画面文字 OCR 边界精修完成：检出 {len(merged)} 段硬红线画面条/标题")
        return merged
