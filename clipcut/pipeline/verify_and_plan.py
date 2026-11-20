from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..llm import ClipCutLLM, DEFAULT_MULTIMODAL_FALLBACK_MODELS, DEFAULT_TEXT_FALLBACK_MODELS
from ..models import (
    Candidate,
    Chapter,
    ClipItem,
    MultimodalVerdict,
    ProjectManifest,
    RemixItem,
    TopicMapItem,
    Utterance,
)
from ..receipts import ProjectStore

DANGLING_START_PATTERNS = [
    re.compile(r"^(?:所以|但是|不过|而且|因此|也就是说|这就|那个|这个就是|刚才说的|第二点|第三点)"),
    re.compile(r"^(?:and\s+that'?s|so\s+that'?s|that'?s\s+why|but\s+that|which\s+is\s+why|like\s+i\s+said)", re.IGNORECASE),
]
TERMINAL_PUNCT_RE = re.compile(r"[。！？.!?…\"'”’\)]$")


def _load_audio_envelope(store: ProjectStore) -> tuple[np.ndarray, float]:
    p = store.work_dir / "audio_envelope.npz"
    if not p.exists():
        return np.zeros(0, dtype=np.float32), 0.02
    data = np.load(p)
    return data["rms"], float(data["hop_s"][0])


def snap_bounds_with_acoustic_guard(
    *,
    start_utt: int,
    end_utt: int,
    utterances: list[Utterance],
    rms: np.ndarray,
    hop_s: float,
    video_duration: float,
    cfg: dict[str, Any],
) -> tuple[float, float, int, int, list[str]]:
    """
    BaoCut-inspired pause-midpoint snapping with <40ms short-pause protection.
    Never cuts into neighboring words; snaps cleanly to lowest acoustic energy valley inside inter-utterance pause.
    """
    b_cfg = cfg.get("boundary", {})
    max_pre_s = float(b_cfg.get("max_pre_roll_ms", 350)) / 1000.0
    max_post_s = float(b_cfg.get("max_post_roll_ms", 450)) / 1000.0
    short_guard_s = float(b_cfg.get("short_pause_guard_ms", 40)) / 1000.0

    u_start = utterances[start_utt]
    u_end = utterances[end_utt]
    notes: list[str] = []

    # ── Start boundary ──
    raw_start = u_start.start_s
    prev_end = utterances[start_utt - 1].end_s if start_utt > 0 else 0.0
    gap_before = raw_start - prev_end

    if start_utt == 0:
        snapped_start = max(0.0, raw_start - min(0.15, raw_start))
    elif gap_before < short_guard_s:
        # Short pause (<40ms): keep strictly inside pause, favor not eating previous word
        snapped_start = max(prev_end, raw_start - max(0.0, gap_before * 0.25))
        notes.append(f"句首停顿仅 {int(max(0, gap_before) * 1000)}ms (<40ms)，已启用防吞字边界保护")
    else:
        win_lo = max(prev_end + 0.02, raw_start - min(max_pre_s, gap_before * 0.75))
        win_hi = raw_start
        midpoint = prev_end + min(gap_before * 0.5, max_pre_s)
        if rms.size > 0 and win_hi > win_lo + hop_s:
            i0 = max(0, int(win_lo / hop_s))
            i1 = min(rms.size, max(i0 + 1, int(win_hi / hop_s)))
            valley_idx = i0 + int(np.argmin(rms[i0:i1]))
            snapped_start = valley_idx * hop_s
        else:
            snapped_start = midpoint

    # ── End boundary ──
    raw_end = u_end.end_s
    next_start = utterances[end_utt + 1].start_s if end_utt + 1 < len(utterances) else video_duration
    gap_after = next_start - raw_end

    if end_utt + 1 >= len(utterances):
        snapped_end = min(video_duration, raw_end + min(0.25, max(0.0, video_duration - raw_end)))
    elif gap_after < short_guard_s:
        # Short pause (<40ms): strictly avoid eating next sentence start
        snapped_end = min(next_start, raw_end + max(0.0, gap_after * 0.25))
        notes.append(f"句尾停顿仅 {int(max(0, gap_after) * 1000)}ms (<40ms)，已启用防串入下句保护")
    else:
        win_lo = raw_end
        win_hi = min(next_start - 0.02, raw_end + min(max_post_s, gap_after * 0.75))
        midpoint = raw_end + min(gap_after * 0.5, max_post_s)
        if rms.size > 0 and win_hi > win_lo + hop_s:
            i0 = max(0, int(win_lo / hop_s))
            i1 = min(rms.size, max(i0 + 1, int(win_hi / hop_s)))
            valley_idx = i0 + int(np.argmin(rms[i0:i1]))
            snapped_end = valley_idx * hop_s
        else:
            snapped_end = midpoint

    snapped_start = round(max(0.0, snapped_start), 3)
    snapped_end = round(min(video_duration if video_duration > 0 else snapped_end, max(snapped_start + 0.5, snapped_end)), 3)
    delta_start_ms = int(round((snapped_start - raw_start) * 1000))
    delta_end_ms = int(round((snapped_end - raw_end) * 1000))
    return snapped_start, snapped_end, delta_start_ms, delta_end_ms, notes


def ensure_sentence_completeness(
    start_utt: int,
    end_utt: int,
    utterances: list[Utterance],
) -> tuple[int, int, list[str]]:
    """Expand start/end by 1-2 utterances if opening is dangling or closing is mid-sentence."""
    n = len(utterances)
    notes: list[str] = []
    su, eu = start_utt, end_utt

    # Leading dangling check (at most 2 steps backward)
    for _ in range(2):
        if su <= 0:
            break
        txt = (utterances[su].zh or utterances[su].en).strip()
        is_dangling = any(p.search(txt) for p in DANGLING_START_PATTERNS)
        prev_txt = (utterances[su - 1].zh or utterances[su - 1].en).strip()
        prev_unfinished = bool(prev_txt) and not TERMINAL_PUNCT_RE.search(prev_txt)
        if is_dangling or prev_unfinished:
            su -= 1
            notes.append(f"句首存在上下文承接/断句，自动前移至 u{su:04d} 保持完整开场")
        else:
            break

    # Trailing completeness check (at most 2 steps forward)
    for _ in range(2):
        if eu + 1 >= n:
            break
        txt = (utterances[eu].zh or utterances[eu].en).strip()
        if txt and not TERMINAL_PUNCT_RE.search(txt):
            eu += 1
            notes.append(f"句尾未落在完整终结标点，自动后延至 u{eu:04d}")
        else:
            break

    return su, eu, notes


def compute_clip_render_hash(clip: ClipItem, render_video: str) -> str:
    payload = {
        "video": render_video,
        "start_s": round(clip.start_s, 3),
        "end_s": round(clip.end_s, 3),
        "title": clip.title,
        "title_en": clip.title_en,
        "slug": clip.slug,
        "index": clip.index,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def compute_remix_render_hash(manifest: ProjectManifest) -> str:
    r = manifest.remix
    payload = {
        "video": manifest.source.render_video,
        "card_template": "v2-vision-pip",
        "published_date": manifest.metadata.published_date,
        "card_mode": r.card_mode,
        "card_duration_s": r.card_duration_s,
        "card_fade_s": r.card_fade_s,
        "enable_sfx": r.enable_sfx,
        "enable_bgm": r.enable_bgm,
        "bgm_volume": r.bgm_volume,
        "sfx_volume": r.sfx_volume,
        "bgm_path": r.bgm_path,
        "sfx_path": r.sfx_path,
        "items": [
            {
                "id": it.remix_id,
                "order": it.order,
                "theme_id": it.theme_id,
                "group": it.group_label,
                "card": it.card_title,
                "start_s": round(it.start_s, 3),
                "end_s": round(it.end_s, 3),
            }
            for it in sorted(r.items, key=lambda x: x.order)
            if it.enabled
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _extract_verify_preview(video_path: Path, start_s: float, end_s: float, out_path: Path) -> bool:
    """Cut a tiny 480p 2fps + 64k mono clip (capped at 75s around peak) for cheap multimodal verification."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dur = min(75.0, max(5.0, end_s - start_s))
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{dur:.3f}",
        "-vf",
        "fps=2,scale=-2:480",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "32",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        str(out_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1024


def _verify_exciting_candidates(
    manifest: ProjectManifest,
    store: ProjectStore,
    cfg: dict[str, Any],
    llm: ClipCutLLM,
    run_id: str,
) -> None:
    models_cfg = cfg.get("models", {})
    mm_provider = models_cfg.get("multimodal_provider", "dashscope")
    verify_model = models_cfg.get("multimodal_model") or models_cfg.get("verify_model") or "qwen3-vl-plus"
    mm_fallbacks = list(models_cfg.get("multimodal_fallback_models") or DEFAULT_MULTIMODAL_FALLBACK_MODELS)
    max_checks = int(models_cfg.get("max_multimodal_checks", 8))

    to_verify = [c for c in manifest.candidates if c.needs_multimodal_verify]
    to_verify.sort(key=lambda c: c.score, reverse=True)
    to_verify = to_verify[:max_checks]

    if not to_verify:
        store.append_receipt(
            run_id=run_id,
            stage="verify_and_plan",
            actor="system",
            action="multimodal_verify_skip",
            summary="无需要多模态复核的情绪/动作候选段",
        )
        return

    analysis_video = Path(manifest.source.analysis_video)
    preview_dir = store.work_dir / "verify_previews"
    utts = manifest.utterances

    verdicts_summary: list[dict[str, Any]] = []
    for cand in to_verify:
        s_t = utts[cand.start_utt].start_s
        e_t = utts[cand.end_utt].end_s
        preview_path = preview_dir / f"{cand.candidate_id}.mp4"
        store.log(f"[verify_and_plan] 多模态复核 {cand.candidate_id}: {cand.title} ({s_t:.1f}s-{e_t:.1f}s)")

        ok = _extract_verify_preview(analysis_video, s_t, e_t, preview_path)
        if not ok:
            cand.multimodal = MultimodalVerdict(
                status="unverified",
                error="ffmpeg 提取低码率复核预览失败",
            )
            continue

        video_bytes = preview_path.read_bytes()
        prompt = f"""你正在复核一段被标记为「精彩对话（情绪高涨/大笑/大幅度动作/激烈论辩）」的视频片段。
片段标题：{cand.title}
初筛理由：{cand.reason}

请真实观看画面并聆听声音，严格判断该片段是否确实具备以下至少一项强表现力：
1. 明显大笑、发自内心的幽默互动或强烈表情变化；
2. 明显的大幅度肢体动作、手势强调或情绪激动；
3. 激烈的论辩、打断、反问、提高音量或锋利交锋。

严格输出 JSON：
{{
  "verdict": "pass | downgrade | reject",
  "intensity_score": 0.0,
  "seen_motion": "具体描述你在画面中实际看到的表情、手势或肢体动作",
  "heard_emotion": "具体描述你在声音中听到的笑声、语调起伏、抢话或辩论张力",
  "evidence": "一句总结判定依据（若只是平淡讲述，应给 downgrade 或 reject）"
}}"""
        try:
            res = llm.generate_json(
                prompt=prompt,
                model=verify_model,
                provider=mm_provider,
                fallback_models=mm_fallbacks,
                temperature=0.1,
                max_output_tokens=1024,
                media_bytes=video_bytes,
                media_mime="video/mp4",
            )
            d = res.parsed if isinstance(res.parsed, dict) else {}
            v = str(d.get("verdict", "pass")).lower()
            if v not in ("pass", "downgrade", "reject"):
                v = "pass"
            cand.multimodal = MultimodalVerdict(
                status=v,  # type: ignore[arg-type]
                model=f"{res.provider}:{res.model}",
                seen_motion=str(d.get("seen_motion") or ""),
                heard_emotion=str(d.get("heard_emotion") or ""),
                intensity_score=float(d.get("intensity_score") or 0.0),
                evidence=str(d.get("evidence") or ""),
                clip_preview_path=str(preview_path),
            )
            verdicts_summary.append(
                {
                    "candidate_id": cand.candidate_id,
                    "title": cand.title,
                    "verdict": v,
                    "intensity_score": cand.multimodal.intensity_score,
                    "seen_motion": cand.multimodal.seen_motion,
                    "heard_emotion": cand.multimodal.heard_emotion,
                    "evidence": cand.multimodal.evidence,
                }
            )
            store.append_receipt(
                run_id=run_id,
                stage="verify_and_plan",
                actor="ai",
                action="multimodal_verify",
                model=f"{res.provider}:{res.model}",
                prompt_tokens=res.prompt_tokens,
                output_tokens=res.output_tokens,
                elapsed_ms=res.elapsed_ms,
                summary=f"多模态复核 {cand.candidate_id} [{v.upper()}] (强度 {cand.multimodal.intensity_score:.1f}): {cand.multimodal.evidence[:90]}",
                details=verdicts_summary[-1],
            )
        except Exception as exc:
            cand.multimodal = MultimodalVerdict(
                status="unverified",
                clip_preview_path=str(preview_path),
                error=str(exc),
            )
            store.append_receipt(
                run_id=run_id,
                stage="verify_and_plan",
                actor="system",
                action="multimodal_verify",
                summary=f"多模态复核 {cand.candidate_id} 超时或失败，标记 unverified 继续流程",
                error=str(exc),
            )


def _interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    if inter <= 0:
        return 0.0
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def _merge_overlapping_candidates(
    cands: list[Candidate],
    utterances: list[Utterance],
    iou_thresh: float,
) -> list[Candidate]:
    # Higher score wins primary metadata; merge overlapping intervals
    ordered = sorted(cands, key=lambda c: c.score, reverse=True)
    kept: list[Candidate] = []
    for c in ordered:
        c0, c1 = utterances[c.start_utt].start_s, utterances[c.end_utt].end_s
        merged = False
        for k in kept:
            k0, k1 = utterances[k.start_utt].start_s, utterances[k.end_utt].end_s
            iou = _interval_iou(c0, c1, k0, k1)
            if iou >= iou_thresh:
                # Merge categories, hot topics, and topic IDs into stronger candidate
                for cat in c.categories:
                    if cat not in k.categories:
                        k.categories.append(cat)
                for ht in c.matched_hot_topics:
                    if ht not in k.matched_hot_topics:
                        k.matched_hot_topics.append(ht)
                for tid in c.topic_ids:
                    if tid not in k.topic_ids:
                        k.topic_ids.append(tid)
                c.disposition = "merged"
                c.disposition_reason = f"与更强候选 {k.candidate_id} 重叠度 {iou:.2f} >= {iou_thresh}，已合并标签"
                merged = True
                break
        if not merged:
            kept.append(c)
    kept.sort(key=lambda c: c.start_utt)
    return kept


REMIX_SYSTEM = """你是顶级财经与科技视频总编，负责把一部深度长访谈的已定稿精华切片，策展并浓缩成一支结构清晰、论证充分、有叙事节奏的高光集锦。
核心策展原则：
1. 精选策展，不做全片简介：宁缺毋滥，主动舍弃平淡、常规历史回顾或重复段落（填入 dropped_clips），绝不为了覆盖每条切片而均匀切碎。
2. 宏观立意优先，严禁“把载体当主张”：结合全片主题、章节标题与前情话题地图提炼宏观核心命题。嘉宾常借具体部门/产品/生活案例（如借 HR 问卷或代码库分析演示“超越简单问数的企业决策 AI 能力”）论证更大的技术或商业判断；板块标题与卡片文案必须提炼宏观核心观点，绝不能把表层举例场景写成嘉宾对该领域的主张。
3. 非均匀弹性节奏与完整叙事单元：不需要开场 Hook。按内容性质自然伸缩——独立尖锐论断（punchline，约 8~22 秒）、设问与完整论证（qa_argument，约 25~65 秒，严禁只留问题不给答案）、完整案例演示（case_demo，约 35~85 秒，含场景前提、反直觉转折与价值落点）。
4. 主题板块聚合插卡：每个主题板块（theme）可串联 1~2 段紧密衔接的小节并共用同一张板块转场卡片，避免频繁黑屏打断叙事。
5. 严格基于给定字幕句号选段，总时长严格不超过原片总时长的 1/10。严格输出合法 JSON。"""


def _fmt_ts(sec: float) -> str:
    s = max(0, int(sec))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _parse_utt_int(val: Any, default: int = 0) -> int:
    if isinstance(val, int):
        return val
    digits = re.findall(r"\d+", str(val or ""))
    return int(digits[0]) if digits else default


def _clean_match_text(text: str) -> str:
    return re.sub(r"\s+|[^\w\u4e00-\u9fff]", "", text.lower())


def _match_span_pair(
    q_clean: str,
    utterances: list[Utterance],
    lo: int,
    hi: int,
    max_span: int,
    max_span_s: float,
) -> tuple[int, int]:
    """Find best matching utterance window in [lo, hi] using substring coverage and 2-gram Dice F1."""
    if not q_clean or lo > hi:
        return lo, lo
    q_grams = {q_clean[i : i + 2] for i in range(max(1, len(q_clean) - 1))}
    best_score = -1.0
    best_pair = (lo, min(hi, lo))

    for i in range(lo, hi + 1):
        limit = min(max_span, hi - i + 1)
        for span in range(1, limit + 1):
            j = i + span - 1
            if span > 1 and (utterances[j].end_s - utterances[i].start_s) > max_span_s:
                break
            cand_txt = "".join((utterances[k].zh or "") + (utterances[k].en or "") for k in range(i, j + 1))
            cand_clean = _clean_match_text(cand_txt)
            if not cand_clean:
                continue
            if q_clean in cand_clean or cand_clean in q_clean:
                coverage = min(len(q_clean), len(cand_clean)) / max(len(q_clean), len(cand_clean), 1)
                score = 5.0 + 5.0 * coverage - 0.05 * span
            else:
                c_grams = {cand_clean[k : k + 2] for k in range(max(1, len(cand_clean) - 1))}
                inter = len(q_grams & c_grams)
                score = (2.0 * inter) / max(1, len(q_grams) + len(c_grams))
            if score > best_score:
                best_score = score
                best_pair = (i, j)
    return best_pair


def _match_quote_to_utterances(
    quote: str,
    utterances: list[Utterance],
    allowed_start: int,
    allowed_end: int,
    max_span_s: float = 90.0,
) -> tuple[int, int]:
    """Find the utterance window inside [allowed_start, allowed_end] best matching quote (supports '...' anchors)."""
    if not utterances:
        return 0, 0
    lo = max(0, min(allowed_start, len(utterances) - 1))
    hi = max(lo, min(allowed_end, len(utterances) - 1))

    # Handle "opening quote...closing quote" format used for longer multi-sentence excerpts
    parts = [_clean_match_text(p) for p in re.split(r"\.{3,}|…+", quote) if _clean_match_text(p)]
    if len(parts) >= 2:
        su, _ = _match_span_pair(parts[0], utterances, lo, hi, max_span=2, max_span_s=max_span_s)
        _, eu = _match_span_pair(parts[-1], utterances, su, hi, max_span=2, max_span_s=max_span_s)
        while eu > su and (utterances[eu].end_s - utterances[su].start_s) > max_span_s:
            eu -= 1
        return su, max(su, eu)

    q_clean = parts[0] if parts else _clean_match_text(quote)
    if not q_clean:
        return lo, min(hi, lo + 1)
    return _match_span_pair(q_clean, utterances, lo, hi, max_span=12, max_span_s=max_span_s)


def _compute_clip_bridge_window(
    clip: ClipItem,
    all_clips: list[ClipItem],
    topic_map: list[TopicMapItem],
    chapters: list[Chapter],
    utterances: list[Utterance],
) -> int:
    """
    Compute optional same-chapter pre-context boundary before clip.start_utt.
    Excludes all prior clips (including disabled ones) and filler topics, capped at 25 utts / 95s.
    """
    if clip.start_utt <= 0 or not utterances:
        return clip.start_utt

    prev_boundary = 0
    for other in all_clips:
        if other.clip_id == clip.clip_id:
            continue
        if other.start_utt < clip.start_utt:
            prev_boundary = max(
                prev_boundary,
                other.end_utt + 1 if other.end_utt < clip.start_utt else clip.start_utt,
            )

    for topic in topic_map:
        if topic.is_filler and topic.start_utt < clip.start_utt:
            prev_boundary = max(
                prev_boundary,
                topic.end_utt + 1 if topic.end_utt < clip.start_utt else clip.start_utt,
            )

    matched_ch: Chapter | None = None
    if clip.chapter_title and chapters:
        target_ch = clip.chapter_title.strip()
        matched_ch = next((ch for ch in chapters if ch.title.strip() == target_ch), None)

    bridge_lo = max(prev_boundary, clip.start_utt - 25)
    clip_start_s = utterances[clip.start_utt].start_s
    while bridge_lo < clip.start_utt:
        if (clip_start_s - utterances[bridge_lo].start_s) > 95.0:
            bridge_lo += 1
        elif matched_ch is not None and utterances[bridge_lo].start_s < matched_ch.start_s:
            bridge_lo += 1
        else:
            break

    return min(clip.start_utt, max(0, bridge_lo))


def _card_fee_for_item(card_mode: str, card_dur_base: float, has_items: bool, next_key: str, last_key: str | None) -> float:
    if card_mode == "every":
        return card_dur_base
    if card_mode == "group":
        return card_dur_base if (not has_items or next_key != last_key) else 0.0
    return 0.0


def _build_remix_macro_prompt(
    manifest: ProjectManifest,
    cfg: dict[str, Any],
    enabled_clips: list[ClipItem],
    clip_bounds: dict[str, tuple[int, int]],
) -> tuple[str, int, int, float]:
    utts = manifest.utterances
    r_cfg = cfg.get("remix", {})
    src_dur = max(60.0, float(manifest.source.duration_s or 600.0))
    max_ratio = float(r_cfg.get("max_duration_ratio", 0.10))
    min_ratio = float(r_cfg.get("target_min_ratio", 0.065))
    hard_cap_s = src_dur * max_ratio
    target_max = max(35, int(round(hard_cap_s)))
    target_min = max(25, int(round(src_dur * min_ratio)))
    max_excerpt_s = float(r_cfg.get("max_excerpt_seconds", 90))
    max_themes_cfg = int(r_cfg.get("max_themes", 6))
    max_excerpts_cfg = int(r_cfg.get("max_excerpts", 12))

    est_max_themes = min(max_themes_cfg, max(2, int(round(target_max / 65))))
    est_min_themes = max(2, min(est_max_themes, est_max_themes - 2))
    est_max_excerpts = min(max_excerpts_cfg, max(3, int(round(target_max / 34))))
    est_min_excerpts = max(2, min(est_max_excerpts, int(round(target_max / 60))))

    ch_lines = [
        f"- Ch{ch.index:02d} ({_fmt_ts(ch.start_s)}–{_fmt_ts(ch.end_s)}): {ch.title}"
        for ch in manifest.metadata.chapters
    ]
    topic_clip_map: dict[str, str] = {}
    for c in enabled_clips:
        for t in manifest.topic_map:
            if not (t.end_utt < c.start_utt or t.start_utt > c.end_utt):
                topic_clip_map.setdefault(t.topic_id, c.clip_id)

    tm_lines = []
    for t in manifest.topic_map:
        if t.is_filler:
            continue
        link = t.selected_clip_id or topic_clip_map.get(t.topic_id) or "未单切(可作同章前情)"
        tm_lines.append(
            f"- {t.topic_id} [{link}] u{t.start_utt}..u{t.end_utt} ({_fmt_ts(t.start_s)}–{_fmt_ts(t.end_s)}) "
            f"| 章节:{t.chapter_title or '无'} | {t.title} — {t.summary}"
        )

    blocks: list[str] = []
    for c in enabled_clips:
        allowed_lo, allowed_hi = clip_bounds[c.clip_id]
        lines = []
        if allowed_lo < c.start_utt:
            lines.append(f"  [同章前情铺垫（句号 {allowed_lo}..{c.start_utt - 1}，仅当正文缺乏主语/工具背景时从此处起选）]")
            for uid in range(allowed_lo, c.start_utt):
                u = utts[uid]
                dur_u = max(0.5, u.end_s - u.start_s)
                lines.append(f"  [句号={u.id} | 单句长{dur_u:.1f}s] {u.zh or u.en}")
            lines.append(f"  [切片正文（句号 {c.start_utt}..{c.end_utt}）]")
        for uid in range(c.start_utt, allowed_hi + 1):
            u = utts[uid]
            dur_u = max(0.5, u.end_s - u.start_s)
            sig = f" <<{','.join(u.signal_tags)}>>" if u.signal_tags else ""
            lines.append(f"  [句号={u.id} | 单句长{dur_u:.1f}s] {u.zh or u.en}{sig}")

        mm_note = c.multimodal_status if c.multimodal_status != "not_required" else "未需复核"
        blocks.append(
            f"### Clip {c.clip_id} (可用句号 {allowed_lo}..{allowed_hi} | 章节: {c.chapter_title or '无'} | "
            f"分类: {','.join(c.categories)} | 多模态: {mm_note}) | {c.title}\n"
            f"- 切片简介: {c.summary}\n"
            f"- 入选理由: {c.reason}\n"
            + "\n".join(lines)
        )

    for p in manifest.remix_only_pool:
        lines = []
        for uid in range(p.start_utt, p.end_utt + 1):
            u = utts[uid]
            dur_u = max(0.5, u.end_s - u.start_s)
            lines.append(f"  [句号={u.id} | 单句长{dur_u:.1f}s] {u.zh or u.en}")
        blocks.append(
            f"### ShortHighlight {p.candidate_id} (可用句号 {p.start_utt}..{p.end_utt} | 分类: {','.join(p.categories)}) | {p.title}\n"
            f"- 简介: {p.summary}\n"
            + "\n".join(lines)
        )

    prompt = f"""请以首席总编视角，基于全片宏观背景与以下定稿切片，策划一支总时长约 {target_min}~{target_max} 秒（硬上限 ≤{target_max} 秒，即原片 1/10）、分为 {est_min_themes}~{est_max_themes} 个主题板块、共 {est_min_excerpts}~{est_max_excerpts} 个弹性完整小段的 Remix 高光集锦。

【全片宏观语境】
- 原视频标题: {manifest.metadata.original_title}
- 核心中文主题: {manifest.metadata.original_theme_zh or manifest.metadata.original_title}
- 嘉宾: {manifest.metadata.guest_zh or '未知'} | 频道: {manifest.metadata.channel_zh or manifest.metadata.channel or '未知'}
- 主话题去重集合: {manifest.primary_topics}
- 原片总长: {int(src_dur)} 秒（1/10 上限 = {target_max} 秒）
- 原片简介:
{manifest.metadata.description[:2000] or '（无简介）'}

【官方章节结构】
{chr(10).join(ch_lines) or '（无官方章节）'}

【全片话题脉络（非寒暄部分，含未单切的章节铺垫）】
{chr(10).join(tm_lines) or '（见下方切片正文）'}

【四项硬性策展铁律】
1. **精选策展，不做全片简介**：主动舍弃平淡过渡、常识性历史回顾或内容重复的切片并写入 `dropped_clips`；把时间预算留给全片最犀利、反直觉、数据密度最高或交锋最精彩的 {est_min_themes}~{est_max_themes} 个板块。
2. **宏观立意优先，严禁“把载体当主张”**：提炼 `group_label` 和 `card_title` 时，必须站在**所属章节与全片主旨**的高度归纳嘉宾真正要证明的技术/商业命题！例如若嘉宾在“Agentic AI 实机演示”章节用内部 HR 调查、代码库或组织沟通记录作为分析样本来演示 AI 如何靠 Ontology 打破问答瓶颈、超越传统“问数”实现企业级决策推理，板块卡片必须总结“企业决策AI / 超越问数的智能分析”这类宏观洞察，严禁把表层举例误写成嘉宾对“HR管理/员工福利”有某种主张，并在 `surface_example_guard` 字段自检。
3. **非均匀弹性节奏与完整叙事单元**：
   - 不需要刻意设计开场 Hook，直接进入第 1 个主题板块（每个主题板块前插入 1 张转场卡片）。
   - 根据论证需要自由控制单段时长（单段上限 {int(max_excerpt_s)} 秒），严禁机械均匀切成 10 秒碎片：
     * `punchline`（8~22秒）：独立成立的尖锐判断或反直觉金句；
     * `qa_argument`（25~65秒）：完整保留“主持提问/反驳 + 嘉宾核心回应与论据”（有问必有答，不能只抛问题）；
     * `case_demo`（35~85秒）：完整保留“背景与工具前提 + 具体数据/对比转折 + 最终价值落点”；如果切片正文一开头直接进入举例缺少前提（如未交代演示的工具是什么），请从对应块的 `[同章前情铺垫]` 起选！
   - 开头严禁悬空代词/承接词起句，结尾必须停在完整结论句之后。
4. **板块聚合与句号规范**：
   - 同一 `theme` 内的 `segments`（1~2 段）必须同一章节/同一宏观议题，它们**共用该板块的 1 张转场标题卡**；
   - `start_utt` / `end_utt` 必须填写方括号中的**整数句号**（如 `[句号=336 | ...]` 填 `336`），落在对应 `source_clip_id` 的可用句号范围内，严禁重复句号；`quote` 用 `"开头原文...结尾原文"` 锚定首尾。

【输出 JSON Schema】
{{
  "remix_title": "整支高光集锦中文标题（≤22字）",
  "editorial_thesis": "2~3句总编策展思路（说明重点突出哪几大看点、为何舍弃其余平淡或背景段）",
  "dropped_clips": [
    {{
      "clip_id": "clip_01",
      "reason": "为何不在精华高光混剪中占用时长"
    }}
  ],
  "themes": [
    {{
      "theme_id": "TH01",
      "group_label": "3~7字板块短标签",
      "card_title": "6~18字宏观观点卡片标题",
      "macro_intent": "嘉宾在这一部分真正想证明的宏观命题",
      "surface_example_guard": "这段用到了什么表层举例/场景，确认卡片未把举例写成该领域主张",
      "segments": [
        {{
          "source_clip_id": "clip_03",
          "role": "core | hot_topic | exciting",
          "beat_type": "punchline | qa_argument | case_demo",
          "start_utt": 320,
          "end_utt": 377,
          "quote": "开头原话...结尾原话",
          "narrative_check": "确认前提/设问与结论/反转完整，独立观看能听懂"
        }}
      ]
    }}
  ]
}}

【可选定稿切片与前情正文】
{chr(10).join(blocks)}"""
    return prompt, target_min, target_max, hard_cap_s


def plan_remix_from_clips(
    manifest: ProjectManifest,
    store: ProjectStore,
    cfg: dict[str, Any],
    run_id: str,
) -> ProjectManifest:
    """Plan curated macro-context Remix AFTER clips are finalized (also callable after human edits)."""
    utts = manifest.utterances
    enabled_clips = [c for c in manifest.clips if c.enabled]
    if not enabled_clips and not manifest.remix_only_pool:
        manifest.remix.items = []
        manifest.remix.editorial_thesis = ""
        manifest.remix.render_hash = compute_remix_render_hash(manifest)
        store.save(manifest)
        return manifest

    r_cfg = cfg.get("remix", {})
    max_excerpt_s = float(r_cfg.get("max_excerpt_seconds", 90))
    clip_bounds: dict[str, tuple[int, int]] = {}
    for c in enabled_clips:
        bridge_lo = _compute_clip_bridge_window(c, manifest.clips, manifest.topic_map, manifest.metadata.chapters, utts)
        allowed_lo = bridge_lo if (c.start_utt - bridge_lo >= 2) else c.start_utt
        clip_bounds[c.clip_id] = (allowed_lo, c.end_utt)
    for p in manifest.remix_only_pool:
        clip_bounds[p.candidate_id] = (p.start_utt, p.end_utt)

    prompt, _, _, hard_cap_s = _build_remix_macro_prompt(manifest, cfg, enabled_clips, clip_bounds)

    models_cfg = cfg.get("models", {})
    text_provider = models_cfg.get("text_provider", "dashscope")
    remix_model = models_cfg.get("remix_model") or models_cfg.get("text_model") or "qwen3.8-max"
    text_fallbacks = list(models_cfg.get("text_fallback_models") or DEFAULT_TEXT_FALLBACK_MODELS)
    llm = ClipCutLLM(models_cfg)
    res = llm.generate_json(
        prompt=prompt,
        model=remix_model,
        provider=text_provider,
        fallback_models=text_fallbacks,
        system_instruction=REMIX_SYSTEM,
        temperature=0.25,
        max_output_tokens=8192,
    )
    data = res.parsed if isinstance(res.parsed, dict) else {}

    # Flatten nested themes -> segments (fallback to flat items for backwards compatibility)
    flat_raw_items: list[dict[str, Any]] = []
    theme_audit_logs: list[dict[str, Any]] = []
    raw_themes = data.get("themes")
    if isinstance(raw_themes, list) and raw_themes:
        for t_idx, th in enumerate(raw_themes):
            if not isinstance(th, dict):
                continue
            t_id = str(th.get("theme_id") or f"TH{t_idx + 1:02d}").strip()
            g_label = str(th.get("group_label") or "高光精选").strip()[:14]
            c_title = str(th.get("card_title") or g_label).strip()[:24]
            m_intent = str(th.get("macro_intent") or "").strip()
            s_guard = str(th.get("surface_example_guard") or "").strip()
            segs = th.get("segments") or []
            theme_audit_logs.append(
                {
                    "theme_id": t_id,
                    "group_label": g_label,
                    "card_title": c_title,
                    "macro_intent": m_intent,
                    "surface_example_guard": s_guard,
                    "segment_count": len(segs) if isinstance(segs, list) else 0,
                }
            )
            if not isinstance(segs, list):
                continue
            for seg in segs:
                if isinstance(seg, dict):
                    flat_raw_items.append(
                        {
                            **seg,
                            "theme_id": t_id,
                            "group_label": g_label,
                            "card_title": c_title,
                            "macro_intent": m_intent,
                        }
                    )
    else:
        for raw_it in data.get("items") or []:
            if isinstance(raw_it, dict):
                flat_raw_items.append(raw_it)

    rms, hop_s = _load_audio_envelope(store)
    n_utts = len(utts)
    items: list[RemixItem] = []
    used_utts: set[int] = set()
    valid_roles = {"core", "hot_topic", "exciting", "hook", "finale"}
    valid_beats = {"punchline", "qa_argument", "case_demo"}

    card_mode = str(r_cfg.get("card_mode", manifest.remix.card_mode) or "group")
    card_dur_base = float(r_cfg.get("card_duration_s", manifest.remix.card_duration_s)) if card_mode != "none" else 0.0
    accumulated_s = 0.0
    last_group_key: str | None = None

    for raw_it in flat_raw_items:
        src_cid = str(raw_it.get("source_clip_id") or "").strip()
        quote_raw = str(raw_it.get("quote") or "").strip()
        allowed_lo, allowed_hi = clip_bounds.get(src_cid, (0, n_utts - 1))

        su = _parse_utt_int(raw_it.get("start_utt"), -1)
        eu = _parse_utt_int(raw_it.get("end_utt"), su)
        if eu < su:
            su, eu = eu, su

        in_bounds = allowed_lo <= su <= allowed_hi and allowed_lo <= eu <= allowed_hi
        raw_dur = (utts[eu].end_s - utts[su].start_s) if in_bounds else 0.0
        # Safeguard A: only realign via quote text if outside source bounds or >1.5x max_excerpt_s
        if not in_bounds or raw_dur > max_excerpt_s * 1.5:
            if quote_raw:
                su, eu = _match_quote_to_utterances(quote_raw, utts, allowed_lo, allowed_hi, max_span_s=max_excerpt_s)
            else:
                su = max(allowed_lo, min(su, allowed_hi))
                eu = min(allowed_hi, su + 3)

        # Safeguard B: enforce max_excerpt_s by trimming trailing utterances
        while eu > su and (utts[eu].end_s - utts[su].start_s) > max_excerpt_s:
            eu -= 1

        # Sentence completeness guard without creeping into used_utts or before allowed_lo
        cand_su, cand_eu, _ = ensure_sentence_completeness(su, eu, utts)
        cand_su = max(allowed_lo, cand_su)
        cand_eu = min(n_utts - 1, cand_eu)
        if (
            not any(u_idx in used_utts for u_idx in range(cand_su, cand_eu + 1))
            and (utts[cand_eu].end_s - utts[cand_su].start_s) <= max_excerpt_s + 8.0
        ):
            su, eu = cand_su, cand_eu

        # Safeguard C: trim edge overlap against used_utts; skip if >50% overlapped
        orig_count = max(1, eu - su + 1)
        overlap_count = sum(1 for u_idx in range(su, eu + 1) if u_idx in used_utts)
        if overlap_count / orig_count > 0.5:
            continue
        while su <= eu and su in used_utts:
            su += 1
        while eu >= su and eu in used_utts:
            eu -= 1
        if su > eu or any(u_idx in used_utts for u_idx in range(su, eu + 1)):
            continue

        group_label_val = str(raw_it.get("group_label") or "高光精选").strip()[:14]
        theme_id_val = str(raw_it.get("theme_id") or "").strip()
        next_group_key = group_label_val or theme_id_val or f"rmx_{len(items) + 1:02d}"
        item_card_fee = _card_fee_for_item(card_mode, card_dur_base, bool(items), next_group_key, last_group_key)

        s_t, e_t, _, _, _ = snap_bounds_with_acoustic_guard(
            start_utt=su,
            end_utt=eu,
            utterances=utts,
            rms=rms,
            hop_s=hop_s,
            video_duration=manifest.source.duration_s,
            cfg=cfg,
        )
        seg_dur = max(0.5, e_t - s_t)

        # Safeguard D: strictly enforce <= 1/10 hard cap without chopping full arguments into 1-sentence fragments
        if items and (accumulated_s + seg_dur + item_card_fee) > hard_cap_s:
            trimmed_fit = False
            cand_eu = eu
            max_trim_s = seg_dur * 0.20
            while cand_eu > su:
                cand_eu -= 1
                t_s, t_e, _, _, _ = snap_bounds_with_acoustic_guard(
                    start_utt=su,
                    end_utt=cand_eu,
                    utterances=utts,
                    rms=rms,
                    hop_s=hop_s,
                    video_duration=manifest.source.duration_s,
                    cfg=cfg,
                )
                new_dur = t_e - t_s
                if (seg_dur - new_dur) > max_trim_s or new_dur < 15.0:
                    break
                tail_txt = (utts[cand_eu].zh or utts[cand_eu].en).strip()
                if (accumulated_s + new_dur + item_card_fee) <= hard_cap_s and tail_txt and TERMINAL_PUNCT_RE.search(tail_txt):
                    eu, s_t, e_t, seg_dur = cand_eu, t_s, t_e, new_dur
                    trimmed_fit = True
                    break
            if not trimmed_fit:
                continue

        for u_idx in range(su, eu + 1):
            used_utts.add(u_idx)
        accumulated_s += seg_dur + item_card_fee
        last_group_key = next_group_key

        role = str(raw_it.get("role", "core"))
        if role not in valid_roles or role in ("hook", "finale"):
            role = "core"
        beat_type = str(raw_it.get("beat_type") or "").strip()
        if beat_type not in valid_beats:
            beat_type = ""
        order_num = len(items) + 1
        actual_quote = " ".join((utts[k].zh or utts[k].en).strip() for k in range(su, eu + 1)).strip()
        items.append(
            RemixItem(
                remix_id=f"rmx_{order_num:02d}",
                order=order_num,
                enabled=True,
                source_clip_id=src_cid or None,
                theme_id=theme_id_val,
                beat_type=beat_type,  # type: ignore[arg-type]
                role=role,  # type: ignore[arg-type]
                group_label=group_label_val,
                card_title=str(raw_it.get("card_title") or group_label_val).strip()[:24],
                macro_intent=str(raw_it.get("macro_intent") or "").strip(),
                start_utt=su,
                end_utt=eu,
                start_s=s_t,
                end_s=e_t,
                quote=actual_quote or quote_raw,
                reason=str(raw_it.get("narrative_check") or raw_it.get("reason") or "").strip(),
            )
        )

    manifest.remix.title = str(data.get("remix_title") or f"{manifest.metadata.original_theme_zh or '全片'} · 高光混剪")
    manifest.remix.editorial_thesis = str(data.get("editorial_thesis") or "").strip()
    manifest.remix.card_mode = r_cfg.get("card_mode", manifest.remix.card_mode)
    manifest.remix.card_duration_s = float(r_cfg.get("card_duration_s", manifest.remix.card_duration_s))
    manifest.remix.card_fade_s = float(r_cfg.get("card_fade_s", manifest.remix.card_fade_s))
    manifest.remix.enable_sfx = bool(r_cfg.get("enable_sfx", manifest.remix.enable_sfx))
    manifest.remix.enable_bgm = bool(r_cfg.get("enable_bgm", manifest.remix.enable_bgm))
    manifest.remix.bgm_volume = float(r_cfg.get("bgm_volume", manifest.remix.bgm_volume))
    manifest.remix.sfx_volume = float(r_cfg.get("sfx_volume", manifest.remix.sfx_volume))
    manifest.remix.items = items
    manifest.remix.render_hash = compute_remix_render_hash(manifest)

    total_speech_s = sum(max(0.0, it.end_s - it.start_s) for it in items)
    theme_count = len(theme_audit_logs) or len({it.group_label for it in items if it.group_label})
    store.save(manifest)
    store.append_receipt(
        run_id=run_id,
        stage="verify_and_plan",
        actor="ai",
        action="plan_remix",
        model=f"{res.provider}:{res.model}",
        prompt_tokens=res.prompt_tokens,
        output_tokens=res.output_tokens,
        elapsed_ms=res.elapsed_ms,
        summary=f"Remix 策展完成：{theme_count} 个板块 / {len(items)} 小段，纯人声 {total_speech_s:.1f}s，预估总长 {accumulated_s:.1f}s（上限 {hard_cap_s:.1f}s）",
        details={
            "remix_title": manifest.remix.title,
            "editorial_thesis": manifest.remix.editorial_thesis,
            "dropped_clips": data.get("dropped_clips") or [],
            "theme_count": theme_count,
            "themes": theme_audit_logs,
            "excerpt_count": len(items),
            "total_speech_seconds": round(total_speech_s, 1),
            "estimated_total_seconds": round(accumulated_s, 1),
            "hard_cap_seconds": round(hard_cap_s, 1),
        },
    )
    return manifest


def run_verify_and_plan_stage(
    manifest: ProjectManifest,
    store: ProjectStore,
    cfg: dict[str, Any],
    run_id: str,
) -> ProjectManifest:
    models_cfg = cfg.get("models", {})
    llm = ClipCutLLM(models_cfg)
    utts = manifest.utterances
    rms, hop_s = _load_audio_envelope(store)

    # 1. Targeted multimodal verification for exciting_dialogue candidates
    _verify_exciting_candidates(manifest, store, cfg, llm, run_id)

    # 2. Filter rejected pure-exciting candidates
    surviving: list[Candidate] = []
    remix_only_pool: list[Candidate] = []
    for c in manifest.candidates:
        if c.multimodal.status == "reject":
            other_cats = [x for x in c.categories if x != "exciting_dialogue"]
            if not other_cats:
                c.disposition = "dropped"
                c.disposition_reason = f"多模态复核淘汰: {c.multimodal.evidence}"
                continue
            else:
                c.categories = other_cats  # type: ignore[arg-type]
        if c.disposition == "remix_only":
            remix_only_pool.append(c)
        else:
            surviving.append(c)

    # 3. Overlap merge
    dur_cfg = cfg.get("clip_duration", {})
    iou_thresh = float(dur_cfg.get("overlap_merge_iou", 0.35))
    min_s = float(dur_cfg.get("min_seconds", 60))
    max_s = float(dur_cfg.get("max_seconds", 300))

    merged_cands = _merge_overlapping_candidates(surviving, utts, iou_thresh)

    # 4. Sentence completeness + acoustic pause-midpoint snapping + duration checks
    clips: list[ClipItem] = []
    snap_logs: list[dict[str, Any]] = []

    for cand in merged_cands:
        su, eu, comp_notes = ensure_sentence_completeness(cand.start_utt, cand.end_utt, utts)
        s_t, e_t, d_start, d_end, guard_notes = snap_bounds_with_acoustic_guard(
            start_utt=su,
            end_utt=eu,
            utterances=utts,
            rms=rms,
            hop_s=hop_s,
            video_duration=manifest.source.duration_s,
            cfg=cfg,
        )
        dur = e_t - s_t

        # Route very short (<48s) high-impact clips to remix_only_pool rather than padding artificially
        if dur < min(48.0, min_s * 0.8):
            cand.disposition = "remix_only"
            cand.disposition_reason = f"时长 {dur:.1f}s 短于独立切片下限，转入 Remix 精选池（不硬凑铺垫）"
            remix_only_pool.append(cand)
            continue

        notes = comp_notes + guard_notes
        if cand.multimodal.status == "pass":
            notes.append(f"多模态已看画面确认：{cand.multimodal.evidence[:70]}")
        elif cand.multimodal.status == "downgrade":
            notes.append(f"多模态复核强度偏弱（已降权保留）：{cand.multimodal.evidence[:70]}")
        elif cand.multimodal.status == "unverified":
            notes.append("多模态复核未完成（超时/错误），当前仅凭声画指标与文本保留")

        if dur > max_s + 20:
            notes.append(f"时长 {dur:.0f}s 略超 {int(max_s)}s 上限（为保持完整论证未强行腰斩，可在工作台微调）")

        idx = len(clips) + 1
        clip_id = f"clip_{idx:02d}"
        clip = ClipItem(
            clip_id=clip_id,
            index=idx,
            enabled=True,
            categories=list(cand.categories),
            matched_hot_topics=list(cand.matched_hot_topics),
            chapter_title=cand.chapter_title,
            title=cand.title,
            title_en=cand.title_en,
            slug=cand.slug,
            hook=cand.hook or (utts[su].zh or utts[su].en),
            reason=cand.reason,
            summary=cand.summary,
            key_quote=cand.key_quote,
            notes=notes,
            start_utt=su,
            end_utt=eu,
            start_s=s_t,
            end_s=e_t,
            snap_delta_start_ms=d_start,
            snap_delta_end_ms=d_end,
            multimodal_status=cand.multimodal.status,
            score=float(cand.score),
        )
        clip.render_hash = compute_clip_render_hash(clip, manifest.source.render_video)
        clips.append(clip)
        cand.disposition = "selected"
        cand.disposition_reason = f"定稿为 {clip_id} ({dur:.1f}s)"

        # Link back to topic_map
        for t in manifest.topic_map:
            if t.topic_id in cand.topic_ids:
                t.selected_clip_id = clip_id
                t.dropped_reason = ""

        snap_logs.append(
            {
                "clip_id": clip_id,
                "title": clip.title,
                "orig_utts": [cand.start_utt, cand.end_utt],
                "final_utts": [su, eu],
                "snap_delta_ms": [d_start, d_end],
                "duration_s": round(dur, 1),
            }
        )

    manifest.clips = clips
    manifest.remix_only_pool = remix_only_pool

    sel_cfg = cfg.get("clip_selection", {})
    default_max = int(sel_cfg.get("default_selected_max", 6))
    if len(clips) > default_max:
        curated = select_curated_top_clips(manifest, n=default_max)
        curated_ids = {c.clip_id: tag for c, tag in curated}
        for clip in clips:
            if clip.clip_id in curated_ids:
                clip.enabled = True
                clip.curation_tag = curated_ids[clip.clip_id]
            else:
                clip.enabled = False
                clip.curation_tag = ""
                clip.notes.append(f"长片候选较多（共 {len(clips)} 支）：已智能策划预选 Top {default_max}，本切片保留策略，可在工作台按需启用")
    else:
        for clip in clips:
            clip.enabled = True
            clip.curation_tag = ""

    store.save(manifest)
    store.append_receipt(
        run_id=run_id,
        stage="verify_and_plan",
        actor="system",
        action="finalize_clips_and_snap",
        summary=f"定稿独立切片 {len(clips)} 支（短高光入 Remix 池 {len(remix_only_pool)} 支），全部完成完整句校验与静音中点吸附",
        details={"clips_snapping": snap_logs},
    )

    # 5. Plan Remix strictly after clips are finalized


def select_curated_top_clips(
    manifest: ProjectManifest,
    n: int = 5,
) -> list[tuple[ClipItem, str]]:
    """
    Select Top N clips covering diverse editorial pillars rather than flat chronological order:
    1. Core Theme (最反映全片主旨/核心痛点)
    2. Controversy & Debunking (最具争议/打破行业共识)
    3. Exciting Dialogue (最精彩交锋/声画表现力)
    4. Hardcore Production Use Case (硬核落地实操/量化价值)
    5. Highest Overall (综合王牌高分)
    """
    clips = manifest.clips
    if not clips:
        return []
    if len(clips) <= n:
        return [(c, "⭐ 精选") for c in clips]

    orig_title = manifest.metadata.original_title or ""

    def _score_theme(c: ClipItem) -> float:
        text = f"{c.title} {c.summary} {c.reason} {c.hook}"
        s = c.score
        # Keyword overlap with original title
        for word in re.findall(r"[\w]+", orig_title):
            if len(word) >= 2 and word.lower() in text.lower():
                s += 0.4
        # Core thematic pain points / keywords
        if re.search(r"看不到|机密|数据|合规|隐私|看不见|无法查看|无法看到|数据室|data room", text, flags=re.IGNORECASE):
            s += 2.0
        if re.search(r"智能体|工作流|编排|Agent|系统架构|模拟世界", text, flags=re.IGNORECASE):
            s += 1.0
        return s

    def _score_controversy(c: ClipItem) -> float:
        text = f"{c.title} {c.summary} {c.reason} {c.hook} {c.key_quote}"
        s = c.score
        if re.search(r"初级律师|消失|计费|按小时|合伙制|不可拥有|阻碍|危机|辩论|存亡|冲击|替代|证伪|抗拒|瓶颈|收费", text, flags=re.IGNORECASE):
            s += 2.5
        return s

    def _score_exciting(c: ClipItem) -> float:
        s = c.score
        if "exciting_dialogue" in c.categories:
            s += 2.2
        if c.multimodal_status == "pass":
            s += 1.2
        return s

    def _score_practice(c: ClipItem) -> float:
        s = c.score
        if "hot_topic" in c.categories:
            s += 1.5
        text = f"{c.title} {c.summary} {c.reason} {c.hook}"
        if re.search(r"FDE|前线部署|表格审查|开放权重|检索|起草|万份合同|原语|落地|实操|量化|成本|ROI|准确率", text, flags=re.IGNORECASE):
            s += 1.5
        return s

    selected: list[tuple[ClipItem, str]] = []
    used_ids: set[str] = set()

    slots = [
        ("🔥 核心主题", _score_theme),
        ("⚡ 行业争议", _score_controversy),
        ("💥 精彩交锋", _score_exciting),
        ("🛠 硬核实操", _score_practice),
        ("⭐ 综合王牌", lambda c: c.score),
    ]

    for label, scoring_fn in slots:
        if len(selected) >= n:
            break
        available = [c for c in clips if c.clip_id not in used_ids]
        if not available:
            break
        best = max(available, key=scoring_fn)
        selected.append((best, label))
        used_ids.add(best.clip_id)

    # Fill remaining slots up to n from remaining highest scored
    while len(selected) < n:
        available = [c for c in clips if c.clip_id not in used_ids]
        if not available:
            break
        best = max(available, key=lambda c: c.score)
        selected.append((best, "⭐ 推荐精选"))
        used_ids.add(best.clip_id)

    return selected

    manifest = plan_remix_from_clips(manifest, store, cfg, run_id)
    return manifest
