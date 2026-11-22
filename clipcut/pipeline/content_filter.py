from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Literal

import numpy as np

from ..cancel import JobCancelledError, check_cancelled, run_cancellable_subprocess
from ..config import load_editorial_config
from ..llm import ClipCutLLM, DEFAULT_TEXT_FALLBACK_MODELS
from ..models import CutSegment, ProjectManifest, Utterance, utc_now_iso
from ..receipts import ProjectStore


VALID_SEVERITIES = ("high", "medium", "low")
VALID_POLITICAL_ACTIONS = ("auto", "delete", "mask")

_DEFAULT_ZH_TO_EN_TERMS: dict[str, list[str]] = {
    "习近平": ["xi jinping", "xi"],
    "习主席": ["xi jinping", "xi"],
    "中国共产党": ["chinese communist party", "ccp"],
    "共产党": ["communist party", "communist", "ccp"],
    "中共": ["ccp", "chinese communist party", "communist party"],
    "台海": ["taiwan strait", "taiwan"],
    "台湾": ["taiwan", "taiwanese"],
    "香港": ["hong kong"],
    "新疆": ["xinjiang"],
    "西藏": ["tibet", "tibetan"],
    "南海": ["south china sea"],
    "北京": ["beijing"],
    "中国": ["china", "chinese"],
    "政权": ["regime"],
}

DEFAULT_POLITICAL_PROFILES: dict[str, dict[str, Any]] = {
    "strict": {
        "label": "严格从严（推荐视频号/B站）",
        "description": "适配中国大陆主流平台强审核：覆盖涉华及中国友好国家（俄/朝/伊朗与伊斯兰世界等）的硬红线、明确指责对抗及边缘涉政表述。",
        "trigger_severities": ["high", "medium", "low"],
        "rules": [
            {
                "id": "hard_redline",
                "name": "硬性政治红线",
                "severity": "high",
                "description": "直接涉及中国领导人、执政党、政治体制、政权评论或影射；台湾/香港/新疆/西藏/南海主权挑战或台海军事冲突；对中/俄/朝/伊朗等中国友好国家的政权攻击、极端军事打击渲染或直接意识形态攻击。",
            },
            {
                "id": "geopolitics_confrontation",
                "name": "涉华及友好国家对抗站队",
                "severity": "medium",
                "description": "上下文明确指向中国，或指向俄罗斯、朝鲜、伊朗/伊斯兰世界等中国友好国家，且发表针对上述国家的军事威慑、国防军工部署、芯片/科技制裁封锁或地缘政治对抗站队（纯提及美国科技领先或中性中外产业交流不标）。",
            },
            {
                "id": "policy_sensitive",
                "name": "涉华及友好国家政策体制指责",
                "severity": "medium",
                "description": "上下文明确指向中国，或指向俄罗斯、朝鲜、伊朗/伊斯兰世界等中国友好国家，并对其政府政策、监管审查、人权或体制提出明确指责或负面定性批评（泛谈“政府监管/接管”或批评美国/西方政府、NSA及商业公司不标）。",
            },
            {
                "id": "borderline_stance",
                "name": "从严边缘涉政表述复核",
                "severity": "low",
                "description": "确涉及中国或俄/朝/伊朗等中国友好国家的敏感地缘摩擦、制度隐喻或争议政治人物评价，但措辞温和间接的边缘涉政表述（纯商业/科技访谈中中性提及去中国参观企业、对比技术进度、称赞美国领先、批评欧美政府或公司严禁标为低危，应直接忽略不输出）。",
            },
        ],
    },
    "balanced": {
        "label": "平衡",
        "description": "聚焦硬红线及针对中国或友好国家（俄/朝/伊朗等）的明确指责与对抗；放行一般技术对比、商业讨论及欧美内政。",
        "trigger_severities": ["high", "medium"],
        "rules": [
            {
                "id": "hard_redline",
                "name": "硬性政治红线",
                "severity": "high",
                "description": "直接涉及中国领导人、执政党、政治体制或政权评价/影射；台湾/香港/新疆/西藏/南海主权议题；对中/俄/朝/伊朗等中国友好国家的政权攻击或直接意识形态攻击。",
            },
            {
                "id": "geopolitics_confrontation",
                "name": "明确涉华及友好国家军事制裁对抗",
                "severity": "medium",
                "description": "上下文明确指向中国或俄罗斯、朝鲜、伊朗/伊斯兰世界等中国友好国家的军事威慑、台海冲突推演或直接制裁对抗站队（普通中美技术对比、单纯称赞美国领先或非涉华历史类比不标中危或低危）。",
            },
            {
                "id": "policy_sensitive",
                "name": "涉华及友好国家审查与体制指责",
                "severity": "medium",
                "description": "明确指向中国或俄罗斯、朝鲜、伊朗/伊斯兰世界等中国友好国家的审查、监控或人权体制指责；泛指“政府”、批评欧美政府/NSA或商业垄断不标。",
            },
        ],
    },
    "loose": {
        "label": "仅硬红线",
        "description": "仅识别直接触及国内法律法规与平台红线的高危内容；一般地缘竞争、产业讨论、历史类比与欧美内政均放行。",
        "trigger_severities": ["high"],
        "rules": [
            {
                "id": "hard_redline",
                "name": "硬性政治红线",
                "severity": "high",
                "description": "直接涉及中国领导人、执政党、政治体制或政权评论与影射；台湾/香港/新疆/西藏/南海主权挑战；敏感政治历史事件、违禁人物与直接意识形态攻击。",
            },
            {
                "id": "exclude_general_geopolitics",
                "name": "非红线政治/产业评论豁免（指导界定）",
                "severity": "low",
                "description": "普通科技竞争、美国本土政策吐槽、非涉华历史冷战/核威慑类比或开源模型对比不属高危红线，严禁误标为 high。",
            },
        ],
    },
}


def _safe_load_cfg(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    if cfg is not None:
        return cfg
    try:
        loaded = load_editorial_config()
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def get_political_profiles(cfg: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Load, validate, and normalize political sensitivity profiles from config with defaults."""
    resolved = _safe_load_cfg(cfg)
    raw_profiles = resolved.get("content_filter", {}).get("political_profiles")
    out: dict[str, dict[str, Any]] = {}
    for key in ("strict", "balanced", "loose"):
        base = copy.deepcopy(DEFAULT_POLITICAL_PROFILES[key])
        raw_p = raw_profiles.get(key) if isinstance(raw_profiles, dict) else None
        if not isinstance(raw_p, dict):
            out[key] = base
            continue
        label = str(raw_p.get("label") or base["label"]).strip() or base["label"]
        desc = str(raw_p.get("description") if raw_p.get("description") is not None else base["description"]).strip()
        raw_trig = raw_p.get("trigger_severities")
        if isinstance(raw_trig, list):
            trig = [s for s in VALID_SEVERITIES if s in {str(x).strip().lower() for x in raw_trig}]
        else:
            trig = list(base["trigger_severities"])
        if not trig:
            trig = list(base["trigger_severities"])

        raw_rules = raw_p.get("rules")
        rules: list[dict[str, str]] = []
        if isinstance(raw_rules, list):
            seen_ids: set[str] = set()
            for idx, r in enumerate(raw_rules, 1):
                if not isinstance(r, dict):
                    continue
                name = str(r.get("name") or "").strip()
                rdesc = str(r.get("description") or "").strip()
                if not name and not rdesc:
                    continue
                rid = str(r.get("id") or f"rule_{idx}").strip() or f"rule_{idx}"
                if rid in seen_ids:
                    rid = f"{rid}_{idx}"
                seen_ids.add(rid)
                sev = str(r.get("severity") or "medium").strip().lower()
                if sev not in VALID_SEVERITIES:
                    sev = "medium"
                rules.append(
                    {
                        "id": rid,
                        "name": name or rid,
                        "severity": sev,
                        "description": rdesc,
                    }
                )
        else:
            rules = list(base["rules"])

        out[key] = {
            "label": label,
            "description": desc,
            "trigger_severities": trig,
            "rules": rules,
        }
    return out


def get_profile_trigger_severities(
    sensitivity: str,
    cfg: dict[str, Any] | None = None,
) -> set[str]:
    profiles = get_political_profiles(cfg)
    key = str(sensitivity or "strict").strip().lower()
    if key not in profiles:
        key = "strict"
    trig = profiles[key].get("trigger_severities") or DEFAULT_POLITICAL_PROFILES[key]["trigger_severities"]
    return {str(s).lower() for s in trig if str(s).lower() in VALID_SEVERITIES}


def segment_matches_sensitivity(
    seg: CutSegment,
    sensitivity: str,
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Manual blocks and ads always match; LLM/visual political segments follow active profile trigger_severities."""
    if seg.kind != "political_sensitive" or seg.detector == "manual":
        return True
    sev = str(seg.severity or "high").strip().lower()
    if sev not in VALID_SEVERITIES:
        sev = "high"
    return sev in get_profile_trigger_severities(sensitivity, cfg)


CONTENT_FILTER_SYSTEM = """你是面向中国大陆视频平台（微信视频号、B站、小红书）的精准合规剪辑审核员。
你的任务是精确识别长视频字幕中的两类需要处理的片段：
1. 商业广告与推广植入（kind = "ad"，必须包含明确第三方品牌/赞助产品或具体付费导流诉求）
2. 针对中国、台海/港澳疆藏南海主权，或针对俄罗斯、朝鲜、伊朗/伊斯兰世界等中国友好国家的实质性政治红线或明确政权/政策指责对抗（kind = "political_sensitive"）
严禁进行“可能引发对中国政府/平台监管联想”“可能引发群体对立/社会争议”之类的滑坡式过度推断；不得将泛指“政府”、商业垄断批评（Apple/Google/Meta）、欧美内政文化争议（woke/Elon/Trump）或粗口标为涉政。
你必须精准定位字幕句号范围 `[start_utt, end_utt]`，只输出严格的 JSON，不要输出 Markdown 代码块或多余文字。"""


def _fmt_ts(sec: float) -> str:
    s = max(0, int(round(sec)))
    h, rem = divmod(s, 3600)
    m, r = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{r:02d}" if h else f"{m:02d}:{r:02d}"


def has_clean_source_video(manifest: ProjectManifest) -> bool:
    """Return True if manifest points to an existing clean (non-hardsub) analysis video."""
    raw_clean = (manifest.source.analysis_video or "").strip()
    if not raw_clean:
        return False
    clean_p = Path(raw_clean).expanduser()
    if not clean_p.is_file():
        return False
    raw_render = (manifest.source.render_video or "").strip()
    if raw_render and "_hardsub" in clean_p.stem.lower():
        try:
            if clean_p.resolve() == Path(raw_render).expanduser().resolve():
                return False
        except Exception:
            if raw_clean == raw_render:
                return False
    return True


def resolve_segment_action(
    seg: CutSegment,
    manifest: ProjectManifest,
    *,
    allow_mask: bool | None = None,
) -> Literal["delete", "mask"]:
    """Resolve effective handling action ('delete' or 'mask') for a CutSegment."""
    if seg.kind != "political_sensitive":
        return "delete"
    can_mask = has_clean_source_video(manifest) if allow_mask is None else bool(allow_mask)
    if not can_mask:
        return "delete"
    seg_act = str(getattr(seg, "action", "inherit") or "inherit").strip().lower()
    if seg_act == "mask":
        return "mask"
    if seg_act == "delete":
        return "delete"
    global_act = str(getattr(manifest.clean_cut, "political_action", "delete") or "delete").strip().lower()
    if global_act == "mask":
        return "mask"
    if global_act == "auto":
        rec = str(getattr(seg, "recommended_action", "delete") or "delete").strip().lower()
        return "mask" if rec == "mask" else "delete"
    return "delete"


def get_active_cut_segments(
    manifest: ProjectManifest,
    cfg: dict[str, Any] | None = None,
) -> list[CutSegment]:
    """Return enabled CutSegments whose category toggle and active political standard match."""
    cc = manifest.clean_cut
    out: list[CutSegment] = []
    for seg in manifest.cut_segments:
        if not seg.enabled:
            continue
        if seg.kind == "ad" and not cc.cut_ads:
            continue
        if seg.kind == "political_sensitive":
            if not cc.cut_political_sensitive:
                continue
            if not segment_matches_sensitivity(seg, cc.sensitivity, cfg):
                continue
        if seg.end_s <= seg.start_s:
            continue
        out.append(seg)
    return out


def get_active_delete_segments(
    manifest: ProjectManifest,
    cfg: dict[str, Any] | None = None,
    *,
    allow_mask: bool | None = None,
) -> list[CutSegment]:
    """Return active CutSegments resolved to physical video deletion."""
    return [
        seg
        for seg in get_active_cut_segments(manifest, cfg)
        if resolve_segment_action(seg, manifest, allow_mask=allow_mask) == "delete"
    ]


def get_active_mask_segments(
    manifest: ProjectManifest,
    cfg: dict[str, Any] | None = None,
    *,
    allow_mask: bool | None = None,
) -> list[CutSegment]:
    """Return active CutSegments resolved to subtitle redaction + audio bleep masking."""
    return [
        seg
        for seg in get_active_cut_segments(manifest, cfg)
        if resolve_segment_action(seg, manifest, allow_mask=allow_mask) == "mask"
    ]


def merge_spans(spans: list[tuple[float, float]], max_gap_s: float = 0.15) -> list[tuple[float, float]]:
    """Sort and merge overlapping or near-adjacent time spans."""
    cleaned = sorted((max(0.0, float(s)), max(0.0, float(e))) for s, e in spans if e > s)
    merged: list[tuple[float, float]] = []
    for s, e in cleaned:
        if merged and s <= merged[-1][1] + max_gap_s:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _segments_to_merged_spans(
    segs: list[CutSegment],
    duration_s: float,
    *,
    max_gap_s: float = 0.15,
) -> list[tuple[float, float]]:
    raw: list[tuple[float, float]] = []
    for seg in segs:
        s = max(0.0, float(seg.start_s))
        e = float(seg.end_s)
        if duration_s > 0:
            e = min(duration_s, e)
        if e - s > 0.05:
            raw.append((round(s, 3), round(e, 3)))
    return merge_spans(raw, max_gap_s=max_gap_s)


def get_active_cut_spans(
    manifest: ProjectManifest,
    cfg: dict[str, Any] | None = None,
) -> list[tuple[float, float]]:
    """Return sorted, merged (start_s, end_s) spans of all active cuts clamped to video duration."""
    dur = float(manifest.source.duration_s or 0.0)
    return _segments_to_merged_spans(get_active_cut_segments(manifest, cfg), dur, max_gap_s=0.15)


def get_active_delete_spans(
    manifest: ProjectManifest,
    cfg: dict[str, Any] | None = None,
    *,
    allow_mask: bool | None = None,
) -> list[tuple[float, float]]:
    """Return sorted, merged (start_s, end_s) spans for segments that physically cut the timeline."""
    dur = float(manifest.source.duration_s or 0.0)
    return _segments_to_merged_spans(
        get_active_delete_segments(manifest, cfg, allow_mask=allow_mask),
        dur,
        max_gap_s=0.15,
    )


def _find_transcribe_state_path(manifest: ProjectManifest) -> Path | None:
    seen: set[str] = set()
    for raw in (manifest.source.analysis_video, manifest.source.render_video, manifest.source.work_dir):
        if not raw:
            continue
        p = Path(raw).expanduser()
        cand_dir = p if p.is_dir() else p.parent
        key = str(cand_dir)
        if key in seen:
            continue
        seen.add(key)
        ts = cand_dir / "transcribe_state.json"
        if ts.is_file():
            return ts
    return None


def _normalize_en_tokens(text: str) -> list[str]:
    norm = re.sub(r"([a-z0-9])['’]s\b", r"\1", str(text or "").lower())
    return re.findall(r"[a-z0-9]+", norm)


def _load_whisper_words(ts_path: Path) -> list[tuple[str, float, float]]:
    try:
        raw = json.loads(ts_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    segs = raw.get("segments") if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    if not isinstance(segs, list):
        return []
    out: list[tuple[str, float, float]] = []
    for seg in segs:
        if not isinstance(seg, dict):
            continue
        words = seg.get("words") or []
        if not isinstance(words, list):
            continue
        for w in words:
            if not isinstance(w, dict):
                continue
            try:
                ws = float(w.get("start"))
                we = float(w.get("end"))
            except (TypeError, ValueError):
                continue
            if we <= ws:
                continue
            toks = _normalize_en_tokens(str(w.get("word") or w.get("text") or ""))
            for tok in toks:
                out.append((tok, ws, we))
    return out


def _tokens_match(word_tok: str, term_tok: str) -> bool:
    if word_tok == term_tok:
        return True
    if len(term_tok) >= 3 and (word_tok == term_tok + "s" or term_tok == word_tok + "s"):
        return True
    return False


def _collect_search_en_terms(seg: CutSegment) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for t in seg.sensitive_terms_en or []:
        s = str(t or "").strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            terms.append(s)
    if not terms:
        for zt in seg.sensitive_terms_zh or []:
            z_clean = str(zt or "").strip()
            if not z_clean:
                continue
            for zh_k, en_list in _DEFAULT_ZH_TO_EN_TERMS.items():
                if zh_k in z_clean or z_clean in zh_k:
                    for en_t in en_list:
                        if en_t.lower() not in seen:
                            seen.add(en_t.lower())
                            terms.append(en_t)
    return terms


def _compile_en_term_pattern(term: str, *, allow_empty_gap: bool = False) -> re.Pattern[str] | None:
    parts = [re.escape(p) for p in term.split() if p]
    if not parts:
        return None
    joiner = r"\s*" if allow_empty_gap else r"\s+"
    body = joiner.join(parts)
    return re.compile(
        rf"(?<![A-Za-z0-9]){body}(?:['’]s|s)?(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def resolve_mask_audio_spans(
    manifest: ProjectManifest,
    mask_segments: list[CutSegment],
) -> list[tuple[float, float]]:
    """Resolve word-level audio mute/bleep time spans for masked political segments."""
    if not mask_segments:
        return []
    dur = float(manifest.source.duration_s or (manifest.utterances[-1].end_s if manifest.utterances else 0.0))
    ts_path = _find_transcribe_state_path(manifest)
    whisper_words = _load_whisper_words(ts_path) if ts_path is not None else []
    utts = manifest.utterances

    raw_spans: list[tuple[float, float]] = []
    for seg in mask_segments:
        terms_zh = [str(x).strip() for x in (seg.sensitive_terms_zh or []) if str(x).strip()]
        en_terms = _collect_search_en_terms(seg)
        if not terms_zh and not (seg.sensitive_terms_en or []):
            raw_spans.append((seg.start_s, seg.end_s))
            continue
        if not en_terms:
            raw_spans.append((seg.start_s, seg.end_s))
            continue

        win_lo = max(0.0, seg.start_s - 0.5)
        win_hi = seg.end_s + 0.5
        win_words = [(tok, ws, we) for tok, ws, we in whisper_words if we >= win_lo and ws <= win_hi]

        unmatched_terms: list[str] = []
        seg_matched = False
        for term in en_terms:
            t_toks = _normalize_en_tokens(term)
            if not t_toks:
                continue
            found_in_whisper = False
            n_t = len(t_toks)
            if len(win_words) >= n_t:
                for i in range(len(win_words) - n_t + 1):
                    if all(_tokens_match(win_words[i + j][0], t_toks[j]) for j in range(n_t)):
                        raw_spans.append(
                            (
                                max(0.0, win_words[i][1] - 0.04),
                                win_words[i + n_t - 1][2] + 0.05,
                            )
                        )
                        found_in_whisper = True
                        seg_matched = True
            if not found_in_whisper:
                unmatched_terms.append(term)

        if unmatched_terms and utts:
            seg_utts = [
                u
                for u in utts
                if (
                    seg.start_utt is not None
                    and seg.end_utt is not None
                    and seg.start_utt <= u.id <= seg.end_utt
                )
                or (u.end_s > seg.start_s - 0.3 and u.start_s < seg.end_s + 0.3)
            ]
            for term in unmatched_terms:
                pat = _compile_en_term_pattern(term)
                if pat is None:
                    continue
                for u in seg_utts:
                    en_txt = u.en or ""
                    if not en_txt:
                        continue
                    L = max(1, len(en_txt))
                    u_dur = max(0.2, u.end_s - u.start_s)
                    for m in pat.finditer(en_txt):
                        ws = u.start_s + (m.start() / L) * u_dur - 0.12
                        we = u.start_s + (m.end() / L) * u_dur + 0.12
                        raw_spans.append((max(win_lo, ws), min(win_hi, we)))
                        seg_matched = True

        if not seg_matched and not (seg.sensitive_terms_en or []):
            raw_spans.append((seg.start_s, seg.end_s))

    clamped: list[tuple[float, float]] = []
    for s, e in raw_spans:
        cs = max(0.0, float(s))
        ce = min(dur, float(e)) if dur > 0 else float(e)
        if ce - cs > 0.02:
            clamped.append((round(cs, 3), round(ce, 3)))
    return merge_spans(clamped, max_gap_s=0.12)


def _ass_ts_to_seconds(ts: str) -> float:
    parts = ts.strip().split(":")
    if len(parts) != 3:
        return 0.0
    try:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return 0.0


def redact_ass_dialogue_text(text: str, terms_zh: list[str], terms_en: list[str]) -> str:
    """Redact sensitive Chinese/English terms inside ASS Dialogue text while preserving override tags."""
    clean_zh = sorted({str(t).strip() for t in (terms_zh or []) if str(t).strip()}, key=len, reverse=True)
    clean_en = sorted({str(t).strip() for t in (terms_en or []) if str(t).strip()}, key=len, reverse=True)

    # Tokenize ASS override tags `{...}` and escapes `\N`, `\n`, `\h`
    n = len(text)
    en_chars: list[str] = []
    en_map: list[int] = []
    zh_chars: list[str] = []
    zh_map: list[int] = []
    en_split_pos: int | None = None

    i = 0
    while i < n:
        ch = text[i]
        if ch == "{":
            close = text.find("}", i + 1)
            if close != -1:
                tag_body = text[i : close + 1]
                if en_split_pos is None and ("\\fs" in tag_body or "\\c&H" in tag_body) and i > 0:
                    en_split_pos = i
                i = close + 1
                continue
        if ch == "\\" and i + 1 < n and text[i + 1] in ("N", "n", "h"):
            en_chars.append(" ")
            en_map.append(-1)
            i += 2
            continue
        en_chars.append(ch)
        en_map.append(i)
        zh_chars.append(ch)
        zh_map.append(i)
        i += 1

    if not clean_zh and not clean_en:
        # Blanket redaction keeping inline ASS styling intact
        head_tags = "".join(re.findall(r"^(\{[^}]*\})+", text))
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", "".join(zh_chars)))
        if en_split_pos is not None:
            m_tag = re.match(r"(\{[^}]+\})+", text[en_split_pos:])
            en_tag = m_tag.group(0) if m_tag else ""
            zh_prefix = f"{head_tags}[已脱敏]" if has_cjk else ""
            return f"{zh_prefix}\\N{en_tag}[REDACTED]" if zh_prefix else f"{en_tag}[REDACTED]"
        return f"{head_tags}[已脱敏]" if has_cjk else f"{head_tags}[REDACTED]"

    repl: dict[int, str] = {}

    en_vis = "".join(en_chars)
    for term in clean_en:
        pat = _compile_en_term_pattern(term)
        if pat is None:
            continue
        for m in pat.finditer(en_vis):
            for k in range(m.start(), m.end()):
                ri = en_map[k]
                if ri >= 0 and not en_vis[k].isspace():
                    repl[ri] = "*"

    zh_vis = "".join(zh_chars)
    for term in clean_zh:
        if re.fullmatch(r"[A-Za-z0-9\s.'’\-]+", term):
            pat_z = _compile_en_term_pattern(term, allow_empty_gap=True)
            if pat_z is None:
                continue
            for m in pat_z.finditer(zh_vis):
                for k in range(m.start(), m.end()):
                    repl[zh_map[k]] = "*"
        else:
            pat_z = re.compile(re.escape(term), re.IGNORECASE)
            for m in pat_z.finditer(zh_vis):
                for k in range(m.start(), m.end()):
                    ri = zh_map[k]
                    orig = text[ri]
                    repl[ri] = "＊" if ord(orig) > 127 else "*"

    if not repl:
        return text
    return "".join(repl.get(idx, c) for idx, c in enumerate(text))


def _resolve_base_ass_path(manifest: ProjectManifest, work_dir: Path) -> Path | None:
    sub_raw = (manifest.source.subtitle_path or "").strip()
    if not sub_raw:
        return None
    sub_p = Path(sub_raw).expanduser()
    if sub_p.suffix.lower() == ".ass" and sub_p.is_file():
        return sub_p
    ass_cand = sub_p.with_suffix(".ass")
    if ass_cand.is_file():
        return ass_cand
    if sub_p.stem.endswith(".bi"):
        ass_cand2 = sub_p.with_name(sub_p.stem + ".ass")
        if ass_cand2.is_file():
            return ass_cand2
    if sub_p.is_file() and sub_p.suffix.lower() == ".srt":
        try:
            from ..subtitle_repair import _generate_ass

            tmp_ass = work_dir / f"base_{manifest.project_id}.ass"
            tmp_ass.parent.mkdir(parents=True, exist_ok=True)
            _generate_ass(sub_p, tmp_ass, manifest.source.width or 1920, manifest.source.height or 1080)
            if tmp_ass.is_file():
                return tmp_ass
        except Exception:
            return None
    return None


def build_masked_ass_file(
    manifest: ProjectManifest,
    mask_segments: list[CutSegment],
    work_dir: Path,
    hash_key: str = "",
) -> Path | None:
    """Create a temporary masked ASS copy in work_dir without modifying source subtitles."""
    if not mask_segments:
        return None
    base_ass = _resolve_base_ass_path(manifest, work_dir)
    if base_ass is None or not base_ass.is_file():
        return None

    work_dir.mkdir(parents=True, exist_ok=True)
    out_ass = work_dir / f"masked_{hash_key or 'active'}.ass"
    lines = base_ass.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    out_lines: list[str] = []

    in_events = False
    start_col = 1
    end_col = 2
    style_col = 3
    num_cols = 10

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            in_events = stripped.lower() == "[events]"
            out_lines.append(line)
            continue
        if in_events and line.startswith("Format:"):
            cols = [c.strip().lower() for c in line[7:].split(",")]
            num_cols = len(cols)
            if "start" in cols:
                start_col = cols.index("start")
            if "end" in cols:
                end_col = cols.index("end")
            if "style" in cols:
                style_col = cols.index("style")
            out_lines.append(line)
            continue

        if in_events and line.startswith("Dialogue:"):
            body = line[9:]
            parts = body.split(",", num_cols - 1)
            if len(parts) >= num_cols:
                style_name = parts[style_col].strip() if style_col < len(parts) else ""
                ev_text = parts[-1]
                if style_name != "BoxBase" and "\\p1" not in ev_text:
                    ev_s = _ass_ts_to_seconds(parts[start_col])
                    ev_e = _ass_ts_to_seconds(parts[end_col])
                    hitting = [
                        seg
                        for seg in mask_segments
                        if min(ev_e, seg.end_s) - max(ev_s, seg.start_s) > 0.05
                    ]
                    if hitting:
                        zh_terms: list[str] = []
                        en_terms: list[str] = []
                        for seg in hitting:
                            zh_terms.extend(seg.sensitive_terms_zh or [])
                            en_terms.extend(seg.sensitive_terms_en or [])
                        if not en_terms and zh_terms:
                            for zt in zh_terms:
                                for zh_k, en_l in _DEFAULT_ZH_TO_EN_TERMS.items():
                                    if zh_k in zt or zt in zh_k:
                                        en_terms.extend(en_l)
                        parts[-1] = redact_ass_dialogue_text(ev_text, zh_terms, en_terms)
                        out_lines.append("Dialogue:" + ",".join(parts))
                        continue
        out_lines.append(line)

    out_ass.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return out_ass


def get_overlapping_mask_payload(
    start_s: float,
    end_s: float,
    manifest: ProjectManifest,
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return stable hashable mask specs overlapping [start_s, end_s]."""
    mask_segs = get_active_mask_segments(manifest, cfg)
    if not mask_segs or end_s <= start_s:
        return []
    hitting = [
        seg
        for seg in mask_segs
        if min(end_s, float(seg.end_s)) - max(start_s, float(seg.start_s)) > 0.05
    ]
    if not hitting:
        return []
    audio_spans = resolve_mask_audio_spans(manifest, hitting)
    return [
        {
            "cut_id": seg.cut_id,
            "start_s": round(seg.start_s, 3),
            "end_s": round(seg.end_s, 3),
            "terms_zh": sorted({str(x).strip() for x in (seg.sensitive_terms_zh or []) if str(x).strip()}),
            "terms_en": sorted({str(x).strip() for x in (seg.sensitive_terms_en or []) if str(x).strip()}),
            "audio_spans": [[round(s, 3), round(e, 3)] for s, e in audio_spans if e > start_s and s < end_s],
        }
        for seg in hitting
    ]


def describe_applied_filter(
    manifest: ProjectManifest,
    spans: list[tuple[float, float]] | None = None,
    cfg: dict[str, Any] | None = None,
) -> str:
    """Summarize the content filter state applied to target intervals (or whole project if spans is None)."""
    cc = manifest.clean_cut
    sens_map = {"strict": "严格", "balanced": "平衡", "loose": "红线"}
    act_map = {"auto": "智能", "delete": "删段", "mask": "遮蔽"}
    sens_zh = sens_map.get(str(cc.sensitivity or "strict"), "严格")

    def _overlaps(seg: CutSegment) -> bool:
        if spans is None:
            return True
        return any(min(float(e), float(seg.end_s)) - max(float(s), float(seg.start_s)) > 0.05 for s, e in spans)

    all_segs = [s for s in manifest.cut_segments if _overlaps(s)]
    active_segs = [s for s in get_active_cut_segments(manifest, cfg) if _overlaps(s)]
    ad_cnt = sum(1 for s in active_segs if s.kind == "ad")
    del_pol_cnt = sum(
        1 for s in active_segs if s.kind == "political_sensitive" and resolve_segment_action(s, manifest) == "delete"
    )
    mask_pol_cnt = sum(
        1 for s in active_segs if s.kind == "political_sensitive" and resolve_segment_action(s, manifest) == "mask"
    )

    parts: list[str] = []
    if ad_cnt > 0:
        parts.append("去广")
    if del_pol_cnt > 0 and mask_pol_cnt > 0:
        parts.append(f"脱敏(智能·{sens_zh})")
    elif mask_pol_cnt > 0:
        parts.append(f"脱敏(遮蔽·{sens_zh})")
    elif del_pol_cnt > 0:
        parts.append(f"脱敏(删段·{sens_zh})")

    if parts:
        return "已" + "+".join(parts)

    # No active cut applied in these spans
    has_pol_in_span = any(s.kind == "political_sensitive" and s.enabled for s in all_segs)
    if has_pol_in_span:
        if not cc.cut_political_sensitive:
            return "原片直出(未脱敏)"
        return f"原片直出({sens_zh}已放行)"
    if any(s.kind == "ad" and s.enabled for s in all_segs):
        return "原片直出(未去广)"
    return "原片直出(无敏感)" if cc.scanned_at else ""


def clip_overlapping_cut_spans(
    start_s: float,
    end_s: float,
    cut_spans: list[tuple[float, float]] | None,
) -> list[tuple[float, float]]:
    """Return merged sub-spans of cut_spans that intersect [start_s, end_s]."""
    if not cut_spans or end_s <= start_s:
        return []
    hits: list[tuple[float, float]] = []
    for cs, ce in cut_spans:
        ov_s = max(start_s, float(cs))
        ov_e = min(end_s, float(ce))
        if ov_e - ov_s > 0.05:
            hits.append((round(ov_s, 3), round(ov_e, 3)))
    return merge_spans(hits, max_gap_s=0.05)


def subtract_cut_intervals(
    start_s: float,
    end_s: float,
    cut_spans: list[tuple[float, float]] | None,
    min_keep_s: float = 0.4,
) -> list[tuple[float, float]]:
    """Subtract cut_spans from [start_s, end_s] and return kept sub-intervals."""
    if end_s <= start_s:
        return []
    if not cut_spans:
        return [(round(start_s, 3), round(end_s, 3))]

    merged_cuts = merge_spans(cut_spans, max_gap_s=0.05)
    keep: list[tuple[float, float]] = []
    cursor = float(start_s)
    for cs, ce in merged_cuts:
        if ce <= cursor:
            continue
        if cs >= end_s:
            break
        if cs - cursor >= min_keep_s:
            keep.append((round(cursor, 3), round(cs, 3)))
        cursor = max(cursor, ce)
    if end_s - cursor >= min_keep_s:
        keep.append((round(cursor, 3), round(end_s, 3)))
    return keep


def should_use_precise_copy(
    manifest: ProjectManifest,
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Use fast stream-copy only when explicitly requested or auto without active political cuts/masks."""
    mode = (manifest.clean_cut.cut_mode or "auto").lower()
    if mode == "precise":
        return True
    if mode == "reencode":
        return False
    return not any(seg.kind == "political_sensitive" for seg in get_active_cut_segments(manifest, cfg))


def compute_clean_cut_render_hash(
    manifest: ProjectManifest,
    cfg: dict[str, Any] | None = None,
) -> str:
    cc = manifest.clean_cut
    dur = float(manifest.source.duration_s or (manifest.utterances[-1].end_s if manifest.utterances else 0.0))
    payload = {
        "video": manifest.source.render_video,
        "title": cc.title,
        "cover_title": cc.cover_title,
        "sensitivity": cc.sensitivity,
        "political_action": cc.political_action,
        "cut_mode": cc.cut_mode,
        "precise_copy": should_use_precise_copy(manifest, cfg),
        "cut_spans": [[round(s, 3), round(e, 3)] for s, e in get_active_delete_spans(manifest, cfg)],
        "masks": get_overlapping_mask_payload(0.0, max(1.0, dur), manifest, cfg),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def sync_all_render_hashes(
    manifest: ProjectManifest,
    cfg: dict[str, Any] | None = None,
) -> None:
    """Recompute Clean Cut and linked Clip/Remix/Compilation render hashes."""
    from .verify_and_plan import (
        compute_clip_render_hash,
        compute_compilation_render_hash,
        compute_remix_render_hash,
    )

    manifest.clean_cut.render_hash = compute_clean_cut_render_hash(manifest, cfg)
    apply_links = bool(manifest.clean_cut.apply_to_clips)
    delete_spans = get_active_delete_spans(manifest, cfg) if apply_links else []
    for clip in manifest.clips:
        clip_masks = get_overlapping_mask_payload(clip.start_s, clip.end_s, manifest, cfg) if apply_links else []
        clip.render_hash = compute_clip_render_hash(
            clip,
            manifest.source.render_video,
            delete_spans,
            manifest.source.subtitle_sha256,
            mask_payload=clip_masks,
        )
    manifest.compilation.render_hash = compute_compilation_render_hash(manifest)
    manifest.remix.render_hash = compute_remix_render_hash(manifest)


def detect_visual_ad_spans(
    video_path: Path,
    min_duration: float = 3.0,
    max_duration: float = 300.0,
) -> list[tuple[float, float]]:
    """Detect resolution-anomaly ad inserts from video keyframes via ffprobe."""
    if not video_path.is_file():
        return []
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-skip_frame",
        "nokey",
        "-show_entries",
        "frame=pts_time,width,height",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    try:
        res = run_cancellable_subprocess(cmd, capture_output=True, text=True, timeout=45)
        if res.returncode != 0 or not res.stdout.strip():
            return []
    except JobCancelledError:
        raise
    except Exception:
        return []

    frames: list[tuple[float, int, int]] = []
    for line in res.stdout.splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            frames.append((float(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    if len(frames) < 3:
        return []

    main_res = Counter((w, h) for _, w, h in frames).most_common(1)[0][0]
    last_t = frames[-1][0]
    candidates: list[tuple[float, float, int]] = []
    start: float | None = None
    count = 0
    for t, w, h in frames:
        if (w, h) == main_res:
            if start is not None:
                candidates.append((start, t, count))
                start = None
                count = 0
            continue
        if start is None:
            start = t
        count += 1
    if start is not None:
        candidates.append((start, last_t, count))

    out: list[tuple[float, float]] = []
    for s, e, cnt in candidates:
        dur = e - s
        if (cnt >= 2 or dur >= min_duration) and min_duration <= dur <= max_duration:
            out.append((round(s, 3), round(e, 3)))
    return out


def _utterance_excerpt(utterances: list[Utterance], sp: int, ep: int, max_chars: int = 160) -> str:
    texts = [((utterances[i].zh or utterances[i].en).strip()) for i in range(max(0, sp), min(len(utterances), ep + 1))]
    joined = " ".join(t for t in texts if t)
    if len(joined) <= max_chars:
        return joined
    half = max(20, max_chars // 2 - 3)
    return f"{joined[:half].rstrip()} ... {joined[-half:].lstrip()}"


def _find_chapter_for_time(manifest: ProjectManifest, t_s: float) -> str:
    for ch in manifest.metadata.chapters:
        if ch.start_s <= t_s <= ch.end_s:
            return ch.title
    return ""


def _collect_rules_by_severity(profiles: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {"high": [], "medium": [], "low": []}
    seen: dict[str, set[str]] = {"high": set(), "medium": set(), "low": set()}
    for pkey in ("strict", "balanced", "loose"):
        p = profiles.get(pkey, {})
        for r in p.get("rules") or []:
            sev = str(r.get("severity") or "medium").lower()
            if sev not in grouped:
                continue
            rid = str(r.get("id") or "rule").strip()
            name = str(r.get("name") or rid).strip()
            desc = str(r.get("description") or "").strip()
            sig = f"{rid}:{name}:{desc}"
            if sig in seen[sev]:
                continue
            seen[sev].add(sig)
            grouped[sev].append(f"  * [{rid}] {name}：{desc}")
    return grouped


def _build_filter_prompt(
    manifest: ProjectManifest,
    chunk: list[Utterance],
    *,
    scan_ads: bool,
    scan_political: bool,
    sensitivity: str,
    cfg: dict[str, Any] | None = None,
) -> str:
    tasks: list[str] = []
    if scan_ads:
        tasks.append(
            """1. 广告与推广片段（kind = "ad"）：
   - "sponsor": 有**明确商业品牌名称或具体赞助产品**的赞助商口播、品牌植入、推广折扣码/链接引导；
   - "midroll": 节目中段插播推广、明确指名的播客/课程/商业工具广告；
   - "outro": 片尾频道订阅催促、周边带货、Patreon/会员引导；
   - "intro": 开场片头赞助口播（仅明确带具体品牌名的商业推广）。
   - **明确品牌要求**：广告必须有**具体可识别的品牌名、赞助商名称或明确带货/付费导流诉求**。若说话人只是泛称/调侃“商业赞助”“commercial software / brought to you by commercial”（无任何具体第三方品牌名），**不算广告植入，请勿标注**。
   - **严格仅依据下方【字幕全文】中的实际口播内容判断广告**，绝对不要因为视频简介里包含订阅链接、App 推广或赞助 URL 就把正常欢迎语、开场白或访谈致谢误判为广告。"""
        )

    if scan_political:
        lvl = str(sensitivity or "strict").strip().lower()
        if lvl not in ("strict", "balanced", "loose"):
            lvl = "strict"
        profiles = get_political_profiles(cfg)
        active_profile = profiles[lvl]
        active_rules = active_profile.get("rules") or []
        active_rule_lines = "\n".join(
            f"   - [{r['id']}]（severity={r['severity']}）{r['name']}：{r['description']}"
            for r in active_rules
        ) or "   - （当前档位未单独配置附加规则）"
        trig_text = "/".join(active_profile.get("trigger_severities") or [])
        grouped = _collect_rules_by_severity(profiles)
        high_txt = "\n".join(grouped["high"]) or "  * 仅直接涉及中国领导人/执政党/政治体制与主权红线"
        med_txt = "\n".join(grouped["medium"]) or "  * 无中危规则（不适用中危拦截）"
        low_txt = "\n".join(grouped["low"]) or "  * 未配置低危规则，仅作边界说明时不要滥标"

        pol_rules = f"""2. 政治敏感片段（kind = "political_sensitive"，当前预设标准：【{active_profile['label']}】/{lvl}，触发风险级：{trig_text}）：
   - 请**始终输出全片所有确实命中的 high / medium / low 片段**并准确标注 `severity`（不要为了当前预设而刻意漏报真实存在的低/中危段，前端依赖三档实时切换过滤）。
   【当前档位生效规则】
{active_rule_lines}

   【全系统风险分级与 `severity` 标注铁律（必须严格对齐，前端支持在三档间实时切换过滤）】：
   - `severity = "high"`（高危；「仅硬红线 / 平衡 / 严格从严」三档均拦截）：
{high_txt}
   - `severity = "medium"`（中危；「平衡 / 严格从严」拦截，切到「仅硬红线」时放行）：
{med_txt}
   - `severity = "low"`（低危；仅「严格从严」拦截，切到「平衡」或「仅硬红线」时放行）：
{low_txt}

   【严禁关键词误判与豁免铁律（极重要，不符合以下上下文条件的一律跳过，连 low 都不要标）】：
   - **目标国别与明确指责双重要件**：定为 `medium` 或 `high` **绝不能仅因出现“政府/government”“中国/China”“美国/US”“监控”“核”“垄断”等孤立词汇就触发**。上下文必须明确指向**中国，或俄罗斯、朝鲜、伊朗/伊斯兰世界等中国友好国家**，且内容构成对上述国家的**明确指责、政策/体制批评、主权挑战或军事/制裁对抗**。
   - **严禁滑坡/联想式审查**：禁止以“可能引发对中国政府/平台监管的联想”“可能引发群体对立或社会争议”为由，将未实际讨论目标国家的片段标为涉政；若 `target_country` 只能填 `"非目标"` 或 `is_explicit_criticism` 为 `false`，**直接忽略不输出**。
   - **以下内容严禁误标（直接放行，连 low 都不要输出）**：
     1. 仅泛泛讨论“政府监管 / 政府接管 AI”、或明显批评美国/西方政府、NSA、90 年代美国密码战、欧美反垄断（如 Google/Meta 反乌托邦/垄断）、美国政党、Trump、Elon Musk/Twitter、“觉醒派（woke）/蓝发”等欧美内部政经与文化争议；
     2. 仅仅提及“美国 / US”或称赞美国、硅谷前沿实验室技术领先（只要未涉及打压中国或对华/友好国家体制军事威胁，就绝对不要标为中危或高危，亦勿滥标低危）；
     3. 中性提及中国或中国科技公司（如参访宇树科技 Unitree、客观比较中美开发者对 AGI 时间表的不同判断，不含政治贬损或制度攻击）；
     4. 纯商业平台与产品批评（如批评 Apple 30% 抽成、macOS 弹窗、围墙花园、Google/Meta 双头垄断、AI 安全研究员观点、普通口语粗口如 fucking）；
     5. 非指向中国或友好国家现实政治批评的历史常识/类比（如奥本海默、回溯过去 50 年冷战未核毁灭）。
   - 而直接评论或影射**中国最高领导人（如 Xi / 习近平）或台海军事打击/中国政权更迭**的段落必须标为 `high`。
   - 若整块字幕无实质涉政指责或红线内容，**直接返回空数组，严禁凑数输出中性提及中国/企业的 low 片段**；每条命中必须写出具体指责对象与违规原因（`reason` 不可为空）。

   【脱敏策略推荐（`recommended_action`）与精确词表要求】：
   - `"recommended_action": "mask"`（遮蔽敏感词字幕+哔音）：适用于正常科技/商业/工程讨论中**仅顺带提及敏感地名/国别/机构/人名**（删掉或遮住 1~2 个词，整句技术/商业含义依然成立），例如 "Chinese destroyer"、"China's chip policy"、"Taiwan" 作为例子顺带提到；务必在 `sensitive_terms_zh` / `sensitive_terms_en` 列出该段字幕中实际出现的所有需打码词。
   - `"recommended_action": "delete"`（整段删除）：适用于整段或连续数句**围绕政权倒台/领导人评价/台海军事威慑/体制政治攻击展开论证**，仅打码无法消除敏感语境的片段（仍建议列出核心关键词）。广告段固定填 `"delete"` 且词表填 `[]`。"""
        tasks.append(pol_rules)

    ch_lines = [
        f"- Ch{c.index:02d} ({_fmt_ts(c.start_s)}–{_fmt_ts(c.end_s)}): {c.title}"
        for c in manifest.metadata.chapters
    ]
    utt_lines = [
        f"[u{u.id:04d} {_fmt_ts(u.start_s)}–{_fmt_ts(u.end_s)}] {u.zh or ''} | {u.en or ''}".strip()
        for u in chunk
    ]

    return f"""请审查以下视频字幕片段，找出所有需要剪除或脱敏的目标区间。

【视频信息】
- 标题: {manifest.metadata.original_title}
- 中文主题: {manifest.metadata.original_theme_zh or '无'}
- 简介节选: {(manifest.metadata.description or '')[:1200] or '无'}
- 官方章节:
{chr(10).join(ch_lines) or '（无官方章节）'}

【审查任务】
{chr(10).join(tasks)}

【边界精确定位要求（非常关键）】
- 必填 `"quote"`：从命中的字幕原文（`[uXXXX ...]` 行中 `|` 前后）逐字摘抄触发句（单句/短段摘 15~40 字/词原文；多句连续段落可用 `"首句原文...尾句原文"` 格式），严禁改写或杜撰。
- `start_utt` 填触发段的第一句整数编号（必须与 `[uXXXX]` 完全一致），`end_utt` 填结束前最后一句整数编号。
- 对于政治敏感内容（`political_sensitive`），可能紧挨正常技术/商业对话出现，**务必紧贴敏感论证起止句，不要把前后安全对话一起标进来**；如果单句就能遮蔽处理，就只标这 1~2 句。
- 连续的同类整段敏感论战或广告片段应合并为单一区间，不要碎片化拆分。

【输出 JSON Schema】
{{
  "segments": [
    {{
      "kind": "ad | political_sensitive",
      "subtype": "sponsor | midroll | outro | intro | hard_redline | geopolitics_confrontation | policy_sensitive",
      "brand": "仅 ad 填写具体赞助品牌/产品/平台名（无具体品牌严禁标为广告）；political_sensitive 填 \"\"",
      "target_country": "仅 political_sensitive 填字幕上下文实际谈及的目标国家/地区（如 中国 / 台海 / 俄罗斯 / 朝鲜 / 伊朗 / 伊斯兰世界 等）；若实为泛指政府、美国/欧美或商业公司，填 \"非目标\"",
      "is_explicit_criticism": true,
      "quote": "从对应 [uXXXX] 字幕逐字摘录的触发原句（多句可用 首句...尾句）",
      "start_utt": 12,
      "end_utt": 19,
      "severity": "high | medium | low",
      "recommended_action": "delete | mask",
      "sensitive_terms_zh": ["需打码的中文原文词（从上述中文字幕原样摘录）"],
      "sensitive_terms_en": ["term to mute/redact verbatim from English subtitle"],
      "reason": "简短中文原因说明"
    }}
  ]
}}
如果没有任何命中，请返回 {{"segments": []}}。

【字幕全文】
{chr(10).join(utt_lines)}"""


def _parse_utt_int(val: Any, fallback: int = -1) -> int:
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        s = val.strip().lower().lstrip("u")
        if s.isdigit():
            return int(s)
    return fallback


def _term_in_utterance(term: str, utt: Utterance) -> bool:
    t = str(term or "").strip()
    if not t:
        return False
    zh = utt.zh or ""
    en = utt.en or ""
    if re.fullmatch(r"[A-Za-z0-9\s\-']+", t):
        t_low = t.lower()
        return bool(re.search(r"(?<![a-z0-9])" + re.escape(t_low) + r"(?![a-z0-9])", en.lower()))
    return (t in zh) or (t.lower() in en.lower())


def _single_quote_score(q_clean: str, utts_slice: list[Utterance]) -> float:
    from .verify_and_plan import _clean_match_text

    if not q_clean or not utts_slice:
        return 0.0
    cand_clean = _clean_match_text("".join((u.zh or "") + (u.en or "") for u in utts_slice))
    if not cand_clean:
        return 0.0
    if len(q_clean) >= 4 and (q_clean in cand_clean or cand_clean in q_clean):
        return 1.0
    q_grams = {q_clean[i : i + 2] for i in range(max(1, len(q_clean) - 1))}
    if not q_grams:
        return 0.0

    best = 0.0
    texts_to_check = [cand_clean] + [
        _clean_match_text((u.zh or "") + (u.en or "")) for u in utts_slice if len(utts_slice) > 1
    ]
    for tc in texts_to_check:
        if len(tc) < 2:
            continue
        if len(q_clean) >= 4 and (q_clean in tc or tc in q_clean):
            return 1.0
        c_grams = {tc[i : i + 2] for i in range(max(1, len(tc) - 1))}
        if not c_grams:
            continue
        inter = len(q_grams & c_grams)
        dice = (2.0 * inter) / (len(q_grams) + len(c_grams))
        recall = inter / len(q_grams) if len(q_grams) >= 6 else 0.0
        best = max(best, dice, recall)
    return best


def _quote_similarity(quote: str, utts_slice: list[Utterance]) -> float:
    from .verify_and_plan import _clean_match_text

    if not quote or not utts_slice:
        return 0.0
    parts = [_clean_match_text(p) for p in re.split(r"\.{3,}|…+", quote) if _clean_match_text(p)]
    if not parts:
        return 0.0
    if len(parts) >= 2:
        return min(
            _single_quote_score(parts[0], utts_slice),
            _single_quote_score(parts[-1], utts_slice),
        )
    return _single_quote_score(parts[0], utts_slice)


def _reanchor_hit_utterances(
    utts: list[Utterance],
    su: int,
    eu: int,
    chunk_lo: int,
    chunk_hi: int,
    quote: str,
    terms_zh: list[str],
    terms_en: list[str],
) -> tuple[int, int] | None:
    """Verify [su, eu] matches quote/terms; re-anchor within chunk or drop hallucinated hits."""
    from .verify_and_plan import _match_quote_to_utterances

    q_str = str(quote or "").strip()
    valid_terms = [str(t).strip() for t in [*terms_zh, *terms_en] if str(t).strip()]
    if not q_str and not valid_terms:
        return su, eu
    if not utts:
        return None

    id_to_idx = {u.id: idx for idx, u in enumerate(utts)}
    lo_idx = id_to_idx.get(chunk_lo, 0)
    hi_idx = id_to_idx.get(chunk_hi, len(utts) - 1)
    if lo_idx > hi_idx:
        lo_idx, hi_idx = hi_idx, lo_idx

    s_idx = id_to_idx.get(su, lo_idx)
    e_idx = id_to_idx.get(eu, s_idx)
    s_idx = max(lo_idx, min(s_idx, hi_idx))
    e_idx = max(lo_idx, min(e_idx, hi_idx))
    if s_idx > e_idx:
        s_idx, e_idx = e_idx, s_idx
    orig_slice = utts[s_idx : e_idx + 1]

    if q_str:
        if _quote_similarity(q_str, orig_slice) >= 0.35:
            return utts[s_idx].id, utts[e_idx].id
        m_i, m_j = _match_quote_to_utterances(q_str, utts, lo_idx, hi_idx, max_span_s=90.0)
        matched_slice = utts[m_i : m_j + 1]
        if _quote_similarity(q_str, matched_slice) >= 0.35:
            orig_span = max(0, e_idx - s_idx)
            if valid_terms and orig_span > 0:
                ext_j = m_j
                cap_j = min(hi_idx, m_i + orig_span)
                while ext_j < cap_j and any(_term_in_utterance(t, utts[ext_j + 1]) for t in valid_terms):
                    ext_j += 1
                m_j = ext_j
            return utts[m_i].id, utts[m_j].id

    if valid_terms:
        if any(_term_in_utterance(t, u) for u in orig_slice for t in valid_terms):
            return utts[s_idx].id, utts[e_idx].id
        term_hits = [
            idx
            for idx in range(lo_idx, hi_idx + 1)
            if any(_term_in_utterance(t, utts[idx]) for t in valid_terms)
        ]
        if term_hits:
            best_idx = max(
                term_hits,
                key=lambda idx: (
                    _quote_similarity(q_str, [utts[idx]]) if q_str else 0.0,
                    sum(1 for t in valid_terms if _term_in_utterance(t, utts[idx])),
                    -abs(idx - s_idx),
                ),
            )
            return utts[best_idx].id, utts[best_idx].id

    return None


_GENERIC_NON_BRAND_TERMS = {
    "",
    "无",
    "none",
    "n/a",
    "null",
    "unknown",
    "商业",
    "商业赞助",
    "商业软件",
    "广告",
    "赞助",
    "赞助商",
    "推广",
    "commercial",
    "commercial software",
    "sponsor",
    "sponsorship",
    "ad",
    "ads",
    "business",
    "capitalism",
}

_OUTRO_CTA_KEYWORDS = (
    "patreon",
    "subscribe",
    "订阅",
    "会员",
    "折扣码",
    "promo",
    "coupon",
    "merch",
    "周边",
    "http",
    ".com",
    ".io",
    ".ai",
    "link",
)


def _is_valid_ai_ad_hit(seg: dict[str, Any], quote_txt: str, reason: str) -> bool:
    """Require an identifiable sponsor/brand (or explicit monetization CTA for outro) on AI ad hits."""
    brand_present = "brand" in seg
    brand_norm = str(seg.get("brand") or "").strip().lower()
    subtype = str(seg.get("subtype") or "").strip().lower()
    combined = f"{quote_txt} {reason}".lower()
    has_cta = any(kw in combined for kw in _OUTRO_CTA_KEYWORDS)

    if brand_present:
        if brand_norm in _GENERIC_NON_BRAND_TERMS:
            if subtype == "outro" and has_cta:
                return True
            return False
        return True

    # Legacy fallback when model omits "brand": reject generic commercial-sponsorship jokes lacking a CTA
    if (quote_txt or "").strip() and subtype != "outro":
        if any(g in combined for g in ("商业赞助", "商业软件", "commercial software")) and not has_cta:
            return False
    return True


_NON_TARGET_POLITICAL_TOKENS = (
    "非目标",
    "无",
    "none",
    "n/a",
    "null",
    "unknown",
    "泛指",
    "美国",
    "欧美",
    "西方",
    "全球",
    "商业",
    "公司",
    "群体",
    "文化",
    "通用",
    "general",
    "global",
    "us",
    "usa",
    "united states",
    "america",
    "western",
    "west",
    "apple",
    "google",
    "meta",
    "elon",
    "musk",
    "woke",
)

_POLITICAL_STOP_TOKENS = {"国家", "政府", "世界", "地区", "问题", "政权", "相关", "以及", "等", "and", "or", "the", "of"}

_POLITICAL_TARGET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"中国|中共|共产党|北京|中南海|习近平|习主席|台湾|台海|香港|新疆|西藏|南海|解放军|天安门|文革|法轮功|达赖|维吾尔"
    ),
    re.compile(r"俄罗斯|俄国|普京|莫斯科|苏联|朝鲜|平壤|金正恩|伊朗|德黑兰|哈梅内伊|伊斯兰|穆斯林|巴勒斯坦|哈马斯|叙利亚|阿富汗|塔利班|巴基斯坦|古巴|委内瑞拉|白俄罗斯|塞尔维亚"),
    re.compile(
        r"(?<![a-z0-9])(china|chinese|ccp|prc|beijing|taiwan|taiwanese|hong\s+kong|xinjiang|tibet|tibetan|south\s+china\s+sea|pla|xi\s+jinping|xi|uyghur|tiananmen|dalai|russia|russian|putin|moscow|north\s+korea|pyongyang|kim\s+jong|iran|iranian|tehran|khamenei|islam|islamic|muslim|palestine|palestinian|hamas|syria|afghanistan|taliban|pakistan|cuba|venezuela|belarus|serbia)(?![a-z0-9])",
        re.IGNORECASE,
    ),
)


def _is_valid_political_hit(
    seg: dict[str, Any],
    utts: list[Utterance],
    su: int,
    eu: int,
    quote_txt: str,
    reason: str,
) -> bool:
    """Verify AI political hit targets China/friendly countries (not speculative generic 'government' or US/tech)."""
    if "target_country" not in seg and "is_explicit_criticism" not in seg:
        return True

    crit_val = seg.get("is_explicit_criticism")
    if crit_val is False or (isinstance(crit_val, str) and crit_val.strip().lower() in ("false", "0", "no", "否")):
        return False

    tc_raw = str(seg.get("target_country") or "").strip()
    tc = tc_raw.lower()
    if not tc:
        return False
    for nt in _NON_TARGET_POLITICAL_TOKENS:
        if tc == nt or re.match(r"^" + re.escape(nt) + r"(?:[\s/·,，(（]|$)", tc):
            return False

    id_to_idx = {u.id: idx for idx, u in enumerate(utts)}
    s_idx = id_to_idx.get(su, 0)
    e_idx = id_to_idx.get(eu, s_idx)
    if s_idx > e_idx:
        s_idx, e_idx = e_idx, s_idx
    win_lo = max(0, s_idx - 3)
    win_hi = min(len(utts), e_idx + 4)
    ctx_parts = [f"{u.zh or ''} {u.en or ''}" for u in utts[win_lo:win_hi]]
    if quote_txt:
        ctx_parts.append(quote_txt)
    ctx_text = " ".join(ctx_parts)
    ctx_lower = ctx_text.lower()

    if any(pat.search(ctx_text) for pat in _POLITICAL_TARGET_PATTERNS):
        return True

    for tok in re.split(r"[\s/,，、;；|()（）]+", tc_raw):
        tok_s = tok.strip()
        if not tok_s or tok_s.lower() in _POLITICAL_STOP_TOKENS:
            continue
        if re.fullmatch(r"[A-Za-z0-9\-']+", tok_s):
            if len(tok_s) >= 2 and re.search(
                r"(?<![a-z0-9])" + re.escape(tok_s.lower()) + r"(?![a-z0-9])",
                ctx_lower,
            ):
                return True
        elif len(tok_s) >= 2 and tok_s in ctx_text:
            return True

    return False


def scan_cut_segments(
    manifest: ProjectManifest,
    store: ProjectStore,
    cfg: dict[str, Any],
    run_id: str,
    *,
    scan_ads: bool | None = None,
    scan_political: bool | None = None,
    sensitivity: str | None = None,
) -> ProjectManifest:
    """Scan full transcript (+ video keyframes for ads) to detect ad and political-sensitive cut segments."""
    from .verify_and_plan import _load_audio_envelope, snap_bounds_with_acoustic_guard

    t0 = time.monotonic()
    cf_cfg = cfg.get("content_filter", {})
    eff_ads = bool(cf_cfg.get("cut_ads", True)) if scan_ads is None else bool(scan_ads)
    eff_pol = bool(cf_cfg.get("cut_political_sensitive", True)) if scan_political is None else bool(scan_political)
    eff_sens = str(sensitivity or manifest.clean_cut.sensitivity or cf_cfg.get("sensitivity", "strict"))
    if eff_sens not in ("strict", "balanced", "loose"):
        eff_sens = "strict"

    manifest.clean_cut.sensitivity = eff_sens  # type: ignore[assignment]
    cfg_action = str(cf_cfg.get("political_action") or "").strip().lower()
    if manifest.clean_cut.scanned_at is None and cfg_action in VALID_POLITICAL_ACTIONS:
        manifest.clean_cut.political_action = cfg_action  # type: ignore[assignment]

    utts = manifest.utterances
    if not utts:
        manifest.clean_cut.cut_ads = any(x.kind == "ad" for x in manifest.cut_segments)
        manifest.clean_cut.cut_political_sensitive = any(
            x.kind == "political_sensitive" and segment_matches_sensitivity(x, eff_sens, cfg)
            for x in manifest.cut_segments
        )
        if not get_active_cut_segments(manifest, cfg):
            manifest.clean_cut.enabled = False
        manifest.clean_cut.scanned_at = utc_now_iso()
        sync_all_render_hashes(manifest, cfg)
        store.save(manifest)
        return manifest

    manual_segments = [seg.model_copy(deep=True) for seg in manifest.cut_segments if seg.detector == "manual"]

    models_cfg = cfg.get("models", {})
    text_provider = models_cfg.get("text_provider", "dashscope")
    text_model = models_cfg.get("text_model", "qwen3.8-max")
    text_fallbacks = list(models_cfg.get("text_fallback_models") or DEFAULT_TEXT_FALLBACK_MODELS)
    llm = ClipCutLLM(models_cfg)
    scan_provider = text_provider
    scan_model = text_model
    scan_fallbacks = list(text_fallbacks)
    if eff_pol and getattr(llm, "gemini_key", None):
        scan_provider = "gemini"
        scan_model = "gemini-2.5-flash"
        scan_fallbacks = ["vertex:gemini-2.5-pro", f"{text_provider}:{text_model}"] + [
            fb
            for fb in text_fallbacks
            if fb not in ("gemini:gemini-2.5-flash", "vertex:gemini-2.5-pro")
        ]

    used_model_label = f"{scan_provider}:{scan_model}"
    raw_ai_hits: list[dict[str, Any]] = []
    if eff_ads or eff_pol:
        chunk_size = 380
        overlap = 20
        chunks: list[list[Utterance]] = []
        if len(utts) <= 420:
            chunks.append(utts)
        else:
            step = max(1, chunk_size - overlap)
            for idx in range(0, len(utts), step):
                part = utts[idx : idx + chunk_size]
                if part:
                    chunks.append(part)
                if idx + chunk_size >= len(utts):
                    break

        for ch_idx, chunk in enumerate(chunks):
            check_cancelled()
            chunk_lo = chunk[0].id
            chunk_hi = chunk[-1].id
            prompt = _build_filter_prompt(
                manifest,
                chunk,
                scan_ads=eff_ads,
                scan_political=eff_pol,
                sensitivity=eff_sens,
                cfg=cfg,
            )
            res = llm.generate_json(
                prompt=prompt,
                model=scan_model,
                provider=scan_provider,
                fallback_models=scan_fallbacks,
                system_instruction=CONTENT_FILTER_SYSTEM,
                temperature=0.1,
                max_output_tokens=16384,
            )
            used_model_label = f"{getattr(res, 'provider', scan_provider)}:{getattr(res, 'model', scan_model)}"
            data = res.parsed if isinstance(res.parsed, dict) else {}
            segs = data.get("segments") if isinstance(data, dict) else None
            if not isinstance(segs, list):
                continue
            for seg in segs:
                if not isinstance(seg, dict):
                    continue
                kind = str(seg.get("kind") or "").strip().lower()
                if kind not in ("ad", "political_sensitive"):
                    continue
                if kind == "ad" and not eff_ads:
                    continue
                if kind == "political_sensitive" and not eff_pol:
                    continue
                su = _parse_utt_int(seg.get("start_utt"), -1)
                eu = _parse_utt_int(seg.get("end_utt"), su)
                if su < 0:
                    continue
                if eu < su:
                    su, eu = eu, su
                su = max(chunk_lo, min(su, chunk_hi))
                eu = max(su, min(eu, chunk_hi))
                severity = str(seg.get("severity") or "high").strip().lower()
                if severity not in ("high", "medium", "low"):
                    severity = "high"
                rec_act = str(seg.get("recommended_action") or "delete").strip().lower()
                if kind == "ad" or rec_act not in ("delete", "mask"):
                    rec_act = "delete"
                terms_zh = [
                    str(x).strip()
                    for x in (seg.get("sensitive_terms_zh") or [])
                    if isinstance(x, (str, int, float)) and str(x).strip()
                ]
                terms_en = [
                    str(x).strip()
                    for x in (seg.get("sensitive_terms_en") or [])
                    if isinstance(x, (str, int, float)) and str(x).strip()
                ]
                quote_txt = str(seg.get("quote") or "").strip()
                raw_reason = str(seg.get("reason") or "").strip()
                if kind == "ad" and not _is_valid_ai_ad_hit(seg, quote_txt, raw_reason):
                    store.log(
                        f"[clean_cut] 丢弃无明确品牌广告误报 u{su:04d}-u{eu:04d} (brand={seg.get('brand')!r}, quote={quote_txt[:40]!r})"
                    )
                    continue
                if kind == "political_sensitive" and severity == "low" and not raw_reason:
                    store.log(
                        f"[clean_cut] 丢弃无理由低危凑数命中 u{su:04d}-u{eu:04d} (quote={quote_txt[:40]!r})"
                    )
                    continue
                anchored = _reanchor_hit_utterances(
                    utts,
                    su,
                    eu,
                    chunk_lo,
                    chunk_hi,
                    quote_txt,
                    terms_zh if kind == "political_sensitive" else [],
                    terms_en if kind == "political_sensitive" else [],
                )
                if anchored is None:
                    store.log(
                        f"[clean_cut] 丢弃未锚定命中 u{su:04d}-u{eu:04d} (kind={kind}, quote={quote_txt[:40]!r})"
                    )
                    continue
                su, eu = anchored
                if kind == "political_sensitive" and not _is_valid_political_hit(
                    seg, utts, su, eu, quote_txt, raw_reason
                ):
                    store.log(
                        f"[clean_cut] 丢弃非目标国/无明确指责涉政误报 u{su:04d}-u{eu:04d} "
                        f"(target={seg.get('target_country')!r}, crit={seg.get('is_explicit_criticism')!r}, quote={quote_txt[:40]!r})"
                    )
                    continue
                raw_ai_hits.append(
                    {
                        "kind": kind,
                        "subtype": str(seg.get("subtype") or "").strip(),
                        "su": su,
                        "eu": eu,
                        "severity": severity,
                        "recommended_action": rec_act,
                        "terms_zh": terms_zh if kind == "political_sensitive" else [],
                        "terms_en": terms_en if kind == "political_sensitive" else [],
                        "reason": raw_reason or ("广告内容" if kind == "ad" else "涉政/合规风险"),
                    }
                )
            store.log(f"[clean_cut] AI 扫描分块 {ch_idx + 1}/{len(chunks)} 完成（本块命中 {len(segs)} 处）")

    # Merge overlapping same-kind hits across chunks
    deduped_ai: list[dict[str, Any]] = []
    for kind in ("ad", "political_sensitive"):
        khits = sorted((h for h in raw_ai_hits if h["kind"] == kind), key=lambda x: (x["su"], x["eu"]))
        for h in khits:
            if deduped_ai and deduped_ai[-1]["kind"] == kind and h["su"] <= deduped_ai[-1]["eu"] + 1:
                prev = deduped_ai[-1]
                prev["eu"] = max(prev["eu"], h["eu"])
                if h["severity"] == "high":
                    prev["severity"] = "high"
                elif h["severity"] == "medium" and prev["severity"] == "low":
                    prev["severity"] = "medium"
                if h.get("recommended_action") == "delete":
                    prev["recommended_action"] = "delete"
                for tz in h.get("terms_zh") or []:
                    if tz not in prev["terms_zh"]:
                        prev["terms_zh"].append(tz)
                for te in h.get("terms_en") or []:
                    if te not in prev["terms_en"]:
                        prev["terms_en"].append(te)
                if h["reason"] and h["reason"] not in prev["reason"]:
                    prev["reason"] = f"{prev['reason']}；{h['reason']}"[:160]
            else:
                deduped_ai.append(
                    {
                        **h,
                        "terms_zh": list(h.get("terms_zh") or []),
                        "terms_en": list(h.get("terms_en") or []),
                    }
                )

    rms, hop_s = _load_audio_envelope(store)
    dur = float(manifest.source.duration_s or (utts[-1].end_s if utts else 0.0))
    n_utts = len(utts)

    scanned_segments: list[CutSegment] = []
    for item in deduped_ai:
        su = max(0, min(item["su"], n_utts - 1))
        eu = max(su, min(item["eu"], n_utts - 1))
        s_t, e_t, _, _, _ = snap_bounds_with_acoustic_guard(
            start_utt=su,
            end_utt=eu,
            utterances=utts,
            rms=rms if isinstance(rms, np.ndarray) else np.zeros(0, dtype=np.float32),
            hop_s=hop_s,
            video_duration=dur,
            cfg=cfg,
        )
        # For ads only: absorb silent transition B-roll gaps >2.5s immediately before/after the ad read
        if item["kind"] == "ad":
            if su > 0:
                prev_end = utts[su - 1].end_s
                if s_t - prev_end > 2.5:
                    s_t = round(prev_end + 0.15, 3)
            if eu + 1 < n_utts:
                next_start = utts[eu + 1].start_s
                if next_start - e_t > 2.5:
                    e_t = round(max(e_t, next_start - 0.15), 3)

        scanned_segments.append(
            CutSegment(
                cut_id="",
                kind=item["kind"],  # type: ignore[arg-type]
                subtype=item["subtype"],
                enabled=True,
                detector="llm",
                action="inherit",
                recommended_action=item.get("recommended_action", "delete"),  # type: ignore[arg-type]
                sensitive_terms_zh=list(item.get("terms_zh") or []),
                sensitive_terms_en=list(item.get("terms_en") or []),
                start_utt=su,
                end_utt=eu,
                start_s=round(max(0.0, s_t), 3),
                end_s=round(min(dur if dur > 0 else e_t, max(s_t + 0.2, e_t)), 3),
                quote=_utterance_excerpt(utts, su, eu),
                reason=item["reason"],
                severity=item["severity"],  # type: ignore[arg-type]
                chapter_title=_find_chapter_for_time(manifest, s_t),
            )
        )

    # Visual resolution-switch ad detection (catch unsubtitled sponsor inserts)
    check_cancelled()
    if eff_ads:
        vis_spans = detect_visual_ad_spans(Path(manifest.source.render_video))
        for vs, ve in vis_spans:
            overlaps_existing = any(
                seg.kind == "ad" and max(seg.start_s, vs) < min(seg.end_s, ve)
                for seg in scanned_segments
            )
            if overlaps_existing:
                continue
            scanned_segments.append(
                CutSegment(
                    cut_id="",
                    kind="ad",
                    subtype="visual_insert",
                    enabled=True,
                    detector="visual",
                    start_utt=None,
                    end_utt=None,
                    start_s=round(vs, 3),
                    end_s=round(ve, 3),
                    quote="",
                    reason="画面分辨率切换插播广告",
                    severity="high",
                    chapter_title=_find_chapter_for_time(manifest, vs),
                )
            )

    all_segments = scanned_segments + manual_segments
    all_segments.sort(key=lambda x: (x.start_s, x.end_s))
    for idx, seg in enumerate(all_segments, 1):
        seg.cut_id = f"cut_{idx:02d}"

    manifest.cut_segments = all_segments
    has_ads = any(x.kind == "ad" for x in all_segments)
    has_pol = any(
        x.kind == "political_sensitive" and segment_matches_sensitivity(x, eff_sens, cfg)
        for x in all_segments
    )
    manifest.clean_cut.cut_ads = has_ads
    manifest.clean_cut.cut_political_sensitive = has_pol
    manifest.clean_cut.enabled = bool(get_active_cut_segments(manifest, cfg))
    if all_segments and not manifest.clean_cut.title:
        base_title = (manifest.metadata.original_theme_zh or manifest.metadata.original_title or "全片").strip()
        manifest.clean_cut.title = f"{base_title}（净版）"
    manifest.clean_cut.scanned_at = utc_now_iso()
    check_cancelled()
    sync_all_render_hashes(manifest, cfg)
    store.save(manifest)

    ad_cnt = sum(1 for x in all_segments if x.kind == "ad")
    pol_cnt = sum(1 for x in all_segments if x.kind == "political_sensitive")
    merged_spans = merge_spans([(x.start_s, x.end_s) for x in all_segments], max_gap_s=0.05)
    total_cut_s = sum(max(0.0, e - s) for s, e in merged_spans)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    summary = (
        f"净版合规扫描完成：命中 {len(all_segments)} 段"
        f"（广告 {ad_cnt} 段，政治敏感 {pol_cnt} 段，共 {total_cut_s:.1f}s，标准: {eff_sens}）"
    )
    store.log(f"[clean_cut] {summary}")
    store.append_receipt(
        run_id=run_id,
        stage="clean_cut",
        actor="ai",
        action="scan_cut_segments",
        model=used_model_label,
        elapsed_ms=elapsed_ms,
        summary=summary,
        details={
            "ads": ad_cnt,
            "political_sensitive": pol_cnt,
            "manual_preserved": len(manual_segments),
            "total_cut_seconds": round(total_cut_s, 1),
            "sensitivity": eff_sens,
        },
    )
    return manifest
