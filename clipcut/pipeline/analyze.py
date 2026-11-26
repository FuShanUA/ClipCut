from __future__ import annotations

import re
import time
from typing import Any

from ..cancel import check_cancelled
from ..llm import ClipCutLLM, DEFAULT_TEXT_FALLBACK_MODELS
from ..models import (
    Candidate,
    ChapterAudit,
    ProjectManifest,
    SuppressedHotTopic,
    TopicMapItem,
    Utterance,
)
from ..receipts import ProjectStore


def _fmt_ts(sec: float) -> str:
    s = max(0, int(sec))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def format_transcript_for_llm(utterances: list[Utterance]) -> str:
    lines: list[str] = []
    for u in utterances:
        text_parts = []
        if u.zh:
            text_parts.append(u.zh)
        if u.en and u.en != u.zh:
            text_parts.append(u.en)
        body = " / ".join(text_parts) if text_parts else u.raw
        sig = f" <<{','.join(u.signal_tags)}>>" if u.signal_tags else ""
        lines.append(f"[u{u.id:04d} {_fmt_ts(u.start_s)}-{_fmt_ts(u.end_s)}] {body}{sig}")
    return "\n".join(lines)


def _format_chapters(manifest: ProjectManifest) -> str:
    if not manifest.metadata.chapters:
        return "（原视频无官方章节）"
    rows = []
    for ch in manifest.metadata.chapters:
        dur = max(0.0, ch.end_s - ch.start_s)
        rows.append(f"- Chapter {ch.index}: {_fmt_ts(ch.start_s)} - {_fmt_ts(ch.end_s)} ({int(dur)}s) | {ch.title}")
    return "\n".join(rows)


def _format_hot_topics(cfg: dict[str, Any]) -> str:
    cats = cfg.get("hot_topics", {}).get("categories", [])
    rows = []
    for c in cats:
        kws = ", ".join(c.get("keywords", []))
        rows.append(f"- {c.get('name', '')}: [{kws}] — {c.get('note', '')}")
    return "\n".join(rows)


def _make_slug(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[\s_]+", "-", s.strip())
    s = re.sub(r"-+", "-", s)
    return s[:48].rstrip("-") or "clip"


PASS1_SYSTEM = """你是顶级财经/科技视频总编，负责对长视频做严谨的全文结构解构与宏观话题地图测绘。
你的工作遵循「先画 100% 覆盖全片的宏观话题地图，再做高质量完整切片」铁律：把同一论题的追问、举例与推论合并为完整叙事单元，绝不把长对话拆成大量1分钟碎片，绝不遗漏后半段内容。
你必须严格输出合法 JSON，不附加任何解释或 Markdown 围栏。"""


def _compute_topic_budget(duration_s: float) -> tuple[int, int]:
    """Compute recommended (min, max) topic count for Pass 1 macro segmentation."""
    dur_min = max(1.0, float(duration_s or 0.0) / 60.0)
    tmax = max(6, min(20, int(round(dur_min / 3.0))))
    tmin = max(4, int(round(tmax * 0.55)))
    return tmin, tmax


def _compute_clip_budget(duration_s: float, cfg: dict[str, Any]) -> tuple[int, int]:
    """Compute recommended (min, max) standalone clip count from video duration and config."""
    dur_cfg = cfg.get("clip_duration", {})
    max_per_hour = float(dur_cfg.get("max_clips_per_hour", 12))
    min_floor = int(dur_cfg.get("min_clips_floor", 4))
    dur_hours = max(0.05, float(duration_s or 0.0) / 3600.0)
    bmax = max(min_floor, min(16, int(round(dur_hours * max_per_hour))))
    bmin = max(2, int(round(bmax * 0.55)))
    return bmin, bmax


def _build_pass1_prompt(manifest: ProjectManifest, transcript_text: str) -> str:
    n_utts = len(manifest.utterances)
    dur_s = float(manifest.source.duration_s or 0.0)
    t_min, t_max = _compute_topic_budget(dur_s)
    ch_count = len(manifest.metadata.chapters)
    avg_ch_s = (dur_s / ch_count) if ch_count > 0 and dur_s > 0 else 0.0
    is_dense_chapters = ch_count > 18 or (ch_count > 0 and avg_ch_s < 150.0)
    if is_dense_chapters:
        chapter_grouping_note = (
            f"   - 【密集短章节强制归并】：原视频包含 {ch_count} 个细碎短章节（平均仅 {int(avg_ch_s)} 秒），"
            "严禁按原章节 1:1 切分！必须把探讨同一大议题的相邻短章节归并为 2.5~6 分钟的完整宏观话题。\n"
        )
    elif ch_count > 0:
        chapter_grouping_note = (
            "   - 【章节归并原则】：3~5 分钟的官方章节通常对应 1 个完整话题；6~10 分钟长章节最多拆 2 个宏观话题，避免把同一章节拆成 3~5 段 1 分钟碎片。\n"
        )
    else:
        chapter_grouping_note = ""

    return f"""请通读整部视频的元数据、章节与带编号字幕，完成以下四项测绘任务：

【原视频元数据】
- 原标题: {manifest.metadata.original_title}
- 频道/来源: {manifest.metadata.channel or manifest.metadata.uploader or '未知'}
- 时长: {_fmt_ts(manifest.source.duration_s)} ({int(manifest.source.duration_s)}秒)，字幕编号 u0000 ~ u{n_utts - 1:04d}
- 简介:
{manifest.metadata.description[:4000] or '（无简介）'}

【原视频章节】
{_format_chapters(manifest)}

【任务要求】
1. `primary_topics`：提炼本视频的 1~4 个核心主话题/主角实体（例如若整期访谈就是采访 Palantir CEO 或英伟达高管，则必须把 "Palantir" / "英伟达" / "NVIDIA" 写入 `primary_topics`，供后续热点去重使用）。
2. `meta`：提炼规范中文信息 `guest_zh`、`channel_zh`、`original_theme_zh` 与 `full_summary_zh`。
3. `chapter_audits`：如果原视频有章节，逐一审视每个章节；对于前后较短的片花（Teaser）、开场客套寒暄、赞助广告（Sponsor）、片尾致谢，标为 `"skip"` 并写明理由；有实质内容的标为 `"keep"`。如果原视频无章节，可留空数组 `[]`。
4. `topic_map`：按时间顺序把整部视频切成首尾相接、覆盖 u0000 到 u{n_utts - 1:04d} 的**完整宏观叙事单元**清单，总计严格控制在 **{t_min}~{t_max} 个话题**，实质话题一般 **2~7 分钟**：
   - 同一核心论点下的连续追问、举例佐证、对比论证与推论必须归在同一个话题中，严禁拆成多个 1 分钟左右的碎片。
{chapter_grouping_note}   - 寒暄、片花、广告、片尾也要单独记一行并设 `is_filler: true` + `filler_reason`，不留空当。
   - 每个话题给出准确的 `start_utt`、`end_utt`（整数，对应 `uXXXX`）、`punchline_utt`（落点/金句所在句号）、所属 `chapter_title`、`title`（中文简短主题）、`summary`（1~2句概括论点与结论）。

【输出 JSON Schema】
{{
  "primary_topics": ["字符串"],
  "meta": {{
    "guest_zh": "嘉宾中文名 (英文名)",
    "channel_zh": "频道中文名",
    "original_theme_zh": "原视频核心中文主题",
    "full_summary_zh": "120~220字面向观众的简体中文全片内容提炼介绍，概括人物/节目背景、核心议题与主要观点，自然连贯，严禁出现字幕句数、片段数、广告/脱敏/剪辑等技术参数"
  }},
  "chapter_audits": [
    {{
      "chapter_index": 0,
      "title": "原章节名",
      "decision": "keep | skip",
      "reason": "保留或剔除理由"
    }}
  ],
  "topic_map": [
    {{
      "topic_id": "T01",
      "start_utt": 0,
      "end_utt": 25,
      "chapter_title": "对应章节名（无则空字符串）",
      "title": "话题简短中文标题",
      "summary": "概括这段讲了什么、核心观点/结论是什么",
      "punchline_utt": 22,
      "is_filler": false,
      "filler_reason": ""
    }}
  ]
}}

【完整带信号字幕】
{transcript_text}"""


PASS2_SYSTEM = """你是严肃的中文科技/商业视频切片主编。你必须严格基于已经画好的「话题地图」与原文带信号字幕，执行三类精选、主话题热点去重、反碎片聚合与边界收敛。
切记：
1. 每个独立切片必须是一个完整自洽的独立叙事弧（理想 2~4.5 分钟，允许 1.5~6 分钟），宁缺毋滥，严禁把长访谈按话题逐段连切成十几二十条碎片，同论点或同章节内的相邻子话题必须合并为 1 支完整切片。
2. 开头和结尾必须是完整的句子（开头最好直接落在钩子/明确提问/核心论点，结尾停在完整结论或包袱之后）。
3. 如果某话题因平淡或铺垫未入选，必须在 `dropped_topics` 中如实写明淘汰理由。
4. 严格输出合法 JSON。"""


def _build_pass2_prompt(
    manifest: ProjectManifest,
    cfg: dict[str, Any],
    transcript_text: str,
) -> str:
    dur_cfg = cfg.get("clip_duration", {})
    min_s = int(dur_cfg.get("min_seconds", 90))
    max_s = int(dur_cfg.get("max_seconds", 360))
    ideal_min_s = int(dur_cfg.get("ideal_min_seconds", 120))
    ideal_max_s = int(dur_cfg.get("ideal_max_seconds", 270))
    short_max_s = int(dur_cfg.get("short_highlight_max_seconds", 75))
    b_min, b_max = _compute_clip_budget(manifest.source.duration_s, cfg)
    dur_min_total = max(1, int(round(float(manifest.source.duration_s or 0.0) / 60.0)))

    topic_lines = []
    for t in manifest.topic_map:
        flag = f"[FILLER: {t.filler_reason}]" if t.is_filler else "[SUBSTANTIVE]"
        topic_lines.append(
            f"- {t.topic_id} {flag} u{t.start_utt:04d}-u{t.end_utt:04d} ({_fmt_ts(t.start_s)}-{_fmt_ts(t.end_s)}) "
            f"| 章节:{t.chapter_title or '无'} | {t.title} — {t.summary}"
        )

    kept_chapters = [a for a in manifest.metadata.chapter_audits if a.decision == "keep"]
    kept_ch_text = "\n".join(f"- Ch{a.chapter_index}: {a.title} ({_fmt_ts(a.start_s)}-{_fmt_ts(a.end_s)})" for a in kept_chapters) or "（按话题地图覆盖）"

    return f"""请基于以下「话题地图」、元数据与字幕信号，完成严格的三类精选与主话题热点去重。

【原视频信息】
- 原标题: {manifest.metadata.original_title}
- 核心主话题（primary_topics）: {manifest.primary_topics}
- 简介:
{manifest.metadata.description[:3000] or '（无简介）'}

【审视后保留的有效章节】
{kept_ch_text}

【全片话题地图（100%覆盖）】
{chr(10).join(topic_lines)}

【中国市场热点与编辑关注词表】
{_format_hot_topics(cfg)}

【三类选片严格标准】
1. `key_dialogue`（重点对话）：
   - 围绕原标题核心命题、简介关键要点与全片信息密度最高的宏观议题进行**精选**（宁缺毋滥，绝不按章节或话题 1:1 机械切分）；
   - 服务于同一论点的「提问/前提 -> 案例/证据 -> 推论/总结」相邻子话题必须合并为 1 支完整切片，并在 `topic_ids` 中列出所合并的话题 ID；
   - 背景介绍、平淡过渡、常识性铺垫或观点重复的段落应直接淘汰并写入 `dropped_topics`。
2. `exciting_dialogue`（精彩对话）：
   - 结合字幕语义与 `<<音量峰值/语速激增/大幅动作/高频交锋>>` 声画信号标记，挑出大笑、大幅度动作、激烈论辩、犀利追问与反转交锋段落。
   - 这些候选会被送入下一步多模态视频复核，因此请标出 `needs_multimodal_verify: true`。
3. `hot_topic`（中国热点与编辑关注点）：
   - 命中上述热点词表且有实质观点、对比或数据支撑的段落。
   - 【主话题去重硬规则】：本视频主话题是 `{manifest.primary_topics}`。凡是仅仅在常规讨论主话题本身的内容，**严禁**当作 `hot_topic` 切出（应归为 `key_dialogue` 或不切），并将其记录到 `suppressed_hot_topics` 中说明原因！只有当该段出现了**具体可量化的业务用例/ROI数据**、**明确的中国市场/中美竞争视角**、或**跨界讨论了主话题之外的其他热点词**时，才允许标为 `hot_topic`。

【反碎片化、篇幅预算与边界铁律】
- **全片切片数量预算**：本片时长约 {dur_min_total} 分钟，独立切片总数（`is_short_highlight: false`）应严格控制在 **{b_min}~{b_max} 支**，挑选全片真正值得单条传播的精华叙事单元，**严禁**将全片 75%~90% 时长首尾 0 秒连切成十几二十条碎片。
- **独立切片目标时长**：理想 **{ideal_min_s}~{ideal_max_s} 秒**（2~4.5 分钟），允许区间 **{min_s}~{max_s} 秒**（1.5~6 分钟）；首尾为完整句可略微浮动（最多 ±15 秒）。
- **同章/同论证链聚合**：单个 3~5 分钟章节原则上最多切出 1 支完整切片；6~9 分钟长章节最多 2 支；不要把一个完整论点的前后两半硬拆成 60~80 秒两截。
- **短高光分流（Remix 精选池）**：如果某高能情绪/锋利金句片段天然短于 **{short_max_s} 秒** 且无需或无法自然并入前后完整论证，不要硬凑铺垫、也不要发成破碎的独立长切片，请输出并标注 `is_short_highlight: true`（系统会自动收入 Remix 精选池）。
- 开头不能以依赖前文的残缺指代词起头（如“所以这个也是”“That's why it works”），必须包含必要的问句或前提句；结尾必须停在完整结论句之后。
- 切片标题规范：标题必须立足所属章节与全片大主题，提炼嘉宾借该案例论证的宏观核心命题，避免把表层举例场景写成主标题（例如借某部门问卷或内部案例演示“Agentic AI 决策分析能力”，标题应聚焦 AI 决策能力而非该部门本身）。
- 每个候选可以同时归属多个类别（例如既是 `key_dialogue` 又是 `hot_topic`），避免为同一段拆出重复候选。

【输出 JSON Schema】
{{
  "candidates": [
    {{
      "candidate_id": "C01",
      "topic_ids": ["T03"],
      "categories": ["key_dialogue", "hot_topic"],
      "matched_hot_topics": ["AI具体业务用例与量化价值"],
      "chapter_title": "所属章节名",
      "title": "主标题：具体核心观点（≤22字，禁止震惊/颠覆等标题党词）",
      "title_en": "Concise English subtitle",
      "slug": "concise-english-slug",
      "start_utt": 120,
      "end_utt": 168,
      "hook": "严格照抄 start_utt 附近开头第一句抓人的原话",
      "reason": "明确说明归属哪类规则、为何值得独立切出",
      "summary": "50~80字中文简介",
      "key_quote": "本段最锋利的一句原话（英文或中文原文）",
      "score": 8.8,
      "needs_multimodal_verify": false,
      "is_short_highlight": false
    }}
  ],
  "suppressed_hot_topics": [
    {{
      "topic_id": "T05",
      "start_utt": 210,
      "end_utt": 245,
      "matched_keyword": "Palantir",
      "primary_topic_conflict": "Palantir",
      "reason": "本片主话题即为 Palantir，该段仅常规介绍产品历史，无中国视角或量化业务案例，不按热点单切"
    }}
  ],
  "dropped_topics": [
    {{
      "topic_id": "T01",
      "reason": "片头1分钟客套寒暄，无实质论点"
    }}
  ]
}}

【完整带信号字幕】
{transcript_text}"""


def _is_primary_topic_match(keyword: str, primary_topics: list[str]) -> str | None:
    kw_l = keyword.strip().lower()
    if not kw_l:
        return None
    for pt in primary_topics:
        pt_l = pt.strip().lower()
        if not pt_l:
            continue
        if kw_l in pt_l or pt_l in kw_l:
            return pt
    return None


def _clamp_utt(idx: int, n_utts: int) -> int:
    return max(0, min(int(idx), n_utts - 1))


def run_analyze_stage(
    manifest: ProjectManifest,
    store: ProjectStore,
    cfg: dict[str, Any],
    run_id: str,
) -> ProjectManifest:
    models_cfg = cfg.get("models", {})
    llm = ClipCutLLM(models_cfg)
    text_provider = models_cfg.get("text_provider", "dashscope")
    analyze_model = models_cfg.get("text_model") or models_cfg.get("analyze_model") or "qwen3.8-max"
    fallback_models = list(models_cfg.get("text_fallback_models") or DEFAULT_TEXT_FALLBACK_MODELS)

    utts = manifest.utterances
    n_utts = len(utts)
    transcript_text = format_transcript_for_llm(utts)

    # ── Pass 1: Video Profile, Chapter Audit & 100% Topic Map ──
    check_cancelled()
    store.log(f"[analyze] Pass 1: 测绘全片话题地图与章节审查（{n_utts} 句字幕，模型: {text_provider}:{analyze_model}）")
    res1 = llm.generate_json(
        prompt=_build_pass1_prompt(manifest, transcript_text),
        model=analyze_model,
        provider=text_provider,
        fallback_models=fallback_models,
        system_instruction=PASS1_SYSTEM,
        temperature=0.2,
        max_output_tokens=16384,
    )
    data1 = res1.parsed if isinstance(res1.parsed, dict) else {}

    manifest.primary_topics = [str(x).strip() for x in (data1.get("primary_topics") or []) if str(x).strip()]
    meta1 = data1.get("meta") or {}
    manifest.metadata.guest_zh = str(meta1.get("guest_zh") or manifest.metadata.guest_zh or "")
    manifest.metadata.channel_zh = str(meta1.get("channel_zh") or manifest.metadata.channel_zh or "")
    manifest.metadata.original_theme_zh = str(meta1.get("original_theme_zh") or manifest.metadata.original_title or "")
    manifest.metadata.full_summary_zh = str(
        meta1.get("full_summary_zh") or manifest.metadata.full_summary_zh or ""
    ).strip()
    if manifest.cut_segments or manifest.clean_cut.scanned_at:
        from .content_filter import compute_clean_cut_render_hash
        from .verify_and_plan import build_default_clean_cut_title, is_untranslated_clean_cut_title

        if is_untranslated_clean_cut_title(manifest, manifest.clean_cut.title):
            cand_cc_title = build_default_clean_cut_title(manifest)
            if cand_cc_title != manifest.clean_cut.title:
                cc = manifest.clean_cut
                was_up_to_date = bool(cc.output_file and cc.render_hash == cc.last_rendered_hash)
                cc.title = cand_cc_title
                cc.render_hash = compute_clean_cut_render_hash(manifest, cfg)
                if was_up_to_date:
                    cc.last_rendered_hash = cc.render_hash

    # Map chapter audits
    ch_by_idx = {c.index: c for c in manifest.metadata.chapters}
    audits: list[ChapterAudit] = []
    for raw_a in data1.get("chapter_audits") or []:
        c_idx = int(raw_a.get("chapter_index", len(audits)))
        ch = ch_by_idx.get(c_idx)
        audits.append(
            ChapterAudit(
                chapter_index=c_idx,
                title=str(raw_a.get("title") or (ch.title if ch else f"Chapter {c_idx}")),
                start_s=ch.start_s if ch else 0.0,
                end_s=ch.end_s if ch else 0.0,
                decision="skip" if str(raw_a.get("decision", "keep")).lower() == "skip" else "keep",
                reason=str(raw_a.get("reason") or ""),
            )
        )
    manifest.metadata.chapter_audits = audits

    # Map Topic Map items
    topic_map: list[TopicMapItem] = []
    for idx, raw_t in enumerate(data1.get("topic_map") or []):
        su = _clamp_utt(raw_t.get("start_utt", 0), n_utts)
        eu = _clamp_utt(raw_t.get("end_utt", su), n_utts)
        if eu < su:
            su, eu = eu, su
        pu_raw = raw_t.get("punchline_utt")
        pu = _clamp_utt(pu_raw, n_utts) if pu_raw is not None else None
        topic_map.append(
            TopicMapItem(
                topic_id=str(raw_t.get("topic_id") or f"T{idx + 1:02d}"),
                start_utt=su,
                end_utt=eu,
                start_s=utts[su].start_s,
                end_s=utts[eu].end_s,
                chapter_title=str(raw_t.get("chapter_title") or ""),
                title=str(raw_t.get("title") or f"话题 {idx + 1}"),
                summary=str(raw_t.get("summary") or ""),
                punchline_utt=pu,
                is_filler=bool(raw_t.get("is_filler", False)),
                filler_reason=str(raw_t.get("filler_reason") or ""),
            )
        )
    manifest.topic_map = topic_map

    skipped_chs = [a.model_dump() for a in audits if a.decision == "skip"]
    store.append_receipt(
        run_id=run_id,
        stage="analyze",
        actor="ai",
        action="build_topic_map",
        model=f"{res1.provider}:{res1.model}",
        prompt_tokens=res1.prompt_tokens,
        output_tokens=res1.output_tokens,
        elapsed_ms=res1.elapsed_ms,
        summary=f"完成全片测绘：主话题 {manifest.primary_topics}，话题地图 {len(topic_map)} 段（含 {sum(1 for t in topic_map if t.is_filler)} 段片花/寒暄），章节剔除 {len(skipped_chs)} 个",
        details={
            "primary_topics": manifest.primary_topics,
            "topic_count": len(topic_map),
            "skipped_chapters": skipped_chs,
            "kept_chapter_count": len(audits) - len(skipped_chs),
        },
    )

    # ── Pass 2: Three-Track Selection & Primary-Topic Dedup ──
    check_cancelled()
    store.log(f"[analyze] Pass 2: 执行三类选片（重点对话/精彩对话/中国热点）与主话题去重（模型: {text_provider}:{analyze_model}）")
    res2 = llm.generate_json(
        prompt=_build_pass2_prompt(manifest, cfg, transcript_text),
        model=analyze_model,
        provider=text_provider,
        fallback_models=fallback_models,
        system_instruction=PASS2_SYSTEM,
        temperature=0.25,
        max_output_tokens=16384,
    )
    data2 = res2.parsed if isinstance(res2.parsed, dict) else {}

    suppressed: list[SuppressedHotTopic] = []
    for raw_s in data2.get("suppressed_hot_topics") or []:
        suppressed.append(
            SuppressedHotTopic(
                topic_id=raw_s.get("topic_id"),
                start_utt=_clamp_utt(raw_s.get("start_utt", 0), n_utts),
                end_utt=_clamp_utt(raw_s.get("end_utt", 0), n_utts),
                matched_keyword=str(raw_s.get("matched_keyword") or ""),
                primary_topic_conflict=str(raw_s.get("primary_topic_conflict") or ""),
                reason=str(raw_s.get("reason") or "与主话题重复"),
            )
        )

    valid_cats = {"key_dialogue", "exciting_dialogue", "hot_topic"}
    candidates: list[Candidate] = []
    dedup_enabled = bool(cfg.get("hot_topics", {}).get("primary_topic_dedup", True))

    for idx, raw_c in enumerate(data2.get("candidates") or []):
        su = _clamp_utt(raw_c.get("start_utt", 0), n_utts)
        eu = _clamp_utt(raw_c.get("end_utt", su), n_utts)
        if eu < su:
            su, eu = eu, su
        cats = [c for c in (raw_c.get("categories") or []) if c in valid_cats]
        if not cats:
            cats = ["key_dialogue"]
        matched_hot = [str(x).strip() for x in (raw_c.get("matched_hot_topics") or []) if str(x).strip()]

        # Programmatic guardrail for primary-topic deduplication
        if dedup_enabled and "hot_topic" in cats and manifest.primary_topics:
            non_primary_hot = []
            for kw in matched_hot:
                conflict = _is_primary_topic_match(kw, manifest.primary_topics)
                if conflict:
                    # Only keep if reason explicitly mentions ROI / 量化 / 中国 / 中美
                    reason_text = f"{raw_c.get('reason', '')} {raw_c.get('summary', '')}"
                    if re.search(r"量化|ROI|降本|提效|收入|毛利|中国|中美|China", reason_text, flags=re.IGNORECASE):
                        non_primary_hot.append(kw)
                    else:
                        suppressed.append(
                            SuppressedHotTopic(
                                topic_id=(raw_c.get("topic_ids") or [None])[0],
                                start_utt=su,
                                end_utt=eu,
                                matched_keyword=kw,
                                primary_topic_conflict=conflict,
                                reason="程序化去重规则拦截：命中词与全片主话题一致且无量化ROI或中国视角",
                            )
                        )
                else:
                    non_primary_hot.append(kw)
            matched_hot = non_primary_hot
            if not matched_hot and "hot_topic" in cats:
                cats = [c for c in cats if c != "hot_topic"]
                if not cats:
                    # Pure false hot-topic -> drop it
                    continue

        needs_mm = bool(raw_c.get("needs_multimodal_verify", False)) or ("exciting_dialogue" in cats)
        is_short = bool(raw_c.get("is_short_highlight", False))
        title = str(raw_c.get("title") or f"片段 {idx + 1}").strip()
        title_en = str(raw_c.get("title_en") or "").strip()
        slug = _make_slug(str(raw_c.get("slug") or title_en or title))

        candidates.append(
            Candidate(
                candidate_id=str(raw_c.get("candidate_id") or f"C{idx + 1:02d}"),
                topic_ids=[str(t) for t in (raw_c.get("topic_ids") or [])],
                categories=cats,  # type: ignore[arg-type]
                matched_hot_topics=matched_hot,
                chapter_title=str(raw_c.get("chapter_title") or ""),
                title=title,
                title_en=title_en,
                slug=slug,
                start_utt=su,
                end_utt=eu,
                hook=str(raw_c.get("hook") or (utts[su].zh or utts[su].en)).strip(),
                reason=str(raw_c.get("reason") or "").strip(),
                summary=str(raw_c.get("summary") or "").strip(),
                key_quote=str(raw_c.get("key_quote") or "").strip(),
                score=float(raw_c.get("score") or 7.5),
                needs_multimodal_verify=needs_mm,
                disposition="remix_only" if is_short else "pending",
                disposition_reason="AI标记为短高光，待归入Remix精选池" if is_short else "",
            )
        )

    # Record dropped topic reasons back into topic_map
    dropped_map = {str(d.get("topic_id")): str(d.get("reason") or "") for d in (data2.get("dropped_topics") or [])}
    selected_tids = {tid for c in candidates for tid in c.topic_ids}
    for t in manifest.topic_map:
        if t.topic_id in dropped_map:
            t.dropped_reason = dropped_map[t.topic_id]
        elif t.topic_id not in selected_tids:
            t.dropped_reason = t.filler_reason or "信息密度或独立完整性未达选片门槛"

    check_cancelled()
    manifest.candidates = candidates
    manifest.suppressed_hot_topics = suppressed
    store.save(manifest)

    cat_counts = {
        "key_dialogue": sum(1 for c in candidates if "key_dialogue" in c.categories),
        "exciting_dialogue": sum(1 for c in candidates if "exciting_dialogue" in c.categories),
        "hot_topic": sum(1 for c in candidates if "hot_topic" in c.categories),
    }
    store.append_receipt(
        run_id=run_id,
        stage="analyze",
        actor="ai",
        action="select_three_track_candidates",
        model=f"{res2.provider}:{res2.model}",
        prompt_tokens=res2.prompt_tokens,
        output_tokens=res2.output_tokens,
        elapsed_ms=res2.elapsed_ms,
        summary=f"三类初筛产出 {len(candidates)} 个候选（重点:{cat_counts['key_dialogue']} / 精彩待复核:{cat_counts['exciting_dialogue']} / 热点:{cat_counts['hot_topic']}），主话题去重拦截 {len(suppressed)} 项",
        details={
            "category_counts": cat_counts,
            "suppressed_hot_topics": [s.model_dump() for s in suppressed],
            "dropped_topic_count": sum(1 for t in manifest.topic_map if t.dropped_reason),
        },
    )
    return manifest
