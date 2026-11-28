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
from ..crop import crop_hash_payload, resolve_effective_crop
from ..llm import ClipCutLLM, DEFAULT_TEXT_FALLBACK_MODELS, parse_provider_model
from ..models import CutSegment, ProjectManifest, Utterance, VisualMaskBox, utc_now_iso
from ..receipts import ProjectStore
from .visual_ocr import detect_visual_ocr_redline_segments, match_hard_redline_text


VALID_SEVERITIES = ("high", "medium", "low")
VALID_POLITICAL_ACTIONS = ("auto", "delete", "mask", "precise_mask")
VALID_SEGMENT_ACTIONS = ("inherit", "delete", "mask", "precise_mask")

_DEFAULT_ZH_TO_EN_TERMS: dict[str, list[str]] = {
    "习近平": ["xi jinping", "xi"],
    "习主席": ["xi jinping", "xi"],
    "普京": ["putin"],
    "中南海": ["zhongnanhai"],
    "中国国家安全部": ["mss", "ministry of state security"],
    "国家安全部": ["mss", "ministry of state security"],
    "国安部": ["mss", "ministry of state security"],
    "解放军": ["pla", "people's liberation army"],
    "中国共产党": ["chinese communist party", "ccp"],
    "共产党": ["communist party", "communist", "ccp"],
    "中共": ["ccp", "chinese communist party", "communist party"],
    "中国大陆": ["mainland china"],
    "南京大屠杀纪念馆": ["nanjing massacre memorial"],
    "新疆维吾尔族自治区": ["xinjiang uygur autonomous region"],
    "台海": ["taiwan strait", "taiwan"],
    "台湾": ["taiwan", "taiwanese"],
    "台独": ["taiwan independence"],
    "中华民国": ["republic of china"],
    "台湾总统": ["president of taiwan", "taiwanese president"],
    "香港": ["hong kong"],
    "港独": ["hong kong independence"],
    "新疆": ["xinjiang"],
    "西藏": ["tibet", "tibetan"],
    "南海": ["south china sea"],
    "北京": ["beijing"],
    "中国": ["china", "chinese"],
    "政权": ["regime"],
    "政权倒台": ["regime toppled", "regime collapse", "toppled"],
    "倒台": ["topple", "toppled", "collapse"],
    "推翻": ["overthrow", "topple", "toppled"],
    "独裁": ["dictatorship", "dictator", "autocratic"],
    "独裁者": ["dictator"],
    "清洗": ["purge", "purges", "purged", "purging"],
    "侵略": ["aggression", "invade", "invaded", "invasion"],
    "侵略者": ["aggressor"],
    "入侵": ["invade", "invaded", "invading", "invasion"],
    "邪恶": ["evil", "wicked"],
    "撒谎": ["lie", "lying", "lied", "liar"],
    "谎言": ["lie", "lies"],
    "正义": ["righteous", "justice", "just"],
    "暴政": ["tyranny", "tyrant"],
    "镇压": ["crackdown", "suppress", "suppression", "oppress"],
    "迫害": ["persecute", "persecution"],
    "极端": ["extremist", "extreme"],
    "恐怖": ["terrorist", "terrorism"],
    "野蛮": ["barbaric", "brutal"],
    "强征": ["conscripted", "conscript"],
    "洗脑": ["brainwash", "brainwashed"],
}

PRECISE_MASK_RULES_PROMPT = """【精准遮蔽/消音关键词（`precise_terms_zh` / `precise_terms_en`）提取铁律】：
目标：只给**真正构成政治红线、不规范主权称谓或明确贬损定性的极少数关键词**打码与消音，**最大限度保留前后正常对话与上下文语流，禁止将整句或中性国家名词（如“俄罗斯/乌克兰/美国/伊朗”）打包列入精准词表**。
每段 `political_sensitive`（无论 `recommended_action` 是 `"delete"` 还是 `"mask"`）都**必须**尽量精确提取以下 3 类词；若某句确实只是一般背景叙述、不包含下列 3 类词，则该段精准词表返回 `[]`（表示精准模式下不触发整段消音）：
1. **不妥当的政治体或政治人物称谓**：我国国家领导人、政治人物及敏感党政/政权称谓（如 `习主席` / `习近平` / `Xi Jinping`、`政权` / `regime`、`独裁者` / `dictator`、`特朗普` / `Trump` 等若触发人物政治审查）。
2. **按网信办、外交部与平台规范对中港台地区、国家形象及法定表述的不规范称谓/用语**：如将台湾/香港与中国并列或称“台湾总统 / 中华民国 / 台湾国 / 两个中国 / 一中一台 / 一边一国 / 台独 / 港独”（仅提普通商业/地理事实时不得泛滥标出 `台湾`/`中国`）；英文将“中国大陆”错译为 `mainland China` / `Mainland China`（规范应为 `Chinese mainland` / `China's mainland` / `the mainland of China`）；错将“侵华日军南京大屠杀遇难同胞纪念馆”简称为 `南京大屠杀纪念馆`、错将“新疆维吾尔自治区”写为 `新疆维吾尔族自治区`；以及侮辱/不当使用 `国旗`/`国歌`/`国徽`/`人民币` 或贬损英烈/`解放军`/`武警`/`国安`/`公安`/`司法人员`/`党员` 形象的定性词与称谓词。
3. **对其他国家、政治人物行为或宗教/文化的定性判断动词、形容词、副词及定性名词**（包括假想推演、举例或引用句中的明确行动/定性措辞，例如 `某国入侵/清洗/倒台`）：如 `侵略` / `侵略者` / `入侵`（`aggressor` / `invasion` / `invaded`，若字幕将 `invasion` 错写为 `evasion` 也请原样摘录 `evasion`）、`正义`（`righteous`/`just`）、`邪恶`（`evil`）、`撒谎`/`谎言`（`lie`/`lying`）、`清洗`（`purges`）、`倒台`/`推翻`（`toppled`/`overthrow`）、`强征`（`conscripted`）、`镇压`/`迫害`/`极端`/`恐怖`/`野蛮`/`洗脑`（及其对宗教、文化群体的定性形容词/动词）。
4. **原文精确性约束**：
   - `precise_terms_zh` 必须逐字原样摘自对应句的**中文字幕**；
   - `precise_terms_en` 必须逐字原样摘自对应句的**英文字幕**（保持字幕实际拼写与时态/单复数，否则音频时间戳与打码将无法命中）。
5. **涉中国最高领导人评价的指代还原防漏约束**：
   - 当片段正在评价中国最高领导人的态度、执政能力或外交礼遇（且后续句以代词 `他`/`他们`/`he`/`they` 延续点评）时，不能仅提取 `习主席`/`President Xi` 而漏掉同一语段中暴露领导人外交姿态或评价的核心词句（如 `尊重`/`respect`、`到飞机迎接我`/`met me at the plane`、`信号`/`sign` 等），否则平台审核仍可凭上下文还原指代；此类整段评价在 `recommended_action` 上应优先设为 `"delete"`。"""

DEFAULT_POLITICAL_PROFILES: dict[str, dict[str, Any]] = {
    "strict": {
        "label": "严格从严（推荐视频号/B站）",
        "description": "适配中国大陆主流平台强审核：完整覆盖国家法律法规（中国特色社会主义制度与核心地位、分裂国家与规范称谓/版图、损害国家荣誉与英烈军警形象）及涉华/友好国家实质红线与对抗表述。",
        "trigger_severities": ["high", "medium", "low"],
        "rules": [
            {
                "id": "hard_redline",
                "name": "危害中国特色社会主义制度与领导人红线",
                "severity": "high",
                "description": "攻击、否定、损害、违背中国特色社会主义指导思想、行动指南及马克思主义中国化最新理论成果；攻击、否定、弱化党中央的核心与全党核心地位，以及任何涉及中国最高领导人个人言行、执政或外交礼遇姿态的评论、影射与可还原指代（含连续代词“他/他们”延续句）；历史虚无主义（以一个阶段党和国家发展历史否定另一个阶段）、违背党的百年奋斗重大成就决议、调侃否定新中国成立以来重大方针政策举措、曲解/攻击/调侃宪法及国家重大法律法规；以娱乐化方式篡改解读根本/基本/重要制度、特定政治名词及敏感历史事件；对中/俄/朝/伊朗等中国友好国家的政权攻击、极端军事打击渲染或直接意识形态攻击。",
            },
            {
                "id": "sovereignty_and_normative_terms",
                "name": "分裂国家、国家主权与规范称谓/版图",
                "severity": "high",
                "description": "危害国家统一与主权领土完整，反对、攻击、曲解“一个中国”“一国两制”，反映分裂国家言行、活动、标识、旗帜、标语口号或涉分裂立场艺人/团体节目，以及对领土和历史事件描写不符合国家定论；错用/滥用特定标识、呼号、称谓与用语：杜绝将“中国大陆”英译为 mainland China/Mainland China（应为 Chinese mainland/China's mainland/the mainland of China）、杜绝将“侵华日军南京大屠杀遇难同胞纪念馆”简称为“南京大屠杀纪念馆”、杜绝将“新疆维吾尔自治区”错写为“新疆维吾尔族自治区”、杜绝“台湾总统/中华民国/两个中国/一中一台/一边一国”；涉及中国地图残缺、地名不完整或篡改等国家版图缺失展示。",
            },
            {
                "id": "national_honor_and_public_service",
                "name": "损害国家荣誉利益、英烈及公务人员形象",
                "severity": "high",
                "description": "贬损、玷污、恶搞国家和民族形象、精神气质，焚烧/毁损/涂划/玷污/践踏/恶搞或不适宜娱乐商业场合使用国歌、国旗、国徽、人民币；贬损、恶搞、损害革命领袖、英雄烈士及中国共产党党员形象名誉；贬损、恶搞、损害人民军队（解放军）、武警、国安（国安部/MSS）、公安、行政、司法人员等国家公务人员形象名誉（含恶意截取执法营造暴力执法效果、泛娱乐化夸张展现解放军、传播未证实军装冲突或冒充公务人员）；美化反面人物、对已定性负面人物歌功颂德、曲解国史党史军史。",
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
        "description": "严守法律法规红线（制度与核心地位、分裂主权与规范称谓/版图、国家荣誉与英烈军警形象），并覆盖对华及友好国家的明确指责与对抗；放行一般技术对比、商业讨论及欧美内政。",
        "trigger_severities": ["high", "medium"],
        "rules": [
            {
                "id": "hard_redline",
                "name": "危害中国特色社会主义制度与领导人红线",
                "severity": "high",
                "description": "攻击、否定、违背中国特色社会主义制度、指导思想、党中央权威与核心地位，及涉中国最高领导人言行/外交姿态的评论、影射与连续代词指代；历史虚无主义、否定重大方针政策或曲解宪法法律；敏感历史政治事件；对中/俄/朝/伊朗等中国友好国家的政权攻击或直接意识形态攻击。",
            },
            {
                "id": "sovereignty_and_normative_terms",
                "name": "分裂国家、国家主权与规范称谓/版图",
                "severity": "high",
                "description": "反对“一个中国”“一国两制”、分裂言行标识及不符合定论的领土历史描写；杜绝 mainland China/Mainland China、南京大屠杀纪念馆（应含“侵华日军…遇难同胞”）、新疆维吾尔族自治区（应无“族”字）、台湾总统/中华民国/一边一国等不规范称谓；杜绝中国版图缺失或地名篡改。",
            },
            {
                "id": "national_honor_and_public_service",
                "name": "损害国家荣誉利益、英烈及公务人员形象",
                "severity": "high",
                "description": "侮辱或不当使用国歌、国旗、国徽、人民币；贬损革命领袖、英雄烈士、中国共产党党员；贬损恶搞人民军队、武警、国安（国安部/MSS）、公安、行政、司法人员等公务人员或恶意截取执法画面；美化反面人物或曲解国史党史军史。",
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
        "description": "仅识别直接触及国家法律法规与平台法定红线的高危内容；一般地缘竞争、产业讨论、历史类比与欧美内政均放行。",
        "trigger_severities": ["high"],
        "rules": [
            {
                "id": "hard_redline",
                "name": "危害中国特色社会主义制度与领导人红线",
                "severity": "high",
                "description": "直接危害中国特色社会主义制度、指导思想、党中央权威与核心地位，及涉中国最高领导人言行/外交姿态的评论、影射与连续代词指代；历史虚无主义、曲解宪法法律与敏感政治历史事件；直接意识形态攻击。",
            },
            {
                "id": "sovereignty_and_normative_terms",
                "name": "分裂国家、国家主权与规范称谓/版图",
                "severity": "high",
                "description": "危害国家统一与领土完整、反对“一个中国”“一国两制”、分裂标识言行；错用 mainland China/Mainland China、南京大屠杀纪念馆、新疆维吾尔族自治区、台湾总统/中华民国等不规范称谓；中国地图残缺或地名篡改。",
            },
            {
                "id": "national_honor_and_public_service",
                "name": "损害国家荣誉利益、英烈及公务人员形象",
                "severity": "high",
                "description": "侮辱或不当使用国歌/国旗/国徽/人民币；贬损革命领袖、英雄烈士、中国共产党党员；贬损恶搞人民军队、武警、国安（国安部/MSS）、公安、司法等国家公务人员；美化反面人物或曲解国史党史军史。",
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
2. 危害中国特色社会主义制度与领导人/核心地位、分裂国家与不规范国家称谓/版图（如错用 `mainland China`/`南京大屠杀纪念馆`/`新疆维吾尔族自治区` 或台湾/港澳疆藏南海主权问题）、损害国家荣誉利益与英烈/党员/军警国安公务人员形象，以及针对俄罗斯、朝鲜、伊朗/伊斯兰世界等中国友好国家的实质性政权/政策攻击或对抗（kind = "political_sensitive"）
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


def is_visual_ocr_segment(seg: CutSegment) -> bool:
    """Return True if segment masks on-screen video text rather than speech/subtitles."""
    return str(getattr(seg, "detector", "") or "") == "visual_ocr" or bool(getattr(seg, "visual_boxes", None))


def resolve_segment_action(
    seg: CutSegment,
    manifest: ProjectManifest,
    *,
    allow_mask: bool | None = None,
) -> Literal["delete", "mask", "precise_mask"]:
    """Resolve effective handling action ('delete', 'mask', or 'precise_mask') for a CutSegment."""
    if seg.kind != "political_sensitive":
        return "delete"
    seg_act = str(getattr(seg, "action", "inherit") or "inherit").strip().lower()
    if is_visual_ocr_segment(seg):
        if seg_act == "delete":
            return "delete"
        if seg_act in ("mask", "precise_mask"):
            return seg_act  # type: ignore[return-value]
        global_act = str(getattr(manifest.clean_cut, "political_action", "delete") or "delete").strip().lower()
        return "precise_mask" if global_act == "precise_mask" else "mask"
    can_mask = has_clean_source_video(manifest) if allow_mask is None else bool(allow_mask)
    if not can_mask:
        return "delete"
    if seg_act in ("mask", "precise_mask", "delete"):
        return seg_act  # type: ignore[return-value]
    global_act = str(getattr(manifest.clean_cut, "political_action", "delete") or "delete").strip().lower()
    if global_act == "precise_mask":
        return "precise_mask"
    if global_act == "mask":
        return "mask"
    if global_act == "auto":
        rec = str(getattr(seg, "recommended_action", "delete") or "delete").strip().lower()
        return "mask" if rec == "mask" else "delete"
    return "delete"


def resolve_visual_mask_regions(
    manifest: ProjectManifest,
    mask_segments: list[CutSegment],
) -> list[dict[str, Any]]:
    """Return active on-screen visual blur regions [{cut_id, start_s, end_s, boxes}]."""
    _ = manifest
    regions: list[dict[str, Any]] = []
    for seg in mask_segments:
        if not is_visual_ocr_segment(seg) or seg.end_s <= seg.start_s:
            continue
        boxes = [
            {
                "x": round(float(b.x), 4),
                "y": round(float(b.y), 4),
                "w": round(float(b.w), 4),
                "h": round(float(b.h), 4),
            }
            for b in (getattr(seg, "visual_boxes", None) or [])
            if float(b.w) > 0 and float(b.h) > 0
        ]
        if not boxes:
            continue
        regions.append(
            {
                "cut_id": seg.cut_id,
                "start_s": round(float(seg.start_s), 3),
                "end_s": round(float(seg.end_s), 3),
                "boxes": boxes,
            }
        )
    return regions


def resolve_segment_mask_terms(
    seg: CutSegment,
    action: Literal["mask", "precise_mask"],
) -> tuple[list[str], list[str]]:
    """Return (zh_terms, en_terms) active for the segment's resolved masking mode."""
    p_zh = [str(x).strip() for x in (getattr(seg, "precise_terms_zh", None) or []) if str(x).strip()]
    p_en = [str(x).strip() for x in (getattr(seg, "precise_terms_en", None) or []) if str(x).strip()]
    s_zh = [str(x).strip() for x in (seg.sensitive_terms_zh or []) if str(x).strip()]
    s_en = [str(x).strip() for x in (seg.sensitive_terms_en or []) if str(x).strip()]
    if action == "precise_mask":
        if p_zh or p_en or bool(getattr(seg, "precise_terms_extracted", False)):
            return p_zh, p_en
        return s_zh, s_en
    return s_zh, s_en


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
        if resolve_segment_action(seg, manifest, allow_mask=allow_mask) in ("mask", "precise_mask")
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


def _en_base_stems(tok: str) -> set[str]:
    stems = {tok}
    if len(tok) < 4:
        return stems
    if tok.endswith("ies") and len(tok) > 4:
        stems.add(tok[:-3] + "y")
        stems.add(tok[:-1])
    elif tok.endswith("ying") and len(tok) > 4:
        stems.add(tok[:-4] + "ie")
        stems.add(tok[:-3])
    elif tok.endswith("ing") and len(tok) > 5:
        stems.add(tok[:-3])
        stems.add(tok[:-3] + "e")
    elif tok.endswith("ied") and len(tok) > 4:
        stems.add(tok[:-3] + "y")
        stems.add(tok[:-1])
    elif tok.endswith("ed") and len(tok) > 4:
        stems.add(tok[:-2])
        stems.add(tok[:-1])
    elif tok.endswith("es") and len(tok) > 4:
        stems.add(tok[:-2])
        stems.add(tok[:-1])
    elif tok.endswith("s") and not tok.endswith("ss"):
        stems.add(tok[:-1])
    return stems


def _tokens_match(word_tok: str, term_tok: str) -> bool:
    if word_tok == term_tok:
        return True
    if len(term_tok) >= 3 and (word_tok == term_tok + "s" or term_tok == word_tok + "s"):
        return True
    if min(len(word_tok), len(term_tok)) >= 4 and (_en_base_stems(word_tok) & _en_base_stems(term_tok)):
        return True
    return False


def _collect_search_en_terms(
    seg: CutSegment,
    terms_zh: list[str] | None = None,
    terms_en: list[str] | None = None,
) -> list[str]:
    src_en = seg.sensitive_terms_en if terms_en is None else terms_en
    src_zh = seg.sensitive_terms_zh if terms_zh is None else terms_zh
    terms: list[str] = []
    seen: set[str] = set()
    for t in src_en or []:
        s = str(t or "").strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            terms.append(s)
    if not terms:
        for zt in src_zh or []:
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
        if is_visual_ocr_segment(seg) and seg.start_utt is None:
            continue
        eff_action = resolve_segment_action(seg, manifest, allow_mask=True)
        is_precise = eff_action == "precise_mask"
        terms_zh, terms_en = resolve_segment_mask_terms(seg, "precise_mask" if is_precise else "mask")
        en_terms = _collect_search_en_terms(seg, terms_zh=terms_zh, terms_en=terms_en)
        if not terms_zh and not terms_en:
            if not is_precise:
                raw_spans.append((seg.start_s, seg.end_s))
            continue
        if not en_terms and not is_precise:
            raw_spans.append((seg.start_s, seg.end_s))
            continue

        win_lo = max(0.0, seg.start_s - 0.5)
        win_hi = seg.end_s + 0.5
        win_words = [(tok, ws, we) for tok, ws, we in whisper_words if we >= win_lo and ws <= win_hi]

        unmatched_terms: list[str] = []
        seg_matched = False
        seg_spans: list[tuple[float, float]] = []
        for term in en_terms:
            t_toks = _normalize_en_tokens(term)
            if not t_toks:
                continue
            found_in_whisper = False
            n_t = len(t_toks)
            if len(win_words) >= n_t:
                for i in range(len(win_words) - n_t + 1):
                    if all(_tokens_match(win_words[i + j][0], t_toks[j]) for j in range(n_t)):
                        sp = (
                            max(0.0, win_words[i][1] - 0.04),
                            win_words[i + n_t - 1][2] + 0.05,
                        )
                        raw_spans.append(sp)
                        seg_spans.append(sp)
                        found_in_whisper = True
                        seg_matched = True
            if not found_in_whisper:
                unmatched_terms.append(term)

        seg_utts = (
            [
                u
                for u in utts
                if (
                    seg.start_utt is not None
                    and seg.end_utt is not None
                    and seg.start_utt <= u.id <= seg.end_utt
                )
                or (u.end_s > seg.start_s - 0.3 and u.start_s < seg.end_s + 0.3)
            ]
            if utts
            else []
        )

        if unmatched_terms and seg_utts:
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
                        sp = (max(win_lo, ws), min(win_hi, we))
                        raw_spans.append(sp)
                        seg_spans.append(sp)
                        seg_matched = True

        # Surgical precise_mask fallback: if Whisper/EN left a matched Chinese term utterance uncovered,
        # emit only a tight short proportional window around the term instead of muting the whole segment.
        if is_precise and terms_zh and seg_utts:
            for u in seg_utts:
                zh_txt = u.zh or ""
                if not zh_txt:
                    continue
                has_u_span = any(min(u.end_s, pe) - max(u.start_s, ps) > 0.02 for ps, pe in seg_spans)
                if has_u_span:
                    continue
                L_zh = max(1, len(zh_txt))
                u_dur = max(0.2, u.end_s - u.start_s)
                for tz in terms_zh:
                    pat_zh = re.compile(re.escape(tz), re.IGNORECASE)
                    for m in pat_zh.finditer(zh_txt):
                        center = u.start_s + (((m.start() + m.end()) / 2.0) / L_zh) * u_dur
                        half = min(1.0, max(0.4, (len(tz) * 0.35 + 0.5) / 2.0))
                        ws = max(win_lo, max(0.0, u.start_s - 0.08), center - half)
                        we = min(win_hi, u.end_s + 0.08, center + half)
                        if we - ws > 0.05:
                            sp = (ws, we)
                            raw_spans.append(sp)
                            seg_spans.append(sp)
                            seg_matched = True

        if not is_precise and not seg_matched and not terms_en:
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


def redact_ass_dialogue_text(
    text: str,
    terms_zh: list[str],
    terms_en: list[str],
    *,
    allow_blanket: bool = True,
) -> str:
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
        if not allow_blanket:
            return text
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
    *,
    allow_visual_passthrough: bool = False,
) -> Path | None:
    """Create a temporary masked ASS copy in work_dir without modifying source subtitles."""
    if not mask_segments:
        return None
    text_segs = [seg for seg in mask_segments if not is_visual_ocr_segment(seg)]
    if not text_segs and not allow_visual_passthrough:
        return None
    base_ass = _resolve_base_ass_path(manifest, work_dir)
    if base_ass is None or not base_ass.is_file():
        return None
    if not text_segs:
        return base_ass

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
                        for seg in text_segs
                        if min(ev_e, seg.end_s) - max(ev_s, seg.start_s) > 0.05
                    ]
                    if hitting:
                        zh_terms: list[str] = []
                        en_terms: list[str] = []
                        want_blanket = False
                        for seg in hitting:
                            eff_act = resolve_segment_action(seg, manifest, allow_mask=True)
                            mode_key: Literal["mask", "precise_mask"] = (
                                "precise_mask" if eff_act == "precise_mask" else "mask"
                            )
                            t_zh, t_en = resolve_segment_mask_terms(seg, mode_key)
                            if mode_key == "mask" and not t_zh and not t_en:
                                want_blanket = True
                            zh_terms.extend(t_zh)
                            en_terms.extend(t_en)
                        if not en_terms and zh_terms:
                            for zt in zh_terms:
                                for zh_k, en_l in _DEFAULT_ZH_TO_EN_TERMS.items():
                                    if zh_k in zt or zt in zh_k:
                                        en_terms.extend(en_l)
                        parts[-1] = redact_ass_dialogue_text(
                            ev_text,
                            zh_terms,
                            en_terms,
                            allow_blanket=want_blanket,
                        )
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
    out: list[dict[str, Any]] = []
    for seg in hitting:
        eff_act = resolve_segment_action(seg, manifest, allow_mask=True)
        mode_key: Literal["mask", "precise_mask"] = (
            "precise_mask" if eff_act == "precise_mask" else "mask"
        )
        is_vis_only = is_visual_ocr_segment(seg) and seg.start_utt is None
        t_zh, t_en = resolve_segment_mask_terms(seg, mode_key)
        item_payload: dict[str, Any] = {
            "cut_id": seg.cut_id,
            "mode": mode_key,
            "start_s": round(seg.start_s, 3),
            "end_s": round(seg.end_s, 3),
            "terms_zh": sorted({str(x).strip() for x in t_zh if str(x).strip()}) if not is_vis_only else [],
            "terms_en": sorted({str(x).strip() for x in t_en if str(x).strip()}) if not is_vis_only else [],
            "audio_spans": [[round(s, 3), round(e, 3)] for s, e in audio_spans if e > start_s and s < end_s],
        }
        vis_boxes = [
            [round(float(b.x), 4), round(float(b.y), 4), round(float(b.w), 4), round(float(b.h), 4)]
            for b in (getattr(seg, "visual_boxes", None) or [])
        ]
        if vis_boxes:
            item_payload["visual_boxes"] = vis_boxes
        out.append(item_payload)
    return out


def describe_applied_filter(
    manifest: ProjectManifest,
    spans: list[tuple[float, float]] | None = None,
    cfg: dict[str, Any] | None = None,
) -> str:
    """Summarize the content filter state applied to target intervals (or whole project if spans is None)."""
    cc = manifest.clean_cut
    sens_map = {"strict": "严格", "balanced": "平衡", "loose": "红线"}
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
    precise_pol_cnt = sum(
        1
        for s in active_segs
        if s.kind == "political_sensitive" and resolve_segment_action(s, manifest) == "precise_mask"
    )

    parts: list[str] = []
    if ad_cnt > 0:
        parts.append("去广")
    any_mask_cnt = mask_pol_cnt + precise_pol_cnt
    if del_pol_cnt > 0 and any_mask_cnt > 0:
        parts.append(f"脱敏(智能·{sens_zh})")
    elif precise_pol_cnt > 0 and mask_pol_cnt > 0:
        parts.append(f"脱敏(混合遮蔽·{sens_zh})")
    elif precise_pol_cnt > 0:
        parts.append(f"脱敏(精准遮蔽·{sens_zh})")
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
    proj_crop = resolve_effective_crop(manifest)
    payload: dict[str, Any] = {
        "video": manifest.source.render_video,
        "title": cc.title,
        "cover_title": cc.cover_title,
        "sensitivity": cc.sensitivity,
        "political_action": cc.political_action,
        "cut_mode": cc.cut_mode,
        "precise_copy": should_use_precise_copy(manifest, cfg) and (proj_crop is None),
        "cut_spans": [[round(s, 3), round(e, 3)] for s, e in get_active_delete_spans(manifest, cfg)],
        "masks": get_overlapping_mask_payload(0.0, max(1.0, dur), manifest, cfg),
    }
    if (cp := crop_hash_payload(proj_crop)) is not None:
        payload["crop"] = cp
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
            crop_rect=resolve_effective_crop(manifest, clip),
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
   - 而直接评论或影射**中国最高领导人（如 Xi / 习近平）、中国特色社会主义制度/核心地位、分裂国家与不规范法定表述（如 `mainland China`/`Mainland China` 误译、`南京大屠杀纪念馆`、`新疆维吾尔族自治区`）、损害国歌/国旗/国徽/人民币/英烈/党员/人民军队/武警/国安（国安部/MSS）/公安/司法形象、国家版图缺失或台海军事打击/中国政权更迭**的段落必须标为 `high`。
   - 若整块字幕无实质涉政指责或红线内容，**直接返回空数组，严禁凑数输出中性提及中国/企业的 low 片段**；每条命中必须写出具体指责对象与违规原因（`reason` 不可为空）。

   【脱敏策略推荐（`recommended_action`）与两级精确词表要求】：
   - `"recommended_action": "mask"`（遮蔽敏感词字幕+哔音）：适用于正常科技/商业/工程讨论中**仅顺带提及敏感地名/国别/机构/不规范用语**（删掉或遮住 1~2 个词，整句技术/商业含义依然成立），例如 "Chinese destroyer"、"China's chip policy"、"Taiwan" 作为例子顺带提到或单独错用 `mainland China`；务必在 `sensitive_terms_zh` / `sensitive_terms_en` 列出该段字幕中实际出现的所有需打码词。
   - `"recommended_action": "delete"`（整段删除）：适用于整段或连续数句**围绕中国最高领导人评价与外交姿态点评、政权倒台/领导人评价、台海军事威慑、国安军警/体制政治攻击展开论证**，仅打码姓氏/人名仍可通过上下文代词还原语境的片段（仍建议列出 `sensitive_terms_zh` / `sensitive_terms_en`）。广告段固定填 `"delete"` 且词表均填 `[]`。
   - **重要**：每条 `political_sensitive`（包括 `recommended_action="delete"`）还必须额外提取 **精准模式词表**（`precise_terms_zh` / `precise_terms_en`），遵循以下铁律：
{PRECISE_MASK_RULES_PROMPT}"""
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
- **代词延续完整性铁律**：当某句提及**中国最高领导人**或触发法定硬红线（制度/主权/国家荣誉与国安军警形象），且后续连续句子仍用代词（`他` / `他们` / `he` / `him` / `they`）延续评价其态度、外交礼遇或行为（如“他之前并不尊重……后来他到飞机迎接我……”）时，**`end_utt` 必须连续延伸至该人物/事件点评彻底结束、话题切换前的最后一句**，严禁在代词延续句中途截断。
- 连续的同类整段敏感论战或广告片段应合并为单一区间，不要碎片化拆分。

【输出 JSON Schema】
{{
  "segments": [
    {{
      "kind": "ad | political_sensitive",
      "subtype": "sponsor | midroll | outro | intro | hard_redline | sovereignty_and_normative_terms | national_honor_and_public_service | geopolitics_confrontation | policy_sensitive",
      "brand": "仅 ad 填写具体赞助品牌/产品/平台名（无具体品牌严禁标为广告）；political_sensitive 填 \"\"",
      "target_country": "仅 political_sensitive 填字幕上下文实际谈及的目标国家/地区（如 中国 / 台海 / 俄罗斯 / 朝鲜 / 伊朗 / 伊斯兰世界 等）；若实为泛指政府、美国/欧美或商业公司，填 \"非目标\"",
      "is_explicit_criticism": true,
      "quote": "从对应 [uXXXX] 字幕逐字摘录的触发原句（多句可用 首句...尾句）",
      "start_utt": 12,
      "end_utt": 19,
      "severity": "high | medium | low",
      "recommended_action": "delete | mask",
      "sensitive_terms_zh": ["常规遮蔽需打码的中文原文词（从上述中文字幕原样摘录）"],
      "sensitive_terms_en": ["term for normal mask verbatim from English subtitle"],
      "precise_terms_zh": ["精准模式仅遮蔽政治体/领导人/不规范中港台称谓/定性判断词的中文原文"],
      "precise_terms_en": ["surgical precise-mode keyword verbatim from English subtitle"],
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
    if re.fullmatch(r"[A-Za-z0-9\s.'’\-]+", t):
        pat = _compile_en_term_pattern(t)
        if pat is None:
            return False
        return bool(pat.search(en) or pat.search(zh))
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
        r"中国|中共|共产党|北京|中南海|习近平|习主席|台湾|台海|香港|新疆|西藏|南海|解放军|武警|国安部|公安|司法|烈士|英烈|革命领袖|国旗|国歌|国徽|人民币|南京大屠杀|新疆维吾尔|宪法|一国两制|一个中国|中国大陆|党史|军史|国史|天安门|文革|法轮功|达赖|维吾尔"
    ),
    re.compile(r"俄罗斯|俄国|普京|莫斯科|苏联|朝鲜|平壤|金正恩|伊朗|德黑兰|哈梅内伊|伊斯兰|穆斯林|巴勒斯坦|哈马斯|叙利亚|阿富汗|塔利班|巴基斯坦|古巴|委内瑞拉|白俄罗斯|塞尔维亚"),
    re.compile(
        r"(?<![a-z0-9])(china|chinese|ccp|prc|beijing|mainland\s+china|taiwan|taiwanese|hong\s+kong|xinjiang|tibet|tibetan|south\s+china\s+sea|pla|mss|one\s+china|nanjing\s+massacre|renminbi|rmb|xi\s+jinping|xi|uyghur|tiananmen|dalai|russia|russian|putin|moscow|north\s+korea|pyongyang|kim\s+jong|iran|iranian|tehran|khamenei|islam|islamic|muslim|palestine|palestinian|hamas|syria|afghanistan|taliban|pakistan|cuba|venezuela|belarus|serbia)(?![a-z0-9])",
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
    """Verify AI political hit targets China/friendly countries or matches hard redlines regardless of tone."""
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

    # Zero-tolerance hard redlines & statutory Bilibili redlines stay valid even if non-critical
    sev_norm = str(seg.get("severity") or "").strip().lower()
    sub_norm = str(seg.get("subtype") or "").strip().lower()
    if sev_norm == "high" and match_hard_redline_text(ctx_text):
        return True
    if (
        sev_norm == "high"
        and sub_norm in ("hard_redline", "sovereignty_and_normative_terms", "national_honor_and_public_service")
        and any(pat.search(ctx_text) for pat in _POLITICAL_TARGET_PATTERNS)
    ):
        return True

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


def _select_compliance_llm(
    cfg: dict[str, Any] | None,
    *,
    prefer_gemini: bool = False,
) -> tuple[ClipCutLLM, str, str, list[str]]:
    del prefer_gemini  # Auto-switching to Gemini is disabled; follow configured models strictly.
    models_cfg = (cfg or {}).get("models", {})
    text_provider = str(models_cfg.get("text_provider") or "dashscope")
    text_model = str(models_cfg.get("text_model") or "qwen3.8-max")
    p_main, m_main = parse_provider_model(text_model, text_provider)
    primary_spec = f"{p_main}:{m_main}"
    raw_fallbacks = list(models_cfg.get("text_fallback_models") or DEFAULT_TEXT_FALLBACK_MODELS)

    normalized_text_fallbacks: list[str] = []
    for fb in raw_fallbacks:
        if not fb:
            continue
        p_fb, m_fb = parse_provider_model(str(fb), p_main)
        spec = f"{p_fb}:{m_fb}"
        if spec != primary_spec and spec not in normalized_text_fallbacks:
            normalized_text_fallbacks.append(spec)

    llm = ClipCutLLM(models_cfg)
    return llm, p_main, m_main, normalized_text_fallbacks


def _clean_term_list(raw: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in raw or []:
        if not isinstance(x, (str, int, float)):
            continue
        s = str(x).strip()
        if not s:
            continue
        low = s.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(s)
    return out


def _scan_deterministic_hard_redlines(utts: list[Utterance]) -> list[dict[str, Any]]:
    """Deterministic zero-tolerance subtitle guard ensuring hard redlines are never missed by LLM."""
    guard_hits: list[dict[str, Any]] = []
    for u in utts:
        hits_zh = match_hard_redline_text(u.zh or "")
        hits_en = match_hard_redline_text(u.en or "")
        if not hits_zh and not hits_en:
            continue
        t_zh: list[str] = []
        seen_zh: set[str] = set()
        for h in hits_zh:
            m = h["matched"]
            if m.lower() not in seen_zh:
                seen_zh.add(m.lower())
                t_zh.append(m)
        t_en: list[str] = []
        seen_en: set[str] = set()
        for h in hits_en:
            m = h["matched"]
            if m.lower() not in seen_en:
                seen_en.add(m.lower())
                t_en.append(m)
        labels: list[str] = []
        for h in (*hits_zh, *hits_en):
            if h["label"] not in labels:
                labels.append(h["label"])
        guard_hits.append(
            {
                "kind": "political_sensitive",
                "subtype": "hard_redline_guard",
                "su": u.id,
                "eu": u.id,
                "severity": "high",
                "recommended_action": "mask",
                "terms_zh": list(t_zh),
                "terms_en": list(t_en),
                "precise_zh": list(t_zh),
                "precise_en": list(t_en),
                "reason": f"硬红线词兜底命中（{'、'.join(labels)}）",
            }
        )
    return guard_hits


def _filter_terms_in_utterances(terms: list[str], utts_slice: list[Utterance]) -> list[str]:
    if not terms or not utts_slice:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw_t in terms:
        t = str(raw_t or "").strip()
        if not t or t.lower() in seen:
            continue
        if any(_term_in_utterance(t, u) for u in utts_slice):
            seen.add(t.lower())
            out.append(t)
    return out


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
    cfg_sens = str(cf_cfg.get("sensitivity") or "strict").strip().lower()
    if cfg_sens not in ("strict", "balanced", "loose"):
        cfg_sens = "strict"
    if sensitivity:
        eff_sens = str(sensitivity).strip().lower()
    elif manifest.clean_cut.scanned_at is None:
        eff_sens = cfg_sens
    else:
        eff_sens = str(manifest.clean_cut.sensitivity or cfg_sens).strip().lower()
    if eff_sens not in ("strict", "balanced", "loose"):
        eff_sens = cfg_sens

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
        if not get_active_cut_segments(manifest, cfg) and resolve_effective_crop(manifest) is None:
            manifest.clean_cut.enabled = False
        manifest.clean_cut.scanned_at = utc_now_iso()
        sync_all_render_hashes(manifest, cfg)
        store.save(manifest)
        return manifest

    manual_segments = [seg.model_copy(deep=True) for seg in manifest.cut_segments if seg.detector == "manual"]

    llm, scan_provider, scan_model, scan_fallbacks = _select_compliance_llm(cfg, prefer_gemini=False)

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
                terms_zh = _clean_term_list(seg.get("sensitive_terms_zh"))
                terms_en = _clean_term_list(seg.get("sensitive_terms_en"))
                precise_zh = _clean_term_list(seg.get("precise_terms_zh"))
                precise_en = _clean_term_list(seg.get("precise_terms_en"))
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
                    [*terms_zh, *precise_zh] if kind == "political_sensitive" else [],
                    [*terms_en, *precise_en] if kind == "political_sensitive" else [],
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
                seg_utts = [u for u in utts if su <= u.id <= eu]
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
                        "precise_zh": _filter_terms_in_utterances(precise_zh, seg_utts)
                        if kind == "political_sensitive"
                        else [],
                        "precise_en": _filter_terms_in_utterances(precise_en, seg_utts)
                        if kind == "political_sensitive"
                        else [],
                        "reason": raw_reason or ("广告内容" if kind == "ad" else "涉政/合规风险"),
                    }
                )
            store.log(f"[clean_cut] AI 扫描分块 {ch_idx + 1}/{len(chunks)} 完成（本块命中 {len(segs)} 处）")

    if eff_pol:
        guard_hits = _scan_deterministic_hard_redlines(utts)
        if guard_hits:
            raw_ai_hits.extend(guard_hits)
            store.log(f"[clean_cut] 硬红线字幕兜底命中 {len(guard_hits)} 句，已纳入强脱敏清单")

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
                for key in ("terms_zh", "terms_en", "precise_zh", "precise_en"):
                    for val in h.get(key) or []:
                        if val not in prev[key]:
                            prev[key].append(val)
                if h["reason"] and h["reason"] not in prev["reason"]:
                    prev["reason"] = f"{prev['reason']}；{h['reason']}"[:160]
            else:
                deduped_ai.append(
                    {
                        **h,
                        "terms_zh": list(h.get("terms_zh") or []),
                        "terms_en": list(h.get("terms_en") or []),
                        "precise_zh": list(h.get("precise_zh") or []),
                        "precise_en": list(h.get("precise_en") or []),
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
                precise_terms_zh=list(item.get("precise_zh") or []),
                precise_terms_en=list(item.get("precise_en") or []),
                precise_terms_extracted=(item["kind"] == "political_sensitive"),
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

    # On-screen embedded banner/title OCR scan for hard redlines
    check_cancelled()
    visual_ocr_count = 0
    if eff_pol:
        clean_exists = has_clean_source_video(manifest)
        ocr_video = (
            Path(manifest.source.analysis_video).expanduser()
            if clean_exists
            else Path(manifest.source.render_video).expanduser()
        )
        vis_ocr_hits = detect_visual_ocr_redline_segments(
            ocr_video,
            dur,
            exclude_bottom_subtitle_band=not clean_exists,
            log_fn=store.log,
        )
        for vh in vis_ocr_hits:
            vis_boxes = [VisualMaskBox(**b) for b in (vh.get("visual_boxes") or [])]
            if not vis_boxes:
                continue
            vs = round(float(vh["start_s"]), 3)
            ve = round(float(vh["end_s"]), 3)
            visual_ocr_count += 1
            scanned_segments.append(
                CutSegment(
                    cut_id="",
                    kind="political_sensitive",
                    subtype="onscreen_text",
                    enabled=True,
                    detector="visual_ocr",
                    action="mask",
                    recommended_action="mask",
                    precise_terms_extracted=True,
                    visual_boxes=vis_boxes,
                    start_utt=None,
                    end_utt=None,
                    start_s=vs,
                    end_s=ve,
                    quote=str(vh.get("quote") or ""),
                    reason=f"画面文字命中硬红线（{vh.get('matched_labels', '硬红线')}）：局部模糊画面文字，保留原声与字幕",
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
    manifest.clean_cut.enabled = bool(get_active_cut_segments(manifest, cfg)) or (
        resolve_effective_crop(manifest) is not None and manifest.clean_cut.enabled
    )
    if all_segments:
        from .verify_and_plan import (
            build_default_clean_cut_title,
            distill_clean_cut_title,
            is_untranslated_clean_cut_title,
        )

        if is_untranslated_clean_cut_title(manifest, manifest.clean_cut.title):
            cand_title = build_default_clean_cut_title(manifest)
            if is_untranslated_clean_cut_title(manifest, cand_title):
                cand_title = distill_clean_cut_title(
                    manifest,
                    None,
                    cfg,
                    run_id=run_id,
                    llm=ClipCutLLM(cfg.get("models", {})),
                )
            manifest.clean_cut.title = cand_title
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
        f"（广告 {ad_cnt} 段，政治敏感 {pol_cnt} 段 [含画面文字 {visual_ocr_count} 段]，共 {total_cut_s:.1f}s，标准: {eff_sens}）"
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
            "visual_ocr_count": visual_ocr_count,
            "manual_preserved": len(manual_segments),
            "total_cut_seconds": round(total_cut_s, 1),
            "sensitivity": eff_sens,
        },
    )
    return manifest


def _segment_overlapping_utterances(manifest: ProjectManifest, seg: CutSegment) -> list[Utterance]:
    if not manifest.utterances:
        return []
    if seg.start_utt is not None and seg.end_utt is not None:
        su = max(0, min(seg.start_utt, seg.end_utt))
        eu = max(seg.start_utt, seg.end_utt)
        hits = [u for u in manifest.utterances if su <= u.id <= eu]
        if hits:
            return hits
    return [
        u
        for u in manifest.utterances
        if u.end_s > seg.start_s - 0.3 and u.start_s < seg.end_s + 0.3
    ]


def extract_precise_terms_for_segments(
    manifest: ProjectManifest,
    store: ProjectStore | None,
    cfg: dict[str, Any] | None = None,
    run_id: str = "precise_terms",
    *,
    cut_ids: list[str] | None = None,
    force: bool = False,
) -> tuple[ProjectManifest, int]:
    """Extract surgical keyword-only redaction/mute terms (`precise_terms_*`) for existing political segments."""
    t0 = time.monotonic()
    resolved_cfg = _safe_load_cfg(cfg)
    want_ids = {str(cid).strip() for cid in (cut_ids or []) if str(cid).strip()} if cut_ids else None
    targets: list[CutSegment] = []
    for seg in manifest.cut_segments:
        if seg.kind != "political_sensitive" or is_visual_ocr_segment(seg):
            continue
        if want_ids is not None and seg.cut_id not in want_ids:
            continue
        has_existing = bool(seg.precise_terms_extracted or seg.precise_terms_zh or seg.precise_terms_en)
        if not force and has_existing:
            continue
        targets.append(seg)

    if not targets:
        return manifest, 0

    seg_blocks: list[str] = []
    seg_utts_by_id: dict[str, list[Utterance]] = {}
    for seg in targets:
        seg_utts = _segment_overlapping_utterances(manifest, seg)
        seg_utts_by_id[seg.cut_id] = seg_utts
        if seg_utts:
            lines = "\n".join(
                f"  [u{u.id:04d} {_fmt_ts(u.start_s)}–{_fmt_ts(u.end_s)}] {u.zh or ''} | {u.en or ''}".strip()
                for u in seg_utts
            )
        else:
            lines = f"  [{_fmt_ts(seg.start_s)}–{_fmt_ts(seg.end_s)}] {seg.quote}"
        seg_blocks.append(
            f"### Segment `{seg.cut_id}` ({_fmt_ts(seg.start_s)}–{_fmt_ts(seg.end_s)}, severity={seg.severity})\n"
            f"- 违规说明: {seg.reason or '涉政敏感'}\n"
            f"- 既有宽泛词表参考: zh={seg.sensitive_terms_zh} / en={seg.sensitive_terms_en}\n"
            f"- 对话原文:\n{lines}"
        )

    prompt = f"""请为以下已标出的涉政/合规片段提取**精准遮蔽与消音模式（Surgical Keyword Mode）**的最小核心关键词清单。

{PRECISE_MASK_RULES_PROMPT}

【附加实操提示】
- 你只输出每个片段字幕中**真正必须遮蔽/消音的少数核心关键词**（例如 `习主席`/`Xi Jinping`、`政权`/`regime`、`侵略者`/`aggressor`、`入侵`/`invaded`、`清洗`/`purges`、`台湾总统`/`中华民国`、`强征`/`conscripted`、`台湾`/`Taiwan` 等具体构成主权/领导人/定性行为风险的词）；
- 若整段只是背景叙事或泛谈战略（不包含上述 3 类核心词），该段可以输出空数组 `[]`；
- 禁止把中性国名（如纯地名 `俄罗斯`/`Russia`、`乌克兰`/`Ukraine`、`伊朗`/`Iran`、`中国`/`China`）与正常商业科技词混入精准词表，除非该词本身就是不规范主权并列称谓的核心组成部分。

【待提取片段】
{chr(10).join(seg_blocks)}

【输出 JSON Schema】
{{
  "items": [
    {{
      "cut_id": "cut_01",
      "precise_terms_zh": ["逐字摘自中文字幕的核心敏感词"],
      "precise_terms_en": ["verbatim English keyword from English subtitle"]
    }}
  ]
}}"""

    llm, provider, model, fallbacks = _select_compliance_llm(resolved_cfg, prefer_gemini=False)
    res = llm.generate_json(
        prompt=prompt,
        model=model,
        provider=provider,
        fallback_models=fallbacks,
        system_instruction="你是严格执行最小影响原则的视频合规字幕/音轨词级脱敏提取器，只输出合法 JSON。",
        temperature=0.05,
        max_output_tokens=8192,
    )
    used_model_label = f"{getattr(res, 'provider', provider)}:{getattr(res, 'model', model)}"
    parsed = res.parsed if isinstance(res.parsed, dict) else {}
    items = parsed.get("items") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        items = []

    by_id = {seg.cut_id: seg for seg in targets}
    updated_segs: list[CutSegment] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("cut_id") or "").strip()
        seg = by_id.get(cid)
        if seg is None:
            continue
        seg_utts = seg_utts_by_id.get(cid, [])
        raw_zh = _clean_term_list(item.get("precise_terms_zh"))
        raw_en = _clean_term_list(item.get("precise_terms_en"))
        if seg_utts:
            seg.precise_terms_zh = _filter_terms_in_utterances(raw_zh, seg_utts)
            seg.precise_terms_en = _filter_terms_in_utterances(raw_en, seg_utts)
        else:
            seg.precise_terms_zh = raw_zh
            seg.precise_terms_en = raw_en
        seg.precise_terms_extracted = True
        updated_segs.append(seg)
    for seg in targets:
        seg.precise_terms_extracted = True

    sync_all_render_hashes(manifest, resolved_cfg)
    if store is not None:
        store.save(manifest)
        mute_spans = resolve_mask_audio_spans(
            manifest,
            [
                seg.model_copy(update={"action": "precise_mask"})
                for seg in updated_segs
            ],
        )
        mute_s = sum(max(0.0, e - s) for s, e in mute_spans)
        raw_s = sum(max(0.0, seg.end_s - seg.start_s) for seg in updated_segs)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        summary = (
            f"精准脱敏关键词提取完成：更新 {len(updated_segs)}/{len(targets)} 段"
            f"（原段总长 {raw_s:.1f}s -> 精准消音约 {mute_s:.2f}s）"
        )
        store.log(f"[clean_cut] {summary}")
        store.append_receipt(
            run_id=run_id,
            stage="clean_cut",
            actor="ai",
            action="extract_precise_terms",
            model=used_model_label,
            elapsed_ms=elapsed_ms,
            summary=summary,
            details={
                "updated_count": len(updated_segs),
                "target_count": len(targets),
                "raw_seconds": round(raw_s, 1),
                "precise_mute_seconds": round(mute_s, 2),
            },
        )
    return manifest, len(updated_segs)


def ensure_precise_terms_if_needed(
    manifest: ProjectManifest,
    store: ProjectStore | None,
    cfg: dict[str, Any] | None = None,
) -> None:
    """Best-effort extract surgical terms before rendering when any active precise_mask segment lacks them."""
    if store is None:
        return
    missing_ids = [
        seg.cut_id
        for seg in get_active_mask_segments(manifest, cfg)
        if not is_visual_ocr_segment(seg)
        and resolve_segment_action(seg, manifest, allow_mask=True) == "precise_mask"
        and not seg.precise_terms_extracted
        and not seg.precise_terms_zh
        and not seg.precise_terms_en
    ]
    if not missing_ids:
        return
    try:
        extract_precise_terms_for_segments(
            manifest,
            store,
            cfg,
            run_id=f"precise_auto_{int(time.time())}",
            cut_ids=missing_ids,
            force=False,
        )
    except Exception as exc:
        store.log(f"[clean_cut] 精准遮蔽关键词自动提取失败，将仅按已填写关键词执行: {exc}")
