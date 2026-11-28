import {
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { api } from "./api";
import VideoCropOverlay, {
  computeEvenOutputDims,
  sanitizeRect,
  type NormalizedCropRect,
} from "./components/VideoCropOverlay";
import type {
  CleanCutConfig,
  ClipCropOverride,
  ClipItem,
  ContentFilterConfig,
  CutSegment,
  NavigationConfig,
  PlatformStatus,
  PoliticalAction,
  PoliticalProfile,
  ProjectManifest,
  ProjectSummary,
  PublishJobInfo,
  PublishRecord,
  Receipt,
  RemixItem,
  RenderedAssetsInfo,
  ScanResult,
  VideoFileCandidate,
} from "./types";

const DEFAULT_SENS_TRIGGERS: Record<string, string[]> = {
  strict: ["high", "medium", "low"],
  balanced: ["high", "medium"],
  loose: ["high"],
};

const SENS_SHORT_LABELS: Record<"strict" | "balanced" | "loose", string> = {
  strict: "从严",
  balanced: "平衡",
  loose: "仅硬红线",
};

function isFullFrameRect(r: NormalizedCropRect): boolean {
  return r.x <= 0.001 && r.y <= 0.001 && r.w >= 0.999 && r.h >= 0.999;
}

function resolveEffectiveCropFrontend(
  manifest: ProjectManifest | null,
  clip: ClipItem | null
): NormalizedCropRect | null {
  if (!manifest) return null;
  if (clip?.crop_override) {
    if (clip.crop_override.mode === "none") return null;
    if (clip.crop_override.mode === "custom") {
      const s = sanitizeRect(clip.crop_override);
      return isFullFrameRect(s) ? null : s;
    }
  }
  if (manifest.crop?.enabled) {
    const s = sanitizeRect(manifest.crop);
    return isFullFrameRect(s) ? null : s;
  }
  return null;
}

interface ComplianceSegmentHit {
  seg: CutSegment;
  effAction: "delete" | "mask" | "precise_mask" | "passed" | "off";
  reasonLabel: string;
}

interface ComplianceSummary {
  adDelS: number;
  polDelS: number;
  maskCount: number;
  maskTermsCount: number;
  maskTermsPreview: string;
  passedBySensCount: number;
  unfilteredPolCount: number;
  unfilteredAdCount: number;
  isFullyCut: boolean;
  netDurS: number;
  rawDurS: number;
  overlappingSegs: ComplianceSegmentHit[];
}

function isVisualOcrSeg(seg: CutSegment): boolean {
  return seg.detector === "visual_ocr" || Boolean(seg.visual_boxes && seg.visual_boxes.length > 0);
}

function resolveSegActionFrontend(
  seg: CutSegment,
  cc: CleanCutConfig | undefined,
  canMask: boolean
): "delete" | "mask" | "precise_mask" {
  if (seg.kind !== "political_sensitive") return "delete";
  if (isVisualOcrSeg(seg)) {
    if (seg.action === "delete") return "delete";
    if (seg.action === "mask" || seg.action === "precise_mask") return seg.action;
    return cc?.political_action === "precise_mask" ? "precise_mask" : "mask";
  }
  if (!canMask) return "delete";
  if (seg.action === "precise_mask" || seg.action === "mask" || seg.action === "delete") return seg.action;
  const g = cc?.political_action || "delete";
  if (g === "precise_mask") return "precise_mask";
  if (g === "mask") return "mask";
  if (g === "auto") return seg.recommended_action === "mask" ? "mask" : "delete";
  return "delete";
}

function getEffectiveMaskTerms(
  seg: CutSegment,
  effAct: "delete" | "mask" | "precise_mask"
): { zh: string[]; en: string[] } {
  const pZh = seg.precise_terms_zh || [];
  const pEn = seg.precise_terms_en || [];
  const sZh = seg.sensitive_terms_zh || [];
  const sEn = seg.sensitive_terms_en || [];
  if (effAct === "precise_mask") {
    if (pZh.length > 0 || pEn.length > 0 || Boolean(seg.precise_terms_extracted)) {
      return { zh: pZh, en: pEn };
    }
    return { zh: sZh, en: sEn };
  }
  if (effAct === "mask") {
    return { zh: sZh, en: sEn };
  }
  return { zh: [], en: [] };
}

function mergeSpanPairs(spans: [number, number][], gap = 0.05): [number, number][] {
  if (spans.length === 0) return [];
  const sorted = [...spans].filter(([s, e]) => e > s).sort((a, b) => a[0] - b[0]);
  const out: [number, number][] = [];
  for (const [s, e] of sorted) {
    const last = out[out.length - 1];
    if (last && s <= last[1] + gap) {
      last[1] = Math.max(last[1], e);
    } else {
      out.push([s, e]);
    }
  }
  return out;
}

function overlapSeconds(intervals: [number, number][], cutSpans: [number, number][]): number {
  if (intervals.length === 0 || cutSpans.length === 0) return 0;
  let total = 0;
  for (const [is, ie] of intervals) {
    const hits: [number, number][] = [];
    for (const [cs, ce] of cutSpans) {
      const os = Math.max(is, cs);
      const oe = Math.min(ie, ce);
      if (oe - os > 0.05) hits.push([os, oe]);
    }
    for (const [hs, he] of mergeSpanPairs(hits, 0.05)) {
      total += he - hs;
    }
  }
  return total;
}

function summarizeIntervalsCompliance(
  intervals: [number, number][],
  segs: CutSegment[],
  cc: CleanCutConfig | undefined,
  matchesSeg: (s: CutSegment) => boolean,
  canMask: boolean
): ComplianceSummary {
  const cleanIntervals = mergeSpanPairs(intervals, 0.01);
  const rawDurS = cleanIntervals.reduce((acc, [s, e]) => acc + (e - s), 0);
  const adSpans: [number, number][] = [];
  const polDelSpans: [number, number][] = [];
  const maskTerms = new Set<string>();
  const overlappingSegs: ComplianceSegmentHit[] = [];

  let maskCount = 0;
  let passedBySensCount = 0;
  let unfilteredPolCount = 0;
  let unfilteredAdCount = 0;

  for (const seg of segs) {
    if (!seg.enabled) continue;
    const touches = cleanIntervals.some(([is, ie]) => Math.min(ie, seg.end_s) - Math.max(is, seg.start_s) > 0.05);
    if (!touches) continue;

    if (seg.kind === "ad") {
      if (cc?.cut_ads) {
        adSpans.push([seg.start_s, seg.end_s]);
        overlappingSegs.push({ seg, effAction: "delete", reasonLabel: "去广·将删" });
      } else {
        unfilteredAdCount += 1;
        overlappingSegs.push({ seg, effAction: "off", reasonLabel: "广告·未勾选去广" });
      }
      continue;
    }

    if (!matchesSeg(seg)) {
      passedBySensCount += 1;
      overlappingSegs.push({ seg, effAction: "passed", reasonLabel: "当前标准已放行" });
      continue;
    }
    if (!cc?.cut_political_sensitive) {
      unfilteredPolCount += 1;
      overlappingSegs.push({ seg, effAction: "off", reasonLabel: "涉敏·未勾选脱敏" });
      continue;
    }

    const act = resolveSegActionFrontend(seg, cc, canMask);
    if (act === "mask" || act === "precise_mask") {
      maskCount += 1;
      if (!isVisualOcrSeg(seg)) {
        const effTerms = getEffectiveMaskTerms(seg, act);
        for (const tz of effTerms.zh) if (tz.trim()) maskTerms.add(tz.trim());
        for (const te of effTerms.en) if (te.trim()) maskTerms.add(te.trim());
      }
      overlappingSegs.push({
        seg,
        effAction: act,
        reasonLabel: isVisualOcrSeg(seg)
          ? "脱敏·画面模糊"
          : act === "precise_mask"
            ? "脱敏·精准遮蔽"
            : "脱敏·常规遮蔽",
      });
    } else {
      polDelSpans.push([seg.start_s, seg.end_s]);
      overlappingSegs.push({ seg, effAction: "delete", reasonLabel: "脱敏·将删段" });
    }
  }

  const mergedAd = mergeSpanPairs(adSpans, 0.05);
  const mergedPolDel = mergeSpanPairs(polDelSpans, 0.05);
  const allDel = mergeSpanPairs([...mergedAd, ...mergedPolDel], 0.05);
  const adDelS = overlapSeconds(cleanIntervals, mergedAd);
  const polDelS = overlapSeconds(cleanIntervals, mergedPolDel);
  const totalDelS = overlapSeconds(cleanIntervals, allDel);
  const netDurS = Math.max(0, rawDurS - totalDelS);
  const isFullyCut = rawDurS > 0.5 && (netDurS < 0.4 || (allDel.length > 0 && netDurS < Math.min(1.0, rawDurS * 0.1)));
  const previewArr = Array.from(maskTerms);

  return {
    adDelS,
    polDelS,
    maskCount,
    maskTermsCount: previewArr.length,
    maskTermsPreview: previewArr.slice(0, 6).join(" / "),
    passedBySensCount,
    unfilteredPolCount,
    unfilteredAdCount,
    isFullyCut,
    netDurS,
    rawDurS,
    overlappingSegs,
  };
}

function renderComplianceTags(comp: ComplianceSummary, isScanned: boolean) {
  if (!isScanned) return null;
  if (comp.isFullyCut) {
    return (
      <span
        className="tag"
        style={{ background: "rgba(239, 68, 68, 0.24)", color: "#fecaca", borderColor: "rgba(239, 68, 68, 0.55)" }}
        title="全部内容落入当前生效的广告或删段脱敏区间，渲染时将跳过"
      >
        整段命中·将跳过
      </span>
    );
  }
  const tags: ReactNode[] = [];
  if (comp.adDelS > 0.3) {
    tags.push(
      <span
        key="ad"
        className="tag"
        style={{ background: "rgba(245, 158, 11, 0.18)", color: "#fcd34d" }}
        title="渲染时将自动剪除重叠广告片段"
      >
        去广 -{Math.max(1, Math.round(comp.adDelS))}s
      </span>
    );
  }
  if (comp.polDelS > 0.3) {
    tags.push(
      <span
        key="pdel"
        className="tag"
        style={{ background: "rgba(239, 68, 68, 0.18)", color: "#fca5a5" }}
        title="渲染时将直接删除重叠的政治敏感片段"
      >
        脱敏·删 -{Math.max(1, Math.round(comp.polDelS))}s
      </span>
    );
  }
  if (comp.maskCount > 0) {
    tags.push(
      <span
        key="pmask"
        className="tag"
        style={{ background: "rgba(56, 189, 248, 0.18)", color: "#7dd3fc" }}
        title={
          comp.maskTermsPreview
            ? `将在干净原片重烧打码字幕并对敏感词哔音消音（${comp.maskTermsPreview}）`
            : "将在干净原片重烧脱敏字幕并消音"
        }
      >
        脱敏·遮蔽 {comp.maskCount}处
      </span>
    );
  }
  if (comp.unfilteredPolCount > 0) {
    tags.push(
      <span
        key="unpol"
        className="tag"
        style={{ background: "rgba(251, 191, 36, 0.16)", color: "#fde68a" }}
        title="当前未勾选「去除政治敏感」，敏感内容将保留在成品中"
      >
        含敏·未脱敏
      </span>
    );
  }
  if (comp.unfilteredAdCount > 0) {
    tags.push(
      <span
        key="unad"
        className="tag"
        style={{ background: "rgba(251, 191, 36, 0.16)", color: "#fde68a" }}
        title="当前未勾选「去除广告」，广告片段将保留在成品中"
      >
        含广·未去广
      </span>
    );
  }
  if (tags.length === 0 && comp.passedBySensCount > 0) {
    tags.push(
      <span
        key="passed"
        className="tag"
        style={{ background: "rgba(148, 163, 184, 0.18)", color: "#cbd5e1" }}
        title="检测到低/中危敏感讨论，已按当前政审标准自动放行"
      >
        含敏·已放行{comp.passedBySensCount}段
      </span>
    );
  }
  if (tags.length === 0) {
    tags.push(
      <span key="clean" className="tag" title="经全片合规扫描，本条目不含广告或命中的敏感片段">
        无广/无敏
      </span>
    );
  }
  return tags;
}

function getRenderBadge(
  item: {
    output_file?: string | null;
    render_hash: string;
    last_rendered_hash: string;
    rendered_filter_tag?: string;
  },
  comp: ComplianceSummary,
  isScanned: boolean,
  emptyLabel: string
): { label: string; cls: "cached" | "dirty"; title: string } {
  if (item.output_file && item.render_hash === item.last_rendered_hash) {
    if (!isScanned) {
      return { label: "已渲染", cls: "cached", title: "已完成渲染" };
    }
    const hasAd = comp.adDelS > 0.3;
    const hasPol = comp.polDelS > 0.3 || comp.maskCount > 0;
    const suffix =
      hasAd && hasPol
        ? "·去广脱敏"
        : hasAd
          ? "·已去广"
          : hasPol
            ? "·已脱敏"
            : "";
    return {
      label: `已渲染${suffix}`,
      cls: "cached",
      title: item.rendered_filter_tag || "已按当前策略完成渲染",
    };
  }
  if (item.output_file) {
    return {
      label: "待重渲",
      cls: "dirty",
      title: item.rendered_filter_tag
        ? `当前去广/脱敏或参数与磁盘成品（${item.rendered_filter_tag}）不一致，需重渲生效`
        : "当前去广/脱敏策略或片段参数已变更，需重渲生效",
    };
  }
  return { label: emptyLabel, cls: "dirty", title: "尚未渲染" };
}

const STAGE_SHORT_LABELS: Record<string, string> = {
  ingest: "载入",
  signals: "信号",
  scan_cuts: "广敏扫描",
  analyze: "分析",
  verify_and_plan: "复核",
  render: "渲染",
};

const CAT_LABELS: Record<string, string> = {
  key_dialogue: "重点对话",
  exciting_dialogue: "精彩对话",
  hot_topic: "中国热点",
};

const BEAT_LABELS: Record<string, string> = {
  punchline: "论断",
  qa_argument: "问答论证",
  case_demo: "案例演示",
};

const PLAT_LABELS: Record<string, string> = {
  wechat: "视频号",
  bilibili: "B站",
  xiaohongshu: "小红书",
};

function fmtTime(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(r)}` : `${pad(m)}:${pad(r)}`;
}

function fmtSize(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function hasCjkText(text: string): boolean {
  return /[\u4e00-\u9fff]/.test(text || "");
}

function formatProgramContext(manifest: ProjectManifest): string {
  const guest = (manifest.metadata.guest_zh || "").trim();
  const channel = (manifest.metadata.channel_zh || manifest.metadata.channel || "").trim();
  const theme = (manifest.metadata.original_theme_zh || "").trim();
  if (guest && channel) return `${guest}做客${channel}的访谈`;
  if (channel) return `${channel}本期节目`;
  if (guest) return `${guest}本期访谈`;
  if (theme) return `《${theme}》`;
  return "本期节目";
}

function pickSpreadItems<T>(items: T[], maxCount: number): T[] {
  if (maxCount <= 0 || items.length === 0) return [];
  if (items.length <= maxCount) return [...items];
  if (maxCount === 1) return [items[0]];
  const indices: number[] = [];
  for (let i = 0; i < maxCount; i += 1) {
    const idx = Math.round((i * (items.length - 1)) / (maxCount - 1));
    if (!indices.includes(idx)) indices.push(idx);
  }
  return indices.map((idx) => items[idx]);
}

function dedupeLabels(labels: string[], maxCount: number): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const raw of labels) {
    const s = (raw || "").trim();
    if (!s) continue;
    const key = s.replace(/\s+/g, "").toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(s);
    if (out.length >= maxCount) break;
  }
  return out;
}

function buildDefaultCleanCutDesc(manifest: ProjectManifest): string {
  const cached = (manifest.clean_cut?.publish_desc || "").trim();
  if (cached) return cached;
  const fullSummary = (manifest.metadata.full_summary_zh || "").trim();
  if (fullSummary) return fullSummary;

  const lead = formatProgramContext(manifest);
  const theme = (manifest.metadata.original_theme_zh || "").trim();
  let opening = `本期为${lead}完整内容。`;
  if (theme && !lead.includes(theme)) {
    opening =
      lead.endsWith("访谈") || lead.endsWith("节目")
        ? `本期为${lead}完整版，核心围绕“${theme}”展开深入对话。`
        : `本期带来${lead}完整内容，核心围绕“${theme}”展开深入对话。`;
  }

  const topics = (manifest.topic_map || []).filter(
    (t) => !t.is_filler && ((t.title || "").trim() || (t.summary || "").trim())
  );
  let detailLine = "";
  if (topics.length > 0) {
    if (topics.length <= 3) {
      const parts: string[] = [];
      let totalLen = 0;
      for (const t of topics) {
        const frag = (t.summary || t.title || "").trim().replace(/[。；;]+$/, "");
        if (!frag) continue;
        if (totalLen + frag.length > 260 && parts.length > 0) break;
        parts.push(frag);
        totalLen += frag.length + 1;
      }
      if (parts.length > 0) detailLine = `全片重点探讨：${parts.join("；")}。`;
    } else {
      const picked = pickSpreadItems(topics, 6);
      const titles = dedupeLabels(
        picked.map((t) => (t.title || (t.summary || "").slice(0, 24) || "").trim().replace(/[。；;]+$/, "")),
        6
      );
      if (titles.length > 0) detailLine = `全片涵盖：${titles.join("；")}等核心议题。`;
    }
  } else if (manifest.clips.length > 0) {
    const pickedClips = pickSpreadItems(manifest.clips, 6);
    const titles = dedupeLabels(
      pickedClips.map((c) => (c.title || "").trim().replace(/[。；;]+$/, "")),
      6
    );
    if (titles.length > 0) detailLine = `全片涵盖：${titles.join("；")}等核心议题。`;
  } else {
    const rawDesc = (manifest.metadata.description || "").trim();
    if (rawDesc && hasCjkText(rawDesc)) {
      detailLine = rawDesc.slice(0, 220);
    }
  }
  return detailLine ? `${opening}\n${detailLine}`.trim() : opening;
}

function buildDefaultRemixDesc(manifest: ProjectManifest): string {
  const cached = (manifest.remix.publish_desc || "").trim();
  if (cached) return cached;

  const sortedItems = [...(manifest.remix.items || [])].sort((a, b) => a.order - b.order);
  const activeItems = sortedItems.filter((x) => x.enabled);
  const usedItems = activeItems.length > 0 ? activeItems : sortedItems;
  const clipById = new Map(manifest.clips.map((c) => [c.clip_id, c]));

  const rawLabels: string[] = [];
  for (const it of usedItems) {
    const srcTitle = it.source_clip_id ? (clipById.get(it.source_clip_id)?.title || "").trim() : "";
    const cand =
      (it.card_title || "").trim() ||
      (it.group_label || "").trim() ||
      srcTitle ||
      (it.quote || "").trim().slice(0, 36);
    if (cand) rawLabels.push(cand);
  }
  const uniqueLabels = dedupeLabels(rawLabels, 8);
  const lead = formatProgramContext(manifest);
  if (uniqueLabels.length > 0) {
    const opening = `本视频为${lead}的高光混剪，浓缩串联全片核心观点与高能片段：`;
    const bullets = uniqueLabels.map((lbl, idx) => `${idx + 1}. ${lbl}`).join("\n");
    return `${opening}\n${bullets}`;
  }
  return `本视频为${lead}的高光混剪，提炼全片最具看点的核心观点与精彩对话。`;
}

function buildDefaultCompilationDesc(manifest: ProjectManifest, compilationClips: ClipItem[]): string {
  const cached = (manifest.compilation?.publish_desc || "").trim();
  if (cached) return cached;

  const titles = dedupeLabels(
    compilationClips.map((c) => (c.title || c.cover_title || "").trim()),
    12
  );
  const lead = formatProgramContext(manifest);
  if (titles.length > 0) {
    const opening = `本视频为${lead}的精选片段合辑，完整收录以下核心内容：`;
    const bullets = titles.map((t, idx) => `${idx + 1}. ${t}`).join("\n");
    return `${opening}\n${bullets}`;
  }
  return `本视频为${lead}的精选片段合辑，收录全片核心精彩对话。`;
}

function bindOutsideClickDismiss(dialog: HTMLDialogElement | null, onClose: () => void) {
  if (!dialog || "closedBy" in HTMLDialogElement.prototype) return () => {};
  let pressedOutside = false;
  const isOutside = (e: PointerEvent) => {
    const r = dialog.getBoundingClientRect();
    return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
  };
  const onDown = (e: PointerEvent) => {
    pressedOutside = isOutside(e);
  };
  const onUp = (e: PointerEvent) => {
    if (pressedOutside && isOutside(e)) {
      dialog.close();
      onClose();
    }
  };
  dialog.addEventListener("pointerdown", onDown);
  dialog.addEventListener("pointerup", onUp);
  return () => {
    dialog.removeEventListener("pointerdown", onDown);
    dialog.removeEventListener("pointerup", onUp);
  };
}

function formatPublishErrorView(rec: PublishRecord): { errorText: string; solutionText: string } {
  const platLabel = PLAT_LABELS[rec.platform] || rec.platform;
  const rawErr = (rec.error || "").trim();
  const isLegacyRawCmdTimeout = rawErr.startsWith("Command '[") && rawErr.includes("timed out");
  let errorText = rawErr || rec.message || `${platLabel}上传失败`;
  if (isLegacyRawCmdTimeout) {
    const secMatch = rawErr.match(/timed out after\s+([\d.]+)\s+seconds/i);
    const mins = secMatch ? Math.max(1, Math.round(Number(secMatch[1]) / 60)) : 30;
    errorText = `${platLabel}上传超过 ${mins} 分钟无响应已超时终止（上传进程在传输阶段卡死）。`;
  }
  let solutionText = (rec.solution || "").trim();
  if (!solutionText) {
    if (rec.platform === "bilibili") {
      solutionText =
        "1) 检查网络或临时关闭/分流 VPN 代理（B站 UPOS 走国内直连，代理易导致分块卡死）；2) 点击「复位状态」后单独重试 B站上传；3) 若多次失败，点击「重扫码」刷新 Cookie 或到 B站创作中心网页端手动投稿。";
    } else if (rec.platform === "wechat") {
      solutionText =
        "1) 先到视频号助手草稿箱确认是否已上传；2) 确认未开启代理拦截 channels.weixin.qq.com 并关闭其他微信登录窗口；3) 点击「重扫码」完成登录后再点「复位状态」并重试。";
    } else {
      solutionText = "1) 检查登录态与网络后点击「复位状态」；2) 重新发起上传。";
    }
  }
  return { errorText, solutionText };
}

function formatShortClock(iso?: string): string {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "";
    return d.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false });
  } catch {
    return "";
  }
}

function renderPubBadges(records?: PublishRecord[]) {
  if (!records || records.length === 0) return null;
  return records.map((rec) => {
    const label = PLAT_LABELS[rec.platform] || rec.platform;
    const pct =
      rec.status === "running" && typeof rec.progress_pct === "number"
        ? Math.max(0, Math.min(100, Math.round(rec.progress_pct)))
        : null;
    const runningStage = (rec.progress_stage || "上传中").trim();
    const statusText =
      rec.status === "published"
        ? "已发"
        : rec.status === "draft"
          ? "草稿"
          : rec.status === "running"
            ? pct !== null
              ? `${runningStage} ${pct}%`
              : runningStage
            : "失败";
    const tooltip =
      rec.status === "error"
        ? (() => {
            const view = formatPublishErrorView(rec);
            return [rec.message, view.errorText, `解决方案：${view.solutionText}`]
              .filter(Boolean)
              .join("\n");
          })()
        : rec.message || rec.error || "";
    if (rec.url_or_id) {
      return (
        <a
          key={rec.platform}
          href={rec.url_or_id}
          target="_blank"
          rel="noreferrer"
          className={`pub-badge ${rec.status}`}
          onClick={(e) => e.stopPropagation()}
          title={tooltip}
        >
          {label}:{statusText}
        </a>
      );
    }
    const badgeStyle =
      rec.status === "running" && pct !== null
        ? ({ "--pub-pct": `${pct}%` } as CSSProperties)
        : undefined;
    return (
      <span
        key={rec.platform}
        className={`pub-badge ${rec.status}`}
        style={badgeStyle}
        title={tooltip}
      >
        {label}:{statusText}
      </span>
    );
  });
}

function hasRemixCardFor(m: ProjectManifest | null): boolean {
  return Boolean(
    m &&
      (m.remix.items.length > 0 ||
        m.clips.length > 0 ||
        (m.remix_only_pool?.length || 0) > 0 ||
        Boolean(m.remix.output_file))
  );
}

function hasCompilationCardFor(m: ProjectManifest | null): boolean {
  return Boolean(m && (m.clips.length > 0 || Boolean(m.compilation?.output_file)));
}

function hasCleanCutCardFor(m: ProjectManifest | null): boolean {
  return Boolean(
    m && ((m.cut_segments || []).length > 0 || Boolean(m.clean_cut?.output_file) || Boolean(m.crop?.enabled))
  );
}

function pickDefaultSelectedId(m: ProjectManifest): string | null {
  if (m.remix.items.length > 0) return "REMIX";
  if (m.clips.length > 0) return m.clips[0].clip_id;
  if (hasRemixCardFor(m)) return "REMIX";
  if (hasCompilationCardFor(m)) return "COMPILATION";
  if (hasCleanCutCardFor(m)) return "CLEAN_CUT";
  return null;
}

function reconcileSelectedId(prev: string | null, m: ProjectManifest): string | null {
  if (prev === "REMIX" && hasRemixCardFor(m)) return prev;
  if (prev === "COMPILATION" && hasCompilationCardFor(m)) return prev;
  if (prev === "CLEAN_CUT" && hasCleanCutCardFor(m)) return prev;
  if (prev && m.clips.some((c) => c.clip_id === prev)) return prev;
  return pickDefaultSelectedId(m);
}

export default function App() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [inputPath, setInputPath] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [manifest, setManifest] = useState<ProjectManifest | null>(null);
  const [jobRunning, setJobRunning] = useState(false);
  const [jobCancelling, setJobCancelling] = useState(false);
  const [jobStages, setJobStages] = useState<string[]>([]);
  const [publishJobs, setPublishJobs] = useState<PublishJobInfo[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [singleRendering, setSingleRendering] = useState(false);

  // Clean 2-column by default; right logs drawer hidden unless user opens it
  const [showLogs, setShowLogs] = useState(false);

  // Resizable panel dimensions (persisted to localStorage)
  const [leftPaneWidth, setLeftPaneWidth] = useState<number>(() => {
    try {
      const v = Number(window.localStorage.getItem("clipcut.leftPaneWidth"));
      return Number.isFinite(v) && v >= 220 && v <= 620 ? v : 285;
    } catch {
      return 285;
    }
  });
  const [bottomPaneHeight, setBottomPaneHeight] = useState<number>(() => {
    try {
      const v = Number(window.localStorage.getItem("clipcut.bottomPaneHeight"));
      return Number.isFinite(v) && v >= 160 && v <= 720 ? v : 380;
    } catch {
      return 380;
    }
  });
  const [resizingAxis, setResizingAxis] = useState<"col" | "row" | null>(null);
  const workbenchRef = useRef<HTMLElement | null>(null);
  const centerPaneRef = useRef<HTMLElement | null>(null);

  // Selection state: clip_id or "REMIX"
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [playMode, setPlayMode] = useState<"source" | "rendered">("source");
  const [loopClip, setLoopClip] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // Interactive frame crop state
  const [cropEditing, setCropEditing] = useState(false);
  const [cropEditScope, setCropEditScope] = useState<"project" | "clip">("project");
  const [cropDraft, setCropDraft] = useState<NormalizedCropRect>({ x: 0, y: 0, w: 1, h: 0.85 });
  const [cropPreviewOnly, setCropPreviewOnly] = useState(true);
  const [cropSaving, setCropSaving] = useState(false);

  // Right pane logs state (loaded only when opened / job running)
  const [rightTab, setRightTab] = useState<"receipts" | "topics" | "logs">("receipts");
  const [stageFilter, setStageFilter] = useState<string>("");
  const [receipts, setReceipts] = useState<Receipt[]>([]);
  const [pipelineLog, setPipelineLog] = useState<string[]>([]);

  // Dialog refs
  const [scanCandidates, setScanCandidates] = useState<ScanResult | null>(null);
  const pickerDialogRef = useRef<HTMLDialogElement | null>(null);
  const settingsDialogRef = useRef<HTMLDialogElement | null>(null);
  const configDialogRef = useRef<HTMLDialogElement | null>(null);
  const publishDialogRef = useRef<HTMLDialogElement | null>(null);
  const batchPublishDialogRef = useRef<HTMLDialogElement | null>(null);
  const replanDialogRef = useRef<HTMLDialogElement | null>(null);
  const policyDialogRef = useRef<HTMLDialogElement | null>(null);
  const metadataDialogRef = useRef<HTMLDialogElement | null>(null);
  const [metaChannelDraft, setMetaChannelDraft] = useState("");
  const [metaGuestDraft, setMetaGuestDraft] = useState("");
  const [metaDateDraft, setMetaDateDraft] = useState("");
  const [metaSaving, setMetaSaving] = useState(false);
  const [replanAssets, setReplanAssets] = useState<RenderedAssetsInfo | null>(null);
  const [projectAssets, setProjectAssets] = useState<RenderedAssetsInfo | null>(null);
  const [canMask, setCanMask] = useState(true);
  const [replanForceSignals, setReplanForceSignals] = useState(false);
  const [replanSubmitting, setReplanSubmitting] = useState(false);

  // Political sensitivity policy profiles & editing draft
  const [contentFilterCfg, setContentFilterCfg] = useState<ContentFilterConfig | null>(null);
  const [navCfg, setNavCfg] = useState<NavigationConfig>({
    style: "top_nav",
    remix_style: "top_nav",
    compilation_style: "top_nav",
    screen_layout: "portrait",
    color_scheme: "tiffany",
    custom_accent: "#0ABAB5",
  });
  const [policyDraft, setPolicyDraft] = useState<ContentFilterConfig | null>(null);
  const [policyTab, setPolicyTab] = useState<"strict" | "balanced" | "loose">("strict");
  const [policySaving, setPolicySaving] = useState(false);
  const [extractingPreciseTerms, setExtractingPreciseTerms] = useState(false);

  // Settings & YAML state
  const [yamlText, setYamlText] = useState("");
  const [yamlSaving, setYamlSaving] = useState(false);
  const [modelCatalog, setModelCatalog] = useState<{
    text: { provider: string; model: string; label: string }[];
    multimodal: { provider: string; model: string; label: string }[];
  }>({ text: [], multimodal: [] });
  const [textModelSpec, setTextModelSpec] = useState("dashscope:qwen3.8-max");
  const [mmModelSpec, setMmModelSpec] = useState("dashscope:qwen3.5-omni-plus");

  // Inline drafts for selected item title + cover title
  const [titleDraft, setTitleDraft] = useState("");
  const [coverTitleDraft, setCoverTitleDraft] = useState("");
  const [remixPromptDraft, setRemixPromptDraft] = useState("");
  const [coverRefreshing, setCoverRefreshing] = useState(false);
  const [coverNonce, setCoverNonce] = useState(0);
  const [compTitleSuggesting, setCompTitleSuggesting] = useState(false);
  const [cleanCutTitleSuggesting, setCleanCutTitleSuggesting] = useState(false);
  const [subtitleDrafts, setSubtitleDrafts] = useState<Record<number, { zh: string; en: string }>>({});
  const [subtitleRepairSaving, setSubtitleRepairSaving] = useState(false);
  const [subtitleRepairFullBurn, setSubtitleRepairFullBurn] = useState(false);

  // Multi-platform publish dialog state
  const [platStatus, setPlatStatus] = useState<Record<string, PlatformStatus>>({});
  const [pubPlatforms, setPubPlatforms] = useState<Record<string, boolean>>({
    wechat: true,
    bilibili: false,
    xiaohongshu: false,
  });
  const [pubTitle, setPubTitle] = useState("");
  const [pubCoverTitle, setPubCoverTitle] = useState("");
  const [pubDesc, setPubDesc] = useState("");
  const [pubDescSuggesting, setPubDescSuggesting] = useState(false);
  const [pubTags, setPubTags] = useState("");
  const [pubPublic, setPubPublic] = useState(false);
  const [pubCollection, setPubCollection] = useState("");
  const [pubLoginNotice, setPubLoginNotice] = useState<string | null>(null);
  const [pubResettingKey, setPubResettingKey] = useState<string | null>(null);
  const [pubLoginWatchPlat, setPubLoginWatchPlat] = useState<string | null>(null);

  // Batch publish standalone clips dialog state
  const [batchPlatforms, setBatchPlatforms] = useState<Record<string, boolean>>({
    wechat: true,
    bilibili: false,
    xiaohongshu: false,
  });
  const [batchCollection, setBatchCollection] = useState("");
  const [batchPublic, setBatchPublic] = useState(false);
  const [batchSkipPublished, setBatchSkipPublished] = useState(true);
  const [batchClipPick, setBatchClipPick] = useState<Record<string, boolean>>({});
  const [batchSubmitting, setBatchSubmitting] = useState(false);

  useEffect(() => {
    api
      .getModels()
      .then((res) => {
        setModelCatalog(res.catalog);
        setTextModelSpec(`${res.selected.text_provider}:${res.selected.text_model}`);
        setMmModelSpec(`${res.selected.multimodal_provider}:${res.selected.multimodal_model}`);
      })
      .catch(() => {});
    api
      .listProjects()
      .then((res) => {
        setProjects(res.projects);
        if (res.projects.length > 0 && !manifest) {
          loadProject(res.projects[0].project_id);
        }
      })
      .catch(() => {});
    api
      .getPublishStatus()
      .then((res) => setPlatStatus(res.platforms))
      .catch(() => {});
    api
      .getContentFilterConfig()
      .then((res) => setContentFilterCfg(res))
      .catch(() => {});
    api
      .getNavigationConfig()
      .then((res) => setNavCfg(res))
      .catch(() => {});
  }, []);

  useEffect(() => {
    const c1 = bindOutsideClickDismiss(pickerDialogRef.current, () => setScanCandidates(null));
    const c2 = bindOutsideClickDismiss(settingsDialogRef.current, () => {});
    const c3 = bindOutsideClickDismiss(configDialogRef.current, () => {});
    const c4 = bindOutsideClickDismiss(publishDialogRef.current, () => {});
    const c5 = bindOutsideClickDismiss(replanDialogRef.current, () => setReplanAssets(null));
    const c6 = bindOutsideClickDismiss(policyDialogRef.current, () => {});
    const c7 = bindOutsideClickDismiss(batchPublishDialogRef.current, () => {});
    return () => {
      c1();
      c2();
      c3();
      c4();
      c5();
      c6();
      c7();
    };
  }, []);

  async function loadProject(projectId: string) {
    try {
      const data = await api.getProject(projectId);
      setManifest(data.manifest);
      setProjectAssets(data.assets ?? null);
      setCanMask(data.can_mask ?? true);
      const running = Boolean(data.job?.running);
      setJobRunning(running);
      setJobCancelling(running && Boolean(data.job?.cancelling));
      setJobStages(running ? data.job?.stages || [] : []);
      setPublishJobs((data.publish_jobs || []).filter((j) => j.running));
      setBatchCollection("");
      if (data.job?.error) setErrorMsg(data.job.error);
      setSelectedId((prev) => reconcileSelectedId(prev, data.manifest));
      if (showLogs) {
        refreshReceipts(projectId, stageFilter);
      }
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function refreshReceipts(projectId: string, stage = "") {
    try {
      const r = await api.getReceipts(projectId, stage || undefined);
      setReceipts(r.receipts);
      setPipelineLog(r.pipeline_log);
    } catch {
      // ignore
    }
  }

  // Poll while pipeline or publish background job is running
  useEffect(() => {
    if (!manifest || (!jobRunning && publishJobs.length === 0)) return;
    const pid = manifest.project_id;
    const timer = window.setInterval(async () => {
      try {
        const data = await api.getProject(pid);
        setManifest(data.manifest);
        if (data.assets) setProjectAssets(data.assets);
        if (typeof data.can_mask === "boolean") setCanMask(data.can_mask);
        if (showLogs) refreshReceipts(pid, stageFilter);
        setJobCancelling(Boolean(data.job?.cancelling));
        if (data.job?.stages) setJobStages(data.job.stages);
        const nextPubJobs = (data.publish_jobs || []).filter((j) => j.running);
        setPublishJobs(nextPubJobs);
        if (!data.job?.running && jobRunning) {
          setJobRunning(false);
          setJobCancelling(false);
          setJobStages([]);
          if (data.job?.error) setErrorMsg(data.job.error);
          setCoverNonce((n) => n + 1);
          setSelectedId((prev) => reconcileSelectedId(prev, data.manifest));
          api.listProjects().then((r) => setProjects(r.projects)).catch(() => {});
        }
      } catch {
        // ignore transient errors
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [manifest?.project_id, jobRunning, publishJobs.length, stageFilter, showLogs]);

  useEffect(() => {
    setRemixPromptDraft(manifest?.remix?.custom_prompt || "");
  }, [manifest?.project_id, manifest?.remix?.custom_prompt]);

  useEffect(() => {
    if (!manifest) return;
    setSelectedId((prev) => reconcileSelectedId(prev, manifest));
  }, [
    manifest?.project_id,
    manifest?.clips.length,
    manifest?.remix.items.length,
    manifest?.remix_only_pool?.length,
    manifest?.remix.output_file,
    manifest?.compilation?.output_file,
    manifest?.cut_segments?.length,
    manifest?.clean_cut?.output_file,
    manifest?.crop?.enabled,
    selectedId,
  ]);

  useEffect(() => {
    if (manifest && showLogs) {
      refreshReceipts(manifest.project_id, stageFilter);
    }
  }, [stageFilter, showLogs]);

  const activeClip: ClipItem | null = useMemo(() => {
    if (!manifest || !selectedId || selectedId === "REMIX" || selectedId === "COMPILATION" || selectedId === "CLEAN_CUT") {
      return null;
    }
    return manifest.clips.find((c) => c.clip_id === selectedId) || null;
  }, [manifest, selectedId]);

  useEffect(() => {
    if (selectedId === "REMIX" && manifest) {
      setTitleDraft(manifest.remix.title || "");
      setCoverTitleDraft(manifest.remix.cover_title || "");
    } else if (selectedId === "COMPILATION" && manifest) {
      setTitleDraft(manifest.compilation?.title || "切片精选集");
      setCoverTitleDraft(manifest.compilation?.cover_title || "");
    } else if (selectedId === "CLEAN_CUT" && manifest) {
      const defTitle =
        manifest.clean_cut?.title ||
        `${manifest.metadata.original_theme_zh || manifest.metadata.original_title || "全片"}（净版）`;
      setTitleDraft(defTitle);
      setCoverTitleDraft(manifest.clean_cut?.cover_title || "");
    } else if (activeClip) {
      setTitleDraft(activeClip.title || "");
      setCoverTitleDraft(activeClip.cover_title || "");
    }
  }, [
    selectedId,
    activeClip?.clip_id,
    activeClip?.title,
    activeClip?.cover_title,
    manifest?.remix.title,
    manifest?.remix.cover_title,
    manifest?.compilation?.title,
    manifest?.compilation?.cover_title,
    manifest?.clean_cut?.title,
    manifest?.clean_cut?.cover_title,
  ]);

  useEffect(() => {
    if (!manifest || !activeClip) {
      setSubtitleDrafts({});
      return;
    }
    const entries = manifest.utterances
      .slice(activeClip.start_utt, activeClip.end_utt + 1)
      .map((u) => [u.id, { zh: u.zh || "", en: u.en || "" }] as const);
    setSubtitleDrafts(Object.fromEntries(entries));
  }, [manifest?.utterances, activeClip?.clip_id, activeClip?.start_utt, activeClip?.end_utt]);

  useEffect(() => {
    setCropEditing(false);
  }, [manifest?.project_id, selectedId]);

  useEffect(() => {
    if (activeClip && playMode === "source" && videoRef.current) {
      videoRef.current.currentTime = activeClip.start_s;
    }
  }, [selectedId, activeClip?.clip_id]);

  const effectiveCrop = useMemo(
    () => resolveEffectiveCropFrontend(manifest, activeClip),
    [manifest?.crop, activeClip]
  );

  const effectiveCropDims = useMemo(() => {
    if (!effectiveCrop || !manifest) return null;
    return computeEvenOutputDims(
      effectiveCrop,
      manifest.source.width || 1920,
      manifest.source.height || 1080
    );
  }, [effectiveCrop, manifest?.source.width, manifest?.source.height]);

  function openCropEditor(scopeOverride?: "project" | "clip") {
    if (!manifest) return;
    setPlayMode("source");
    const nextScope: "project" | "clip" =
      scopeOverride ||
      (activeClip?.crop_override?.mode === "custom" ? "clip" : "project");
    let initRect: NormalizedCropRect = { x: 0, y: 0, w: 1, h: 0.85 };
    if (nextScope === "clip" && activeClip?.crop_override?.mode === "custom") {
      initRect = sanitizeRect(activeClip.crop_override);
    } else if (manifest.crop?.enabled) {
      initRect = sanitizeRect(manifest.crop);
    }
    setCropEditScope(nextScope);
    setCropDraft(initRect);
    setCropEditing(true);
  }

  function handleCropScopeChange(nextScope: "project" | "clip") {
    setCropEditScope(nextScope);
    if (nextScope === "clip" && activeClip?.crop_override?.mode === "custom") {
      setCropDraft(sanitizeRect(activeClip.crop_override));
    } else if (manifest?.crop?.enabled) {
      setCropDraft(sanitizeRect(manifest.crop));
    }
  }

  async function handleApplyCrop() {
    if (!manifest) return;
    setCropSaving(true);
    setErrorMsg(null);
    const rect = sanitizeRect(cropDraft);
    try {
      if (cropEditScope === "clip" && activeClip) {
        const override: ClipCropOverride = isFullFrameRect(rect)
          ? { mode: "none", x: 0, y: 0, w: 1, h: 1 }
          : { mode: "custom", ...rect };
        const res = await api.patchClip(manifest.project_id, activeClip.clip_id, {
          crop_override: override,
        });
        setManifest(res.manifest);
      } else {
        const res = await api.patchCrop(manifest.project_id, {
          enabled: !isFullFrameRect(rect),
          ...rect,
        });
        setManifest(res.manifest);
        if (!isFullFrameRect(rect) && res.manifest.clips.length === 0) {
          setSelectedId("CLEAN_CUT");
        }
      }
      setCropEditing(false);
      setCropPreviewOnly(true);
    } catch (err) {
      setErrorMsg((err as Error).message);
    } finally {
      setCropSaving(false);
    }
  }

  async function handleDisableCropForScope() {
    if (!manifest) return;
    setCropSaving(true);
    setErrorMsg(null);
    try {
      if (cropEditScope === "clip" && activeClip) {
        const res = await api.patchClip(manifest.project_id, activeClip.clip_id, {
          crop_override: { mode: "none", x: 0, y: 0, w: 1, h: 1 },
        });
        setManifest(res.manifest);
      } else {
        const res = await api.patchCrop(manifest.project_id, { enabled: false });
        setManifest(res.manifest);
      }
      setCropEditing(false);
    } catch (err) {
      setErrorMsg((err as Error).message);
    } finally {
      setCropSaving(false);
    }
  }

  async function handleInheritClipCrop() {
    if (!manifest || !activeClip) return;
    setCropSaving(true);
    setErrorMsg(null);
    try {
      const res = await api.patchClip(manifest.project_id, activeClip.clip_id, {
        crop_override: { mode: "inherit", x: 0, y: 0, w: 1, h: 1 },
      });
      setManifest(res.manifest);
      setCropEditing(false);
      setCropPreviewOnly(true);
    } catch (err) {
      setErrorMsg((err as Error).message);
    } finally {
      setCropSaving(false);
    }
  }

  function handleTimeUpdate() {
    const v = videoRef.current;
    if (!v) return;
    setCurrentTime(v.currentTime);
    if (playMode === "source" && loopClip && activeClip && v.currentTime >= activeClip.end_s) {
      v.currentTime = activeClip.start_s;
    }
  }

  function seekTo(sec: number) {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = Math.max(0, sec);
    v.play().catch(() => {});
  }

  function clampLeftWidth(w: number): number {
    const totalW = workbenchRef.current?.clientWidth || window.innerWidth;
    const reserve = showLogs ? 780 : 440;
    const maxW = Math.min(620, Math.max(240, totalW - reserve));
    return Math.round(Math.max(220, Math.min(maxW, w)));
  }

  function clampBottomHeight(h: number): number {
    const totalH = centerPaneRef.current?.clientHeight || window.innerHeight - 120;
    const maxH = Math.max(180, totalH - 168);
    return Math.round(Math.max(160, Math.min(maxH, h)));
  }

  function persistLeftWidth(w: number) {
    try {
      window.localStorage.setItem("clipcut.leftPaneWidth", String(w));
    } catch {
      // ignore storage errors
    }
  }

  function persistBottomHeight(h: number) {
    try {
      window.localStorage.setItem("clipcut.bottomPaneHeight", String(h));
    } catch {
      // ignore storage errors
    }
  }

  useEffect(() => {
    const wb = workbenchRef.current;
    const cp = centerPaneRef.current;
    if (!wb && !cp) return;
    const clampOnResize = () => {
      setLeftPaneWidth((prev) => clampLeftWidth(prev));
      setBottomPaneHeight((prev) => clampBottomHeight(prev));
    };
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(clampOnResize) : null;
    if (ro) {
      if (wb) ro.observe(wb);
      if (cp) ro.observe(cp);
    } else {
      window.addEventListener("resize", clampOnResize);
    }
    return () => {
      if (ro) ro.disconnect();
      else window.removeEventListener("resize", clampOnResize);
    };
  }, [manifest?.project_id, showLogs]);

  function startLeftResize(e: ReactPointerEvent<HTMLDivElement>) {
    if (e.button !== 0 && e.pointerType === "mouse") return;
    e.preventDefault();
    const handle = e.currentTarget;
    try {
      handle.setPointerCapture(e.pointerId);
    } catch {
      // synthetic pointer fallback
    }
    const startX = e.clientX;
    const startW = leftPaneWidth;
    let latestW = startW;
    setResizingAxis("col");

    const onMove = (ev: PointerEvent) => {
      latestW = clampLeftWidth(startW + (ev.clientX - startX));
      setLeftPaneWidth(latestW);
    };
    const cleanup = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      try {
        if (handle.hasPointerCapture(e.pointerId)) handle.releasePointerCapture(e.pointerId);
      } catch {
        // ignore
      }
      setResizingAxis(null);
      persistLeftWidth(latestW);
    };
    const onUp = () => cleanup();
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
  }

  function startBottomResize(e: ReactPointerEvent<HTMLDivElement>) {
    if (e.button !== 0 && e.pointerType === "mouse") return;
    e.preventDefault();
    const handle = e.currentTarget;
    try {
      handle.setPointerCapture(e.pointerId);
    } catch {
      // synthetic pointer fallback
    }
    const startY = e.clientY;
    const startH = bottomPaneHeight;
    let latestH = startH;
    setResizingAxis("row");

    const onMove = (ev: PointerEvent) => {
      // Dragging handle up (clientY decreases) grows the bottom panel
      latestH = clampBottomHeight(startH - (ev.clientY - startY));
      setBottomPaneHeight(latestH);
    };
    const cleanup = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      try {
        if (handle.hasPointerCapture(e.pointerId)) handle.releasePointerCapture(e.pointerId);
      } catch {
        // ignore
      }
      setResizingAxis(null);
      persistBottomHeight(latestH);
    };
    const onUp = () => cleanup();
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
  }

  function handleLeftHandleKeyDown(e: ReactKeyboardEvent<HTMLDivElement>) {
    const step = e.shiftKey ? 40 : 16;
    let next: number | null = null;
    if (e.key === "ArrowLeft") next = clampLeftWidth(leftPaneWidth - step);
    else if (e.key === "ArrowRight") next = clampLeftWidth(leftPaneWidth + step);
    else if (e.key === "Home" || e.key === "Enter") next = clampLeftWidth(285);
    if (next !== null) {
      e.preventDefault();
      setLeftPaneWidth(next);
      persistLeftWidth(next);
    }
  }

  function handleBottomHandleKeyDown(e: ReactKeyboardEvent<HTMLDivElement>) {
    const step = e.shiftKey ? 40 : 16;
    let next: number | null = null;
    if (e.key === "ArrowUp") next = clampBottomHeight(bottomPaneHeight + step);
    else if (e.key === "ArrowDown") next = clampBottomHeight(bottomPaneHeight - step);
    else if (e.key === "Home" || e.key === "Enter") next = clampBottomHeight(380);
    if (next !== null) {
      e.preventDefault();
      setBottomPaneHeight(next);
      persistBottomHeight(next);
    }
  }

  async function handlePickPath(kind: "file" | "folder") {
    try {
      const res = await api.pickPath(kind);
      if (!res.cancelled && res.path) {
        setInputPath(res.path);
        await handleScanAndOpen(res.path);
      }
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handleScanAndOpen(pathOverride?: string) {
    const target = (pathOverride ?? inputPath).trim();
    if (!target) return;
    setErrorMsg(null);
    try {
      const scan = await api.scan(target);
      if (scan.videos.length === 0) {
        setErrorMsg("未找到支持的视频文件 (.mp4/.mov/.mkv/.webm)");
        return;
      }
      if (scan.videos.length === 1) {
        await openCandidate(scan.videos[0]);
      } else {
        setScanCandidates(scan);
        pickerDialogRef.current?.showModal();
      }
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function openCandidate(cand: VideoFileCandidate) {
    pickerDialogRef.current?.close();
    settingsDialogRef.current?.close();
    setScanCandidates(null);
    setErrorMsg(null);
    try {
      const res = await api.openProject({
        video_path: cand.path,
        subtitle_path: cand.suggested_subtitle,
        info_json_path: cand.suggested_info_json,
        source_url: sourceUrl.trim() || null,
      });
      setManifest(res.manifest);
      setProjectAssets(null);
      setSelectedId(pickDefaultSelectedId(res.manifest));
      setPlayMode("source");
      api.listProjects().then((r) => setProjects(r.projects)).catch(() => {});
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handleRun(stages: string[], forceRender = false, renderTargets?: string[]) {
    if (!manifest) return;
    setErrorMsg(null);
    try {
      await api.runStages(manifest.project_id, stages, forceRender, renderTargets);
      setJobCancelling(false);
      setJobStages(stages);
      setJobRunning(true);
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handleCancelJob() {
    if (!manifest || !jobRunning || jobCancelling) return;
    setErrorMsg(null);
    setJobCancelling(true);
    try {
      const res = await api.cancelJob(manifest.project_id);
      if (!res.cancelled) {
        setJobCancelling(false);
        const data = await api.getProject(manifest.project_id);
        setManifest(data.manifest);
        setJobRunning(Boolean(data.job?.running));
        if (!data.job?.running) setJobStages([]);
      }
    } catch (err) {
      setJobCancelling(false);
      setErrorMsg((err as Error).message);
    }
  }

  async function openReplanDialog() {
    if (!manifest || jobRunning) return;
    setErrorMsg(null);
    setReplanForceSignals(false);
    try {
      const assets = await api.getRenderedAssets(manifest.project_id);
      setReplanAssets(assets);
      setProjectAssets(assets);
      replanDialogRef.current?.showModal();
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handleConfirmReplan(deleteRenderedFiles: boolean) {
    if (!manifest) return;
    setErrorMsg(null);
    setReplanSubmitting(true);
    try {
      const res = await api.replanStrategy(manifest.project_id, {
        delete_rendered_files: deleteRenderedFiles,
        force_signals: replanForceSignals,
      });
      replanDialogRef.current?.close();
      setReplanAssets(null);
      setProjectAssets(null);
      setManifest(res.manifest);
      setSelectedId(null);
      setPlayMode("source");
      setJobCancelling(false);
      setJobStages(res.stages || ["scan_cuts", "analyze", "verify_and_plan"]);
      setJobRunning(true);
    } catch (err) {
      setErrorMsg((err as Error).message);
    } finally {
      setReplanSubmitting(false);
    }
  }

  async function handlePatchClip(
    clipId: string,
    patch: { enabled?: boolean; title?: string; cover_title?: string; start_utt?: number; end_utt?: number }
  ) {
    if (!manifest) return;
    setErrorMsg(null);
    try {
      // Materialize independent Compilation clip_ids on legacy manifests before changing standalone clip flags
      if (patch.enabled !== undefined && manifest.compilation?.clip_ids == null) {
        const legacyIds = manifest.clips.filter((c) => c.enabled).map((c) => c.clip_id);
        await api.patchCompilation(manifest.project_id, { clip_ids: legacyIds });
      }
      const res = await api.patchClip(manifest.project_id, clipId, patch);
      setManifest(res.manifest);
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handleBatchClips(action: "enable_all" | "disable_all" | "top_n", n = 5) {
    if (!manifest) return;
    setErrorMsg(null);
    try {
      // Materialize independent Compilation clip_ids on legacy manifests so standalone batch toggles don't wipe Compilation
      if (manifest.compilation?.clip_ids == null) {
        const legacyIds = manifest.clips.filter((c) => c.enabled).map((c) => c.clip_id);
        await api.patchCompilation(manifest.project_id, { clip_ids: legacyIds });
      }
      const res = await api.batchClips(manifest.project_id, action, n);
      setManifest(res.manifest);
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handleMoveClip(fromIdx: number, direction: -1 | 1) {
    if (!manifest) return;
    const toIdx = fromIdx + direction;
    if (toIdx < 0 || toIdx >= manifest.clips.length) return;
    const reordered = [...manifest.clips];
    const [moved] = reordered.splice(fromIdx, 1);
    reordered.splice(toIdx, 0, moved);
    const clipIds = reordered.map((c) => c.clip_id);
    try {
      const res = await api.reorderClips(manifest.project_id, clipIds);
      setManifest(res.manifest);
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handleResetClipOrder() {
    if (!manifest) return;
    try {
      const res = await api.resetClipOrder(manifest.project_id);
      setManifest(res.manifest);
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handleRenderSingleClip(clipId: string) {
    if (!manifest) return;
    setErrorMsg(null);
    setSingleRendering(true);
    try {
      const res = await api.renderClip(manifest.project_id, clipId, true);
      setManifest(res.manifest);
      setPlayMode("rendered");
    } catch (err) {
      setErrorMsg((err as Error).message);
    } finally {
      setSingleRendering(false);
    }
  }

  function openMetadataDialog() {
    if (!manifest) return;
    setMetaChannelDraft(manifest.metadata.channel_zh || manifest.metadata.channel || "");
    setMetaGuestDraft(manifest.metadata.guest_zh || "");
    setMetaDateDraft(manifest.metadata.published_date || "");
    metadataDialogRef.current?.showModal();
  }

  async function handleSaveMetadata() {
    if (!manifest) return;
    setMetaSaving(true);
    setErrorMsg(null);
    try {
      const res = await api.patchMetadata(manifest.project_id, {
        channel_zh: metaChannelDraft.trim(),
        guest_zh: metaGuestDraft.trim(),
        published_date: metaDateDraft.trim(),
      });
      setManifest(res.manifest);
      metadataDialogRef.current?.close();
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setMetaSaving(false);
    }
  }

  async function handlePatchRemix(patch: Record<string, unknown>) {
    if (!manifest) return;
    setErrorMsg(null);
    try {
      const res = await api.patchRemix(manifest.project_id, patch);
      setManifest(res.manifest);
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handlePatchCompilation(patch: Record<string, unknown>) {
    if (!manifest) return;
    setErrorMsg(null);
    try {
      const res = await api.patchCompilation(manifest.project_id, patch);
      setManifest(res.manifest);
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handleRenderRemix(force = true) {
    if (!manifest) return;
    setErrorMsg(null);
    try {
      await api.renderRemix(manifest.project_id, force);
      setJobCancelling(false);
      setJobStages(["render"]);
      setJobRunning(true);
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handleRenderCompilation(force = true) {
    if (!manifest) return;
    setErrorMsg(null);
    try {
      await api.renderCompilation(manifest.project_id, force);
      setJobCancelling(false);
      setJobStages(["render"]);
      setJobRunning(true);
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handleScanCleanCut() {
    if (!manifest) return;
    setErrorMsg(null);
    try {
      await api.scanCleanCut(manifest.project_id, {
        cut_ads: true,
        cut_political_sensitive: true,
        sensitivity: manifest.clean_cut?.sensitivity ?? "strict",
      });
      setJobCancelling(false);
      setJobStages(["clean_cut_scan"]);
      setJobRunning(true);
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handlePatchCleanCut(patch: Record<string, unknown>) {
    if (!manifest) return;
    setErrorMsg(null);
    try {
      const res = await api.patchCleanCut(manifest.project_id, patch);
      setManifest(res.manifest);
      if ("title" in patch || "cover_title" in patch) {
        setCoverNonce((n) => n + 1);
      }
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handleRenderCleanCut(force = true) {
    if (!manifest) return;
    setErrorMsg(null);
    try {
      await api.renderCleanCut(manifest.project_id, force);
      setJobCancelling(false);
      setJobStages(["render"]);
      setJobRunning(true);
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handleSuggestCompilationTitle() {
    if (!manifest) return;
    setErrorMsg(null);
    setCompTitleSuggesting(true);
    try {
      const res = await api.suggestCompilationTitle(manifest.project_id);
      setManifest(res.manifest);
      setTitleDraft(res.title || res.manifest.compilation?.title || "切片精选集");
      setCoverNonce((n) => n + 1);
    } catch (err) {
      setErrorMsg((err as Error).message);
    } finally {
      setCompTitleSuggesting(false);
    }
  }

  async function handleSuggestCleanCutTitle() {
    if (!manifest) return;
    setErrorMsg(null);
    setCleanCutTitleSuggesting(true);
    try {
      const res = await api.suggestCleanCutTitle(manifest.project_id);
      setManifest(res.manifest);
      setTitleDraft(res.title || res.manifest.clean_cut?.title || "全片（净版）");
      setCoverNonce((n) => n + 1);
    } catch (err) {
      setErrorMsg((err as Error).message);
    } finally {
      setCleanCutTitleSuggesting(false);
    }
  }

  async function handleSaveCoverTitleAndRefresh() {
    if (!manifest || !selectedId) return;
    setErrorMsg(null);
    setCoverRefreshing(true);
    try {
      const nextText = coverTitleDraft.trim();
      if (selectedId === "REMIX") {
        const prevCt = (manifest.remix.cover_title || "").trim();
        const res = await api.patchRemix(manifest.project_id, { cover_title: nextText });
        if (!res.manifest.remix.cover_file || nextText === prevCt) {
          const r2 = await api.refreshCover(manifest.project_id, "remix");
          setManifest(r2.manifest);
        } else {
          setManifest(res.manifest);
        }
      } else if (selectedId === "COMPILATION") {
        const prevCt = (manifest.compilation?.cover_title || "").trim();
        const res = await api.patchCompilation(manifest.project_id, { cover_title: nextText });
        if (!res.manifest.compilation?.cover_file || nextText === prevCt) {
          const r2 = await api.refreshCover(manifest.project_id, "compilation");
          setManifest(r2.manifest);
        } else {
          setManifest(res.manifest);
        }
      } else if (selectedId === "CLEAN_CUT") {
        const prevCt = (manifest.clean_cut?.cover_title || "").trim();
        const res = await api.patchCleanCut(manifest.project_id, { cover_title: nextText });
        if (!res.manifest.clean_cut?.cover_file || nextText === prevCt) {
          const r2 = await api.refreshCover(manifest.project_id, "clean_cut");
          setManifest(r2.manifest);
        } else {
          setManifest(res.manifest);
        }
      } else if (activeClip) {
        const prevCt = (activeClip.cover_title || "").trim();
        const res = await api.patchClip(manifest.project_id, activeClip.clip_id, { cover_title: nextText });
        if (!res.manifest.clips.find((c) => c.clip_id === activeClip.clip_id)?.cover_file || nextText === prevCt) {
          const r2 = await api.refreshCover(manifest.project_id, activeClip.clip_id);
          setManifest(r2.manifest);
        } else {
          setManifest(res.manifest);
        }
      }
      setCoverNonce((n) => n + 1);
    } catch (err) {
      setErrorMsg((err as Error).message);
    } finally {
      setCoverRefreshing(false);
    }
  }

  function moveRemixItem(idx: number, delta: number) {
    if (!manifest) return;
    const sorted = [...manifest.remix.items].sort((a, b) => a.order - b.order);
    const targetIdx = idx + delta;
    if (targetIdx < 0 || targetIdx >= sorted.length) return;
    const tmp = sorted[idx];
    sorted[idx] = sorted[targetIdx];
    sorted[targetIdx] = tmp;
    const reordered = sorted.map((it, i) => ({ remix_id: it.remix_id, order: i + 1 }));
    handlePatchRemix({ items: reordered });
  }

  async function openEditorialConfig() {
    try {
      const res = await api.getEditorialYaml();
      setYamlText(res.raw_yaml);
      settingsDialogRef.current?.close();
      configDialogRef.current?.showModal();
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function openPolicyDialog() {
    setErrorMsg(null);
    try {
      const cfgData = await api.getContentFilterConfig();
      setContentFilterCfg(cfgData);
      setPolicyDraft(JSON.parse(JSON.stringify(cfgData)) as ContentFilterConfig);
      const activeKey = manifest?.clean_cut?.sensitivity;
      if (activeKey === "strict" || activeKey === "balanced" || activeKey === "loose") {
        setPolicyTab(activeKey);
      } else {
        setPolicyTab(cfgData.default_sensitivity || "strict");
      }
      settingsDialogRef.current?.close();
      policyDialogRef.current?.showModal();
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  function mutatePolicyProfile(
    key: "strict" | "balanced" | "loose",
    updater: (prev: PoliticalProfile) => PoliticalProfile
  ) {
    setPolicyDraft((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        political_profiles: {
          ...prev.political_profiles,
          [key]: updater(prev.political_profiles[key]),
        },
      };
    });
  }

  async function handleSavePolicyConfig() {
    if (!policyDraft) return;
    setPolicySaving(true);
    setErrorMsg(null);
    try {
      const saved = await api.saveContentFilterConfig(policyDraft);
      setContentFilterCfg({
        default_sensitivity: saved.default_sensitivity,
        default_political_action: saved.default_political_action,
        political_profiles: saved.political_profiles,
      });
      policyDialogRef.current?.close();
      if (manifest) {
        api
          .patchCleanCut(manifest.project_id, {})
          .then((r) => setManifest(r.manifest))
          .catch(() => {});
      }
    } catch (err) {
      setErrorMsg((err as Error).message);
    } finally {
      setPolicySaving(false);
    }
  }

  async function handleResetPolicyProfile(profileKey?: "strict" | "balanced" | "loose") {
    setPolicySaving(true);
    setErrorMsg(null);
    try {
      const res = await api.resetContentFilterConfig(profileKey);
      const nextCfg: ContentFilterConfig = {
        default_sensitivity: res.default_sensitivity,
        default_political_action: res.default_political_action,
        political_profiles: res.political_profiles,
      };
      setContentFilterCfg(nextCfg);
      setPolicyDraft(JSON.parse(JSON.stringify(nextCfg)) as ContentFilterConfig);
      if (manifest) {
        api
          .patchCleanCut(manifest.project_id, {})
          .then((r) => setManifest(r.manifest))
          .catch(() => {});
      }
    } catch (err) {
      setErrorMsg((err as Error).message);
    } finally {
      setPolicySaving(false);
    }
  }

  async function handleExtractPreciseTerms(force = false) {
    if (!manifest) return;
    setExtractingPreciseTerms(true);
    setErrorMsg(null);
    try {
      const res = await api.extractPreciseTerms(manifest.project_id, { force });
      setManifest(res.manifest);
    } catch (err) {
      setErrorMsg((err as Error).message);
    } finally {
      setExtractingPreciseTerms(false);
    }
  }

  async function handleChangeProjectPoliticalAction(nextAction: PoliticalAction) {
    if (!manifest) return;
    setErrorMsg(null);
    try {
      const res = await api.patchCleanCut(manifest.project_id, { political_action: nextAction });
      setManifest(res.manifest);
      const missingCount = (res.manifest.cut_segments || []).filter(
        (s) =>
          s.kind === "political_sensitive" &&
          !s.precise_terms_extracted &&
          !(s.precise_terms_zh && s.precise_terms_zh.length > 0) &&
          !(s.precise_terms_en && s.precise_terms_en.length > 0)
      ).length;
      if (nextAction === "precise_mask" && missingCount > 0) {
        await handleExtractPreciseTerms(false);
      }
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handleChangeDefaultPoliticalAction(nextAction: PoliticalAction) {
    setErrorMsg(null);
    try {
      const saved = await api.saveContentFilterConfig({ default_political_action: nextAction });
      setContentFilterCfg({
        default_sensitivity: saved.default_sensitivity,
        default_political_action: saved.default_political_action,
        political_profiles: saved.political_profiles,
      });
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handleChangeDefaultSensitivity(nextSens: "strict" | "balanced" | "loose") {
    setErrorMsg(null);
    try {
      const saved = await api.saveContentFilterConfig({ default_sensitivity: nextSens });
      setContentFilterCfg({
        default_sensitivity: saved.default_sensitivity,
        default_political_action: saved.default_political_action,
        political_profiles: saved.political_profiles,
      });
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handleModelChange(kind: "text" | "multimodal", spec: string) {
    setErrorMsg(null);
    if (kind === "text") {
      setTextModelSpec(spec);
      try {
        await api.saveModels({ text_model: spec });
      } catch (err) {
        setErrorMsg((err as Error).message);
      }
    } else {
      setMmModelSpec(spec);
      try {
        await api.saveModels({ multimodal_model: spec });
      } catch (err) {
        setErrorMsg((err as Error).message);
      }
    }
  }

  async function saveEditorialConfig() {
    setYamlSaving(true);
    try {
      await api.saveEditorialYaml(yamlText);
      configDialogRef.current?.close();
    } catch (err) {
      setErrorMsg((err as Error).message);
    } finally {
      setYamlSaving(false);
    }
  }

  function getPublishJobForTarget(target: string | null): PublishJobInfo | null {
    if (!target) return null;
    return publishJobs.find((j) => j.running && j.targets.includes(target)) || null;
  }

  const activeBatchPublishJob = useMemo(
    () => publishJobs.find((j) => j.running && j.kind === "batch") || null,
    [publishJobs]
  );

  async function handleCancelPublishJob(jobId: string) {
    if (!manifest || !jobId) return;
    setErrorMsg(null);
    setPublishJobs((prev) =>
      prev.map((j) => (j.job_id === jobId ? { ...j, cancelling: true } : j))
    );
    try {
      await api.cancelJob(manifest.project_id, { job_id: jobId });
      const data = await api.getProject(manifest.project_id);
      setManifest(data.manifest);
      setPublishJobs((data.publish_jobs || []).filter((j) => j.running));
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  function defaultCollectionName(m: ProjectManifest): string {
    const raw = (m.metadata.original_theme_zh || m.metadata.original_title || "").trim();
    if (!raw) return "";
    const firstClause = raw.split(/[：:。；;，,]+/)[0].trim() || raw;
    return (firstClause.length > 10 ? firstClause.slice(0, 10) : firstClause).trim();
  }

  const activePublishTarget = useMemo(() => {
    if (!selectedId) return null;
    if (selectedId === "REMIX") return "remix";
    if (selectedId === "COMPILATION") return "compilation";
    if (selectedId === "CLEAN_CUT") return "clean_cut";
    return activeClip?.clip_id || null;
  }, [selectedId, activeClip?.clip_id]);

  const activePublishJob = useMemo(
    () => getPublishJobForTarget(activePublishTarget),
    [publishJobs, activePublishTarget]
  );

  const activePublishRecords = useMemo(() => {
    if (!manifest || !activePublishTarget) return [] as PublishRecord[];
    if (activePublishTarget === "remix") return manifest.remix.publish_records || [];
    if (activePublishTarget === "compilation") return manifest.compilation?.publish_records || [];
    if (activePublishTarget === "clean_cut") return manifest.clean_cut?.publish_records || [];
    return activeClip?.publish_records || [];
  }, [manifest, activePublishTarget, activeClip]);

  async function handleRefreshPublishStatus() {
    try {
      const res = await api.getPublishStatus();
      setPlatStatus(res.platforms);
      return res.platforms;
    } catch {
      return null;
    }
  }

  async function handleResetPublish(target: string, platform?: string, silent = false) {
    if (!manifest) return;
    const key = platform ? `${target}:${platform}` : `${target}:all`;
    setPubResettingKey(key);
    if (!silent) setErrorMsg(null);
    try {
      const res = await api.resetPublishRecords(manifest.project_id, {
        target,
        platform: platform ?? null,
        only_errors: !platform,
      });
      setManifest(res.manifest);
      if (res.removed.length > 0) {
        const names = res.removed.map((p) => PLAT_LABELS[p] || p).join("、");
        setPubLoginNotice(`已复位 ${names} 发布状态`);
      } else if (!silent) {
        setPubLoginNotice("当前无可复位的失败记录");
      }
    } catch (err) {
      if (!silent) {
        setErrorMsg((err as Error).message);
      }
    } finally {
      setPubResettingKey(null);
    }
  }

  useEffect(() => {
    if (!pubLoginWatchPlat) return;
    const watchPlat = pubLoginWatchPlat;
    let cancelled = false;
    const timer = window.setInterval(async () => {
      try {
        const res = await api.getPublishStatus();
        if (cancelled) return;
        setPlatStatus(res.platforms);
        const st = res.platforms[watchPlat];
        if (!st || st.login_running) return;
        setPubLoginWatchPlat(null);
        const platLabel = PLAT_LABELS[watchPlat] || watchPlat;
        if (st.login_returncode === 0) {
          setPubLoginNotice(`${platLabel} 扫码登录已完成`);
          if (
            manifest &&
            activePublishTarget &&
            !jobRunning &&
            !activePublishJob &&
            activePublishRecords.some((r) => r.platform === watchPlat && r.status === "error")
          ) {
            await handleResetPublish(activePublishTarget, watchPlat, true);
            setPubLoginNotice(`${platLabel} 扫码登录已完成，失败状态已自动复位`);
          }
        } else if (st.login_message) {
          setPubLoginNotice(st.login_message);
        }
      } catch {
        // ignore transient polling error
      }
    }, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [pubLoginWatchPlat, manifest?.project_id, activePublishTarget, jobRunning, activePublishJob, activePublishRecords]);

  function openPublishModal(preselectPlatform?: string) {
    if (!manifest || !selectedId) return;
    setErrorMsg(null);
    if (!pubLoginWatchPlat) setPubLoginNotice(null);
    handleRefreshPublishStatus();
    if (preselectPlatform) {
      setPubPlatforms({
        wechat: preselectPlatform === "wechat",
        bilibili: preselectPlatform === "bilibili",
        xiaohongshu: preselectPlatform === "xiaohongshu",
      });
    }

    if (selectedId === "REMIX") {
      const rmx = manifest.remix;
      const baseTitle = (
        titleDraft.trim() ||
        rmx.title ||
        manifest.metadata.original_theme_zh ||
        rmx.cover_title ||
        "高光混剪"
      ).trim();
      const baseCover = (coverTitleDraft.trim() || rmx.cover_title || baseTitle).trim();
      setPubTitle(baseTitle);
      setPubCoverTitle(baseCover);
      setPubDesc(buildDefaultRemixDesc(manifest));
      setPubTags([...(manifest.primary_topics || []), "AI", "深度访谈"].slice(0, 6).join(", "));
    } else if (selectedId === "COMPILATION") {
      const cmp = manifest.compilation;
      const baseTitle = (titleDraft.trim() || cmp?.title || "切片精选集").trim();
      const baseCover = (coverTitleDraft.trim() || cmp?.cover_title || baseTitle).trim();
      setPubTitle(baseTitle);
      setPubCoverTitle(baseCover);
      setPubDesc(buildDefaultCompilationDesc(manifest, compilationClips));
      setPubTags([...(manifest.primary_topics || []), "AI", "深度访谈", "合集"].slice(0, 6).join(", "));
    } else if (selectedId === "CLEAN_CUT") {
      const cc = manifest.clean_cut;
      const fallbackT = `${manifest.metadata.original_theme_zh || manifest.metadata.original_title || "全片"}（净版）`;
      const baseTitle = (titleDraft.trim() || cc?.title || fallbackT).trim();
      const baseCover = (coverTitleDraft.trim() || cc?.cover_title || baseTitle).trim();
      setPubTitle(baseTitle);
      setPubCoverTitle(baseCover);
      setPubDesc(buildDefaultCleanCutDesc(manifest));
      setPubTags([...(manifest.primary_topics || []), "AI", "深度访谈", "全片"].slice(0, 6).join(", "));
    } else if (activeClip) {
      const baseTitle = (titleDraft.trim() || activeClip.title || activeClip.cover_title || "").trim();
      const baseCover = (coverTitleDraft.trim() || activeClip.cover_title || baseTitle).trim();
      setPubTitle(baseTitle);
      setPubCoverTitle(baseCover);
      setPubDesc((activeClip.summary || activeClip.reason || baseTitle).trim());
      setPubTags(
        Array.from(new Set([...(manifest.primary_topics || []), ...activeClip.matched_hot_topics, "AI"]))
          .slice(0, 6)
          .join(", ")
      );
    }
    setPubCollection(defaultCollectionName(manifest));
    setPubPublic(false);
    publishDialogRef.current?.showModal();
  }

  async function handleSuggestPublishDesc() {
    if (!manifest || !activePublishTarget) return;
    setErrorMsg(null);
    setPubDescSuggesting(true);
    try {
      const res = await api.suggestPublishDesc(manifest.project_id, activePublishTarget);
      setManifest(res.manifest);
      setPubDesc(res.desc || "");
    } catch (err) {
      setErrorMsg((err as Error).message);
    } finally {
      setPubDescSuggesting(false);
    }
  }

  function openBatchPublishModal() {
    if (!manifest) return;
    setErrorMsg(null);
    if (!pubLoginWatchPlat) setPubLoginNotice(null);
    handleRefreshPublishStatus();
    const initPick: Record<string, boolean> = {};
    manifest.clips
      .filter((c) => c.enabled)
      .forEach((c) => {
        initPick[c.clip_id] = Boolean(c.output_file);
      });
    setBatchClipPick(initPick);
    setBatchCollection((prev) => prev || defaultCollectionName(manifest));
    setBatchPublic(false);
    setBatchSkipPublished(true);
    batchPublishDialogRef.current?.showModal();
  }

  async function handleConfirmBatchPublish() {
    if (!manifest) return;
    const chosenPlatforms = Object.entries(batchPlatforms)
      .filter(([, on]) => on)
      .map(([k]) => k);
    if (chosenPlatforms.length === 0) {
      setErrorMsg("请至少选择一个发布平台");
      return;
    }
    const chosenTargets = manifest.clips
      .filter((c) => c.enabled && Boolean(c.output_file) && batchClipPick[c.clip_id])
      .map((c) => c.clip_id);
    if (chosenTargets.length === 0) {
      setErrorMsg("请至少选择 1 个已渲染的独立切片");
      return;
    }
    setErrorMsg(null);
    setBatchSubmitting(true);
    try {
      const colName = batchCollection.trim();
      const res = await api.publishBatch(manifest.project_id, {
        targets: chosenTargets,
        platforms: chosenPlatforms,
        is_public: batchPublic,
        collection: colName,
        skip_already_published: batchSkipPublished,
      });
      batchPublishDialogRef.current?.close();
      setPublishJobs((prev) => [
        ...prev.filter((j) => j.job_id !== res.job_id),
        {
          job_id: res.job_id,
          kind: "batch",
          running: true,
          cancelling: false,
          targets: res.targets,
          current_target: res.targets[0] || null,
          completed_targets: [],
          failed_targets: [],
          skipped_targets: [],
          platforms: res.platforms,
          collection: res.collection,
        },
      ]);
      api
        .getProject(manifest.project_id)
        .then((d) => {
          setManifest(d.manifest);
          setPublishJobs((d.publish_jobs || []).filter((j) => j.running));
        })
        .catch(() => {});
    } catch (err) {
      setErrorMsg((err as Error).message);
    } finally {
      setBatchSubmitting(false);
    }
  }

  async function handleLaunchLogin(plat: string) {
    setErrorMsg(null);
    try {
      const res = await api.launchPublishLogin(plat);
      setPubLoginNotice(res.message);
      setPubLoginWatchPlat(plat);
      await handleRefreshPublishStatus();
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  function renderPublishDiagnostics(target: string, records?: PublishRecord[]) {
    const failed = (records || []).filter((r) => r.status === "error");
    if (failed.length === 0) return null;
    const targetBusy = Boolean(getPublishJobForTarget(target));
    return (
      <div className="pub-error-box">
        {failed.map((rec) => {
          const platLabel = PLAT_LABELS[rec.platform] || rec.platform;
          const { errorText, solutionText } = formatPublishErrorView(rec);
          const resetKey = `${target}:${rec.platform}`;
          const loginSt = platStatus[rec.platform];
          return (
            <div key={rec.platform} className="pub-error-item">
              <div className="pub-error-head">
                <span className="pub-error-title">
                  {rec.message || `${platLabel} 发布失败`}
                </span>
                <div className="pub-error-actions">
                  <button
                    type="button"
                    className="sm"
                    disabled={jobRunning || targetBusy || pubResettingKey !== null}
                    onClick={() => handleLaunchLogin(rec.platform)}
                  >
                    {loginSt?.login_running ? "扫码中..." : "重扫码"}
                  </button>
                  <button
                    type="button"
                    className="sm"
                    disabled={jobRunning || targetBusy || pubResettingKey !== null}
                    onClick={() => handleResetPublish(target, rec.platform)}
                  >
                    {pubResettingKey === resetKey ? "复位中..." : "复位状态"}
                  </button>
                  <button
                    type="button"
                    className="sm primary"
                    disabled={jobRunning || targetBusy || pubResettingKey !== null}
                    onClick={() => openPublishModal(rec.platform)}
                  >
                    重试{platLabel}
                  </button>
                </div>
              </div>
              <div className="pub-error-msg">原因：{errorText}</div>
              <div className="pub-error-solution">解决方案：{solutionText}</div>
            </div>
          );
        })}
      </div>
    );
  }

  async function handleConfirmPublish() {
    if (!manifest || !selectedId) return;
    const chosen = Object.entries(pubPlatforms)
      .filter(([, on]) => on)
      .map(([k]) => k);
    if (chosen.length === 0) {
      setErrorMsg("请至少选择一个发布平台");
      return;
    }
    setErrorMsg(null);
    const targetKey =
      selectedId === "REMIX"
        ? "remix"
        : selectedId === "COMPILATION"
        ? "compilation"
        : selectedId === "CLEAN_CUT"
        ? "clean_cut"
        : selectedId;
    try {
      const tagsList = pubTags
        .split(/[,，、]/)
        .map((x) => x.trim())
        .filter(Boolean);
      const res = await api.publishTarget(manifest.project_id, {
        target: targetKey,
        platforms: chosen,
        title: pubTitle.trim(),
        cover_title: pubCoverTitle.trim(),
        desc: pubDesc.trim(),
        tags: tagsList,
        is_public: pubPublic,
        collection: pubCollection.trim(),
        ui_schema: 2,
      });
      publishDialogRef.current?.close();
      setPublishJobs((prev) => [
        ...prev.filter((j) => j.job_id !== res.job_id),
        {
          job_id: res.job_id,
          kind: "single",
          running: true,
          cancelling: false,
          targets: [targetKey],
          current_target: targetKey,
          completed_targets: [],
          failed_targets: [],
          platforms: chosen,
          collection: pubCollection.trim(),
        },
      ]);
      api
        .getProject(manifest.project_id)
        .then((d) => {
          setManifest(d.manifest);
          setPublishJobs((d.publish_jobs || []).filter((j) => j.running));
        })
        .catch(() => {});
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  const contextUtterances = useMemo(() => {
    if (!manifest || !activeClip) return [];
    const pad = 6;
    const lo = Math.max(0, activeClip.start_utt - pad);
    const hi = Math.min(manifest.utterances.length - 1, activeClip.end_utt + pad);
    return manifest.utterances.slice(lo, hi + 1);
  }, [manifest, activeClip]);

  const repairUtterances = useMemo(() => {
    if (!manifest || !activeClip) return [];
    return manifest.utterances.slice(activeClip.start_utt, activeClip.end_utt + 1);
  }, [manifest, activeClip]);

  async function handleRepairSubtitles() {
    if (!manifest || !activeClip) return;
    const repairs = repairUtterances
      .map((u) => {
        const draft = subtitleDrafts[u.id] || { zh: u.zh || "", en: u.en || "" };
        return {
          utterance_id: u.id,
          zh: draft.zh,
          en: draft.en,
        };
      })
      .filter((item) => {
        const u = manifest.utterances[item.utterance_id];
        return item.zh !== (u.zh || "") || item.en !== (u.en || "");
      });
    if (repairs.length === 0) {
      setErrorMsg("没有需要保存的字幕修改");
      return;
    }
    setSubtitleRepairSaving(true);
    try {
      const res = await api.repairClipSubtitles(manifest.project_id, activeClip.clip_id, {
        repairs,
        rebuild_full_subtitle: true,
        render_clip: true,
        render_full_video: subtitleRepairFullBurn,
      });
      setManifest(res.manifest);
      setPlayMode("rendered");
    } catch (err) {
      setErrorMsg((err as Error).message);
    } finally {
      setSubtitleRepairSaving(false);
    }
  }

  const mediaSrc = useMemo(() => {
    if (!manifest) return "";
    if (selectedId === "REMIX") {
      if (playMode === "rendered" && manifest.remix.output_file) {
        return `/api/projects/${manifest.project_id}/media/remix?v=${manifest.remix.last_rendered_hash}`;
      }
      return `/api/projects/${manifest.project_id}/media/source`;
    }
    if (selectedId === "COMPILATION") {
      if (playMode === "rendered" && manifest.compilation?.output_file) {
        return `/api/projects/${manifest.project_id}/media/compilation?v=${manifest.compilation.last_rendered_hash}`;
      }
      return `/api/projects/${manifest.project_id}/media/source`;
    }
    if (selectedId === "CLEAN_CUT") {
      if (playMode === "rendered" && manifest.clean_cut?.output_file) {
        return `/api/projects/${manifest.project_id}/media/clean_cut?v=${manifest.clean_cut.last_rendered_hash}`;
      }
      return `/api/projects/${manifest.project_id}/media/source`;
    }
    if (playMode === "rendered" && activeClip?.output_file) {
      return `/api/projects/${manifest.project_id}/media/clip?clip_id=${activeClip.clip_id}&v=${activeClip.last_rendered_hash}`;
    }
    return `/api/projects/${manifest.project_id}/media/source`;
  }, [manifest, selectedId, playMode, activeClip]);

  const dirtyCount = useMemo(() => {
    if (!manifest) return 0;
    let n = manifest.clips.filter((c) => c.enabled && c.render_hash !== c.last_rendered_hash).length;
    if (
      manifest.remix.enabled &&
      manifest.remix.items.some((x) => x.enabled) &&
      manifest.remix.render_hash !== manifest.remix.last_rendered_hash
    ) {
      n += 1;
    }
    if (
      manifest.compilation?.enabled &&
      manifest.clips.some((c) => c.enabled) &&
      manifest.compilation.render_hash !== manifest.compilation.last_rendered_hash
    ) {
      n += 1;
    }
    if (
      manifest.clean_cut?.enabled &&
      manifest.clean_cut.render_hash !== manifest.clean_cut.last_rendered_hash
    ) {
      n += 1;
    }
    return n;
  }, [manifest]);

  const cleanCutStats = useMemo(() => {
    const segs = manifest?.cut_segments || [];
    const cc = manifest?.clean_cut;
    const sens = (cc?.sensitivity || contentFilterCfg?.default_sensitivity || "balanced") as
      | "strict"
      | "balanced"
      | "loose";
    const trigList =
      contentFilterCfg?.political_profiles?.[sens]?.trigger_severities ||
      DEFAULT_SENS_TRIGGERS[sens] ||
      DEFAULT_SENS_TRIGGERS.strict;
    const triggerSet = new Set(trigList);
    const matchesSeg = (s: CutSegment) =>
      s.kind !== "political_sensitive" ||
      s.detector === "manual" ||
      triggerSet.has((s.severity || "high").toLowerCase());

    const adTotal = segs.filter((s) => s.kind === "ad").length;
    const allPolSegs = segs.filter((s) => s.kind === "political_sensitive");
    const polAllTotal = allPolSegs.length;
    const matchedPolSegs = allPolSegs.filter((s) => matchesSeg(s));
    const polTotal = matchedPolSegs.length;
    const hasAds = adTotal > 0;
    const hasPol = polTotal > 0;
    const isScanned = Boolean(cc?.scanned_at || manifest?.stages?.scan_cuts?.status === "completed");
    const active = segs.filter(
      (s) =>
        s.enabled &&
        matchesSeg(s) &&
        ((s.kind === "ad" && Boolean(cc?.cut_ads)) ||
          (s.kind === "political_sensitive" && Boolean(cc?.cut_political_sensitive)))
    );
    const adCount = active.filter((s) => s.kind === "ad").length;
    const polActive = active.filter((s) => s.kind === "political_sensitive");
    const polCount = polActive.length;
    const delActive = active.filter((s) => resolveSegActionFrontend(s, cc, canMask) === "delete");
    const normalMaskActive = active.filter((s) => resolveSegActionFrontend(s, cc, canMask) === "mask");
    const preciseMaskActive = active.filter((s) => resolveSegActionFrontend(s, cc, canMask) === "precise_mask");
    const maskActive = [...normalMaskActive, ...preciseMaskActive];
    const missingPreciseCount = allPolSegs.filter(
      (s) =>
        !isVisualOcrSeg(s) &&
        !s.precise_terms_extracted &&
        !(s.precise_terms_zh && s.precise_terms_zh.length > 0) &&
        !(s.precise_terms_en && s.precise_terms_en.length > 0)
    ).length;
    const delPolCount = Math.max(0, polCount - maskActive.length);
    const spans = delActive
      .map((s) => [Math.max(0, s.start_s), Math.max(0, s.end_s)] as [number, number])
      .filter(([s, e]) => e > s);
    const merged = mergeSpanPairs(spans, 0.15);
    const cutS = merged.reduce((acc, [s, e]) => acc + (e - s), 0);
    const totalS = manifest?.source.duration_s || 0;
    const keptS = Math.max(0, totalS - cutS);
    const politicalAction = (cc?.political_action || contentFilterCfg?.default_political_action || "auto") as PoliticalAction;
    const hasGlobalCrop = Boolean(manifest?.crop?.enabled);
    const summaryLabel =
      active.length > 0
        ? `广告${adCount}/${adTotal} · 删敏${delPolCount} · ${preciseMaskActive.length > 0 && normalMaskActive.length === 0 ? "精准" : "遮蔽"}${maskActive.length}`
        : !isScanned && segs.length === 0
          ? hasGlobalCrop
            ? "仅画面裁剪"
            : "待分析"
          : polAllTotal > 0
            ? `广告${adCount}/${adTotal} · 当前标准放行${polAllTotal}段`
            : hasGlobalCrop
              ? "画面裁剪"
              : "无广告/敏感";
    return {
      total: segs.length,
      adTotal,
      polTotal,
      polAllTotal,
      hasAds,
      hasPol,
      isScanned,
      activeCount: active.length,
      adCount,
      polCount,
      delPolCount,
      maskCount: maskActive.length,
      normalMaskCount: normalMaskActive.length,
      preciseCount: preciseMaskActive.length,
      missingPreciseCount,
      cutS,
      keptS,
      merged,
      summaryLabel,
      matchesSeg,
      sensitivity: sens,
      politicalAction,
    };
  }, [manifest, contentFilterCfg, canMask]);

  const remixCompliance = useMemo(() => {
    const empty = summarizeIntervalsCompliance([], [], manifest?.clean_cut, cleanCutStats.matchesSeg, canMask);
    if (!manifest) return empty;
    const intervals = manifest.remix.items
      .filter((it) => it.enabled)
      .map((it) => [it.start_s, it.end_s] as [number, number]);
    return summarizeIntervalsCompliance(
      intervals,
      manifest.cut_segments || [],
      manifest.clean_cut,
      cleanCutStats.matchesSeg,
      canMask
    );
  }, [manifest, cleanCutStats.matchesSeg, canMask]);

  const compilationClips: ClipItem[] = useMemo(() => {
    if (!manifest) return [];
    const rawIds = manifest.compilation?.clip_ids;
    if (Array.isArray(rawIds)) {
      const byId = new Map(manifest.clips.map((c) => [c.clip_id, c]));
      return rawIds.map((cid) => byId.get(cid)).filter((c): c is ClipItem => Boolean(c));
    }
    return manifest.clips.filter((c) => c.enabled);
  }, [manifest]);

  const compilationCompliance = useMemo(() => {
    const empty = summarizeIntervalsCompliance([], [], manifest?.clean_cut, cleanCutStats.matchesSeg, canMask);
    if (!manifest) return empty;
    const intervals = compilationClips.map((c) => [c.start_s, c.end_s] as [number, number]);
    return summarizeIntervalsCompliance(
      intervals,
      manifest.cut_segments || [],
      manifest.clean_cut,
      cleanCutStats.matchesSeg,
      canMask
    );
  }, [manifest, compilationClips, cleanCutStats.matchesSeg, canMask]);

  const activeClipCompliance = useMemo(() => {
    const empty = summarizeIntervalsCompliance([], [], manifest?.clean_cut, cleanCutStats.matchesSeg, canMask);
    if (!manifest || !activeClip) return empty;
    return summarizeIntervalsCompliance(
      [[activeClip.start_s, activeClip.end_s]],
      manifest.cut_segments || [],
      manifest.clean_cut,
      cleanCutStats.matchesSeg,
      canMask
    );
  }, [manifest, activeClip, cleanCutStats.matchesSeg, canMask]);

  const enabledClipCount = useMemo(() => {
    return manifest ? manifest.clips.filter((c) => c.enabled).length : 0;
  }, [manifest]);

  const selectedStandaloneClips = useMemo(() => {
    return manifest ? manifest.clips.filter((c) => c.enabled) : [];
  }, [manifest]);

  const readySelectedClips = useMemo(() => {
    return selectedStandaloneClips.filter((c) => Boolean(c.output_file));
  }, [selectedStandaloneClips]);

  const compilationTotalS = useMemo(() => {
    if (!manifest) return 0;
    const baseSpeechS =
      cleanCutStats.isScanned
        ? compilationCompliance.netDurS
        : compilationClips.reduce((acc, c) => acc + (c.end_s - c.start_s), 0);
    const compStyle = navCfg.compilation_style || navCfg.style;
    if (compStyle === "title_card" || compStyle === "both") {
      const cardDur = manifest.compilation?.card_duration_s ?? 1.8;
      return baseSpeechS + compilationClips.length * cardDur;
    }
    return baseSpeechS;
  }, [manifest, compilationClips, cleanCutStats.isScanned, compilationCompliance.netDurS, navCfg.compilation_style, navCfg.style]);

  const enabledRemixItemCount = useMemo(() => {
    return manifest ? manifest.remix.items.filter((x) => x.enabled).length : 0;
  }, [manifest]);

  const hasSubtitles = Boolean(manifest && manifest.utterances.length > 0);
  const hasGlobalCrop = Boolean(manifest?.crop?.enabled);
  const showRemixCard = hasRemixCardFor(manifest);
  const showCompilationCard = hasCompilationCardFor(manifest);
  const hasActiveRemixTarget = Boolean(
    manifest?.remix.enabled &&
      (enabledRemixItemCount > 0 || (manifest.remix.items.length === 0 && (manifest.clips.length > 0 || (manifest.remix_only_pool?.length || 0) > 0)))
  );
  const hasActiveCompilationTarget = Boolean(manifest?.compilation?.enabled && compilationClips.length > 0);
  const hasActiveCleanCutTarget = Boolean(manifest?.clean_cut?.enabled && (cleanCutStats.activeCount > 0 || hasGlobalCrop));

  const renderTargetSummary = useMemo(() => {
    if (!manifest) return "0 项";
    const parts: string[] = [];
    if (enabledClipCount > 0) parts.push(`${enabledClipCount}切片`);
    if (hasActiveRemixTarget) parts.push("AI混剪");
    if (hasActiveCompilationTarget) parts.push("精选集");
    if (hasActiveCleanCutTarget) parts.push(cleanCutStats.activeCount === 0 && hasGlobalCrop ? "画面裁剪" : "全片净版");
    return parts.join(" + ") || "0 项";
  }, [manifest, enabledClipCount, hasActiveRemixTarget, hasActiveCompilationTarget, hasActiveCleanCutTarget, cleanCutStats.activeCount, hasGlobalCrop]);

  const isRendering = Boolean(jobRunning && (jobStages.includes("render") || manifest?.stages?.render?.status === "running"));
  const isPlanningRunning = Boolean(manifest && jobRunning && !isRendering);
  const cleanCutPubJob = useMemo(() => getPublishJobForTarget("clean_cut"), [publishJobs]);
  const remixPubJob = useMemo(() => getPublishJobForTarget("remix"), [publishJobs]);
  const compPubJob = useMemo(() => getPublishJobForTarget("compilation"), [publishJobs]);
  const clipPubJob = useMemo(
    () => getPublishJobForTarget(activeClip?.clip_id || null),
    [publishJobs, activeClip?.clip_id]
  );
  const hasStrategy = useMemo(() => {
    if (!manifest) return false;
    if (
      manifest.clips.length > 0 ||
      manifest.topic_map.length > 0 ||
      manifest.remix.items.length > 0 ||
      (manifest.cut_segments || []).length > 0 ||
      Boolean(manifest.clean_cut?.scanned_at) ||
      Boolean(manifest.remix.output_file) ||
      Boolean(manifest.compilation?.output_file) ||
      Boolean(manifest.clean_cut?.output_file) ||
      Boolean(projectAssets?.has_strategy || projectAssets?.has_rendered)
    ) {
      return true;
    }
    return ["scan_cuts", "analyze", "verify_and_plan", "render"].some((k) => {
      const st = manifest.stages?.[k];
      return Boolean(st && st.status === "completed");
    });
  }, [manifest, projectAssets]);

  const remixStats = useMemo(() => {
    if (!manifest) {
      return { themeCount: 0, segmentCount: 0, estTotalS: 0, ratioPct: "0.0" };
    }
    const enabled = [...manifest.remix.items].filter((x) => x.enabled).sort((a, b) => a.order - b.order);
    const groups = new Set<string>();
    let cardCount = 0;
    let lastGroup: string | null = null;
    let speechS = 0;
    enabled.forEach((it, idx) => {
      const gk = (it.group_label || "").trim() || it.theme_id || it.remix_id;
      groups.add(gk);
      speechS += Math.max(0, it.end_s - it.start_s);
      if (manifest.remix.card_mode === "every") {
        cardCount += 1;
      } else if (manifest.remix.card_mode === "group" && (idx === 0 || gk !== lastGroup)) {
        cardCount += 1;
      }
      lastGroup = gk;
    });
    const remixStyle = navCfg.remix_style || navCfg.style;
    const cardDur =
      (remixStyle === "title_card" || remixStyle === "both") && manifest.remix.card_mode !== "none"
        ? manifest.remix.card_duration_s || 1.4
        : 0;
    const estTotalS = speechS + cardCount * cardDur;
    const srcDur = manifest.source.duration_s || 0;
    const ratioPct = srcDur > 0 ? ((estTotalS / srcDur) * 100).toFixed(1) : "0.0";
    return { themeCount: groups.size, segmentCount: enabled.length, estTotalS, ratioPct };
  }, [manifest, navCfg.remix_style, navCfg.style]);

  const currentCoverUrl = useMemo(() => {
    if (!manifest) return null;
    if (selectedId === "REMIX" && manifest.remix.cover_file) {
      return `/api/projects/${manifest.project_id}/media/remix_cover?v=${encodeURIComponent(
        manifest.remix.cover_hash || manifest.remix.last_rendered_hash || "1"
      )}&n=${coverNonce}`;
    }
    if (selectedId === "COMPILATION" && manifest.compilation?.cover_file) {
      return `/api/projects/${manifest.project_id}/media/compilation_cover?v=${encodeURIComponent(
        manifest.compilation.cover_hash || manifest.compilation.last_rendered_hash || "1"
      )}&n=${coverNonce}`;
    }
    if (selectedId === "CLEAN_CUT" && manifest.clean_cut?.cover_file) {
      return `/api/projects/${manifest.project_id}/media/clean_cut_cover?v=${encodeURIComponent(
        manifest.clean_cut.cover_hash || manifest.clean_cut.last_rendered_hash || "1"
      )}&n=${coverNonce}`;
    }
    if (activeClip?.cover_file) {
      return `/api/projects/${manifest.project_id}/media/cover?clip_id=${encodeURIComponent(
        activeClip.clip_id
      )}&v=${encodeURIComponent(activeClip.cover_hash || activeClip.last_rendered_hash || "1")}&n=${coverNonce}`;
    }
    return null;
  }, [manifest, selectedId, activeClip, coverNonce]);

  return (
    <div className="app-shell">
      {/* Compact Top Bar */}
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">CC</span>
          <span>ClipCut</span>
        </div>

        <div style={{ display: "flex", gap: 6, alignItems: "center", flex: 1 }}>
          <button type="button" onClick={() => handlePickPath("file")}>
            选视频
          </button>
          <button type="button" onClick={() => handlePickPath("folder")}>
            选文件夹
          </button>
          {projects.length > 0 && (
            <select
              aria-label="切换项目"
              value={manifest?.project_id || ""}
              onChange={(e) => e.target.value && loadProject(e.target.value)}
              style={{ maxWidth: 260 }}
            >
              {projects.map((p) => (
                <option key={p.project_id} value={p.project_id}>
                  {p.title.slice(0, 30)} ({p.clip_count})
                </option>
              ))}
            </select>
          )}
        </div>

        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <button
            type="button"
            className={isPlanningRunning ? "danger" : "primary"}
            disabled={isPlanningRunning ? jobCancelling : (!manifest || jobRunning || !hasSubtitles)}
            onClick={() =>
              isPlanningRunning
                ? handleCancelJob()
                : hasStrategy
                  ? openReplanDialog()
                  : handleRun(["signals", "scan_cuts", "analyze", "verify_and_plan"])
            }
            title={
              isPlanningRunning
                ? "点击立即中止当前规划任务"
                : manifest && !hasSubtitles
                  ? "当前视频未载入字幕文件，无法执行 AI 切片分析；请直接点击播放器下方「裁剪画面」框选并导出成片"
                : hasStrategy
                  ? "重新执行广告/敏感分析、AI 选片与 Remix 规划，并检查是否删除已有成片彻底重做"
                  : "第一步：AI声画测绘、广告/敏感分析、三类选片与Remix规划（不消耗重型算力渲染）"
            }
          >
            {isPlanningRunning
              ? (jobCancelling ? "正在中止..." : "规划中 · 中止")
              : hasStrategy
                ? "重新制定切片策略"
                : "生成切片策略"}
          </button>
          <button
            type="button"
            className={isRendering ? "danger" : undefined}
            disabled={isRendering ? jobCancelling : (!manifest || jobRunning || (enabledClipCount === 0 && !hasActiveRemixTarget && !hasActiveCompilationTarget && !hasActiveCleanCutTarget))}
            onClick={() => (isRendering ? handleCancelJob() : handleRun(["render"]))}
            title={isRendering ? "点击立即中止当前渲染任务" : "第二步：仅对当前勾选的独立切片、AI混剪、切片精选集与全片净版执行渲染"}
          >
            {isRendering ? (jobCancelling ? "正在中止..." : "渲染中 · 中止") : `渲染选中 (${renderTargetSummary})`}
          </button>
          <button
            type="button"
            className="sm"
            disabled={!manifest || jobRunning || !hasSubtitles}
            onClick={() => handleRun(["signals", "scan_cuts", "analyze", "verify_and_plan", "render"])}
            title={
              manifest && !hasSubtitles
                ? "当前视频未载入字幕文件，仅支持手动「裁剪画面」并导出"
                : "全自动无人值守：全流程跑完包含全量渲染"
            }
          >
            全自动跑完
          </button>
          <button type="button" onClick={() => settingsDialogRef.current?.showModal()}>
            设置
          </button>
          <button
            type="button"
            className={showLogs ? "primary" : ""}
            onClick={() => setShowLogs((v) => !v)}
          >
            日志
          </button>
        </div>
      </header>

      {/* Compact Status Strip */}
      <div>
        {errorMsg && (
          <div className="error-banner" role="alert">
            <span>{errorMsg}</span>
            <button type="button" className="sm" onClick={() => setErrorMsg(null)}>
              关闭
            </button>
          </div>
        )}
        <div className="stage-bar">
          <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <div className="stage-pills">
              {Object.entries(STAGE_SHORT_LABELS).map(([k, label]) => {
                const st = manifest?.stages?.[k];
                const status = st?.status || "idle";
                return (
                  <span key={k} className={`stage-pill ${status}`} title={st?.message || ""}>
                    <span>{label}</span>
                  </span>
                );
              })}
            </div>
            <div style={{ display: "inline-flex", alignItems: "center", gap: 8, paddingLeft: 8, borderLeft: "1px solid var(--border)", flexWrap: "wrap" }}>
              <span style={{ fontSize: 11, color: "var(--text-2)", fontWeight: 600 }}>全局:</span>
              <label
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 4,
                  opacity: manifest && cleanCutStats.hasAds ? 1 : 0.45,
                  cursor: manifest && cleanCutStats.hasAds && !jobRunning ? "pointer" : "not-allowed",
                }}
                title={
                  !manifest
                    ? "请先打开视频并生成切片策略"
                    : cleanCutStats.hasAds
                      ? `已识别 ${cleanCutStats.adTotal} 段广告，全局勾选后自动从净版及各切片/混剪/精选集中删去`
                      : cleanCutStats.isScanned
                        ? "本片未检测到广告片段，不适用此项"
                        : "点击「生成切片策略」自动分析广告片段"
                }
              >
                <input
                  type="checkbox"
                  disabled={!manifest || !cleanCutStats.hasAds || jobRunning}
                  checked={Boolean(manifest && cleanCutStats.hasAds && manifest.clean_cut?.cut_ads)}
                  onChange={(e) => handlePatchCleanCut({ cut_ads: e.target.checked })}
                />
                <span>去除广告</span>
                <span style={{ fontSize: 11, color: "var(--text-2)" }}>
                  ({!manifest ? "未载入" : cleanCutStats.hasAds ? `${cleanCutStats.adTotal}段` : cleanCutStats.isScanned ? "无" : "待分析"})
                </span>
              </label>
              <label
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 4,
                  opacity: manifest && cleanCutStats.hasPol ? 1 : 0.45,
                  cursor: manifest && cleanCutStats.hasPol && !jobRunning ? "pointer" : "not-allowed",
                }}
                title={
                  !manifest
                    ? "请先打开视频并生成切片策略"
                    : cleanCutStats.hasPol
                      ? `当前标准命中 ${cleanCutStats.polTotal} 段政治敏感内容（总扫出 ${cleanCutStats.polAllTotal} 段），全局联动净版与全部切片/混剪/精选集`
                      : cleanCutStats.polAllTotal > 0
                        ? `已扫出 ${cleanCutStats.polAllTotal} 段边缘内容，当前政审标准下已全部放行`
                        : cleanCutStats.isScanned
                          ? "本片未检测到政治敏感内容，不适用此项"
                          : "点击「生成切片策略」自动分析政治敏感内容"
                }
              >
                <input
                  type="checkbox"
                  disabled={!manifest || !cleanCutStats.hasPol || jobRunning}
                  checked={Boolean(manifest && cleanCutStats.hasPol && manifest.clean_cut?.cut_political_sensitive)}
                  onChange={(e) => handlePatchCleanCut({ cut_political_sensitive: e.target.checked })}
                />
                <span>去除政治敏感</span>
                <span style={{ fontSize: 11, color: "var(--text-2)" }}>
                  (
                  {!manifest
                    ? "未载入"
                    : cleanCutStats.hasPol
                      ? `${cleanCutStats.polTotal}段`
                      : cleanCutStats.polAllTotal > 0
                        ? `放行${cleanCutStats.polAllTotal}段`
                        : cleanCutStats.isScanned
                          ? "无"
                          : "待分析"}
                  )
                </span>
              </label>
              <select
                aria-label="顶部政审标准"
                disabled={!manifest || jobRunning}
                value={cleanCutStats.sensitivity}
                onChange={(e) => handlePatchCleanCut({ sensitivity: e.target.value })}
                title="调整政审 Level，联动改变受影响切片与命中状态"
                style={{ padding: "2px 6px", fontSize: 11 }}
              >
                {(["strict", "balanced", "loose"] as const).map((k) => (
                  <option key={k} value={k}>
                    政审:{SENS_SHORT_LABELS[k]}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="sm"
                onClick={() => settingsDialogRef.current?.showModal()}
                title="当前脱敏策略（点击前往右上角「设置」调整脱敏策略与政审规则）"
              >
                策略:
                {cleanCutStats.politicalAction === "precise_mask"
                  ? "精准遮蔽"
                  : cleanCutStats.politicalAction === "mask"
                    ? "常规遮蔽"
                    : cleanCutStats.politicalAction === "delete"
                      ? "删段"
                      : "智能"}
              </button>
              {manifest && cleanCutStats.total > 0 && (
                <button
                  type="button"
                  className="sm"
                  onClick={() => {
                    setSelectedId("CLEAN_CUT");
                    setPlayMode(manifest.clean_cut?.output_file ? "rendered" : "source");
                  }}
                >
                  屏蔽明细 ({cleanCutStats.total})
                </button>
              )}
            </div>
          </div>
          {manifest && (
            <div style={{ display: "flex", gap: 10, alignItems: "center", color: "var(--text-2)", flexWrap: "wrap" }}>
              <span
                style={{ cursor: "pointer", display: "inline-flex", gap: 8, alignItems: "center" }}
                onClick={openMetadataDialog}
                title="点击编辑原视频元数据（频道 / 嘉宾 / 发布时间）"
              >
                <span style={{ color: "var(--text-1)" }}>
                  {manifest.metadata.channel_zh || manifest.metadata.channel || "未设频道"}
                </span>
                {manifest.metadata.guest_zh && <span>· {manifest.metadata.guest_zh}</span>}
                {manifest.metadata.published_date && <span className="mono">· {manifest.metadata.published_date}</span>}
              </span>
              <span className="mono">{fmtTime(manifest.source.duration_s)}</span>
              <button
                type="button"
                className="sm"
                onClick={openMetadataDialog}
                title="调整发布频道、被访谈人及发布时间（将同步至竖版画面元数据）"
              >
                编辑元数据
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Main 2-Column (or 3-Column with Logs Drawer) Workbench */}
      {!manifest ? (
        <div className="empty-state">
          <h3>选择视频文件或文件夹开始切片与分发</h3>
          <p>点击左上角「选视频」或「选文件夹」载入项目。</p>
        </div>
      ) : (
        <main
          ref={workbenchRef}
          className={`workbench ${showLogs ? "" : "no-right"} ${resizingAxis ? "is-resizing" : ""}`}
          style={{ "--left-pane-w": `${leftPaneWidth}px` } as CSSProperties}
        >
          {/* LEFT PANE: Compact Clips & Remix List */}
          <section className="pane left-pane">
            <div className="pane-header" style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: "stretch" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span title="左侧勾选仅控制是否单独导出各独立切片，不会清空精选集或锁定后的混剪脚本">
                  独立切片 ({enabledClipCount}/{manifest.clips.length})
                </span>
                <button
                  type="button"
                  className="sm"
                  onClick={() => api.openOutputFolder(manifest.project_id).catch((e) => setErrorMsg(e.message))}
                >
                  打开目录
                </button>
              </div>
              <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                <button
                  type="button"
                  className="sm"
                  disabled={manifest.clips.length === 0 || jobRunning}
                  onClick={() => handleBatchClips("top_n", 5)}
                  title="按【核心主题、行业争议、精彩交锋、硬核实操、综合王牌】5大策展维度精准精选"
                >
                  策划Top5
                </button>
                <button
                  type="button"
                  className="sm"
                  disabled={manifest.clips.length === 0 || jobRunning}
                  onClick={() => handleBatchClips("enable_all")}
                  title="全部启用"
                >
                  全选
                </button>
                <button
                  type="button"
                  className="sm"
                  disabled={manifest.clips.length === 0 || jobRunning}
                  onClick={() => handleBatchClips("disable_all")}
                  title="全部停用"
                >
                  全不选
                </button>
                <button
                  type="button"
                  className="sm"
                  disabled={manifest.clips.length === 0 || jobRunning}
                  onClick={handleResetClipOrder}
                  title="按原片时间轴顺序一键复原切片排序"
                >
                  复原时间轴
                </button>
                {activeBatchPublishJob ? (
                  <button
                    type="button"
                    className="sm danger"
                    disabled={Boolean(activeBatchPublishJob.cancelling)}
                    onClick={() => handleCancelPublishJob(activeBatchPublishJob.job_id)}
                    title="点击中止当前选中切片批量发布"
                  >
                    {activeBatchPublishJob.cancelling
                      ? "正在中止..."
                      : `批量发布中 (${
                          activeBatchPublishJob.completed_targets.length +
                          activeBatchPublishJob.failed_targets.length +
                          (activeBatchPublishJob.skipped_targets?.length || 0)
                        }/${activeBatchPublishJob.targets.length}) · 中止`}
                  </button>
                ) : (
                  <button
                    type="button"
                    className={readySelectedClips.length > 0 ? "sm primary" : "sm"}
                    disabled={readySelectedClips.length === 0 || jobRunning}
                    onClick={openBatchPublishModal}
                    title={
                      readySelectedClips.length > 0
                        ? `将当前选中的 ${readySelectedClips.length} 支已渲染独立切片批量发布到同一合集`
                        : selectedStandaloneClips.length > 0
                          ? "请先点击「渲染选中」生成所选切片的成片后再批量发布"
                          : "请先勾选至少 1 支独立切片并完成渲染"
                    }
                  >
                    批量发布选中 ({readySelectedClips.length})
                  </button>
                )}
              </div>
              {activeBatchPublishJob && (
                <div
                  style={{
                    fontSize: 11,
                    color: "var(--text-1)",
                    background: "var(--bg-2)",
                    border: "1px solid var(--border)",
                    borderRadius: 6,
                    padding: "4px 8px",
                    display: "flex",
                    justifyContent: "space-between",
                    gap: 6,
                    flexWrap: "wrap",
                  }}
                >
                  <span>
                    队列：{activeBatchPublishJob.current_target || "收尾中"}
                    {activeBatchPublishJob.collection ? ` · 合集「${activeBatchPublishJob.collection}」` : ""}
                  </span>
                  <span className="mono">
                    完 {activeBatchPublishJob.completed_targets.length}
                    {activeBatchPublishJob.failed_targets.length > 0
                      ? ` · 败 ${activeBatchPublishJob.failed_targets.length}`
                      : ""}
                  </span>
                </div>
              )}
            </div>
            <div className="pane-body scroller">
              {/* 1. Remix Card (Hidden when project has no planned clips/remix yet) */}
              {showRemixCard && (() => {
                const rb = getRenderBadge(manifest.remix, remixCompliance, cleanCutStats.isScanned, "待渲染");
                const rmxNetS = Math.max(
                  0,
                  remixStats.estTotalS -
                    (cleanCutStats.isScanned ? remixCompliance.adDelS + remixCompliance.polDelS : 0)
                );
                return (
                  <div
                    className={`clip-card ${selectedId === "REMIX" ? "active" : ""} ${!manifest.remix.enabled ? "disabled" : ""}`}
                    onClick={() => {
                      setSelectedId("REMIX");
                      setPlayMode(manifest.remix.output_file ? "rendered" : "source");
                    }}
                  >
                    <div className="clip-card-head">
                      <input
                        type="checkbox"
                        checked={Boolean(manifest.remix.enabled)}
                        aria-label="启用AI混剪"
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => handlePatchRemix({ enabled: e.target.checked })}
                      />
                      <span className="tag hot_topic">AI 混剪</span>
                      <div className="clip-title">{manifest.remix.title || "高光混剪集锦"}</div>
                    </div>
                    <div className="clip-meta">
                      <span>
                        {remixStats.segmentCount === 0 && manifest.clips.length > 0
                          ? enabledClipCount > 0
                            ? `待AI按已选${enabledClipCount}切片策展`
                            : "待AI全片自主策展"
                          : `${remixStats.themeCount}板块 · ${remixStats.segmentCount}段`}
                      </span>
                      <span className="mono">{fmtTime(rmxNetS)}</span>
                      {renderComplianceTags(remixCompliance, cleanCutStats.isScanned)}
                      <span className={`tag ${rb.cls}`} title={rb.title}>
                        {rb.label}
                      </span>
                      {renderPubBadges(manifest.remix.publish_records)}
                    </div>
                  </div>
                );
              })()}

              {/* 2. Compilation Card (Selected Clips Sequential with Title Cards; Hidden when empty) */}
              {showCompilationCard && (() => {
                const compStub = manifest.compilation || {
                  render_hash: "",
                  last_rendered_hash: "",
                  output_file: null,
                };
                const cb = getRenderBadge(compStub, compilationCompliance, cleanCutStats.isScanned, "待渲染");
                return (
                  <div
                    className={`clip-card ${selectedId === "COMPILATION" ? "active" : ""} ${!manifest.compilation?.enabled ? "disabled" : ""}`}
                    onClick={() => {
                      setSelectedId("COMPILATION");
                      setPlayMode(manifest.compilation?.output_file ? "rendered" : "source");
                    }}
                  >
                    <div className="clip-card-head">
                      <input
                        type="checkbox"
                        checked={Boolean(manifest.compilation?.enabled)}
                        aria-label="启用切片精选集"
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => handlePatchCompilation({ enabled: e.target.checked })}
                      />
                      <span className="tag highlight" style={{ background: "rgba(59, 130, 246, 0.18)", color: "#60a5fa", borderColor: "rgba(59, 130, 246, 0.45)" }}>
                        精选集
                      </span>
                      <div className="clip-title">{manifest.compilation?.title || "切片精选集"}</div>
                    </div>
                    <div className="clip-meta">
                      <span>顺序串联 {compilationClips.length} 个切片</span>
                      <span className="mono">{fmtTime(compilationTotalS)}</span>
                      {renderComplianceTags(compilationCompliance, cleanCutStats.isScanned)}
                      <span className={`tag ${cb.cls}`} title={cb.title}>
                        {cb.label}
                      </span>
                      {renderPubBadges(manifest.compilation?.publish_records)}
                    </div>
                  </div>
                );
              })()}

              {/* 3. Clean Cut Card (Ad & Political-Sensitive Removal or Global Crop; Hidden when nothing active) */}
              {(cleanCutStats.activeCount > 0 || hasGlobalCrop || Boolean(manifest.clean_cut?.output_file)) && (
              <div
                className={`clip-card ${selectedId === "CLEAN_CUT" ? "active" : ""} ${!manifest.clean_cut?.enabled ? "disabled" : ""}`}
                onClick={() => {
                  setSelectedId("CLEAN_CUT");
                  setPlayMode(manifest.clean_cut?.output_file ? "rendered" : "source");
                }}
              >
                <div className="clip-card-head">
                  <input
                    type="checkbox"
                    checked={Boolean(manifest.clean_cut?.enabled)}
                    aria-label="启用全片净版"
                    onClick={(e) => e.stopPropagation()}
                    onChange={(e) => handlePatchCleanCut({ enabled: e.target.checked })}
                  />
                  <span
                    className="tag highlight"
                    style={{
                      background: "rgba(220, 38, 38, 0.22)",
                      color: "#f87171",
                      borderColor: "rgba(239, 68, 68, 0.55)",
                    }}
                  >
                    {cleanCutStats.activeCount === 0 && hasGlobalCrop ? "画面裁剪" : "全片净版"}
                  </span>
                  <div className="clip-title" style={{ color: "#fecaca" }}>
                    {manifest.clean_cut?.title ||
                      (cleanCutStats.activeCount === 0 && hasGlobalCrop
                        ? `${manifest.metadata.original_theme_zh || manifest.metadata.original_title || "全片"}（裁剪版）`
                        : `${manifest.metadata.original_theme_zh || manifest.metadata.original_title || "全片"}（净版）`)}
                  </div>
                </div>
                <div className="clip-meta">
                  <span>{cleanCutStats.summaryLabel}</span>
                  {cleanCutStats.cutS > 0.5 && (
                    <span className="mono" style={{ color: "#fca5a5" }}>
                      -{fmtTime(cleanCutStats.cutS)}
                    </span>
                  )}
                  {cleanCutStats.maskCount > 0 && (
                    <span className="tag" style={{ background: "rgba(56, 189, 248, 0.18)", color: "#7dd3fc" }}>
                      遮蔽 {cleanCutStats.maskCount}处
                    </span>
                  )}
                  <span className="mono">{fmtTime(cleanCutStats.keptS)}</span>
                  <span
                    className={`tag ${
                      manifest.clean_cut?.output_file &&
                      manifest.clean_cut.render_hash === manifest.clean_cut.last_rendered_hash
                        ? "cached"
                        : "dirty"
                    }`}
                  >
                    {manifest.clean_cut?.output_file
                      ? manifest.clean_cut.render_hash === manifest.clean_cut.last_rendered_hash
                        ? "已渲染"
                        : "待重渲"
                      : "待渲染"}
                  </span>
                  {renderPubBadges(manifest.clean_cut?.publish_records)}
                </div>
              </div>
              )}

              {/* Standalone Clips */}
              {manifest.clips.length === 0 ? (
                <div className="empty-state" style={{ padding: 20 }}>
                  {!hasSubtitles
                    ? hasGlobalCrop
                      ? "当前为无字幕裁剪模式：已启用全局画面裁剪，点击右侧「渲染裁剪成片」或上方「渲染选中」即可导出裁剪视频。"
                      : "当前视频未附带 .srt/.vtt 字幕，已自动进入纯画面裁剪模式：请点击播放器下方「裁剪画面」框选保留区域并「应用裁剪」后直接导出。"
                    : hasStrategy
                    ? "当前暂无独立切片。点击上方「重新制定切片策略」从头重新规划，或点击「全自动跑完」一次完成规划与渲染。"
                    : "点击上方「生成切片策略」完成广告/敏感分析与选片，再点「渲染选中」输出成片与封面（或直接点「全自动跑完」）。"}
                </div>
              ) : (
                manifest.clips.map((c, idx) => {
                  const clipComp = summarizeIntervalsCompliance(
                    [[c.start_s, c.end_s]],
                    manifest.cut_segments || [],
                    manifest.clean_cut,
                    cleanCutStats.matchesSeg,
                    canMask
                  );
                  const rb = getRenderBadge(c, clipComp, cleanCutStats.isScanned, "待切");
                  return (
                    <div
                      key={c.clip_id}
                      className={`clip-card ${selectedId === c.clip_id ? "active" : ""} ${!c.enabled ? "disabled" : ""}`}
                      onClick={() => {
                        setSelectedId(c.clip_id);
                        setPlayMode("source");
                      }}
                    >
                      <div className="clip-card-head">
                        <input
                          type="checkbox"
                          checked={c.enabled}
                          aria-label={`启用切片 ${c.title}`}
                          onClick={(e) => e.stopPropagation()}
                          onChange={(e) => handlePatchClip(c.clip_id, { enabled: e.target.checked })}
                        />
                        <div className="clip-title">
                          {String(c.index).padStart(2, "0")}. {c.title}
                        </div>
                        <div style={{ display: "inline-flex", gap: 2, marginLeft: "auto" }}>
                          <button
                            type="button"
                            className="sm"
                            style={{ padding: "0 5px", fontSize: 11, minWidth: 20, height: 20, lineHeight: "20px" }}
                            title="上移"
                            disabled={idx === 0}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleMoveClip(idx, -1);
                            }}
                          >
                            ↑
                          </button>
                          <button
                            type="button"
                            className="sm"
                            style={{ padding: "0 5px", fontSize: 11, minWidth: 20, height: 20, lineHeight: "20px" }}
                            title="下移"
                            disabled={idx === manifest.clips.length - 1}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleMoveClip(idx, 1);
                            }}
                          >
                            ↓
                          </button>
                        </div>
                      </div>
                      <div className="clip-meta">
                        <span className="mono">
                          {fmtTime(c.start_s)}–{fmtTime(c.end_s)}
                        </span>
                        {renderComplianceTags(clipComp, cleanCutStats.isScanned)}
                        {c.curation_tag && (
                          <span
                            className="tag highlight"
                            style={{
                              background: "rgba(245, 158, 11, 0.18)",
                              color: "#f59e0b",
                              borderColor: "rgba(245, 158, 11, 0.45)",
                              fontWeight: 600,
                            }}
                          >
                            {c.curation_tag}
                          </span>
                        )}
                        {typeof c.score === "number" && (
                          <span className="tag" title="AI推荐评分">
                            ⭐ {c.score.toFixed(1)}
                          </span>
                        )}
                        {c.categories.map((cat) => (
                          <span key={cat} className={`tag ${cat}`}>
                            {CAT_LABELS[cat] || cat}
                          </span>
                        ))}
                        <span className={`tag ${rb.cls}`} title={rb.title}>
                          {rb.label}
                        </span>
                        {c.crop_override?.mode === "custom" && (
                          <span className="tag" style={{ color: "#22d3ee", borderColor: "rgba(34,211,238,0.45)" }}>
                            单独裁剪
                          </span>
                        )}
                        {c.crop_override?.mode === "none" && (
                          <span className="tag" style={{ color: "#94a3b8" }}>
                            不裁剪
                          </span>
                        )}
                        {renderPubBadges(c.publish_records)}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </section>

          <div
            role="separator"
            aria-orientation="vertical"
            aria-label="调节左侧栏宽度"
            tabIndex={0}
            className={`split-handle split-handle-v ${resizingAxis === "col" ? "dragging" : ""}`}
            onPointerDown={startLeftResize}
            onDoubleClick={() => {
              const defW = clampLeftWidth(285);
              setLeftPaneWidth(defW);
              persistLeftWidth(defW);
            }}
            onKeyDown={handleLeftHandleKeyDown}
            title="左右拖拽调节侧栏宽度，双击恢复默认"
          />

          {/* CENTER PANE: Player + Cover Title & Publish Card + Collapsible Controls */}
          <section
            ref={centerPaneRef}
            className="pane center-pane"
            style={{ "--bottom-pane-h": `${bottomPaneHeight}px` } as CSSProperties}
          >
            <div className="player-wrap">
              <VideoCropOverlay
                videoRef={videoRef}
                mediaSrc={mediaSrc}
                playMode={playMode}
                sourceWidth={manifest.source.width || 1920}
                sourceHeight={manifest.source.height || 1080}
                effectiveCrop={effectiveCrop}
                editing={cropEditing}
                draftRect={cropDraft}
                previewCroppedOnly={cropPreviewOnly}
                onDraftChange={setCropDraft}
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={(e) => {
                  if (activeClip && playMode === "source" && e.currentTarget.currentTime < 0.2) {
                    e.currentTarget.currentTime = activeClip.start_s;
                  }
                }}
              />
              <div className="player-controls">
                <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                  <span className="mono">{fmtTime(currentTime)}</span>
                  <button
                    type="button"
                    className={`sm ${playMode === "source" ? "primary" : ""}`}
                    onClick={() => setPlayMode("source")}
                  >
                    原片
                  </button>
                  <button
                    type="button"
                    className={`sm ${playMode === "rendered" ? "primary" : ""}`}
                    disabled={
                      selectedId === "REMIX"
                        ? !manifest.remix.output_file
                        : selectedId === "COMPILATION"
                        ? !manifest.compilation?.output_file
                        : selectedId === "CLEAN_CUT"
                        ? !manifest.clean_cut?.output_file
                        : !activeClip?.output_file
                    }
                    onClick={() => {
                      setCropEditing(false);
                      setPlayMode("rendered");
                    }}
                  >
                    播成品
                  </button>
                  <button
                    type="button"
                    className={`sm ${cropEditing ? "primary" : ""}`}
                    disabled={jobRunning}
                    onClick={() => (cropEditing ? setCropEditing(false) : openCropEditor())}
                    title="用鼠标在画面上框选保留区域，去除原视频硬字幕"
                  >
                    {cropEditing ? "退出框选" : effectiveCrop ? "调整裁剪框" : "裁剪画面"}
                  </button>
                  {effectiveCrop && !cropEditing && playMode === "source" && (
                    <>
                      <button
                        type="button"
                        className="sm"
                        onClick={() => setCropPreviewOnly((v) => !v)}
                        title="切换仅看框选区域画面或查看原画裁剪框位置"
                      >
                        {cropPreviewOnly ? "仅看框内" : "原画对照"}
                      </button>
                      {effectiveCropDims && (
                        <span className="mono" style={{ color: "#22d3ee", fontSize: 11 }}>
                          {effectiveCropDims.pw}×{effectiveCropDims.ph}
                        </span>
                      )}
                    </>
                  )}
                </div>
                {activeClip && (
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 5, cursor: "pointer" }}>
                    <input type="checkbox" checked={loopClip} onChange={(e) => setLoopClip(e.target.checked)} />
                    <span>区间循环</span>
                  </label>
                )}
              </div>
              {cropEditing && playMode === "source" && (
                <div className="crop-toolbar">
                  <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                    <span style={{ color: "#22d3ee", fontWeight: 600 }}>拖拽画框/调整边缘</span>
                    {activeClip && (
                      <>
                        <select
                          value={cropEditScope}
                          onChange={(e) => handleCropScopeChange(e.target.value as "project" | "clip")}
                          disabled={cropSaving}
                          style={{ fontSize: 12, padding: "2px 6px" }}
                        >
                          <option value="project">全局裁剪（全片/切片默认）</option>
                          <option value="clip">当前切片单独框选</option>
                        </select>
                        {activeClip.crop_override && (
                          <button
                            type="button"
                            className="sm"
                            disabled={cropSaving}
                            onClick={handleInheritClipCrop}
                            title="清除此切片的单独设置，恢复跟随全局裁剪"
                          >
                            跟随全局
                          </button>
                        )}
                      </>
                    )}
                    <button
                      type="button"
                      className="sm"
                      onClick={() => setCropDraft({ x: 0, y: 0, w: 1, h: 0.85 })}
                      title="快速裁掉底部 15% 硬字幕区"
                    >
                      去底15%
                    </button>
                    <button
                      type="button"
                      className="sm"
                      onClick={() => setCropDraft({ x: 0, y: 0, w: 1, h: 0.8 })}
                      title="快速裁掉底部 20% 硬字幕区"
                    >
                      去底20%
                    </button>
                    <button
                      type="button"
                      className="sm"
                      onClick={() => setCropDraft({ x: 0, y: 0, w: 1, h: 1 })}
                    >
                      全画幅
                    </button>
                  </div>
                  <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                    <button
                      type="button"
                      className="sm primary"
                      disabled={cropSaving}
                      onClick={handleApplyCrop}
                    >
                      {cropSaving ? "保存中..." : "应用裁剪"}
                    </button>
                    {(cropEditScope === "clip" ? activeClip?.crop_override?.mode !== "none" : manifest.crop?.enabled) && (
                      <button
                        type="button"
                        className="sm danger"
                        disabled={cropSaving}
                        onClick={handleDisableCropForScope}
                      >
                        {cropEditScope === "clip" ? "此切片不裁剪" : "关闭全局裁剪"}
                      </button>
                    )}
                    <button
                      type="button"
                      className="sm"
                      disabled={cropSaving}
                      onClick={() => setCropEditing(false)}
                    >
                      取消
                    </button>
                  </div>
                </div>
              )}
            </div>

            <div
              role="separator"
              aria-orientation="horizontal"
              aria-label="调节右下面板高度"
              tabIndex={0}
              className={`split-handle split-handle-h ${resizingAxis === "row" ? "dragging" : ""}`}
              onPointerDown={startBottomResize}
              onDoubleClick={() => {
                const defH = clampBottomHeight(380);
                setBottomPaneHeight(defH);
                persistBottomHeight(defH);
              }}
              onKeyDown={handleBottomHandleKeyDown}
              title="上下拖拽调节右下面板高度，双击恢复默认"
            />

            <div className="center-bottom-pane">
            {selectedId === "CLEAN_CUT" ? (
              <>
                {/* Unified Clean Cut Inspector & Publish Card */}
                <div className="clip-inspector-bar" style={{ flexDirection: "row", gap: 14, alignItems: "center" }}>
                  <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 7 }}>
                    <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <span style={{ color: "#fca5a5", fontSize: 12, width: 56, fontWeight: 600 }}>净版标题</span>
                      <input
                        type="text"
                        style={{ flex: 1, fontWeight: 600, color: "#fee2e2" }}
                        value={titleDraft}
                        onChange={(e) => setTitleDraft(e.target.value)}
                        onBlur={() =>
                          titleDraft.trim() &&
                          titleDraft !== (manifest.clean_cut?.title || "") &&
                          handlePatchCleanCut({ title: titleDraft.trim() })
                        }
                      />
                      <button
                        type="button"
                        className="sm"
                        disabled={cleanCutTitleSuggesting || jobRunning}
                        onClick={handleSuggestCleanCutTitle}
                        title="由 AI 根据全片中文主题与节目内容重新生成全片净版中文标题"
                      >
                        {cleanCutTitleSuggesting ? "生成中..." : "AI重新生成标题"}
                      </button>
                    </div>
                    <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <span style={{ color: "var(--text-2)", fontSize: 12, width: 56 }}>封面标题</span>
                      <input
                        type="text"
                        style={{ flex: 1 }}
                        placeholder={`留空则跟随净版标题（${titleDraft || manifest.clean_cut?.title || "全片净版"}）`}
                        value={coverTitleDraft}
                        onChange={(e) => setCoverTitleDraft(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleSaveCoverTitleAndRefresh()}
                      />
                      <button
                        type="button"
                        className="sm"
                        disabled={coverRefreshing || jobRunning}
                        onClick={handleSaveCoverTitleAndRefresh}
                      >
                        {coverRefreshing ? "生成中..." : "更新封面"}
                      </button>
                    </div>
                    <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                      <button
                        type="button"
                        className="sm primary"
                        disabled={jobRunning || !hasSubtitles}
                        onClick={handleScanCleanCut}
                        title={
                          hasSubtitles
                            ? "AI 结合视频关键帧与全片字幕扫描广告与政治敏感内容"
                            : "无字幕文件时不可执行基于字幕的广敏扫描，仍可直接渲染画面裁剪成片"
                        }
                      >
                        {jobRunning ? "处理中..." : "AI 扫描广告与敏感段"}
                      </button>
                      <button
                        type="button"
                        className="sm"
                        disabled={jobRunning || (cleanCutStats.activeCount === 0 && !manifest.crop?.enabled)}
                        onClick={() => handleRenderCleanCut(true)}
                        title={
                          cleanCutStats.activeCount === 0 && !manifest.crop?.enabled
                            ? "当前标准/开关下无生效剪切、脱敏或裁剪项，无需输出全片净版"
                            : "对勾选的广告/政治敏感片段执行删段或词级遮蔽（及全局裁剪），输出全片净版 MP4 与封面"
                        }
                      >
                        {cleanCutStats.activeCount === 0 && hasGlobalCrop
                          ? (manifest.clean_cut?.output_file ? "重渲裁剪成片" : "渲染裁剪成片")
                          : (manifest.clean_cut?.output_file ? "重渲全片净版" : "渲染全片净版")}
                      </button>
                      <button
                        type="button"
                        className={cleanCutPubJob ? "sm danger" : "sm primary"}
                        disabled={
                          cleanCutPubJob
                            ? Boolean(cleanCutPubJob.cancelling)
                            : (!manifest.clean_cut?.output_file || jobRunning)
                        }
                        onClick={() =>
                          cleanCutPubJob ? handleCancelPublishJob(cleanCutPubJob.job_id) : openPublishModal()
                        }
                      >
                        {cleanCutPubJob
                          ? cleanCutPubJob.cancelling
                            ? "正在中止发布..."
                            : "发布中 · 中止"
                          : "发布到 视频号 / B站 / 小红书…"}
                      </button>
                      {renderPubBadges(manifest.clean_cut?.publish_records)}
                    </div>
                    {renderPublishDiagnostics("clean_cut", manifest.clean_cut?.publish_records)}
                  </div>
                  {currentCoverUrl && (
                    <a
                      href={currentCoverUrl}
                      target="_blank"
                      rel="noreferrer"
                      title="点击查看高清封面"
                      style={{ flexShrink: 0, display: "block" }}
                    >
                      <img
                        src={currentCoverUrl}
                        alt="全片净版封面"
                        style={{
                          width: 160,
                          height: 90,
                          objectFit: "cover",
                          borderRadius: 6,
                          border: "1px solid var(--border)",
                          display: "block",
                        }}
                      />
                    </a>
                  )}
                </div>

                {/* Compact Clean Cut Segment Summary & Manual Marker */}
                <div className="clean-cut-toolbar">
                  <span>
                    共 {cleanCutStats.total} 段 · 生效 {cleanCutStats.activeCount} 段（去广 {cleanCutStats.adCount} / 删敏 {cleanCutStats.delPolCount} / 常规遮蔽 {cleanCutStats.normalMaskCount} / 精准遮蔽 {cleanCutStats.preciseCount}
                    {cleanCutStats.polAllTotal > cleanCutStats.polTotal
                      ? ` · 已按当前标准放行 ${cleanCutStats.polAllTotal - cleanCutStats.polTotal} 段`
                      : ""}
                    {" · "}净删 {fmtTime(cleanCutStats.cutS)} · 净版约 {fmtTime(cleanCutStats.keptS)}）
                  </span>
                  <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                    {cleanCutStats.polAllTotal > 0 && canMask && (
                      <button
                        type="button"
                        className="sm"
                        disabled={extractingPreciseTerms || jobRunning}
                        onClick={() => handleExtractPreciseTerms(cleanCutStats.missingPreciseCount === 0)}
                        title="用 AI 仅提取不妥当政治人物/政体/中港台不规范称谓与定性判断词，用于「精准遮蔽+消音」模式"
                      >
                        {extractingPreciseTerms
                          ? "提取精准词中..."
                          : cleanCutStats.missingPreciseCount > 0
                            ? `AI提取精准词 (${cleanCutStats.missingPreciseCount})`
                            : "重提精准词"}
                      </button>
                    )}
                    <button
                      type="button"
                      className="sm"
                      onClick={() => {
                        const s = Math.max(0, Math.round(currentTime * 10) / 10);
                        const e = Math.min(manifest.source.duration_s || s + 15, s + 15);
                        handlePatchCleanCut({
                          add_segment: {
                            kind: (manifest.clean_cut?.cut_political_sensitive ?? true) ? "political_sensitive" : "ad",
                            start_s: s,
                            end_s: e,
                            reason: "人工从播放指针添加屏蔽段",
                          },
                        });
                      }}
                      title="以当前播放器时间为起点添加 15 秒人工屏蔽段（添加后可微调）"
                    >
                      + 在当前指针({fmtTime(currentTime)})添加屏蔽段
                    </button>
                  </div>
                </div>

                {/* Cut Segments Review List */}
                <div className="subtitle-list scroller">
                  {(manifest.cut_segments || []).length === 0 ? (
                    <div className="empty-state" style={{ padding: 28 }}>
                      {!hasSubtitles
                        ? hasGlobalCrop
                          ? "无字幕纯画幅裁剪模式：已设置全局裁剪区域，点击上方「渲染裁剪成片」即可导出（也可点击下方「+ 在当前指针添加屏蔽段」手动剪切不需要的时间段）。"
                          : "无字幕模式：请点击播放器下方「裁剪画面」框选保留画幅区域并应用，即可直接渲染导出。"
                        : "尚未扫描广告与敏感片段。点击上方「AI 扫描广告与敏感段」自动标记，或点击「在当前指针添加屏蔽段」手动标记。"}
                    </div>
                  ) : (
                    (manifest.cut_segments || []).map((seg: CutSegment) => {
                      const categoryActive =
                        seg.kind === "ad"
                          ? (manifest.clean_cut?.cut_ads ?? true)
                          : (manifest.clean_cut?.cut_political_sensitive ?? true);
                      const matchesProfile = cleanCutStats.matchesSeg(seg);
                      const isEffective = seg.enabled && categoryActive && matchesProfile;
                      const segEffAction = resolveSegActionFrontend(seg, manifest.clean_cut, canMask);
                      const isPreciseSeg = segEffAction === "precise_mask";
                      const segActionLabel =
                        segEffAction === "precise_mask"
                          ? "精准遮蔽"
                          : segEffAction === "mask"
                            ? "常规遮蔽"
                            : "删段";
                      return (
                        <div
                          key={seg.cut_id}
                          className="remix-row"
                          style={{ opacity: isEffective ? 1 : 0.45 }}
                        >
                          <input
                            type="checkbox"
                            checked={seg.enabled}
                            onChange={(e) =>
                              handlePatchCleanCut({
                                segments: [{ cut_id: seg.cut_id, enabled: e.target.checked }],
                              })
                            }
                          />
                          <div>
                            <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                              <select
                                aria-label="广敏类别"
                                value={seg.kind}
                                onChange={(e) =>
                                  handlePatchCleanCut({
                                    segments: [{ cut_id: seg.cut_id, kind: e.target.value }],
                                  })
                                }
                                style={{
                                  padding: "2px 6px",
                                  fontSize: 11,
                                  fontWeight: 600,
                                  background:
                                    seg.kind === "political_sensitive"
                                      ? "rgba(239, 68, 68, 0.2)"
                                      : "rgba(245, 158, 11, 0.2)",
                                  color: seg.kind === "political_sensitive" ? "#fca5a5" : "#fcd34d",
                                }}
                              >
                                <option value="ad">广告</option>
                                <option value="political_sensitive">政治敏感</option>
                              </select>
                              {seg.kind === "political_sensitive" && (
                                <select
                                  aria-label="风险级别"
                                  value={seg.severity || "high"}
                                  onChange={(e) =>
                                    handlePatchCleanCut({
                                      segments: [{ cut_id: seg.cut_id, severity: e.target.value }],
                                    })
                                  }
                                  style={{ padding: "2px 5px", fontSize: 11 }}
                                  title="内容本身的风险等级（高中低危），是否处理取决于顶部政审标准"
                                >
                                  <option value="high">高危</option>
                                  <option value="medium">中危</option>
                                  <option value="low">低危</option>
                                </select>
                              )}
                              {seg.kind === "political_sensitive" ? (
                                !matchesProfile ? (
                                  <span
                                    className="tag"
                                    style={{ background: "rgba(148, 163, 184, 0.16)", color: "#cbd5e1" }}
                                    title="当前政审标准不拦截该风险级别，本段将保留不处理"
                                  >
                                    已放行
                                  </span>
                                ) : !categoryActive ? (
                                  <span
                                    className="tag"
                                    style={{ background: "rgba(148, 163, 184, 0.16)", color: "#cbd5e1" }}
                                    title="顶部未勾选「去除政治敏感」"
                                  >
                                    未启用
                                  </span>
                                ) : isVisualOcrSeg(seg) ? (
                                  <select
                                    aria-label="画面文字处理方式"
                                    value={segEffAction === "delete" ? "delete" : "mask"}
                                    onChange={(e) =>
                                      handlePatchCleanCut({
                                        segments: [{ cut_id: seg.cut_id, action: e.target.value }],
                                      })
                                    }
                                    style={{ padding: "2px 5px", fontSize: 11 }}
                                    title="画面文字默认仅对识别坐标框做局部马赛克模糊，不影响原声与翻译字幕"
                                  >
                                    <option value="mask">画面模糊</option>
                                    <option value="delete">删段</option>
                                  </select>
                                ) : (
                                  <select
                                    aria-label="单段脱敏方式"
                                    value={seg.action || "inherit"}
                                    onChange={async (e) => {
                                      const v = e.target.value;
                                      await handlePatchCleanCut({
                                        segments: [{ cut_id: seg.cut_id, action: v }],
                                      });
                                      if (
                                        v === "precise_mask" &&
                                        !seg.precise_terms_extracted &&
                                        !(seg.precise_terms_zh && seg.precise_terms_zh.length > 0) &&
                                        !(seg.precise_terms_en && seg.precise_terms_en.length > 0)
                                      ) {
                                        await handleExtractPreciseTerms(false);
                                      }
                                    }}
                                    style={{ padding: "2px 5px", fontSize: 11 }}
                                    title={`覆盖本段处理方式（AI建议：${seg.recommended_action === "mask" ? "遮蔽" : "删段"}）`}
                                  >
                                    <option value="inherit">
                                      跟随全局（{segActionLabel}）
                                    </option>
                                    <option value="delete">删段</option>
                                    <option value="mask" disabled={!canMask}>
                                      常规遮蔽{!canMask ? "（无干净原片）" : ""}
                                    </option>
                                    <option value="precise_mask" disabled={!canMask}>
                                      精准遮蔽{!canMask ? "（无干净原片）" : ""}
                                    </option>
                                  </select>
                                )
                              ) : !categoryActive ? (
                                <span
                                  className="tag"
                                  style={{ background: "rgba(148, 163, 184, 0.16)", color: "#cbd5e1" }}
                                  title="顶部未勾选「去除广告」"
                                >
                                  未启用
                                </span>
                              ) : (
                                <span
                                  className="tag"
                                  style={{ background: "rgba(245, 158, 11, 0.16)", color: "#fcd34d" }}
                                >
                                  删段
                                </span>
                              )}
                              {seg.detector !== "llm" && (
                                <span className="tag">
                                  {seg.detector === "visual_ocr" ? "画面文字" : seg.detector === "visual" ? "画面" : "手动"}
                                </span>
                              )}
                              <span
                                className="utt-time"
                                onClick={() => {
                                  setPlayMode("source");
                                  seekTo(seg.start_s);
                                }}
                                title="点击在原片播放器跳转试看"
                              >
                                {fmtTime(seg.start_s)}–{fmtTime(seg.end_s)} ({(seg.end_s - seg.start_s).toFixed(1)}s)
                              </span>
                              {seg.chapter_title && (
                                <span style={{ color: "var(--text-2)", fontSize: 11 }}>
                                  [{seg.chapter_title}]
                                </span>
                              )}
                            </div>
                            <div style={{ marginTop: 4, color: "var(--text-0)", fontSize: 12 }}>
                              说明：{seg.reason || "命中合规/广告规则"}
                            </div>
                            {seg.quote && (
                              <div style={{ marginTop: 3, color: "var(--text-2)", fontSize: 12 }}>
                                原文：{seg.quote}
                              </div>
                            )}
                            {seg.kind === "political_sensitive" && (
                              <div style={{ marginTop: 5, display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", fontSize: 11 }}>
                                {isVisualOcrSeg(seg) ? (
                                  <span style={{ color: "#7dd3fc" }}>
                                    局部模糊画面文字 ({seg.visual_boxes?.length || 0} 处区域 · 保留原声与字幕)
                                  </span>
                                ) : isPreciseSeg ? (
                                  <>
                                    <label style={{ display: "inline-flex", alignItems: "center", gap: 4, color: "var(--text-2)" }}>
                                      精准中文词:
                                      <input
                                        key={`pzh-${seg.cut_id}-${(seg.precise_terms_zh || []).join(",")}`}
                                        type="text"
                                        defaultValue={(seg.precise_terms_zh || []).join(", ")}
                                        placeholder="仅屏蔽政治体/不规范称谓/定性词（留空不整段消音）"
                                        style={{ width: 200, padding: "2px 6px", fontSize: 11 }}
                                        onBlur={(e) => {
                                          const parsed = e.target.value
                                            .split(/[,，、]/)
                                            .map((x) => x.trim())
                                            .filter(Boolean);
                                          if (parsed.join(",") !== (seg.precise_terms_zh || []).join(",")) {
                                            handlePatchCleanCut({
                                              segments: [{ cut_id: seg.cut_id, precise_terms_zh: parsed }],
                                            });
                                          }
                                        }}
                                      />
                                    </label>
                                    <label style={{ display: "inline-flex", alignItems: "center", gap: 4, color: "var(--text-2)" }}>
                                      精准英文消音词:
                                      <input
                                        key={`pen-${seg.cut_id}-${(seg.precise_terms_en || []).join(",")}`}
                                        type="text"
                                        defaultValue={(seg.precise_terms_en || []).join(", ")}
                                        placeholder="例: Xi Jinping, aggressor, purges"
                                        style={{ width: 200, padding: "2px 6px", fontSize: 11 }}
                                        onBlur={(e) => {
                                          const parsed = e.target.value
                                            .split(/[,，、]/)
                                            .map((x) => x.trim())
                                            .filter(Boolean);
                                          if (parsed.join(",") !== (seg.precise_terms_en || []).join(",")) {
                                            handlePatchCleanCut({
                                              segments: [{ cut_id: seg.cut_id, precise_terms_en: parsed }],
                                            });
                                          }
                                        }}
                                      />
                                    </label>
                                  </>
                                ) : (
                                  <>
                                    <label style={{ display: "inline-flex", alignItems: "center", gap: 4, color: "var(--text-2)" }}>
                                      中文屏蔽词:
                                      <input
                                        key={`tzh-${seg.cut_id}-${(seg.sensitive_terms_zh || []).join(",")}`}
                                        type="text"
                                        defaultValue={(seg.sensitive_terms_zh || []).join(", ")}
                                        placeholder="留空则整句替换[已脱敏]"
                                        style={{ width: 170, padding: "2px 6px", fontSize: 11 }}
                                        onBlur={(e) => {
                                          const parsed = e.target.value
                                            .split(/[,，、]/)
                                            .map((x) => x.trim())
                                            .filter(Boolean);
                                          if (parsed.join(",") !== (seg.sensitive_terms_zh || []).join(",")) {
                                            handlePatchCleanCut({
                                              segments: [{ cut_id: seg.cut_id, sensitive_terms_zh: parsed }],
                                            });
                                          }
                                        }}
                                      />
                                    </label>
                                    <label style={{ display: "inline-flex", alignItems: "center", gap: 4, color: "var(--text-2)" }}>
                                      英文屏蔽/消音词:
                                      <input
                                        key={`ten-${seg.cut_id}-${(seg.sensitive_terms_en || []).join(",")}`}
                                        type="text"
                                        defaultValue={(seg.sensitive_terms_en || []).join(", ")}
                                        placeholder="例: Xi Jinping, China"
                                        style={{ width: 190, padding: "2px 6px", fontSize: 11 }}
                                        onBlur={(e) => {
                                          const parsed = e.target.value
                                            .split(/[,，、]/)
                                            .map((x) => x.trim())
                                            .filter(Boolean);
                                          if (parsed.join(",") !== (seg.sensitive_terms_en || []).join(",")) {
                                            handlePatchCleanCut({
                                              segments: [{ cut_id: seg.cut_id, sensitive_terms_en: parsed }],
                                            });
                                          }
                                        }}
                                      />
                                    </label>
                                  </>
                                )}
                              </div>
                            )}
                          </div>
                          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                            {typeof seg.start_utt === "number" && typeof seg.end_utt === "number" ? (
                              <>
                                <div style={{ display: "flex", gap: 3 }}>
                                  <button
                                    type="button"
                                    className="sm"
                                    title="起点提前1句字幕"
                                    onClick={() =>
                                      handlePatchCleanCut({
                                        segments: [{ cut_id: seg.cut_id, start_utt: Math.max(0, (seg.start_utt || 0) - 1) }],
                                      })
                                    }
                                  >
                                    起-1句
                                  </button>
                                  <button
                                    type="button"
                                    className="sm"
                                    title="起点延后1句字幕"
                                    onClick={() =>
                                      handlePatchCleanCut({
                                        segments: [
                                          {
                                            cut_id: seg.cut_id,
                                            start_utt: Math.min(seg.end_utt || 0, (seg.start_utt || 0) + 1),
                                          },
                                        ],
                                      })
                                    }
                                  >
                                    起+1句
                                  </button>
                                </div>
                                <div style={{ display: "flex", gap: 3 }}>
                                  <button
                                    type="button"
                                    className="sm"
                                    title="终点提前1句字幕"
                                    onClick={() =>
                                      handlePatchCleanCut({
                                        segments: [
                                          {
                                            cut_id: seg.cut_id,
                                            end_utt: Math.max(seg.start_utt || 0, (seg.end_utt || 0) - 1),
                                          },
                                        ],
                                      })
                                    }
                                  >
                                    止-1句
                                  </button>
                                  <button
                                    type="button"
                                    className="sm"
                                    title="终点延后1句字幕"
                                    onClick={() =>
                                      handlePatchCleanCut({
                                        segments: [{ cut_id: seg.cut_id, end_utt: (seg.end_utt || 0) + 1 }],
                                      })
                                    }
                                  >
                                    止+1句
                                  </button>
                                </div>
                              </>
                            ) : (
                              <>
                                <div style={{ display: "flex", gap: 3 }}>
                                  <button
                                    type="button"
                                    className="sm"
                                    onClick={() =>
                                      handlePatchCleanCut({
                                        segments: [{ cut_id: seg.cut_id, start_s: Math.max(0, seg.start_s - 2) }],
                                      })
                                    }
                                  >
                                    起-2s
                                  </button>
                                  <button
                                    type="button"
                                    className="sm"
                                    onClick={() =>
                                      handlePatchCleanCut({
                                        segments: [{ cut_id: seg.cut_id, start_s: Math.min(seg.end_s - 0.5, seg.start_s + 2) }],
                                      })
                                    }
                                  >
                                    起+2s
                                  </button>
                                </div>
                                <div style={{ display: "flex", gap: 3 }}>
                                  <button
                                    type="button"
                                    className="sm"
                                    onClick={() =>
                                      handlePatchCleanCut({
                                        segments: [{ cut_id: seg.cut_id, end_s: Math.max(seg.start_s + 0.5, seg.end_s - 2) }],
                                      })
                                    }
                                  >
                                    止-2s
                                  </button>
                                  <button
                                    type="button"
                                    className="sm"
                                    onClick={() =>
                                      handlePatchCleanCut({
                                        segments: [{ cut_id: seg.cut_id, end_s: seg.end_s + 2 }],
                                      })
                                    }
                                  >
                                    止+2s
                                  </button>
                                </div>
                              </>
                            )}
                            <button
                              type="button"
                              className="sm"
                              style={{ color: "#fca5a5" }}
                              onClick={() => handlePatchCleanCut({ delete_ids: [seg.cut_id] })}
                            >
                              删除此段
                            </button>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              </>
            ) : selectedId === "REMIX" ? (
              <>
                {/* Compact Remix Mode & One-Click Generate/Render Bar */}
                {(() => {
                  const explicitSrc = Array.isArray(manifest.remix.source_clip_ids)
                    ? manifest.remix.source_clip_ids
                    : null;
                  const srcMode: "compilation" | "manual_clips" | "prompt" | "ai_auto" =
                    manifest.remix.source_mode &&
                    ["compilation", "manual_clips", "prompt", "ai_auto"].includes(manifest.remix.source_mode)
                      ? manifest.remix.source_mode
                      : explicitSrc !== null
                        ? explicitSrc.length > 0
                          ? "manual_clips"
                          : "ai_auto"
                        : "compilation";
                  const activeSrcSet = new Set(
                    explicitSrc !== null
                      ? explicitSrc
                      : manifest.clips.filter((c) => c.enabled).map((c) => c.clip_id)
                  );
                  const remainingToAdd = manifest.clips.filter(
                    (c) => !manifest.remix.items.some((it) => it.source_clip_id === c.clip_id)
                  );
                  const canGenerate =
                    !jobRunning &&
                    (srcMode === "compilation"
                      ? compilationClips.length > 0
                      : srcMode === "manual_clips"
                        ? activeSrcSet.size > 0 || (Boolean(manifest.remix.manual_script_locked) && enabledRemixItemCount > 0)
                        : srcMode === "prompt"
                          ? Boolean(remixPromptDraft.trim()) && manifest.clips.length > 0
                          : manifest.clips.length > 0 || (manifest.remix_only_pool?.length || 0) > 0);

                  const handleGenerateRemix = async () => {
                    if (srcMode === "prompt") {
                      await handlePatchRemix({
                        source_mode: "prompt",
                        custom_prompt: remixPromptDraft.trim(),
                      });
                      handleRun(["replan_remix", "render"], true, ["remix"]);
                    } else if (srcMode === "manual_clips" && manifest.remix.manual_script_locked && enabledRemixItemCount > 0) {
                      handleRenderRemix(true);
                    } else {
                      handleRun(["replan_remix", "render"], true, ["remix"]);
                    }
                  };

                  return (
                    <div
                      className="clean-cut-toolbar"
                      style={{ flexDirection: "column", alignItems: "stretch", gap: 6 }}
                    >
                      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                        <span style={{ fontWeight: 600, color: "var(--text-1)", whiteSpace: "nowrap" }}>混剪模式：</span>
                        <select
                          aria-label="混剪模式"
                          disabled={jobRunning}
                          value={srcMode}
                          onChange={(e) => {
                            const nextMode = e.target.value as
                              | "compilation"
                              | "manual_clips"
                              | "prompt"
                              | "ai_auto";
                            if (nextMode === "manual_clips") {
                              const fallback = manifest.clips
                                .filter((c) => c.enabled)
                                .map((c) => c.clip_id);
                              const nextIds =
                                fallback.length > 0
                                  ? fallback
                                  : manifest.clips.slice(0, 1).map((c) => c.clip_id);
                              handlePatchRemix({ source_mode: nextMode, source_clip_ids: nextIds });
                            } else {
                              handlePatchRemix({ source_mode: nextMode, clear_source_clip_ids: true });
                            }
                          }}
                          style={{ padding: "3px 6px", fontSize: 12 }}
                        >
                          <option value="compilation">基于集锦片段（默认）</option>
                          <option value="manual_clips">手动勾选片段</option>
                          <option value="prompt">按 Prompt 定向提取</option>
                          <option value="ai_auto">完全由 AI 自主精选</option>
                        </select>

                        {srcMode === "prompt" ? (
                          <input
                            type="text"
                            aria-label="混剪定向提取 Prompt"
                            disabled={jobRunning}
                            value={remixPromptDraft}
                            onChange={(e) => setRemixPromptDraft(e.target.value)}
                            onBlur={() => {
                              if (remixPromptDraft.trim() !== (manifest.remix.custom_prompt || "").trim()) {
                                handlePatchRemix({
                                  source_mode: "prompt",
                                  custom_prompt: remixPromptDraft.trim(),
                                });
                              }
                            }}
                            onKeyDown={(e) => e.key === "Enter" && canGenerate && handleGenerateRemix()}
                            placeholder="输入提取要求，如：只提取个人成长经历与商业思维"
                            style={{ flex: 1, minWidth: 200, padding: "3px 8px", fontSize: 12 }}
                          />
                        ) : (
                          <span style={{ fontSize: 11, color: "var(--text-2)", flex: 1, minWidth: 120 }}>
                            {srcMode === "compilation"
                              ? `同步集锦 ${compilationClips.length} 支切片 · 当前 ${enabledRemixItemCount} 段`
                              : srcMode === "manual_clips"
                                ? manifest.remix.manual_script_locked
                                  ? `已锁定人工段落 (${enabledRemixItemCount}/${manifest.remix.items.length})`
                                  : `已选 ${activeSrcSet.size} 支切片 · 当前 ${enabledRemixItemCount} 段`
                                : `全片 ${manifest.clips.length} 支候选 · 当前 ${enabledRemixItemCount} 段`}
                          </span>
                        )}

                        <button
                          type="button"
                          className="sm primary"
                          style={{ marginLeft: "auto", whiteSpace: "nowrap" }}
                          disabled={!canGenerate}
                          onClick={handleGenerateRemix}
                          title="按当前混剪模式重排脚本并自动渲染混剪成片与封面"
                        >
                          {jobRunning ? "生成中..." : "生成混剪"}
                        </button>
                      </div>

                      {srcMode === "manual_clips" && manifest.clips.length > 0 && (
                        <div style={{ display: "flex", gap: 5, alignItems: "center", flexWrap: "wrap", fontSize: 11 }}>
                          <span style={{ color: "var(--text-2)" }}>切片：</span>
                          {manifest.clips.map((c) => {
                            const on = activeSrcSet.has(c.clip_id);
                            return (
                              <button
                                key={c.clip_id}
                                type="button"
                                className={`sm ${on ? "primary" : ""}`}
                                style={{ padding: "1px 6px", fontSize: 11 }}
                                title={c.title}
                                disabled={jobRunning}
                                onClick={() => {
                                  const nextSet = new Set(activeSrcSet);
                                  if (on) nextSet.delete(c.clip_id);
                                  else nextSet.add(c.clip_id);
                                  handlePatchRemix({
                                    source_mode: "manual_clips",
                                    source_clip_ids: manifest.clips
                                      .filter((x) => nextSet.has(x.clip_id))
                                      .map((x) => x.clip_id),
                                  });
                                }}
                              >
                                #{String(c.index).padStart(2, "0")}
                              </button>
                            );
                          })}
                          <button
                            type="button"
                            className="sm"
                            style={{ padding: "1px 6px", fontSize: 11 }}
                            disabled={jobRunning}
                            onClick={() =>
                              handlePatchRemix({
                                source_mode: "manual_clips",
                                source_clip_ids: manifest.clips.filter((c) => c.enabled).map((c) => c.clip_id),
                              })
                            }
                          >
                            同步左侧
                          </button>
                          <div style={{ display: "flex", gap: 4, marginLeft: "auto", flexWrap: "wrap" }}>
                            <button
                              type="button"
                              className="sm"
                              style={{ padding: "1px 6px", fontSize: 11 }}
                              disabled={jobRunning || manifest.remix.items.length === 0}
                              onClick={() =>
                                handlePatchRemix({
                                  items: manifest.remix.items.map((it) => ({ remix_id: it.remix_id, enabled: true })),
                                })
                              }
                            >
                              全选段落
                            </button>
                            <button
                              type="button"
                              className="sm"
                              style={{ padding: "1px 6px", fontSize: 11 }}
                              disabled={jobRunning || manifest.remix.items.length === 0}
                              onClick={() =>
                                handlePatchRemix({
                                  items: manifest.remix.items.map((it) => ({ remix_id: it.remix_id, enabled: false })),
                                })
                              }
                            >
                              全不选
                            </button>
                            <select
                              aria-label="手动加入切片到混剪"
                              disabled={jobRunning || manifest.clips.length === 0}
                              value=""
                              onChange={(e) => {
                                if (e.target.value) {
                                  handlePatchRemix({ add_clip_id: e.target.value });
                                }
                              }}
                              style={{ padding: "1px 5px", fontSize: 11, maxWidth: 145 }}
                            >
                              <option value="">+ 加入切片…</option>
                              {(remainingToAdd.length > 0 ? remainingToAdd : manifest.clips).map((c) => (
                                <option key={c.clip_id} value={c.clip_id}>
                                  #{String(c.index).padStart(2, "0")} {c.title}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })()}

                {/* Unified Remix Cover Title & Publish Card */}
                <div className="clip-inspector-bar" style={{ flexDirection: "row", gap: 14, alignItems: "center" }}>
                  <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 7 }}>
                    <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <span style={{ color: "var(--text-2)", fontSize: 12, width: 56 }}>混剪标题</span>
                      <input
                        type="text"
                        style={{ flex: 1, fontWeight: 600 }}
                        value={titleDraft}
                        onChange={(e) => setTitleDraft(e.target.value)}
                        onBlur={() =>
                          titleDraft.trim() &&
                          titleDraft !== manifest.remix.title &&
                          handlePatchRemix({ title: titleDraft.trim() })
                        }
                      />
                    </div>
                    <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <span style={{ color: "var(--text-2)", fontSize: 12, width: 56 }}>封面标题</span>
                      <input
                        type="text"
                        style={{ flex: 1 }}
                        placeholder={`留空则跟随混剪标题（${titleDraft || manifest.remix.title || "高光混剪"}）`}
                        value={coverTitleDraft}
                        onChange={(e) => setCoverTitleDraft(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleSaveCoverTitleAndRefresh()}
                      />
                      <button
                        type="button"
                        className="sm"
                        disabled={coverRefreshing || jobRunning}
                        onClick={handleSaveCoverTitleAndRefresh}
                      >
                        {coverRefreshing ? "生成中..." : "更新封面"}
                      </button>
                    </div>
                    <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                      <button
                        type="button"
                        className={remixPubJob ? "sm danger" : "sm primary"}
                        disabled={
                          remixPubJob
                            ? Boolean(remixPubJob.cancelling)
                            : (!manifest.remix.output_file || jobRunning)
                        }
                        onClick={() => (remixPubJob ? handleCancelPublishJob(remixPubJob.job_id) : openPublishModal())}
                      >
                        {remixPubJob
                          ? remixPubJob.cancelling
                            ? "正在中止发布..."
                            : "发布中 · 中止"
                          : "发布到 视频号 / B站 / 小红书…"}
                      </button>
                      {renderPubBadges(manifest.remix.publish_records)}
                    </div>
                    {renderPublishDiagnostics("remix", manifest.remix.publish_records)}
                  </div>
                  {currentCoverUrl && (
                    <a
                      href={currentCoverUrl}
                      target="_blank"
                      rel="noreferrer"
                      title="点击查看高清封面"
                      style={{ flexShrink: 0, display: "block" }}
                    >
                      <img
                        src={currentCoverUrl}
                        alt="混剪封面"
                        style={{
                          width: 160,
                          height: 90,
                          objectFit: "cover",
                          borderRadius: 6,
                          border: "1px solid var(--border)",
                          display: "block",
                        }}
                      />
                    </a>
                  )}
                </div>

                {/* Collapsed Remix Advanced Options */}
                <details className="fold">
                  <summary>
                    ▸ 顶部导航条与音频选项（{remixStats.themeCount} 板块 / {remixStats.segmentCount} 段 · 预估 {fmtTime(remixStats.estTotalS)}）
                  </summary>
                  <div className="fold-body" style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center" }}>
                    <label>
                      {(navCfg.remix_style || navCfg.style) === "title_card" ? "段前卡片：" : "顶部导航条："}
                      <select
                        value={manifest.remix.card_mode}
                        onChange={(e) => handlePatchRemix({ card_mode: e.target.value })}
                        style={{ marginLeft: 4 }}
                      >
                        <option value="group">按主题组</option>
                        <option value="every">每段</option>
                        <option value="none">
                          {(navCfg.remix_style || navCfg.style) === "title_card" ? "不插卡" : "隐藏导航条"}
                        </option>
                      </select>
                    </label>
                    <label style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                      <input
                        type="checkbox"
                        checked={manifest.remix.enable_sfx}
                        onChange={(e) => handlePatchRemix({ enable_sfx: e.target.checked })}
                      />
                      <span>转场音效</span>
                    </label>
                    <label style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                      <input
                        type="checkbox"
                        checked={manifest.remix.enable_bgm}
                        onChange={(e) => handlePatchRemix({ enable_bgm: e.target.checked })}
                      />
                      <span>BGM 闪避</span>
                    </label>
                  </div>
                </details>

                {cleanCutStats.isScanned && (
                  <div
                    style={{
                      display: "flex",
                      gap: 8,
                      alignItems: "center",
                      flexWrap: "wrap",
                      fontSize: 12,
                      padding: "6px 10px",
                      borderRadius: 6,
                      background: "rgba(30, 41, 59, 0.55)",
                      border: "1px solid var(--border)",
                      marginBottom: 8,
                    }}
                  >
                    <span style={{ color: "var(--text-2)" }}>全局合规联动：</span>
                    {renderComplianceTags(remixCompliance, cleanCutStats.isScanned)}
                    {(() => {
                      const rb = getRenderBadge(manifest.remix, remixCompliance, cleanCutStats.isScanned, "待渲染");
                      return (
                        <span className={`tag ${rb.cls}`} title={rb.title}>
                          {rb.label}
                        </span>
                      );
                    })()}
                    {remixCompliance.overlappingSegs.length > 0 && (
                      <button
                        type="button"
                        className="sm"
                        style={{ marginLeft: "auto" }}
                        onClick={() => {
                          setSelectedId("CLEAN_CUT");
                          setPlayMode(manifest.clean_cut?.output_file ? "rendered" : "source");
                        }}
                      >
                        查看/微调明细
                      </button>
                    )}
                  </div>
                )}

                <div className="subtitle-list scroller">
                  {[...manifest.remix.items]
                    .sort((a, b) => a.order - b.order)
                    .map((it: RemixItem, idx) => {
                      const itComp = summarizeIntervalsCompliance(
                        [[it.start_s, it.end_s]],
                        manifest.cut_segments || [],
                        manifest.clean_cut,
                        cleanCutStats.matchesSeg,
                        canMask
                      );
                      return (
                      <div key={it.remix_id} className="remix-row" style={{ opacity: it.enabled && !itComp.isFullyCut ? 1 : 0.45 }}>
                        <input
                          type="checkbox"
                          checked={it.enabled}
                          onChange={(e) =>
                            handlePatchRemix({
                              items: [{ remix_id: it.remix_id, enabled: e.target.checked }],
                            })
                          }
                        />
                        <div>
                          <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                            <span className="tag hot_topic">
                              #{it.order} {BEAT_LABELS[it.beat_type || ""] || it.role}
                            </span>
                            <input
                              key={`grp-${it.remix_id}-${it.group_label}`}
                              type="text"
                              defaultValue={it.group_label}
                              style={{ width: 105, padding: "2px 6px", fontSize: 12 }}
                              onBlur={(e) =>
                                e.target.value !== it.group_label &&
                                handlePatchRemix({
                                  items: [{ remix_id: it.remix_id, group_label: e.target.value }],
                                })
                              }
                            />
                            <input
                              key={`card-${it.remix_id}-${it.card_title}`}
                              type="text"
                              defaultValue={it.card_title}
                              style={{ width: 230, padding: "2px 6px", fontSize: 12 }}
                              onBlur={(e) =>
                                e.target.value !== it.card_title &&
                                handlePatchRemix({
                                  items: [{ remix_id: it.remix_id, card_title: e.target.value }],
                                })
                              }
                            />
                            <span
                              className="utt-time"
                              onClick={() => {
                                setPlayMode("source");
                                seekTo(it.start_s);
                              }}
                            >
                              {fmtTime(it.start_s)}–{fmtTime(it.end_s)} ({(it.end_s - it.start_s).toFixed(1)}s)
                            </span>
                            {itComp.overlappingSegs.length > 0 && renderComplianceTags(itComp, cleanCutStats.isScanned)}
                          </div>
                          <div style={{ marginTop: 4, color: "var(--text-1)" }}>{it.quote}</div>
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                          <button type="button" className="sm" onClick={() => moveRemixItem(idx, -1)}>
                            上移
                          </button>
                          <button type="button" className="sm" onClick={() => moveRemixItem(idx, 1)}>
                            下移
                          </button>
                          <button
                            type="button"
                            className="sm danger"
                            onClick={() => handlePatchRemix({ remove_remix_id: it.remix_id })}
                          >
                            移除
                          </button>
                        </div>
                      </div>
                      );
                    })}
                </div>
              </>
            ) : selectedId === "COMPILATION" ? (
              <>
                {/* Unified Compilation Cover Title & Publish Card */}
                <div className="clip-inspector-bar" style={{ flexDirection: "row", gap: 14, alignItems: "center" }}>
                  <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 7 }}>
                    <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <span style={{ color: "var(--text-2)", fontSize: 12, width: 56 }}>精选集名</span>
                      <input
                        type="text"
                        style={{ flex: 1, fontWeight: 600 }}
                        value={titleDraft}
                        onChange={(e) => setTitleDraft(e.target.value)}
                        onBlur={() =>
                          titleDraft.trim() &&
                          titleDraft !== (manifest.compilation?.title || "切片精选集") &&
                          handlePatchCompilation({ title: titleDraft.trim() })
                        }
                      />
                      <button
                        type="button"
                        className="sm"
                        disabled={compTitleSuggesting || jobRunning || manifest.clips.length === 0}
                        onClick={handleSuggestCompilationTitle}
                        title="由 AI 根据入选切片内容重新提炼精选集主题标题"
                      >
                        {compTitleSuggesting ? "提炼中..." : "AI提炼标题"}
                      </button>
                    </div>
                    <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <span style={{ color: "var(--text-2)", fontSize: 12, width: 56 }}>封面标题</span>
                      <input
                        type="text"
                        style={{ flex: 1 }}
                        placeholder={`留空则跟随精选集名（${titleDraft || manifest.compilation?.title || "切片精选集"}）`}
                        value={coverTitleDraft}
                        onChange={(e) => setCoverTitleDraft(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleSaveCoverTitleAndRefresh()}
                      />
                      <button
                        type="button"
                        className="sm"
                        disabled={coverRefreshing || jobRunning || compilationClips.length === 0}
                        onClick={handleSaveCoverTitleAndRefresh}
                      >
                        {coverRefreshing ? "生成中..." : "更新封面"}
                      </button>
                    </div>
                    <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                      <button
                        type="button"
                        className="sm"
                        disabled={jobRunning || compilationClips.length === 0}
                        onClick={() => handleRenderCompilation(true)}
                        title="仅单独渲染当前精选集（串联下方已勾选的精选切片，不受左侧独立切片勾选影响）"
                      >
                        {isRendering ? "渲染中..." : manifest.compilation?.output_file ? "重渲精选集" : "渲染精选集"}
                      </button>
                      <button
                        type="button"
                        className={compPubJob ? "sm danger" : "sm primary"}
                        disabled={
                          compPubJob
                            ? Boolean(compPubJob.cancelling)
                            : (!manifest.compilation?.output_file || jobRunning)
                        }
                        onClick={() => (compPubJob ? handleCancelPublishJob(compPubJob.job_id) : openPublishModal())}
                      >
                        {compPubJob
                          ? compPubJob.cancelling
                            ? "正在中止发布..."
                            : "发布中 · 中止"
                          : "发布到 视频号 / B站 / 小红书…"}
                      </button>
                      {renderPubBadges(manifest.compilation?.publish_records)}
                    </div>
                    {renderPublishDiagnostics("compilation", manifest.compilation?.publish_records)}
                  </div>
                  {currentCoverUrl && (
                    <a
                      href={currentCoverUrl}
                      target="_blank"
                      rel="noreferrer"
                      title="点击查看高清封面"
                      style={{ flexShrink: 0, display: "block" }}
                    >
                      <img
                        src={currentCoverUrl}
                        alt="精选集封面"
                        style={{
                          width: 160,
                          height: 90,
                          objectFit: "cover",
                          borderRadius: 6,
                          border: "1px solid var(--border)",
                          display: "block",
                        }}
                      />
                    </a>
                  )}
                </div>

                {/* Compilation Title Card & Independent Selection Controls */}
                <div className="clean-cut-toolbar">
                  <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                    <span style={{ color: "var(--text-1)", fontWeight: 600 }}>
                      已选 {compilationClips.length}/{manifest.clips.length} 支 · 约 {fmtTime(compilationTotalS)}
                    </span>
                    {(navCfg.compilation_style || navCfg.style) === "title_card" ||
                    (navCfg.compilation_style || navCfg.style) === "both" ? (
                      <label style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                        段前片头卡：
                        <select
                          value={String(manifest.compilation?.card_duration_s ?? 1.8)}
                          onChange={(e) => handlePatchCompilation({ card_duration_s: Number(e.target.value) })}
                          style={{ padding: "2px 5px", fontSize: 11 }}
                        >
                          <option value="1.2">1.2s</option>
                          <option value="1.8">1.8s</option>
                          <option value="2.5">2.5s</option>
                        </select>
                      </label>
                    ) : (
                      <button
                        type="button"
                        className="sm"
                        onClick={() => settingsDialogRef.current?.showModal()}
                        title="点击在设置中切换常驻导航式 / 段前卡片式及色系"
                      >
                        顶部导航条：常驻高亮
                      </button>
                    )}
                  </div>
                  <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                    <button
                      type="button"
                      className="sm"
                      disabled={jobRunning || manifest.clips.length === 0}
                      onClick={() => handlePatchCompilation({ clip_ids: manifest.clips.map((c) => c.clip_id) })}
                    >
                      全选
                    </button>
                    <button
                      type="button"
                      className="sm"
                      disabled={jobRunning || enabledClipCount === 0}
                      onClick={() =>
                        handlePatchCompilation({
                          clip_ids: manifest.clips.filter((c) => c.enabled).map((c) => c.clip_id),
                        })
                      }
                      title="按当前左侧勾选的独立切片同步一份到精选集"
                    >
                      同步左侧勾选
                    </button>
                    <button
                      type="button"
                      className="sm"
                      disabled={jobRunning || compilationClips.length === 0}
                      onClick={() => handlePatchCompilation({ clear_clip_ids: true })}
                    >
                      清空
                    </button>
                  </div>
                </div>

                {cleanCutStats.isScanned && (
                  <div
                    style={{
                      display: "flex",
                      gap: 8,
                      alignItems: "center",
                      flexWrap: "wrap",
                      fontSize: 12,
                      padding: "6px 10px",
                      borderRadius: 6,
                      background: "rgba(30, 41, 59, 0.55)",
                      border: "1px solid var(--border)",
                      marginBottom: 8,
                    }}
                  >
                    <span style={{ color: "var(--text-2)" }}>全局合规联动：</span>
                    {renderComplianceTags(compilationCompliance, cleanCutStats.isScanned)}
                    {(() => {
                      const compStub = manifest.compilation || {
                        render_hash: "",
                        last_rendered_hash: "",
                        output_file: null,
                      };
                      const cb = getRenderBadge(compStub, compilationCompliance, cleanCutStats.isScanned, "待渲染");
                      return (
                        <span className={`tag ${cb.cls}`} title={cb.title}>
                          {cb.label}
                        </span>
                      );
                    })()}
                    {compilationCompliance.overlappingSegs.length > 0 && (
                      <button
                        type="button"
                        className="sm"
                        style={{ marginLeft: "auto" }}
                        onClick={() => {
                          setSelectedId("CLEAN_CUT");
                          setPlayMode(manifest.clean_cut?.output_file ? "rendered" : "source");
                        }}
                      >
                        查看/微调明细
                      </button>
                    )}
                  </div>
                )}

                {/* Compilation Sequential Clips List (Independent Checkboxes + Ordering) */}
                <div className="subtitle-list scroller">
                  {manifest.clips.length === 0 ? (
                    <div className="empty-state" style={{ padding: 24 }}>
                      尚未生成切片，请先点击上方「生成切片策略」。
                    </div>
                  ) : (
                    (() => {
                      const selectedIds = compilationClips.map((c) => c.clip_id);
                      const selectedSet = new Set(selectedIds);
                      const unselectedClips = manifest.clips.filter((c) => !selectedSet.has(c.clip_id));
                      const displayList = [...compilationClips, ...unselectedClips];
                      return displayList.map((c, listIdx) => {
                        const isSelected = selectedSet.has(c.clip_id);
                        const selPos = isSelected ? listIdx : -1;
                        const partNum = isSelected ? String(selPos + 1).padStart(2, "0") : "";
                        const cComp = summarizeIntervalsCompliance(
                          [[c.start_s, c.end_s]],
                          manifest.cut_segments || [],
                          manifest.clean_cut,
                          cleanCutStats.matchesSeg,
                          canMask
                        );
                        const durSec = cleanCutStats.isScanned ? cComp.netDurS : Math.max(0, c.end_s - c.start_s);
                        return (
                          <div
                            key={c.clip_id}
                            className="remix-row"
                            style={{ alignItems: "center", cursor: "pointer", opacity: isSelected ? 1 : 0.55 }}
                            onClick={() => {
                              setPlayMode("source");
                              seekTo(c.start_s);
                            }}
                          >
                            <input
                              type="checkbox"
                              checked={isSelected}
                              aria-label={`精选集包含切片 ${c.title}`}
                              onClick={(e) => e.stopPropagation()}
                              onChange={(e) => {
                                const on = e.target.checked;
                                const nextIds = on
                                  ? [...selectedIds, c.clip_id]
                                  : selectedIds.filter((id) => id !== c.clip_id);
                                handlePatchCompilation({ clip_ids: nextIds });
                              }}
                            />
                            <div style={{ minWidth: 0 }}>
                              <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                                {isSelected && (
                                  <span
                                    className="tag highlight"
                                    style={{
                                      background: "rgba(59, 130, 246, 0.18)",
                                      color: "#60a5fa",
                                      borderColor: "rgba(59, 130, 246, 0.45)",
                                    }}
                                  >
                                    导航分段 {partNum}
                                  </span>
                                )}
                                <span style={{ fontWeight: 600 }}>
                                  #{String(c.index).padStart(2, "0")} {c.title}
                                </span>
                                <span className="utt-time">
                                  {fmtTime(c.start_s)}–{fmtTime(c.end_s)} ({fmtTime(durSec)})
                                </span>
                                {cComp.overlappingSegs.length > 0 && renderComplianceTags(cComp, cleanCutStats.isScanned)}
                              </div>
                            </div>
                            <div style={{ display: "flex", gap: 4 }}>
                              {isSelected && (
                                <>
                                  <button
                                    type="button"
                                    className="sm"
                                    disabled={selPos === 0}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      const next = [...selectedIds];
                                      [next[selPos - 1], next[selPos]] = [next[selPos], next[selPos - 1]];
                                      handlePatchCompilation({ clip_ids: next });
                                    }}
                                  >
                                    ↑
                                  </button>
                                  <button
                                    type="button"
                                    className="sm"
                                    disabled={selPos === selectedIds.length - 1}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      const next = [...selectedIds];
                                      [next[selPos], next[selPos + 1]] = [next[selPos + 1], next[selPos]];
                                      handlePatchCompilation({ clip_ids: next });
                                    }}
                                  >
                                    ↓
                                  </button>
                                </>
                              )}
                              <button
                                type="button"
                                className="sm"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setSelectedId(c.clip_id);
                                  setPlayMode("source");
                                }}
                              >
                                编辑切片
                              </button>
                            </div>
                          </div>
                        );
                      });
                    })()
                  )}
                </div>
              </>
            ) : activeClip ? (
              <>
                {/* Unified Clip Cover Title & Publish Card */}
                <div className="clip-inspector-bar" style={{ flexDirection: "row", gap: 14, alignItems: "center" }}>
                  <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 7 }}>
                    <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <span style={{ color: "var(--text-2)", fontSize: 12, width: 56 }}>切片标题</span>
                      <input
                        type="text"
                        style={{ flex: 1, fontWeight: 600 }}
                        value={titleDraft}
                        onChange={(e) => setTitleDraft(e.target.value)}
                        onBlur={() =>
                          titleDraft.trim() &&
                          titleDraft !== activeClip.title &&
                          handlePatchClip(activeClip.clip_id, { title: titleDraft.trim() })
                        }
                      />
                    </div>
                    <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <span style={{ color: "var(--text-2)", fontSize: 12, width: 56 }}>封面标题</span>
                      <input
                        type="text"
                        style={{ flex: 1 }}
                        placeholder={`留空则跟随切片标题（${titleDraft || activeClip.title}）`}
                        value={coverTitleDraft}
                        onChange={(e) => setCoverTitleDraft(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleSaveCoverTitleAndRefresh()}
                      />
                      <button
                        type="button"
                        className="sm"
                        disabled={coverRefreshing || jobRunning}
                        onClick={handleSaveCoverTitleAndRefresh}
                      >
                        {coverRefreshing ? "生成中..." : "更新封面"}
                      </button>
                    </div>
                    <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                      <button
                        type="button"
                        className="sm"
                        disabled={singleRendering || jobRunning}
                        onClick={() => handleRenderSingleClip(activeClip.clip_id)}
                        title="只定向渲染当前这个切片及其封面"
                      >
                        {singleRendering ? "渲染中..." : activeClip.output_file ? "重渲此切片" : "渲染此切片"}
                      </button>
                      <button
                        type="button"
                        className={clipPubJob ? "sm danger" : "sm primary"}
                        disabled={
                          clipPubJob
                            ? Boolean(clipPubJob.cancelling)
                            : (!activeClip.output_file || jobRunning)
                        }
                        onClick={() => (clipPubJob ? handleCancelPublishJob(clipPubJob.job_id) : openPublishModal())}
                      >
                        {clipPubJob
                          ? clipPubJob.cancelling
                            ? "正在中止..."
                            : clipPubJob.kind === "batch" && clipPubJob.current_target !== activeClip.clip_id
                              ? "批量队列等待中 · 中止批次"
                              : "发布中 · 中止"
                          : "发布到 视频号 / B站 / 小红书…"}
                      </button>
                      {renderPubBadges(activeClip.publish_records)}
                    </div>
                    {renderPublishDiagnostics(activeClip.clip_id, activeClip.publish_records)}
                  </div>
                  {currentCoverUrl && (
                    <a
                      href={currentCoverUrl}
                      target="_blank"
                      rel="noreferrer"
                      title="点击查看高清封面"
                      style={{ flexShrink: 0, display: "block" }}
                    >
                      <img
                        src={currentCoverUrl}
                        alt={`切片 ${activeClip.index} 封面`}
                        style={{
                          width: 160,
                          height: 90,
                          objectFit: "cover",
                          borderRadius: 6,
                          border: "1px solid var(--border)",
                          display: "block",
                        }}
                      />
                    </a>
                  )}
                </div>

                <div className="center-bottom-body scroller">
                {activeClipCompliance.overlappingSegs.length > 0 && (
                  <div
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: 5,
                      fontSize: 12,
                      padding: "8px 10px",
                      borderRadius: 6,
                      background: "rgba(30, 41, 59, 0.65)",
                      border: "1px solid var(--border)",
                      marginBottom: 8,
                    }}
                  >
                    <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                      <span style={{ fontWeight: 600 }}>全局合规联动：</span>
                      {renderComplianceTags(activeClipCompliance, cleanCutStats.isScanned)}
                      {(() => {
                        const rb = getRenderBadge(activeClip, activeClipCompliance, cleanCutStats.isScanned, "待切");
                        return (
                          <span className={`tag ${rb.cls}`} title={rb.title}>
                            {rb.label}
                          </span>
                        );
                      })()}
                      <button
                        type="button"
                        className="sm"
                        style={{ marginLeft: "auto" }}
                        onClick={() => {
                          setSelectedId("CLEAN_CUT");
                          setPlayMode(manifest.clean_cut?.output_file ? "rendered" : "source");
                        }}
                      >
                        查看/微调明细
                      </button>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 2, color: "var(--text-2)", fontSize: 11 }}>
                      {activeClipCompliance.overlappingSegs.map(({ seg, effAction, reasonLabel }) => {
                        const effTerms =
                          effAction === "mask" || effAction === "precise_mask"
                            ? getEffectiveMaskTerms(seg, effAction)
                            : { zh: [], en: [] };
                        const termTxt = [...effTerms.zh, ...effTerms.en].join(" / ");
                        return (
                          <div key={seg.cut_id}>
                            • [{fmtTime(seg.start_s)}–{fmtTime(seg.end_s)}] {reasonLabel}
                            {(effAction === "mask" || effAction === "precise_mask") && termTxt
                              ? `（词表：${termTxt}）`
                              : ""}{" "}
                            — {seg.reason}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Collapsed Clip Subtitle Repair & Re-burn */}
                <details className="fold" style={{ marginBottom: 8 }}>
                  <summary>▸ 字幕修复与重烧</summary>
                  <div className="fold-body">
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                      <label style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12 }}>
                        <input
                          type="checkbox"
                          checked={subtitleRepairFullBurn}
                          onChange={(e) => setSubtitleRepairFullBurn(e.target.checked)}
                        />
                        同时重烧全片硬字幕
                      </label>
                      <button
                        type="button"
                        className="sm primary"
                        disabled={subtitleRepairSaving || jobRunning || repairUtterances.length === 0}
                        onClick={handleRepairSubtitles}
                      >
                        {subtitleRepairSaving ? "修复中..." : "保存并重烧"}
                      </button>
                    </div>
                    {repairUtterances.map((u) => {
                      const draft = subtitleDrafts[u.id] || { zh: "", en: "" };
                      return (
                        <div
                          key={u.id}
                          style={{
                            display: "grid",
                            gridTemplateColumns: "72px 1fr 1fr",
                            gap: 8,
                            alignItems: "start",
                            padding: "6px 0",
                            borderBottom: "1px solid var(--border)",
                          }}
                        >
                          <span className="utt-time" onClick={() => seekTo(u.start_s)} title="点击跳转">
                            {fmtTime(u.start_s)}
                          </span>
                          <textarea
                            value={draft.zh}
                            rows={2}
                            placeholder="中文"
                            onChange={(e) =>
                              setSubtitleDrafts((prev) => ({
                                ...prev,
                                [u.id]: { ...(prev[u.id] || { zh: "", en: "" }), zh: e.target.value },
                              }))
                            }
                          />
                          <textarea
                            value={draft.en}
                            rows={2}
                            placeholder="English"
                            onChange={(e) =>
                              setSubtitleDrafts((prev) => ({
                                ...prev,
                                [u.id]: { ...(prev[u.id] || { zh: "", en: "" }), en: e.target.value },
                              }))
                            }
                          />
                        </div>
                      );
                    })}
                  </div>
                </details>

                {/* Collapsed Clip Subtitle Trimming & Editorial Notes */}
                <details className="fold">
                  <summary>▸ 微调字幕入出点与编审说明</summary>
                  <div className="fold-body">
                    {activeClip.reason && (
                      <div style={{ fontSize: 12, color: "var(--text-2)", marginBottom: 8 }}>
                        说明：{activeClip.reason}
                      </div>
                    )}
                    {contextUtterances.map((u) => {
                      const inRange = u.id >= activeClip.start_utt && u.id <= activeClip.end_utt;
                      const isPlaying =
                        playMode === "source" && currentTime >= u.start_s && currentTime <= u.end_s + 0.15;
                      const uHits = (manifest.cut_segments || []).filter(
                        (seg) =>
                          seg.enabled &&
                          (Math.min(u.end_s, seg.end_s) - Math.max(u.start_s, seg.start_s) > 0.05 ||
                            (typeof seg.start_utt === "number" &&
                              typeof seg.end_utt === "number" &&
                              u.id >= seg.start_utt &&
                              u.id <= seg.end_utt))
                      );
                      const activeHits = uHits.filter(
                        (seg) =>
                          (seg.kind === "ad" && manifest.clean_cut?.cut_ads) ||
                          (seg.kind === "political_sensitive" &&
                            manifest.clean_cut?.cut_political_sensitive &&
                            cleanCutStats.matchesSeg(seg))
                      );
                      const hasDel = activeHits.some(
                        (seg) => resolveSegActionFrontend(seg, manifest.clean_cut, canMask) === "delete"
                      );
                      const maskHits = activeHits.filter((seg) => {
                        const a = resolveSegActionFrontend(seg, manifest.clean_cut, canMask);
                        return a === "mask" || a === "precise_mask";
                      });
                      const allPrecise =
                        maskHits.length > 0 &&
                        maskHits.every((seg) => resolveSegActionFrontend(seg, manifest.clean_cut, canMask) === "precise_mask");
                      const rowStyle: CSSProperties | undefined = hasDel
                        ? { background: "rgba(239, 68, 68, 0.10)", borderLeft: "3px solid #ef4444" }
                        : maskHits.length > 0
                          ? { background: "rgba(56, 189, 248, 0.10)", borderLeft: "3px solid #38bdf8" }
                          : undefined;
                      return (
                        <div
                          key={u.id}
                          className={`utt-row ${inRange ? "in-range" : "out-range"} ${isPlaying ? "playing" : ""}`}
                          style={rowStyle}
                        >
                          <span className="utt-time" onClick={() => seekTo(u.start_s)} title="点击跳转">
                            {fmtTime(u.start_s)}
                          </span>
                          <div>
                            <div className="utt-zh">
                              {u.zh || u.en}
                              {hasDel && (
                                <span
                                  className="tag"
                                  style={{ marginLeft: 6, background: "rgba(239, 68, 68, 0.2)", color: "#fca5a5" }}
                                >
                                  将删段
                                </span>
                              )}
                              {!hasDel && maskHits.length > 0 && (
                                <span
                                  className="tag"
                                  style={{ marginLeft: 6, background: "rgba(56, 189, 248, 0.2)", color: "#7dd3fc" }}
                                >
                                  {allPrecise ? "精准遮蔽:" : "将遮蔽:"}{" "}
                                  {Array.from(
                                    new Set(
                                      maskHits.flatMap((s) => {
                                        const act = resolveSegActionFrontend(s, manifest.clean_cut, canMask);
                                        const t = getEffectiveMaskTerms(s, act);
                                        return [...t.zh, ...t.en];
                                      })
                                    )
                                  ).join(" / ") || (allPrecise ? "无精准词(放行)" : "整句")}
                                </span>
                              )}
                              {activeHits.length === 0 && uHits.length > 0 && (
                                <span
                                  className="tag"
                                  style={{ marginLeft: 6, background: "rgba(148, 163, 184, 0.18)", color: "#cbd5e1" }}
                                >
                                  {uHits.some((s) => s.kind === "political_sensitive" && !cleanCutStats.matchesSeg(s))
                                    ? "当前标准放行"
                                    : "未勾选过滤"}
                                </span>
                              )}
                            </div>
                            {u.en && u.zh && <div className="utt-en">{u.en}</div>}
                          </div>
                          <div className="utt-actions">
                            <button
                              type="button"
                              className="sm"
                              onClick={() => handlePatchClip(activeClip.clip_id, { start_utt: u.id })}
                            >
                              设为起点
                            </button>
                            <button
                              type="button"
                              className="sm"
                              onClick={() => handlePatchClip(activeClip.clip_id, { end_utt: u.id })}
                            >
                              设为终点
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </details>
                </div>
              </>
            ) : (
              <div className="empty-state">
                {!hasSubtitles
                  ? "无字幕模式：请点击上方播放器控制栏中的「裁剪画面」，拖拽框选保留画幅并点击「应用裁剪」即可直接导出。"
                  : hasStrategy
                  ? "在左侧选择一个切片、精选集或 AI 混剪进行预览与发布。"
                  : "尚未生成切片策略。点击上方「生成切片策略」完成广告/敏感分析、独立切片与高光混剪规划。"}
              </div>
            )}
            </div>
          </section>

          {/* RIGHT PANE: Receipts & Logs Drawer (Hidden by default) */}
          {showLogs && (
            <section className="pane">
              <div className="pane-header">
                <div className="tabs">
                  <button
                    type="button"
                    className={`tab-btn ${rightTab === "receipts" ? "active" : ""}`}
                    onClick={() => setRightTab("receipts")}
                  >
                    回执 ({receipts.length})
                  </button>
                  <button
                    type="button"
                    className={`tab-btn ${rightTab === "topics" ? "active" : ""}`}
                    onClick={() => setRightTab("topics")}
                  >
                    话题 ({manifest.topic_map.length})
                  </button>
                  <button
                    type="button"
                    className={`tab-btn ${rightTab === "logs" ? "active" : ""}`}
                    onClick={() => setRightTab("logs")}
                  >
                    运行日志
                  </button>
                </div>
              </div>

              <div className="pane-body scroller">
                {rightTab === "receipts" && (
                  <>
                    <div style={{ marginBottom: 8, display: "flex", gap: 6 }}>
                      <select
                        aria-label="按阶段筛选回执"
                        value={stageFilter}
                        onChange={(e) => setStageFilter(e.target.value)}
                        style={{ flex: 1 }}
                      >
                        <option value="">全部阶段</option>
                        <option value="ingest">ingest</option>
                        <option value="signals">signals</option>
                        <option value="scan_cuts">scan_cuts</option>
                        <option value="analyze">analyze</option>
                        <option value="verify_and_plan">verify_and_plan</option>
                        <option value="render">render</option>
                        <option value="publish">publish</option>
                        <option value="human_edit">human_edit</option>
                        <option value="replan">replan</option>
                      </select>
                      <button
                        type="button"
                        className="sm"
                        onClick={() => refreshReceipts(manifest.project_id, stageFilter)}
                      >
                        刷新
                      </button>
                    </div>

                    {receipts.map((r) => (
                      <div key={r.receipt_id} className={`receipt-item ${r.error ? "has-error" : ""}`}>
                        <div className="receipt-head">
                          <span className="mono">
                            [{r.stage}] {r.action}
                          </span>
                          <span className="mono">{r.elapsed_ms ? `${r.elapsed_ms}ms` : ""}</span>
                        </div>
                        <div>{r.summary}</div>
                        {r.error && <div style={{ color: "#fca5a5", marginTop: 4 }}>{r.error}</div>}
                        {(Object.keys(r.details || {}).length > 0 || Object.keys(r.diff || {}).length > 0) && (
                          <details style={{ marginTop: 5 }}>
                            <summary style={{ cursor: "pointer", color: "var(--text-2)", fontSize: 11 }}>详情</summary>
                            <pre className="json-pre">
                              {JSON.stringify(Object.keys(r.diff || {}).length > 0 ? r.diff : r.details, null, 2)}
                            </pre>
                          </details>
                        )}
                      </div>
                    ))}
                  </>
                )}

                {rightTab === "topics" &&
                  manifest.topic_map.map((t) => (
                    <div key={t.topic_id} className="receipt-item">
                      <div className="receipt-head">
                        <span className="mono">
                          {t.topic_id} · {fmtTime(t.start_s)}–{fmtTime(t.end_s)}
                        </span>
                        <span className={`tag ${t.selected_clip_id ? "cached" : ""}`}>
                          {t.selected_clip_id ? t.selected_clip_id : t.is_filler ? "片花" : "未切"}
                        </span>
                      </div>
                      <div style={{ fontWeight: 600 }}>{t.title}</div>
                      <div style={{ color: "var(--text-2)", marginTop: 2 }}>{t.summary}</div>
                    </div>
                  ))}

                {rightTab === "logs" && (
                  <pre className="json-pre" style={{ maxHeight: 540 }}>
                    {pipelineLog.join("\n") || "暂无日志"}
                  </pre>
                )}
              </div>
            </section>
          )}
        </main>
      )}

      {/* Multi-Video Folder Picker Dialog */}
      <dialog
        ref={pickerDialogRef}
        aria-labelledby="picker-title"
        {...({ closedby: "any" } as Record<string, string>)}
        onClose={() => setScanCandidates(null)}
      >
        <div className="dialog-head">
          <h2 id="picker-title">目录内发现多个视频，请选择目标视频</h2>
          <form method="dialog">
            <button type="submit" className="sm">
              关闭
            </button>
          </form>
        </div>
        <div className="dialog-body scroller">
          {scanCandidates?.videos.map((v) => (
            <div key={v.path} className="picker-row">
              <div>
                <div style={{ fontWeight: 600 }}>
                  {v.filename} {v.is_hardsub && <span className="tag cached">硬字幕</span>}
                </div>
                <div style={{ fontSize: 12, color: "var(--text-2)", marginTop: 3 }}>
                  {v.width}×{v.height} · {fmtTime(v.duration_s)} · {fmtSize(v.size_bytes)}
                </div>
              </div>
              <button
                type="button"
                className="primary"
                onClick={() => openCandidate(v)}
                autoFocus={v === scanCandidates.videos[0]}
              >
                选择此视频
              </button>
            </div>
          ))}
        </div>
      </dialog>

      {/* Settings Dialog (Desensitization policy, models, manual path, YAML config) */}
      <dialog
        ref={settingsDialogRef}
        aria-labelledby="settings-title"
        {...({ closedby: "any" } as Record<string, string>)}
      >
        <div className="dialog-head">
          <h2 id="settings-title">设置（脱敏策略、模型与手动载入）</h2>
          <form method="dialog">
            <button type="submit" className="sm">
              关闭
            </button>
          </form>
        </div>
        <div className="dialog-body scroller" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div
            style={{
              padding: "10px 12px",
              borderRadius: 8,
              border: "1px solid var(--border)",
              background: "var(--bg-2)",
              display: "flex",
              flexDirection: "column",
              gap: 10,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
              <span style={{ fontWeight: 600 }}>去广与脱敏策略</span>
              <button type="button" className="sm" onClick={openPolicyDialog}>
                配置三档政审规则…
              </button>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 12, color: "var(--text-2)" }}>
                  当前项目脱敏方式
                </span>
                <select
                  aria-label="当前项目脱敏方式"
                  disabled={!manifest || jobRunning}
                  value={manifest ? cleanCutStats.politicalAction : contentFilterCfg?.default_political_action || "auto"}
                  onChange={(e) => handleChangeProjectPoliticalAction(e.target.value as PoliticalAction)}
                >
                  <option value="auto">
                    智能（顺带打码+哔音，整段删剪）{manifest && !canMask ? " [无干净原片]" : ""}
                  </option>
                  <option value="delete">全删片段</option>
                  <option value="mask" disabled={Boolean(manifest && !canMask)}>
                    常规遮蔽+消音（现有方式）{manifest && !canMask ? "（无干净原片）" : ""}
                  </option>
                  <option value="precise_mask" disabled={Boolean(manifest && !canMask)}>
                    精准遮蔽+消音（仅关键词）{manifest && !canMask ? "（无干净原片）" : ""}
                  </option>
                </select>
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 12, color: "var(--text-2)" }}>
                  新建项目默认脱敏方式（写入配置）
                </span>
                <select
                  aria-label="配置中的默认脱敏方式"
                  value={contentFilterCfg?.default_political_action || "auto"}
                  onChange={(e) => handleChangeDefaultPoliticalAction(e.target.value as PoliticalAction)}
                >
                  <option value="auto">智能（顺带打码+哔音，整段删剪）</option>
                  <option value="delete">全删片段</option>
                  <option value="mask">常规遮蔽+消音（现有方式）</option>
                  <option value="precise_mask">精准遮蔽+消音（仅关键词）</option>
                </select>
              </label>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, alignItems: "end" }}>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 12, color: "var(--text-2)" }}>净版剪切模式</span>
                <select
                  aria-label="设置中的净版剪切模式"
                  disabled={!manifest || jobRunning}
                  value={manifest?.clean_cut?.cut_mode || "auto"}
                  onChange={(e) => handlePatchCleanCut({ cut_mode: e.target.value })}
                >
                  <option value="auto">智能（含敏感时帧精准）</option>
                  <option value="precise">极速无损关键帧</option>
                  <option value="reencode">全片硬件帧精准</option>
                </select>
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 12, color: "var(--text-2)" }}>新建项目默认政审标准</span>
                <select
                  aria-label="设置中的默认政审标准"
                  value={contentFilterCfg?.default_sensitivity || "strict"}
                  onChange={(e) =>
                    handleChangeDefaultSensitivity(e.target.value as "strict" | "balanced" | "loose")
                  }
                >
                  <option value="strict">严格从严（推荐视频号/B站）</option>
                  <option value="balanced">平衡</option>
                  <option value="loose">仅硬红线</option>
                </select>
              </label>
            </div>
            <div style={{ fontSize: 11, color: "var(--text-2)", lineHeight: 1.5 }}>
              {manifest && !canMask
                ? "当前项目缺少无字幕干净原片，遮蔽选项已禁用并自动回退删段。"
                : "「精准遮蔽」仅屏蔽领导人/政体/中港台不规范称谓及定性判断词，不消音整段；任一模式均可设为新建项目默认。"}
            </div>
          </div>

          <div
            style={{
              padding: "10px 12px",
              borderRadius: 8,
              border: "1px solid var(--border)",
              background: "var(--bg-2)",
              display: "flex",
              flexDirection: "column",
              gap: 10,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
              <span style={{ fontWeight: 600 }}>导航样式与封面主题色系</span>
              <span style={{ fontSize: 11, color: "var(--text-2)" }}>
                切换后重渲或刷新封面生效
              </span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 12, color: "var(--text-2)" }}>高光混剪导航模式</span>
                <select
                  aria-label="高光混剪导航模式"
                  value={navCfg.remix_style || navCfg.style}
                  onChange={async (e) => {
                    const nextStyle = e.target.value as "top_nav" | "both" | "title_card";
                    setNavCfg((prev) => ({ ...prev, remix_style: nextStyle }));
                    try {
                      const saved = await api.saveNavigationConfig({ remix_style: nextStyle });
                      setNavCfg(saved);
                      if (manifest) {
                        const refreshed = await api.getProject(manifest.project_id);
                        setManifest(refreshed.manifest);
                      }
                    } catch (err) {
                      setErrorMsg(String(err));
                    }
                  }}
                >
                  <option value="top_nav">常驻导航式（默认 · 全程高亮）</option>
                  <option value="both">导航 + 段前卡片（双显）</option>
                  <option value="title_card">段前卡片式（每段前插入片头卡）</option>
                </select>
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 12, color: "var(--text-2)" }}>切片精选集导航模式</span>
                <select
                  aria-label="切片精选集导航模式"
                  value={navCfg.compilation_style || navCfg.style}
                  onChange={async (e) => {
                    const nextStyle = e.target.value as "top_nav" | "both" | "title_card";
                    setNavCfg((prev) => ({ ...prev, compilation_style: nextStyle }));
                    try {
                      const saved = await api.saveNavigationConfig({ compilation_style: nextStyle });
                      setNavCfg(saved);
                      if (manifest) {
                        const refreshed = await api.getProject(manifest.project_id);
                        setManifest(refreshed.manifest);
                      }
                    } catch (err) {
                      setErrorMsg(String(err));
                    }
                  }}
                >
                  <option value="top_nav">常驻导航式（默认 · 全程高亮）</option>
                  <option value="both">导航 + 段前卡片（双显）</option>
                  <option value="title_card">段前卡片式（每段前插入片头卡）</option>
                </select>
              </label>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 12, color: "var(--text-2)" }}>屏幕布局（混剪 / 精选集）</span>
                <select
                  aria-label="屏幕布局"
                  value={navCfg.screen_layout || "portrait"}
                  onChange={async (e) => {
                    const nextLayout = e.target.value as "portrait" | "landscape";
                    setNavCfg((prev) => ({ ...prev, screen_layout: nextLayout }));
                    try {
                      const saved = await api.saveNavigationConfig({ screen_layout: nextLayout });
                      setNavCfg(saved);
                      if (manifest) {
                        const refreshed = await api.getProject(manifest.project_id);
                        setManifest(refreshed.manifest);
                      }
                    } catch (err) {
                      setErrorMsg(String(err));
                    }
                  }}
                >
                  <option value="portrait">竖版 9:16（默认 · 标题与导航置上下黑边）</option>
                  <option value="landscape">横版 16:9（原画比例 · 顶部导航条）</option>
                </select>
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 12, color: "var(--text-2)" }}>主题色系（导航 / 卡片 / 封面）</span>
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <select
                    aria-label="导航主题色系"
                    style={{ flex: 1 }}
                    value={navCfg.color_scheme}
                    onChange={async (e) => {
                      const nextScheme = e.target.value;
                      setNavCfg((prev) => ({ ...prev, color_scheme: nextScheme }));
                      try {
                        const saved = await api.saveNavigationConfig({ color_scheme: nextScheme });
                        setNavCfg(saved);
                        if (manifest) {
                          const target =
                            selectedId === "REMIX"
                              ? "remix"
                              : selectedId === "COMPILATION"
                                ? "compilation"
                                : selectedId === "CLEAN_CUT"
                                  ? "clean_cut"
                                  : selectedId || "remix";
                          try {
                            const cov = await api.refreshCover(manifest.project_id, target);
                            setManifest(cov.manifest);
                            setCoverNonce((n) => n + 1);
                          } catch {
                            const refreshed = await api.getProject(manifest.project_id);
                            setManifest(refreshed.manifest);
                          }
                        }
                      } catch (err) {
                        setErrorMsg(String(err));
                      }
                    }}
                  >
                    {(
                      navCfg.presets || [
                        { id: "tiffany", label: "蒂芙尼青", hex: "#0ABAB5" },
                        { id: "amber", label: "琥珀金", hex: "#F59E0B" },
                        { id: "cobalt", label: "钴蓝科技", hex: "#3B82F6" },
                        { id: "crimson", label: "绯红醒目", hex: "#EF4444" },
                        { id: "emerald", label: "翡翠绿", hex: "#10B981" },
                        { id: "violet", label: "紫罗兰", hex: "#8B5CF6" },
                      ]
                    ).map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.label} ({p.hex})
                      </option>
                    ))}
                    <option value="custom">自定义色值 ({navCfg.custom_accent})</option>
                  </select>
                  <input
                    type="color"
                    aria-label="自定义导航主题色"
                    value={
                      navCfg.color_scheme === "custom"
                        ? navCfg.custom_accent || "#0ABAB5"
                        : (navCfg.presets || []).find((p) => p.id === navCfg.color_scheme)?.hex || "#0ABAB5"
                    }
                    onChange={async (e) => {
                      const hex = e.target.value.toUpperCase();
                      setNavCfg((prev) => ({ ...prev, color_scheme: "custom", custom_accent: hex }));
                      try {
                        const saved = await api.saveNavigationConfig({
                          color_scheme: "custom",
                          custom_accent: hex,
                        });
                        setNavCfg(saved);
                        if (manifest) {
                          const target =
                            selectedId === "REMIX"
                              ? "remix"
                              : selectedId === "COMPILATION"
                                ? "compilation"
                                : selectedId === "CLEAN_CUT"
                                  ? "clean_cut"
                                  : selectedId || "remix";
                          try {
                            const cov = await api.refreshCover(manifest.project_id, target);
                            setManifest(cov.manifest);
                            setCoverNonce((n) => n + 1);
                          } catch {
                            const refreshed = await api.getProject(manifest.project_id);
                            setManifest(refreshed.manifest);
                          }
                        }
                      } catch (err) {
                        setErrorMsg(String(err));
                      }
                    }}
                    title="自定义主题色"
                    style={{
                      width: 32,
                      height: 28,
                      padding: 2,
                      border: "1px solid var(--border)",
                      borderRadius: 4,
                      background: "transparent",
                      cursor: "pointer",
                    }}
                  />
                </div>
              </label>
            </div>
            {(() => {
              const activeHex =
                navCfg.color_scheme === "custom"
                  ? navCfg.custom_accent || "#0ABAB5"
                  : (navCfg.presets || [
                      { id: "tiffany", label: "蒂芙尼青", hex: "#0ABAB5" },
                      { id: "amber", label: "琥珀金", hex: "#F59E0B" },
                      { id: "cobalt", label: "钴蓝科技", hex: "#3B82F6" },
                      { id: "crimson", label: "绯红醒目", hex: "#EF4444" },
                      { id: "emerald", label: "翡翠绿", hex: "#10B981" },
                      { id: "violet", label: "紫罗兰", hex: "#8B5CF6" },
                    ]).find((p) => p.id === navCfg.color_scheme)?.hex || "#0ABAB5";
              return (
                <div
                  style={{
                    display: "flex",
                    gap: 6,
                    padding: "6px 8px",
                    borderRadius: 6,
                    background: "rgba(11, 15, 24, 0.92)",
                    border: `1px solid ${activeHex}55`,
                    fontSize: 11,
                    alignItems: "center",
                  }}
                >
                  <span style={{ color: "#94a3b8", padding: "2px 6px", borderRadius: 4, background: "rgba(255,255,255,0.06)" }}>
                    01 已播分段
                  </span>
                  <span style={{ color: "#fff", fontWeight: 700, padding: "2px 8px", borderRadius: 4, background: activeHex }}>
                    02 当前高亮分段
                  </span>
                  <span style={{ color: "#94a3b8", padding: "2px 6px", borderRadius: 4, background: "rgba(255,255,255,0.04)" }}>
                    03 待播分段
                  </span>
                  <span style={{ marginLeft: "auto", color: "#94a3b8" }}>
                    {(navCfg.screen_layout || "portrait") === "portrait" ? "竖版9:16" : "横版16:9"} · 混剪:
                    {(navCfg.remix_style || navCfg.style) === "both"
                      ? "导航+卡"
                      : (navCfg.remix_style || navCfg.style) === "top_nav"
                        ? "常驻"
                        : "卡片"}
                    /精选集:
                    {(navCfg.compilation_style || navCfg.style) === "both"
                      ? "导航+卡"
                      : (navCfg.compilation_style || navCfg.style) === "top_nav"
                        ? "常驻"
                        : "卡片"}
                  </span>
                </div>
              );
            })()}
          </div>

          <div>
            <div style={{ fontSize: 12, color: "var(--text-2)", marginBottom: 4 }}>手动输入本地视频或目录路径</div>
            <div style={{ display: "flex", gap: 6 }}>
              <input
                type="text"
                style={{ flex: 1 }}
                value={inputPath}
                onChange={(e) => setInputPath(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleScanAndOpen()}
                placeholder="/path/to/video.mp4"
              />
              <input
                type="text"
                style={{ width: 170 }}
                value={sourceUrl}
                onChange={(e) => setSourceUrl(e.target.value)}
                placeholder="可选原链接"
              />
              <button type="button" className="primary" onClick={() => handleScanAndOpen()}>
                载入
              </button>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontSize: 12, color: "var(--text-2)" }}>文本分析模型</span>
              <select value={textModelSpec} onChange={(e) => handleModelChange("text", e.target.value)}>
                {modelCatalog.text.map((m) => (
                  <option key={`${m.provider}:${m.model}`} value={`${m.provider}:${m.model}`}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontSize: 12, color: "var(--text-2)" }}>多模态复核模型</span>
              <select value={mmModelSpec} onChange={(e) => handleModelChange("multimodal", e.target.value)}>
                {modelCatalog.multimodal.map((m) => (
                  <option key={`${m.provider}:${m.model}`} value={`${m.provider}:${m.model}`}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", paddingTop: 6 }}>
            <button
              type="button"
              disabled={!manifest || jobRunning}
              onClick={() => {
                settingsDialogRef.current?.close();
                openReplanDialog();
              }}
            >
              重新制定切片策略…
            </button>
            <button type="button" onClick={openEditorialConfig}>
              编辑热点词表 YAML…
            </button>
          </div>
        </div>
      </dialog>

      {/* Editorial YAML Dialog */}
      <dialog
        ref={configDialogRef}
        aria-labelledby="config-title"
        {...({ closedby: "any" } as Record<string, string>)}
      >
        <div className="dialog-head">
          <h2 id="config-title">热点词表与配置 (editorial.yaml)</h2>
          <form method="dialog">
            <button type="submit" className="sm">
              关闭
            </button>
          </form>
        </div>
        <div className="dialog-body scroller">
          <textarea
            aria-label="编辑配置 YAML"
            value={yamlText}
            onChange={(e) => setYamlText(e.target.value)}
            rows={20}
            className="mono"
            style={{ width: "100%", resize: "vertical" }}
            autoFocus
          />
        </div>
        <div className="dialog-foot">
          <form method="dialog">
            <button type="submit">取消</button>
          </form>
          <button type="button" className="primary" disabled={yamlSaving} onClick={saveEditorialConfig}>
            {yamlSaving ? "保存中..." : "保存配置"}
          </button>
        </div>
      </dialog>

      {/* Multi-Platform Publish Dialog (WeChat Channels / Bilibili / Xiaohongshu) */}
      <dialog
        ref={publishDialogRef}
        aria-labelledby="publish-title"
        {...({ closedby: "any" } as Record<string, string>)}
      >
        <div className="dialog-head">
          <h2 id="publish-title">多平台一键分发（复用 AutoSub 登录态与脚本）</h2>
          <form method="dialog">
            <button type="submit" className="sm">
              关闭
            </button>
          </form>
        </div>
        <div className="dialog-body scroller" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
            <span style={{ fontSize: 12, color: "var(--text-2)" }}>选择发布平台与登录态</span>
            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              {activePublishTarget && activePublishRecords.some((r) => r.status === "error" || r.status === "running") && (
                <button
                  type="button"
                  className="sm"
                  disabled={jobRunning || pubResettingKey !== null}
                  onClick={() => handleResetPublish(activePublishTarget)}
                >
                  {pubResettingKey === `${activePublishTarget}:all` ? "复位中..." : "复位全部失败状态"}
                </button>
              )}
              <button type="button" className="sm" onClick={() => handleRefreshPublishStatus()}>
                刷新登录态
              </button>
            </div>
          </div>

          {pubLoginNotice && (
            <div className="pub-login-notice" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
              <span>{pubLoginNotice}</span>
              <button type="button" className="sm" onClick={() => setPubLoginNotice(null)}>
                关闭
              </button>
            </div>
          )}

          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {(["wechat", "bilibili", "xiaohongshu"] as const).map((plat) => {
              const st = platStatus[plat];
              const logged = Boolean(st?.logged_in);
              const loginRunning = Boolean(st?.login_running);
              const updatedClock = formatShortClock(st?.login_updated_at);
              const targetRec = activePublishRecords.find((r) => r.platform === plat);
              const hasFailedRec = targetRec?.status === "error";
              const errView = hasFailedRec && targetRec ? formatPublishErrorView(targetRec) : null;
              return (
                <div
                  key={plat}
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 6,
                    padding: "8px 10px",
                    borderRadius: 8,
                    border: "1px solid var(--border)",
                    background: "var(--bg-2)",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
                    <label style={{ display: "inline-flex", alignItems: "center", gap: 8, cursor: "pointer", flexWrap: "wrap" }}>
                      <input
                        type="checkbox"
                        checked={Boolean(pubPlatforms[plat])}
                        onChange={(e) => setPubPlatforms((prev) => ({ ...prev, [plat]: e.target.checked }))}
                      />
                      <span style={{ fontWeight: 600 }}>{PLAT_LABELS[plat]}</span>
                      <span className={`tag ${loginRunning ? "running" : logged ? "cached" : "dirty"}`}>
                        {loginRunning ? "扫码中..." : logged ? "已登录" : "未登录"}
                      </span>
                      {updatedClock && (
                        <span style={{ fontSize: 11, color: "var(--text-2)" }}>更新 {updatedClock}</span>
                      )}
                      {hasFailedRec && <span className="tag dirty">上次失败</span>}
                    </label>
                    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                      {hasFailedRec && activePublishTarget && (
                        <button
                          type="button"
                          className="sm"
                          disabled={jobRunning || pubResettingKey !== null}
                          onClick={() => handleResetPublish(activePublishTarget, plat)}
                        >
                          {pubResettingKey === `${activePublishTarget}:${plat}` ? "复位中..." : "复位状态"}
                        </button>
                      )}
                      <button type="button" className="sm" onClick={() => handleLaunchLogin(plat)}>
                        {loginRunning ? "查看登录态" : logged ? "重扫码" : "扫码登录"}
                      </button>
                    </div>
                  </div>
                  {errView && (
                    <div style={{ fontSize: 11, lineHeight: 1.45, color: "#fca5a5", paddingLeft: 22 }}>
                      <div>原因：{errView.errorText}</div>
                      <div style={{ color: "#fde68a" }}>解决方案：{errView.solutionText}</div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 12, color: "var(--text-2)" }}>
              发布标题（视频标题；视频号限30字 · 小红书限20字自动截断）
            </span>
            <input type="text" value={pubTitle} onChange={(e) => setPubTitle(e.target.value)} />
          </label>

          <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 12, color: "var(--text-2)" }}>封面标题（仅渲染在封面图上，可与发布标题不同）</span>
            <input type="text" value={pubCoverTitle} onChange={(e) => setPubCoverTitle(e.target.value)} />
          </label>

          <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
              <span style={{ fontSize: 12, color: "var(--text-2)" }}>简介文案</span>
              <button
                type="button"
                className="sm"
                disabled={pubDescSuggesting || !activePublishTarget}
                onClick={handleSuggestPublishDesc}
                title="由 AI 根据全片/片段内容提炼面向观众的中文简介，不含剪辑参数"
              >
                {pubDescSuggesting ? "提炼中..." : "AI 提炼简介"}
              </button>
            </div>
            <textarea rows={5} value={pubDesc} onChange={(e) => setPubDesc(e.target.value)} />
          </label>

          <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 12, color: "var(--text-2)" }}>标签（逗号分隔）</span>
            <input type="text" value={pubTags} onChange={(e) => setPubTags(e.target.value)} />
          </label>

          <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 12, color: "var(--text-2)" }}>
              合集名称（视频号自动创建/选择，限10字自动截断；B站按名匹配已有合集；小红书自动加入/创建）
            </span>
            <input
              type="text"
              maxLength={30}
              placeholder="默认取本片核心中文主题题眼"
              value={pubCollection}
              onChange={(e) => setPubCollection(e.target.value)}
            />
          </label>

          <label style={{ display: "inline-flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
            <input type="checkbox" checked={pubPublic} onChange={(e) => setPubPublic(e.target.checked)} />
            <span>公开发布（不勾选则安全模式：存草稿 / 仅自己可见）</span>
          </label>
        </div>
        <div className="dialog-foot">
          <form method="dialog">
            <button type="submit">取消</button>
          </form>
          <button type="button" className="primary" onClick={handleConfirmPublish}>
            开始上传
          </button>
        </div>
      </dialog>

      {/* Batch Publish Standalone Clips Dialog */}
      <dialog
        ref={batchPublishDialogRef}
        aria-labelledby="batch-publish-title"
        style={{ width: "min(94vw, 660px)", maxWidth: 660 }}
        {...({ closedby: "any" } as Record<string, string>)}
      >
        <div className="dialog-head">
          <h2 id="batch-publish-title">批量发布选中切片并归入同一合集</h2>
          <form method="dialog">
            <button type="submit" className="sm">
              关闭
            </button>
          </form>
        </div>
        {manifest && (
          <>
            <div className="dialog-body scroller" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: 12, color: "var(--text-1)", fontWeight: 600 }}>
                    合集名称（本次选中切片统一加入此合集）
                  </span>
                  <span style={{ fontSize: 11, color: batchCollection.trim().length > 10 ? "#fcd34d" : "var(--text-2)" }}>
                    {batchCollection.trim().length} 字{batchCollection.trim().length > 10 ? "（视频号自动取前10字）" : ""}
                  </span>
                </div>
                <input
                  type="text"
                  maxLength={30}
                  placeholder="留空则不指定合集，例如：特朗普芯片访谈"
                  value={batchCollection}
                  onChange={(e) => setBatchCollection(e.target.value)}
                />
                <span style={{ fontSize: 11, color: "var(--text-2)" }}>
                  视频号/小红书不存在同名合集时将自动新建；B站按名称匹配已有合集。
                </span>
              </label>

              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                <span style={{ fontSize: 12, color: "var(--text-2)" }}>选择发布平台与登录态</span>
                <button type="button" className="sm" onClick={() => handleRefreshPublishStatus()}>
                  刷新登录态
                </button>
              </div>

              {pubLoginNotice && (
                <div className="pub-login-notice" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
                  <span>{pubLoginNotice}</span>
                  <button type="button" className="sm" onClick={() => setPubLoginNotice(null)}>
                    关闭
                  </button>
                </div>
              )}

              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {(["wechat", "bilibili", "xiaohongshu"] as const).map((plat) => {
                  const st = platStatus[plat];
                  const logged = Boolean(st?.logged_in);
                  const loginRunning = Boolean(st?.login_running);
                  return (
                    <div
                      key={plat}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        gap: 8,
                        padding: "7px 10px",
                        borderRadius: 8,
                        border: "1px solid var(--border)",
                        background: "var(--bg-2)",
                      }}
                    >
                      <label style={{ display: "inline-flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
                        <input
                          type="checkbox"
                          checked={Boolean(batchPlatforms[plat])}
                          onChange={(e) => setBatchPlatforms((prev) => ({ ...prev, [plat]: e.target.checked }))}
                        />
                        <span style={{ fontWeight: 600 }}>{PLAT_LABELS[plat]}</span>
                        <span className={`tag ${loginRunning ? "running" : logged ? "cached" : "dirty"}`}>
                          {loginRunning ? "扫码中..." : logged ? "已登录" : "未登录"}
                        </span>
                      </label>
                      <button type="button" className="sm" onClick={() => handleLaunchLogin(plat)}>
                        {loginRunning ? "查看登录态" : logged ? "重扫码" : "扫码登录"}
                      </button>
                    </div>
                  );
                })}
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: 12, color: "var(--text-1)", fontWeight: 600 }}>
                    待发布独立切片（{selectedStandaloneClips.filter((c) => c.output_file && batchClipPick[c.clip_id]).length}/{selectedStandaloneClips.length}）
                  </span>
                  <div style={{ display: "flex", gap: 6 }}>
                    <button
                      type="button"
                      className="sm"
                      onClick={() => {
                        const next: Record<string, boolean> = {};
                        selectedStandaloneClips.forEach((c) => {
                          next[c.clip_id] = Boolean(c.output_file);
                        });
                        setBatchClipPick(next);
                      }}
                    >
                      全选已渲染
                    </button>
                    <button type="button" className="sm" onClick={() => setBatchClipPick({})}>
                      全不选
                    </button>
                  </div>
                </div>
                <div
                  className="scroller"
                  style={{
                    maxHeight: 210,
                    overflowY: "auto",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    padding: 6,
                    display: "flex",
                    flexDirection: "column",
                    gap: 4,
                    background: "var(--bg-0)",
                  }}
                >
                  {selectedStandaloneClips.length === 0 ? (
                    <div style={{ padding: 12, textAlign: "center", color: "var(--text-2)", fontSize: 12 }}>
                      左侧尚未勾选任何独立切片
                    </div>
                  ) : (
                    selectedStandaloneClips.map((c) => {
                      const ready = Boolean(c.output_file);
                      const checked = ready && Boolean(batchClipPick[c.clip_id]);
                      return (
                        <label
                          key={c.clip_id}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            gap: 8,
                            padding: "5px 8px",
                            borderRadius: 6,
                            background: "var(--bg-2)",
                            opacity: ready ? 1 : 0.5,
                            cursor: ready ? "pointer" : "not-allowed",
                          }}
                        >
                          <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
                            <input
                              type="checkbox"
                              disabled={!ready}
                              checked={checked}
                              onChange={(e) =>
                                setBatchClipPick((prev) => ({ ...prev, [c.clip_id]: e.target.checked }))
                              }
                            />
                            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                              #{String(c.index).padStart(2, "0")} {c.title}
                            </span>
                          </div>
                          <div style={{ display: "flex", gap: 4, alignItems: "center", flexShrink: 0 }}>
                            {!ready && <span className="tag dirty">未渲染·跳过</span>}
                            {renderPubBadges(c.publish_records)}
                          </div>
                        </label>
                      );
                    })
                  )}
                </div>
                <span style={{ fontSize: 11, color: "var(--text-2)" }}>
                  各切片自动使用各自标题、封面标题、简介与标签顺序上传，单条失败不中断其余切片。
                </span>
              </div>

              <label style={{ display: "inline-flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={batchSkipPublished}
                  onChange={(e) => setBatchSkipPublished(e.target.checked)}
                />
                <span>跳过所选平台已有成功/草稿记录的切片（防重复上传）</span>
              </label>

              <label style={{ display: "inline-flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
                <input type="checkbox" checked={batchPublic} onChange={(e) => setBatchPublic(e.target.checked)} />
                <span>公开发布（不勾选则安全模式：存草稿 / 仅自己可见）</span>
              </label>
            </div>
            <div className="dialog-foot">
              <form method="dialog">
                <button type="submit" disabled={batchSubmitting}>
                  取消
                </button>
              </form>
              <button
                type="button"
                className="primary"
                disabled={
                  batchSubmitting ||
                  selectedStandaloneClips.filter((c) => c.output_file && batchClipPick[c.clip_id]).length === 0
                }
                onClick={handleConfirmBatchPublish}
              >
                {batchSubmitting
                  ? "提交中..."
                  : `开始批量上传 (${selectedStandaloneClips.filter((c) => c.output_file && batchClipPick[c.clip_id]).length} 支)`}
              </button>
            </div>
          </>
        )}
      </dialog>

      {/* Replan Strategy Confirmation Dialog */}
      <dialog
        ref={replanDialogRef}
        aria-labelledby="replan-title"
        {...({ closedby: "any" } as Record<string, string>)}
        onClose={() => setReplanAssets(null)}
      >
        <div className="dialog-head">
          <h2 id="replan-title">
            {replanAssets?.has_rendered ? "检测到已有已渲染成片，是否删除并彻底重做？" : "重新制定切片策略"}
          </h2>
          <form method="dialog">
            <button type="submit" className="sm">
              关闭
            </button>
          </form>
        </div>
        <div className="dialog-body scroller" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {replanAssets?.has_rendered ? (
            <>
              <div style={{ lineHeight: 1.6 }}>
                该项目已经执行过渲染，若继续重新制定切片策略，旧的切片边界与混剪规划将被新方案替换。请选择如何处理输出目录中的已有成片文件：
              </div>
              <div
                style={{
                  padding: "10px 12px",
                  borderRadius: 8,
                  border: "1px solid var(--border)",
                  background: "var(--bg-2)",
                  fontSize: 12,
                  display: "flex",
                  flexDirection: "column",
                  gap: 6,
                }}
              >
                <div>
                  输出目录：<span className="mono">{replanAssets.output_dir || "clips"}</span>
                </div>
                <div>
                  检测到：
                  <strong>{replanAssets.video_files.length}</strong> 个成片视频 ·{" "}
                  <strong>{replanAssets.cover_files.length}</strong> 张封面 ·{" "}
                  <strong>{replanAssets.doc_files.length}</strong> 个清单文件（共{" "}
                  {replanAssets.total_files} 项，约 {fmtSize(replanAssets.total_size_bytes)}）
                </div>
                {replanAssets.video_files.length > 0 && (
                  <div style={{ color: "var(--text-2)", maxHeight: 130, overflowY: "auto" }} className="mono">
                    {replanAssets.video_files.slice(0, 8).map((name) => (
                      <div key={name}>• {name}</div>
                    ))}
                    {replanAssets.video_files.length > 8 && (
                      <div>… 其余 {replanAssets.video_files.length - 8} 个视频文件</div>
                    )}
                  </div>
                )}
              </div>
              <div style={{ fontSize: 12, color: "var(--text-2)" }}>
                提示：「删除已有切片，彻底重做」仅清理输出目录的成片、封面、导出清单与临时渲染缓存，不会删除原始视频与字幕文件。
              </div>
            </>
          ) : (
            <div style={{ lineHeight: 1.6 }}>
              当前尚无已渲染成片文件，但检测到已有分析/扫描记录。继续将彻底清空旧的广告敏感识别、话题地图、切片列表与阶段状态，按最新规则从头重新规划。
            </div>
          )}
          <label
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              fontSize: 12,
              cursor: "pointer",
              paddingTop: 4,
            }}
          >
            <input
              type="checkbox"
              checked={replanForceSignals || manifest?.stages?.signals?.status !== "completed"}
              disabled={manifest?.stages?.signals?.status !== "completed"}
              onChange={(e) => setReplanForceSignals(e.target.checked)}
            />
            <span>
              强制重新提取声画信号（彻底从头重跑
              {manifest?.stages?.signals?.status !== "completed" ? " · 当前信号未就绪，将自动重提" : ""}）
            </span>
          </label>
        </div>
        <div className="dialog-foot">
          <form method="dialog">
            <button type="submit" disabled={replanSubmitting}>
              取消
            </button>
          </form>
          {replanAssets?.has_rendered ? (
            <>
              <button
                type="button"
                disabled={replanSubmitting}
                onClick={() => handleConfirmReplan(false)}
              >
                保留已有文件，仅重做策略
              </button>
              <button
                type="button"
                className="danger"
                disabled={replanSubmitting}
                onClick={() => handleConfirmReplan(true)}
                autoFocus
              >
                {replanSubmitting ? "清理并重置中..." : "删除已有切片，彻底重做"}
              </button>
            </>
          ) : (
            <button
              type="button"
              className="primary"
              disabled={replanSubmitting}
              onClick={() => handleConfirmReplan(true)}
              autoFocus
            >
              {replanSubmitting ? "重置中..." : "彻底清空并从头规划"}
            </button>
          )}
        </div>
      </dialog>

      {/* Political Sensitivity Policy Editor Dialog */}
      <dialog
        ref={policyDialogRef}
        aria-labelledby="policy-title"
        style={{ width: "min(94vw, 760px)", maxWidth: 760 }}
        {...({ closedby: "any" } as Record<string, string>)}
      >
        <div className="dialog-head">
          <h2 id="policy-title">政审标准与各档规则配置</h2>
          <form method="dialog">
            <button type="submit" className="sm">
              关闭
            </button>
          </form>
        </div>
        {policyDraft && (
          <>
            <div className="dialog-body scroller" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                <div className="tabs">
                  {(["strict", "balanced", "loose"] as const).map((k) => (
                    <button
                      key={k}
                      type="button"
                      className={`tab-btn ${policyTab === k ? "active" : ""}`}
                      onClick={() => setPolicyTab(k)}
                    >
                      {policyDraft.political_profiles[k]?.label || SENS_SHORT_LABELS[k]}
                    </button>
                  ))}
                </div>
                <label style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12 }}>
                  <span style={{ color: "var(--text-2)" }}>新建项目默认：</span>
                  <select
                    value={policyDraft.default_sensitivity}
                    onChange={(e) =>
                      setPolicyDraft({
                        ...policyDraft,
                        default_sensitivity: e.target.value as "strict" | "balanced" | "loose",
                      })
                    }
                  >
                    <option value="strict">严格从严</option>
                    <option value="balanced">平衡</option>
                    <option value="loose">仅硬红线</option>
                  </select>
                </label>
              </div>

              {(() => {
                const prof = policyDraft.political_profiles[policyTab];
                if (!prof) return null;
                return (
                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                      <span style={{ fontSize: 12, color: "var(--text-2)" }}>档位说明</span>
                      <input
                        type="text"
                        value={prof.description}
                        onChange={(e) =>
                          mutatePolicyProfile(policyTab, (p) => ({ ...p, description: e.target.value }))
                        }
                      />
                    </label>

                    <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap", fontSize: 12 }}>
                      <span style={{ color: "var(--text-2)" }}>生效拦截风险等级：</span>
                      {(
                        [
                          ["high", "高危 (high)"],
                          ["medium", "中危 (medium)"],
                          ["low", "低危 (low)"],
                        ] as const
                      ).map(([sev, label]) => {
                        const checked = prof.trigger_severities.includes(sev);
                        return (
                          <label key={sev} style={{ display: "inline-flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={(e) => {
                                const on = e.target.checked;
                                mutatePolicyProfile(policyTab, (p) => {
                                  const next = new Set(p.trigger_severities);
                                  if (on) next.add(sev);
                                  else if (next.size > 1) next.delete(sev);
                                  const order: ("high" | "medium" | "low")[] = ["high", "medium", "low"];
                                  return {
                                    ...p,
                                    trigger_severities: order.filter((x) => next.has(x)),
                                  };
                                });
                              }}
                            />
                            <span>{label}</span>
                          </label>
                        );
                      })}
                    </div>

                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ fontSize: 12, color: "var(--text-2)" }}>
                        该档判定规则（共 {prof.rules.length} 条 · 修改后切换标准即时过滤，重扫按新规则识别）
                      </span>
                      <button
                        type="button"
                        className="sm primary"
                        onClick={() =>
                          mutatePolicyProfile(policyTab, (p) => ({
                            ...p,
                            rules: [
                              ...p.rules,
                              {
                                id: `custom_${Date.now().toString(36)}`,
                                name: "新规则名称",
                                severity: policyTab === "loose" ? "high" : policyTab === "balanced" ? "medium" : "low",
                                description: "",
                              },
                            ],
                          }))
                        }
                      >
                        + 新增规则
                      </button>
                    </div>

                    {prof.rules.map((rule, rIdx) => (
                      <div
                        key={`${rule.id}-${rIdx}`}
                        style={{
                          padding: "8px 10px",
                          borderRadius: 8,
                          border: "1px solid var(--border)",
                          background: "var(--bg-2)",
                          display: "flex",
                          flexDirection: "column",
                          gap: 6,
                        }}
                      >
                        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                          <input
                            type="text"
                            style={{ flex: 1, fontWeight: 600 }}
                            placeholder="规则名称"
                            value={rule.name}
                            onChange={(e) =>
                              mutatePolicyProfile(policyTab, (p) => ({
                                ...p,
                                rules: p.rules.map((x, i) => (i === rIdx ? { ...x, name: e.target.value } : x)),
                              }))
                            }
                          />
                          <select
                            aria-label="规则风险级别"
                            value={rule.severity}
                            onChange={(e) =>
                              mutatePolicyProfile(policyTab, (p) => ({
                                ...p,
                                rules: p.rules.map((x, i) =>
                                  i === rIdx ? { ...x, severity: e.target.value as "high" | "medium" | "low" } : x
                                ),
                              }))
                            }
                          >
                            <option value="high">高危 (high)</option>
                            <option value="medium">中危 (medium)</option>
                            <option value="low">低危 (low)</option>
                          </select>
                          <button
                            type="button"
                            className="sm danger"
                            onClick={() =>
                              mutatePolicyProfile(policyTab, (p) => ({
                                ...p,
                                rules: p.rules.filter((_, i) => i !== rIdx),
                              }))
                            }
                          >
                            删除
                          </button>
                        </div>
                        <textarea
                          rows={2}
                          placeholder="详细判定标准与放行边界说明（直接注入 AI 审核 Prompt）"
                          value={rule.description}
                          onChange={(e) =>
                            mutatePolicyProfile(policyTab, (p) => ({
                              ...p,
                              rules: p.rules.map((x, i) => (i === rIdx ? { ...x, description: e.target.value } : x)),
                            }))
                          }
                        />
                      </div>
                    ))}
                  </div>
                );
              })()}
            </div>
            <div className="dialog-foot" style={{ justifyContent: "space-between" }}>
              <button
                type="button"
                disabled={policySaving}
                onClick={() => handleResetPolicyProfile(policyTab)}
              >
                恢复本档默认
              </button>
              <div style={{ display: "flex", gap: 8 }}>
                <form method="dialog">
                  <button type="submit" disabled={policySaving}>
                    取消
                  </button>
                </form>
                <button
                  type="button"
                  className="primary"
                  disabled={policySaving}
                  onClick={handleSavePolicyConfig}
                >
                  {policySaving ? "保存中..." : "保存政审规则"}
                </button>
              </div>
            </div>
          </>
        )}
      </dialog>

      {/* Video Metadata Editor Dialog */}
      <dialog ref={metadataDialogRef} className="modal" style={{ maxWidth: 460 }}>
        <div className="dialog-head">
          <span>编辑视频元数据</span>
          <form method="dialog">
            <button type="submit" className="sm">
              关闭
            </button>
          </form>
        </div>
        {manifest && (
          <>
            <div className="dialog-body" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 12, color: "var(--text-1)" }}>发布频道 / 栏目</span>
                <input
                  type="text"
                  value={metaChannelDraft}
                  onChange={(e) => setMetaChannelDraft(e.target.value)}
                  placeholder="例如：Charlie Rose 访谈"
                />
                {manifest.metadata.channel && manifest.metadata.channel !== metaChannelDraft && (
                  <span style={{ fontSize: 11, color: "var(--text-2)" }}>
                    原采集值：{manifest.metadata.channel}
                  </span>
                )}
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 12, color: "var(--text-1)" }}>被访谈人 / 嘉宾</span>
                <input
                  type="text"
                  value={metaGuestDraft}
                  onChange={(e) => setMetaGuestDraft(e.target.value)}
                  placeholder="例如：查理·芒格"
                />
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 12, color: "var(--text-1)" }}>发布时间</span>
                <input
                  type="text"
                  value={metaDateDraft}
                  onChange={(e) => setMetaDateDraft(e.target.value)}
                  placeholder="例如：2015-05-02"
                />
              </label>
            </div>
            <div className="dialog-foot" style={{ justifyContent: "flex-end", gap: 8 }}>
              <form method="dialog">
                <button type="submit" disabled={metaSaving}>
                  取消
                </button>
              </form>
              <button
                type="button"
                className="primary"
                disabled={metaSaving}
                onClick={handleSaveMetadata}
              >
                {metaSaving ? "保存中..." : "保存"}
              </button>
            </div>
          </>
        )}
      </dialog>
    </div>
  );
}
