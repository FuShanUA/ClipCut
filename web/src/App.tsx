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
import type {
  CleanCutConfig,
  ClipItem,
  ContentFilterConfig,
  CutSegment,
  PlatformStatus,
  PoliticalProfile,
  ProjectManifest,
  ProjectSummary,
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

interface ComplianceSegmentHit {
  seg: CutSegment;
  effAction: "delete" | "mask" | "passed" | "off";
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

function resolveSegActionFrontend(
  seg: CutSegment,
  cc: CleanCutConfig | undefined,
  canMask: boolean
): "delete" | "mask" {
  if (seg.kind !== "political_sensitive" || !canMask) return "delete";
  if (seg.action === "mask" || seg.action === "delete") return seg.action;
  const g = cc?.political_action || "delete";
  if (g === "mask") return "mask";
  if (g === "auto") return seg.recommended_action === "mask" ? "mask" : "delete";
  return "delete";
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
    if (act === "mask") {
      maskCount += 1;
      for (const tz of seg.sensitive_terms_zh || []) if (tz.trim()) maskTerms.add(tz.trim());
      for (const te of seg.sensitive_terms_en || []) if (te.trim()) maskTerms.add(te.trim());
      overlappingSegs.push({ seg, effAction: "mask", reasonLabel: "脱敏·遮蔽消音" });
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
  scan_cuts: "广敏分析",
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
    if (rec.url_or_id) {
      return (
        <a
          key={rec.platform}
          href={rec.url_or_id}
          target="_blank"
          rel="noreferrer"
          className={`pub-badge ${rec.status}`}
          onClick={(e) => e.stopPropagation()}
          title={rec.message || rec.error || ""}
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
        title={rec.message || rec.error || ""}
      >
        {label}:{statusText}
      </span>
    );
  });
}

export default function App() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [inputPath, setInputPath] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [manifest, setManifest] = useState<ProjectManifest | null>(null);
  const [jobRunning, setJobRunning] = useState(false);
  const [jobCancelling, setJobCancelling] = useState(false);
  const [jobStages, setJobStages] = useState<string[]>([]);
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
  const replanDialogRef = useRef<HTMLDialogElement | null>(null);
  const policyDialogRef = useRef<HTMLDialogElement | null>(null);
  const [replanAssets, setReplanAssets] = useState<RenderedAssetsInfo | null>(null);
  const [projectAssets, setProjectAssets] = useState<RenderedAssetsInfo | null>(null);
  const [canMask, setCanMask] = useState(true);
  const [replanForceSignals, setReplanForceSignals] = useState(false);
  const [replanSubmitting, setReplanSubmitting] = useState(false);

  // Political sensitivity policy profiles & editing draft
  const [contentFilterCfg, setContentFilterCfg] = useState<ContentFilterConfig | null>(null);
  const [policyDraft, setPolicyDraft] = useState<ContentFilterConfig | null>(null);
  const [policyTab, setPolicyTab] = useState<"strict" | "balanced" | "loose">("strict");
  const [policySaving, setPolicySaving] = useState(false);

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
  const [coverRefreshing, setCoverRefreshing] = useState(false);
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
  const [pubTags, setPubTags] = useState("");
  const [pubPublic, setPubPublic] = useState(false);
  const [pubCollection, setPubCollection] = useState("");

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
  }, []);

  useEffect(() => {
    const c1 = bindOutsideClickDismiss(pickerDialogRef.current, () => setScanCandidates(null));
    const c2 = bindOutsideClickDismiss(settingsDialogRef.current, () => {});
    const c3 = bindOutsideClickDismiss(configDialogRef.current, () => {});
    const c4 = bindOutsideClickDismiss(publishDialogRef.current, () => {});
    const c5 = bindOutsideClickDismiss(replanDialogRef.current, () => setReplanAssets(null));
    const c6 = bindOutsideClickDismiss(policyDialogRef.current, () => {});
    return () => {
      c1();
      c2();
      c3();
      c4();
      c5();
      c6();
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
      if (data.job?.error) setErrorMsg(data.job.error);
      if (data.manifest.remix.items.length > 0) {
        setSelectedId((prev) => prev || "REMIX");
      } else if (data.manifest.clips.length > 0) {
        setSelectedId((prev) => prev || data.manifest.clips[0].clip_id);
      }
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

  // Poll while background job is running
  useEffect(() => {
    if (!manifest || !jobRunning) return;
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
        if (!data.job?.running) {
          setJobRunning(false);
          setJobCancelling(false);
          setJobStages([]);
          if (data.job?.error) setErrorMsg(data.job.error);
          setSelectedId((prev) => {
            if (prev === "REMIX" || prev === "COMPILATION" || prev === "CLEAN_CUT") {
              return prev;
            }
            if (prev && data.manifest.clips.some((c) => c.clip_id === prev)) {
              return prev;
            }
            if (data.manifest.remix.items.length > 0) return "REMIX";
            return data.manifest.clips[0]?.clip_id || null;
          });
          api.listProjects().then((r) => setProjects(r.projects)).catch(() => {});
        }
      } catch {
        // ignore transient errors
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [manifest?.project_id, jobRunning, stageFilter, showLogs]);

  useEffect(() => {
    if (manifest && selectedId === "CLEAN_CUT" && (manifest.cut_segments || []).length === 0) {
      setSelectedId(manifest.remix.items.length > 0 ? "REMIX" : manifest.clips[0]?.clip_id || null);
    }
  }, [manifest?.project_id, selectedId, manifest?.cut_segments?.length]);

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
    if (activeClip && playMode === "source" && videoRef.current) {
      videoRef.current.currentTime = activeClip.start_s;
    }
  }, [selectedId, activeClip?.clip_id]);

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
      setSelectedId(res.manifest.clips[0]?.clip_id || "REMIX");
      setPlayMode("source");
      api.listProjects().then((r) => setProjects(r.projects)).catch(() => {});
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handleRun(stages: string[], forceRender = false) {
    if (!manifest) return;
    setErrorMsg(null);
    try {
      await api.runStages(manifest.project_id, stages, forceRender);
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

  async function handleSaveCoverTitleAndRefresh() {
    if (!manifest || !selectedId) return;
    setErrorMsg(null);
    setCoverRefreshing(true);
    try {
      const nextText = coverTitleDraft.trim();
      if (selectedId === "REMIX") {
        const res = await api.patchRemix(manifest.project_id, { cover_title: nextText });
        if (!res.manifest.remix.cover_file) {
          const r2 = await api.refreshCover(manifest.project_id, "remix");
          setManifest(r2.manifest);
        } else {
          setManifest(res.manifest);
        }
      } else if (selectedId === "COMPILATION") {
        const res = await api.patchCompilation(manifest.project_id, { cover_title: nextText });
        if (!res.manifest.compilation?.cover_file) {
          const r2 = await api.refreshCover(manifest.project_id, "compilation");
          setManifest(r2.manifest);
        } else {
          setManifest(res.manifest);
        }
      } else if (selectedId === "CLEAN_CUT") {
        const res = await api.patchCleanCut(manifest.project_id, { cover_title: nextText });
        if (!res.manifest.clean_cut?.cover_file) {
          const r2 = await api.refreshCover(manifest.project_id, "clean_cut");
          setManifest(r2.manifest);
        } else {
          setManifest(res.manifest);
        }
      } else if (activeClip) {
        const res = await api.patchClip(manifest.project_id, activeClip.clip_id, { cover_title: nextText });
        if (!res.manifest.clips.find((c) => c.clip_id === activeClip.clip_id)?.cover_file) {
          const r2 = await api.refreshCover(manifest.project_id, activeClip.clip_id);
          setManifest(r2.manifest);
        } else {
          setManifest(res.manifest);
        }
      }
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

  async function handleUndo() {
    if (!manifest) return;
    setErrorMsg(null);
    try {
      const res = await api.undo(manifest.project_id);
      setManifest(res.manifest);
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
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

  async function handleChangeGlobalPoliticalAction(nextAction: "auto" | "delete" | "mask") {
    setErrorMsg(null);
    try {
      const saved = await api.saveContentFilterConfig({ default_political_action: nextAction });
      setContentFilterCfg({
        default_sensitivity: saved.default_sensitivity,
        default_political_action: saved.default_political_action,
        political_profiles: saved.political_profiles,
      });
      if (manifest) {
        await handlePatchCleanCut({ political_action: nextAction });
      }
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

  async function openPublishModal() {
    if (!manifest || !selectedId) return;
    setErrorMsg(null);
    api
      .getPublishStatus()
      .then((res) => setPlatStatus(res.platforms))
      .catch(() => {});

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
      const bullets = rmx.items
        .filter((x) => x.enabled && x.card_title)
        .slice(0, 4)
        .map((x) => x.card_title)
        .join(" | ");
      setPubDesc(
        [manifest.metadata.guest_zh, manifest.metadata.channel_zh, bullets].filter(Boolean).join(" · ")
      );
      setPubTags([...(manifest.primary_topics || []), "AI", "深度访谈"].slice(0, 6).join(", "));
    } else if (selectedId === "COMPILATION") {
      const cmp = manifest.compilation;
      const baseTitle = (titleDraft.trim() || cmp?.title || "切片精选集").trim();
      const baseCover = (coverTitleDraft.trim() || cmp?.cover_title || baseTitle).trim();
      setPubTitle(baseTitle);
      setPubCoverTitle(baseCover);
      const summary = compilationClips
        .slice(0, 4)
        .map((c) => c.title)
        .join(" | ");
      setPubDesc(
        `精选 ${compilationClips.length} 支独立切片：\n` + summary
      );
      setPubTags([...(manifest.primary_topics || []), "AI", "深度访谈", "合集"].slice(0, 6).join(", "));
    } else if (selectedId === "CLEAN_CUT") {
      const cc = manifest.clean_cut;
      const fallbackT = `${manifest.metadata.original_theme_zh || manifest.metadata.original_title || "全片"}（净版）`;
      const baseTitle = (titleDraft.trim() || cc?.title || fallbackT).trim();
      const baseCover = (coverTitleDraft.trim() || cc?.cover_title || baseTitle).trim();
      setPubTitle(baseTitle);
      setPubCoverTitle(baseCover);
      setPubDesc(
        [manifest.metadata.guest_zh, manifest.metadata.channel_zh, cleanCutStats.summaryLabel].filter(Boolean).join(" · ")
      );
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
    const defaultCol = (manifest.metadata.original_theme_zh || manifest.metadata.original_title || "")
      .trim()
      .slice(0, 10);
    setPubCollection(defaultCol);
    setPubPublic(false);
    publishDialogRef.current?.showModal();
  }

  async function handleLaunchLogin(plat: string) {
    setErrorMsg(null);
    try {
      const res = await api.launchPublishLogin(plat);
      window.alert(res.message);
      api.getPublishStatus().then((r) => setPlatStatus(r.platforms)).catch(() => {});
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
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
    try {
      const tagsList = pubTags
        .split(/[,，、]/)
        .map((x) => x.trim())
        .filter(Boolean);
      await api.publishTarget(manifest.project_id, {
        target:
          selectedId === "REMIX"
            ? "remix"
            : selectedId === "COMPILATION"
            ? "compilation"
            : selectedId === "CLEAN_CUT"
            ? "clean_cut"
            : selectedId,
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
      setJobCancelling(false);
      setJobStages(["publish"]);
      setJobRunning(true);
      api
        .getProject(manifest.project_id)
        .then((d) => setManifest(d.manifest))
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
    const sens = (cc?.sensitivity || contentFilterCfg?.default_sensitivity || "strict") as
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
    const maskActive = active.filter((s) => resolveSegActionFrontend(s, cc, canMask) === "mask");
    const delPolCount = Math.max(0, polCount - maskActive.length);
    const spans = delActive
      .map((s) => [Math.max(0, s.start_s), Math.max(0, s.end_s)] as [number, number])
      .filter(([s, e]) => e > s);
    const merged = mergeSpanPairs(spans, 0.15);
    const cutS = merged.reduce((acc, [s, e]) => acc + (e - s), 0);
    const totalS = manifest?.source.duration_s || 0;
    const keptS = Math.max(0, totalS - cutS);
    const politicalAction = (cc?.political_action || contentFilterCfg?.default_political_action || "auto") as
      | "auto"
      | "delete"
      | "mask";
    const summaryLabel =
      !isScanned && segs.length === 0
        ? "待分析"
        : active.length > 0
          ? `广告${adCount}/${adTotal} · 删敏${delPolCount} · 遮蔽${maskActive.length}`
          : polAllTotal > 0
            ? `广告${adCount}/${adTotal} · 当前标准放行${polAllTotal}段`
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

  const compilationTotalS = useMemo(() => {
    if (!manifest) return 0;
    const cardDur = manifest.compilation?.card_duration_s ?? 1.8;
    const baseSpeechS =
      cleanCutStats.isScanned
        ? compilationCompliance.netDurS
        : compilationClips.reduce((acc, c) => acc + (c.end_s - c.start_s), 0);
    return baseSpeechS + compilationClips.length * cardDur;
  }, [manifest, compilationClips, cleanCutStats.isScanned, compilationCompliance.netDurS]);

  const enabledRemixItemCount = useMemo(() => {
    return manifest ? manifest.remix.items.filter((x) => x.enabled).length : 0;
  }, [manifest]);

  const hasActiveRemixTarget = Boolean(
    manifest?.remix.enabled &&
      (enabledRemixItemCount > 0 || (manifest.remix.items.length === 0 && (manifest.clips.length > 0 || (manifest.remix_only_pool?.length || 0) > 0)))
  );
  const hasActiveCompilationTarget = Boolean(manifest?.compilation?.enabled && compilationClips.length > 0);
  const hasActiveCleanCutTarget = Boolean(manifest?.clean_cut?.enabled && cleanCutStats.activeCount > 0);

  const renderTargetSummary = useMemo(() => {
    if (!manifest) return "0 项";
    const parts: string[] = [];
    if (enabledClipCount > 0) parts.push(`${enabledClipCount}切片`);
    if (hasActiveRemixTarget) parts.push("AI混剪");
    if (hasActiveCompilationTarget) parts.push("精选集");
    if (hasActiveCleanCutTarget) parts.push("全片净版");
    return parts.join(" + ") || "0 项";
  }, [manifest, enabledClipCount, hasActiveRemixTarget, hasActiveCompilationTarget, hasActiveCleanCutTarget]);

  const isRendering = Boolean(jobRunning && (jobStages.includes("render") || manifest?.stages?.render?.status === "running"));
  const isPublishing = Boolean(jobRunning && jobStages.includes("publish"));
  const isPlanningRunning = Boolean(manifest && jobRunning && !isRendering && !isPublishing);
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
    return ["signals", "scan_cuts", "analyze", "verify_and_plan", "render"].some((k) => {
      const st = manifest.stages?.[k];
      return Boolean(st && (st.status !== "idle" || st.finished_at || st.message));
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
    const cardDur = manifest.remix.card_mode === "none" ? 0 : manifest.remix.card_duration_s || 1.4;
    const estTotalS = speechS + cardCount * cardDur;
    const srcDur = manifest.source.duration_s || 0;
    const ratioPct = srcDur > 0 ? ((estTotalS / srcDur) * 100).toFixed(1) : "0.0";
    return { themeCount: groups.size, segmentCount: enabled.length, estTotalS, ratioPct };
  }, [manifest]);

  const currentCoverUrl = useMemo(() => {
    if (!manifest) return null;
    if (selectedId === "REMIX" && manifest.remix.cover_file) {
      return `/api/projects/${manifest.project_id}/media/remix_cover?v=${encodeURIComponent(
        manifest.remix.cover_hash || manifest.remix.last_rendered_hash || "1"
      )}`;
    }
    if (selectedId === "COMPILATION" && manifest.compilation?.cover_file) {
      return `/api/projects/${manifest.project_id}/media/compilation_cover?v=${encodeURIComponent(
        manifest.compilation.cover_hash || manifest.compilation.last_rendered_hash || "1"
      )}`;
    }
    if (selectedId === "CLEAN_CUT" && manifest.clean_cut?.cover_file) {
      return `/api/projects/${manifest.project_id}/media/clean_cut_cover?v=${encodeURIComponent(
        manifest.clean_cut.cover_hash || manifest.clean_cut.last_rendered_hash || "1"
      )}`;
    }
    if (activeClip?.cover_file) {
      return `/api/projects/${manifest.project_id}/media/cover?clip_id=${encodeURIComponent(
        activeClip.clip_id
      )}&v=${encodeURIComponent(activeClip.cover_hash || activeClip.last_rendered_hash || "1")}`;
    }
    return null;
  }, [manifest, selectedId, activeClip]);

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
            disabled={isPlanningRunning ? jobCancelling : (!manifest || jobRunning)}
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
            disabled={!manifest || jobRunning}
            onClick={() => handleRun(["signals", "scan_cuts", "analyze", "verify_and_plan", "render"])}
            title="全自动无人值守：全流程跑完包含全量渲染"
          >
            全自动跑完
          </button>
          <button type="button" onClick={handleUndo} disabled={!manifest}>
            撤销
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
                策略:{cleanCutStats.politicalAction === "mask" ? "遮蔽" : cleanCutStats.politicalAction === "delete" ? "删段" : "智能"}
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
            <div style={{ display: "flex", gap: 10, alignItems: "center", color: "var(--text-2)" }}>
              <span>{manifest.metadata.channel_zh || manifest.metadata.channel}</span>
              <span className="mono">{fmtTime(manifest.source.duration_s)}</span>
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
                  onClick={() => handleBatchClips("top_n", 5)}
                  title="按【核心主题、行业争议、精彩交锋、硬核实操、综合王牌】5大策展维度精准精选"
                >
                  策划Top5
                </button>
                <button type="button" className="sm" onClick={() => handleBatchClips("enable_all")} title="全部启用">
                  全选
                </button>
                <button type="button" className="sm" onClick={() => handleBatchClips("disable_all")} title="全部停用">
                  全不选
                </button>
                <button
                  type="button"
                  className="sm"
                  onClick={handleResetClipOrder}
                  title="按原片时间轴顺序一键复原切片排序"
                >
                  复原时间轴
                </button>
              </div>
            </div>
            <div className="pane-body scroller">
              {/* 1. Remix Card */}
              {(() => {
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

              {/* 2. Compilation Card (Selected Clips Sequential with Title Cards) */}
              {(() => {
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

              {/* 3. Clean Cut Card (Ad & Political-Sensitive Removal; Hidden when nothing active) */}
              {cleanCutStats.activeCount > 0 && (
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
                      background: "rgba(16, 185, 129, 0.16)",
                      color: "#34d399",
                      borderColor: "rgba(16, 185, 129, 0.45)",
                    }}
                  >
                    全片净版
                  </span>
                  <div className="clip-title">
                    {manifest.clean_cut?.title ||
                      `${manifest.metadata.original_theme_zh || manifest.metadata.original_title || "全片"}（净版）`}
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
                  {hasStrategy
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
              <div className="video-stage">
                <video
                  ref={videoRef}
                  key={mediaSrc}
                  className="video-el"
                  src={mediaSrc}
                  controls
                  onTimeUpdate={handleTimeUpdate}
                />
              </div>
              <div className="player-controls">
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <span className="mono">{fmtTime(currentTime)}</span>
                  <button
                    type="button"
                    className={`sm ${playMode === "source" ? "primary" : ""}`}
                    onClick={() => setPlayMode("source")}
                  >
                    原片试看
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
                    onClick={() => setPlayMode("rendered")}
                  >
                    播成品
                  </button>
                </div>
                {activeClip && (
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 5, cursor: "pointer" }}>
                    <input type="checkbox" checked={loopClip} onChange={(e) => setLoopClip(e.target.checked)} />
                    <span>区间循环</span>
                  </label>
                )}
              </div>
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
                      <span style={{ color: "var(--text-2)", fontSize: 12, width: 56 }}>净版标题</span>
                      <input
                        type="text"
                        style={{ flex: 1, fontWeight: 600 }}
                        value={titleDraft}
                        onChange={(e) => setTitleDraft(e.target.value)}
                        onBlur={() =>
                          titleDraft.trim() &&
                          titleDraft !== (manifest.clean_cut?.title || "") &&
                          handlePatchCleanCut({ title: titleDraft.trim() })
                        }
                      />
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
                        disabled={jobRunning}
                        onClick={handleScanCleanCut}
                        title="AI 结合视频关键帧与全片字幕扫描广告与政治敏感内容"
                      >
                        {jobRunning ? "处理中..." : "AI 扫描广告与敏感段"}
                      </button>
                      <button
                        type="button"
                        className="sm"
                        disabled={jobRunning || cleanCutStats.activeCount === 0}
                        onClick={() => handleRenderCleanCut(true)}
                        title={
                          cleanCutStats.activeCount === 0
                            ? "当前标准/开关下无生效剪切或脱敏项，无需输出全片净版"
                            : "对勾选的广告/政治敏感片段执行删段或词级遮蔽，输出全片净版 MP4 与封面"
                        }
                      >
                        {manifest.clean_cut?.output_file ? "重渲全片净版" : "渲染全片净版"}
                      </button>
                      <button
                        type="button"
                        className="sm primary"
                        disabled={!manifest.clean_cut?.output_file || jobRunning}
                        onClick={openPublishModal}
                      >
                        发布到 视频号 / B站 / 小红书…
                      </button>
                      {renderPubBadges(manifest.clean_cut?.publish_records)}
                    </div>
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
                    共 {cleanCutStats.total} 段 · 生效 {cleanCutStats.activeCount} 段（去广 {cleanCutStats.adCount} / 删敏 {cleanCutStats.delPolCount} / 遮蔽 {cleanCutStats.maskCount}
                    {cleanCutStats.polAllTotal > cleanCutStats.polTotal
                      ? ` · 已按当前标准放行 ${cleanCutStats.polAllTotal - cleanCutStats.polTotal} 段`
                      : ""}
                    {" · "}净删 {fmtTime(cleanCutStats.cutS)} · 净版约 {fmtTime(cleanCutStats.keptS)}）
                  </span>
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

                {/* Cut Segments Review List */}
                <div className="subtitle-list scroller">
                  {(manifest.cut_segments || []).length === 0 ? (
                    <div className="empty-state" style={{ padding: 28 }}>
                      尚未扫描广告与敏感片段。点击上方「AI 扫描广告与敏感段」自动标记，或点击「在当前指针添加屏蔽段」手动标记。
                    </div>
                  ) : (
                    (manifest.cut_segments || []).map((seg: CutSegment) => {
                      const categoryActive =
                        seg.kind === "ad"
                          ? (manifest.clean_cut?.cut_ads ?? true)
                          : (manifest.clean_cut?.cut_political_sensitive ?? true);
                      const matchesProfile = cleanCutStats.matchesSeg(seg);
                      const isEffective = seg.enabled && categoryActive && matchesProfile;
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
                                ) : (
                                  <select
                                    aria-label="单段脱敏方式"
                                    value={seg.action || "inherit"}
                                    onChange={(e) =>
                                      handlePatchCleanCut({
                                        segments: [{ cut_id: seg.cut_id, action: e.target.value }],
                                      })
                                    }
                                    style={{ padding: "2px 5px", fontSize: 11 }}
                                    title={`覆盖本段处理方式（AI建议：${seg.recommended_action === "mask" ? "遮蔽" : "删段"}）`}
                                  >
                                    <option value="inherit">
                                      跟随全局（{resolveSegActionFrontend(seg, manifest.clean_cut, canMask) === "mask" ? "遮蔽" : "删段"}）
                                    </option>
                                    <option value="delete">删段</option>
                                    <option value="mask" disabled={!canMask}>
                                      遮蔽{!canMask ? "（无干净原片）" : ""}
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
                                <span className="tag">{seg.detector === "visual" ? "画面" : "手动"}</span>
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
                        className="sm"
                        disabled={jobRunning || (manifest.remix.items.length > 0 && enabledRemixItemCount === 0) || (manifest.remix.items.length === 0 && manifest.clips.length === 0 && (manifest.remix_only_pool?.length || 0) === 0)}
                        onClick={() => handleRenderRemix(true)}
                        title="仅单独渲染当前 AI 混剪及封面（严格按已勾选段落出片）"
                      >
                        {isRendering ? "渲染中..." : manifest.remix.output_file ? "重渲混剪" : "渲染混剪"}
                      </button>
                      <button
                        type="button"
                        className="sm"
                        disabled={jobRunning || (manifest.clips.length === 0 && (manifest.remix_only_pool?.length || 0) === 0)}
                        onClick={() => handleRun(["replan_remix"])}
                        title="按上方设定的素材范围由 AI 重新规划混剪脚本（只重排脚本不渲染）"
                      >
                        AI 重新规划
                      </button>
                      <button
                        type="button"
                        className="sm primary"
                        disabled={!manifest.remix.output_file || jobRunning}
                        onClick={openPublishModal}
                      >
                        发布到 视频号 / B站 / 小红书…
                      </button>
                      {renderPubBadges(manifest.remix.publish_records)}
                    </div>
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

                {/* Visible Remix Source & Manual Control Bar */}
                {(() => {
                  const explicitSrc = Array.isArray(manifest.remix.source_clip_ids)
                    ? manifest.remix.source_clip_ids
                    : null;
                  const srcMode =
                    explicitSrc !== null
                      ? explicitSrc.length > 0
                        ? "custom"
                        : "auto"
                      : enabledClipCount > 0
                        ? "custom"
                        : "auto";
                  const activeSrcSet = new Set(
                    explicitSrc !== null
                      ? explicitSrc
                      : manifest.clips.filter((c) => c.enabled).map((c) => c.clip_id)
                  );
                  const remainingToAdd = manifest.clips.filter(
                    (c) => !manifest.remix.items.some((it) => it.source_clip_id === c.clip_id)
                  );
                  return (
                    <div
                      className="clean-cut-toolbar"
                      style={{ flexDirection: "column", alignItems: "stretch", gap: 6 }}
                    >
                      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                        <span style={{ fontWeight: 600, color: "var(--text-1)" }}>脚本来源：</span>
                        <select
                          aria-label="混剪素材范围"
                          disabled={jobRunning}
                          value={srcMode}
                          onChange={(e) => {
                            if (e.target.value === "auto") {
                              handlePatchRemix({ source_clip_ids: [] });
                            } else {
                              const fallback = manifest.clips
                                .filter((c) => c.enabled)
                                .map((c) => c.clip_id);
                              const next =
                                fallback.length > 0
                                  ? fallback
                                  : manifest.clips.slice(0, 1).map((c) => c.clip_id);
                              handlePatchRemix({ source_clip_ids: next });
                            }
                          }}
                          style={{ padding: "2px 6px", fontSize: 11 }}
                        >
                          <option value="auto">全片 AI 自主精选</option>
                          <option value="custom">限定参与重排的切片范围</option>
                        </select>
                        {manifest.remix.manual_script_locked ? (
                          <span
                            className="tag cached"
                            title="已手动增删/勾选/排序段落，渲染时严格按下方已勾选段落出片，不会被左侧独立切片改动覆盖"
                          >
                            人工已锁定·按勾选段落出片 ({enabledRemixItemCount}/{manifest.remix.items.length})
                          </span>
                        ) : (
                          <span className="tag">
                            AI 策展脚本 ({enabledRemixItemCount}/{manifest.remix.items.length})
                          </span>
                        )}
                        <div style={{ display: "flex", gap: 4, marginLeft: "auto", flexWrap: "wrap" }}>
                          <button
                            type="button"
                            className="sm"
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
                            style={{ padding: "2px 6px", fontSize: 11, maxWidth: 165 }}
                          >
                            <option value="">+ 手动加入切片…</option>
                            {(remainingToAdd.length > 0 ? remainingToAdd : manifest.clips).map((c) => (
                              <option key={c.clip_id} value={c.clip_id}>
                                #{String(c.index).padStart(2, "0")} {c.title}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                      {srcMode === "custom" && manifest.clips.length > 0 && (
                        <div style={{ display: "flex", gap: 5, alignItems: "center", flexWrap: "wrap", fontSize: 11 }}>
                          <span style={{ color: "var(--text-2)" }}>参与重排切片：</span>
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
                                source_clip_ids: manifest.clips.filter((c) => c.enabled).map((c) => c.clip_id),
                              })
                            }
                          >
                            同步左侧勾选
                          </button>
                        </div>
                      )}
                    </div>
                  );
                })()}

                {/* Collapsed Remix Advanced Options */}
                <details className="fold">
                  <summary>
                    ▸ 画面卡片与音频选项（{remixStats.themeCount} 板块 / {remixStats.segmentCount} 段 · 预估 {fmtTime(remixStats.estTotalS)}）
                  </summary>
                  <div className="fold-body" style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center" }}>
                    <label>
                      标题卡：
                      <select
                        value={manifest.remix.card_mode}
                        onChange={(e) => handlePatchRemix({ card_mode: e.target.value })}
                        style={{ marginLeft: 4 }}
                      >
                        <option value="group">按主题组</option>
                        <option value="every">每段前</option>
                        <option value="none">不插卡</option>
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
                        className="sm primary"
                        disabled={!manifest.compilation?.output_file || jobRunning}
                        onClick={openPublishModal}
                      >
                        发布到 视频号 / B站 / 小红书…
                      </button>
                      {renderPubBadges(manifest.compilation?.publish_records)}
                    </div>
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
                    <label style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                      章节片头卡：
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
                        const cardSec = (manifest.compilation?.card_duration_s ?? 1.8).toFixed(1);
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
                                    片头卡 Part {partNum} ({cardSec}s)
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
                        className="sm primary"
                        disabled={!activeClip.output_file || jobRunning}
                        onClick={openPublishModal}
                      >
                        发布到 视频号 / B站 / 小红书…
                      </button>
                      {renderPubBadges(activeClip.publish_records)}
                    </div>
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
                        const termTxt = [...(seg.sensitive_terms_zh || []), ...(seg.sensitive_terms_en || [])].join(" / ");
                        return (
                          <div key={seg.cut_id}>
                            • [{fmtTime(seg.start_s)}–{fmtTime(seg.end_s)}] {reasonLabel}
                            {effAction === "mask" && termTxt ? `（词表：${termTxt}）` : ""} — {seg.reason}
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
                      const maskHits = activeHits.filter(
                        (seg) => resolveSegActionFrontend(seg, manifest.clean_cut, canMask) === "mask"
                      );
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
                                  将遮蔽:{" "}
                                  {Array.from(
                                    new Set(
                                      maskHits.flatMap((s) => [
                                        ...(s.sensitive_terms_zh || []),
                                        ...(s.sensitive_terms_en || []),
                                      ])
                                    )
                                  ).join(" / ") || "整句"}
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
              <div className="empty-state">在左侧选择一个切片或 Remix 进行预览与发布。</div>
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
                  脱敏策略（{manifest ? "作用于当前项目并存为默认" : "新建项目默认"}）
                </span>
                <select
                  aria-label="设置中的脱敏策略"
                  disabled={jobRunning}
                  value={manifest ? cleanCutStats.politicalAction : contentFilterCfg?.default_political_action || "auto"}
                  onChange={(e) =>
                    handleChangeGlobalPoliticalAction(e.target.value as "auto" | "delete" | "mask")
                  }
                >
                  <option value="auto">
                    智能（顺带打码+哔音，整段删剪）{manifest && !canMask ? " [无干净原片]" : ""}
                  </option>
                  <option value="delete">全删片段</option>
                  <option value="mask" disabled={Boolean(manifest && !canMask)}>
                    敏感词遮蔽+哔音{manifest && !canMask ? "（无干净原片）" : ""}
                  </option>
                </select>
              </label>
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
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, alignItems: "end" }}>
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
              <div style={{ fontSize: 11, color: "var(--text-2)", lineHeight: 1.5 }}>
                {manifest
                  ? canMask
                    ? "当前项目存在无字幕干净原片，可使用词级遮蔽+880Hz哔音并重烧双语字幕。"
                    : "当前项目缺少无字幕干净原片，遮蔽选项已禁用并自动回退删段。"
                  : "载入项目后可在上方状态栏切换本片去广/脱敏开关与政审标准，在此统一配置脱敏方式。"}
              </div>
            </div>
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
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {(["wechat", "bilibili", "xiaohongshu"] as const).map((plat) => {
              const st = platStatus[plat];
              const logged = Boolean(st?.logged_in);
              return (
                <div
                  key={plat}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "8px 10px",
                    borderRadius: 8,
                    border: "1px solid var(--border)",
                    background: "var(--bg-2)",
                  }}
                >
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
                    <input
                      type="checkbox"
                      checked={Boolean(pubPlatforms[plat])}
                      onChange={(e) => setPubPlatforms((prev) => ({ ...prev, [plat]: e.target.checked }))}
                    />
                    <span style={{ fontWeight: 600 }}>{PLAT_LABELS[plat]}</span>
                    <span className={`tag ${logged ? "cached" : "dirty"}`}>{logged ? "已登录" : "未登录"}</span>
                  </label>
                  <button type="button" className="sm" onClick={() => handleLaunchLogin(plat)}>
                    {logged ? "重扫码" : "扫码登录"}
                  </button>
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
            <span style={{ fontSize: 12, color: "var(--text-2)" }}>简介文案</span>
            <textarea rows={3} value={pubDesc} onChange={(e) => setPubDesc(e.target.value)} />
          </label>

          <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 12, color: "var(--text-2)" }}>标签（逗号分隔）</span>
            <input type="text" value={pubTags} onChange={(e) => setPubTags(e.target.value)} />
          </label>

          <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 12, color: "var(--text-2)" }}>视频号合集（上传切片自动创建或存入同名合集，微信限10字）</span>
            <input
              type="text"
              maxLength={10}
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
              当前尚无已渲染成片文件，但检测到已有分析/扫描记录。继续将彻底清空旧的广告敏感识别、话题地图、切片列表与阶段状态，按最新规则从头重新规划（支持事后点「撤销」恢复旧状态）。
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
    </div>
  );
}
