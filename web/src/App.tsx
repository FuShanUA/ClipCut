import { type CSSProperties, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import type {
  ClipItem,
  PlatformStatus,
  ProjectManifest,
  ProjectSummary,
  PublishRecord,
  Receipt,
  RemixItem,
  ScanResult,
  VideoFileCandidate,
} from "./types";

const STAGE_SHORT_LABELS: Record<string, string> = {
  ingest: "载入",
  signals: "信号",
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
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [singleRendering, setSingleRendering] = useState(false);

  // Clean 2-column by default; right logs drawer hidden unless user opens it
  const [showLogs, setShowLogs] = useState(false);

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
  }, []);

  useEffect(() => {
    const c1 = bindOutsideClickDismiss(pickerDialogRef.current, () => setScanCandidates(null));
    const c2 = bindOutsideClickDismiss(settingsDialogRef.current, () => {});
    const c3 = bindOutsideClickDismiss(configDialogRef.current, () => {});
    const c4 = bindOutsideClickDismiss(publishDialogRef.current, () => {});
    return () => {
      c1();
      c2();
      c3();
      c4();
    };
  }, []);

  async function loadProject(projectId: string) {
    try {
      const data = await api.getProject(projectId);
      setManifest(data.manifest);
      setJobRunning(Boolean(data.job?.running));
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
        if (showLogs) refreshReceipts(pid, stageFilter);
        if (!data.job?.running) {
          setJobRunning(false);
          if (data.job?.error) setErrorMsg(data.job.error);
          api.listProjects().then((r) => setProjects(r.projects)).catch(() => {});
        }
      } catch {
        // ignore transient errors
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [manifest?.project_id, jobRunning, stageFilter, showLogs]);

  useEffect(() => {
    if (manifest && showLogs) {
      refreshReceipts(manifest.project_id, stageFilter);
    }
  }, [stageFilter, showLogs]);

  const activeClip: ClipItem | null = useMemo(() => {
    if (!manifest || !selectedId || selectedId === "REMIX" || selectedId === "COMPILATION") return null;
    return manifest.clips.find((c) => c.clip_id === selectedId) || null;
  }, [manifest, selectedId]);

  useEffect(() => {
    if (selectedId === "REMIX" && manifest) {
      setTitleDraft(manifest.remix.title || "");
      setCoverTitleDraft(manifest.remix.cover_title || "");
    } else if (selectedId === "COMPILATION" && manifest) {
      setTitleDraft(manifest.compilation?.title || "切片精选集");
      setCoverTitleDraft(manifest.compilation?.cover_title || "");
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
  ]);

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
      setJobRunning(true);
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function handlePatchClip(
    clipId: string,
    patch: { enabled?: boolean; title?: string; cover_title?: string; start_utt?: number; end_utt?: number }
  ) {
    if (!manifest) return;
    setErrorMsg(null);
    try {
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

  async function handleRenderCompilation() {
    if (!manifest) return;
    setErrorMsg(null);
    setSingleRendering(true);
    try {
      const res = await api.renderCompilation(manifest.project_id, true);
      setManifest(res.manifest);
      setPlayMode("rendered");
    } catch (err) {
      setErrorMsg((err as Error).message);
    } finally {
      setSingleRendering(false);
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
      const enabledClips = manifest.clips.filter((c) => c.enabled);
      const summary = enabledClips
        .slice(0, 4)
        .map((c) => c.title)
        .join(" | ");
      setPubDesc(
        `精选 ${enabledClips.length} 支独立切片：\n` + summary
      );
      setPubTags([...(manifest.primary_topics || []), "AI", "深度访谈", "合集"].slice(0, 6).join(", "));
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
        target: selectedId === "REMIX" ? "remix" : selectedId === "COMPILATION" ? "compilation" : selectedId,
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
    return n;
  }, [manifest]);

  const enabledClipCount = useMemo(() => {
    return manifest ? manifest.clips.filter((c) => c.enabled).length : 0;
  }, [manifest]);

  const compilationTotalS = useMemo(() => {
    if (!manifest) return 0;
    const enabledClips = manifest.clips.filter((c) => c.enabled);
    const cardDur = manifest.compilation?.card_duration_s ?? 1.8;
    return enabledClips.reduce((acc, c) => acc + (c.end_s - c.start_s), 0) + enabledClips.length * cardDur;
  }, [manifest]);

  const renderTargetSummary = useMemo(() => {
    if (!manifest) return "0 项";
    const parts: string[] = [];
    if (enabledClipCount > 0) parts.push(`${enabledClipCount}切片`);
    if (manifest.remix.enabled) parts.push("AI混剪");
    if (manifest.compilation?.enabled) parts.push("精选集");
    return parts.join(" + ") || "0 项";
  }, [manifest, enabledClipCount]);

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
            className="primary"
            disabled={!manifest || jobRunning}
            onClick={() => handleRun(["signals", "analyze", "verify_and_plan"])}
            title="第一步：AI声画测绘、三类选片与Remix规划（不消耗重型算力渲染）"
          >
            {jobRunning ? "规划中..." : "生成切片策略"}
          </button>
          <button
            type="button"
            disabled={!manifest || jobRunning || (enabledClipCount === 0 && !manifest.remix.enabled && !manifest.compilation?.enabled)}
            onClick={() => handleRun(["render"])}
            title="第二步：仅对当前勾选的切片、AI混剪与切片精选集执行物理渲染"
          >
            渲染选中 ({renderTargetSummary})
          </button>
          <button
            type="button"
            className="sm"
            disabled={!manifest || jobRunning}
            onClick={() => handleRun(["signals", "analyze", "verify_and_plan", "render"])}
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
        <main className={`workbench ${showLogs ? "" : "no-right"}`}>
          {/* LEFT PANE: Compact Clips & Remix List */}
          <section className="pane">
            <div className="pane-header" style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: "stretch" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span>切片 ({enabledClipCount}/{manifest.clips.length})</span>
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
                  <span>{remixStats.themeCount}板块 · {remixStats.segmentCount}段</span>
                  <span className="mono">{fmtTime(remixStats.estTotalS)}</span>
                  <span
                    className={`tag ${
                      manifest.remix.output_file && manifest.remix.render_hash === manifest.remix.last_rendered_hash
                        ? "cached"
                        : "dirty"
                    }`}
                  >
                    {manifest.remix.output_file && manifest.remix.render_hash === manifest.remix.last_rendered_hash
                      ? "已渲染"
                      : "待渲染"}
                  </span>
                  {renderPubBadges(manifest.remix.publish_records)}
                </div>
              </div>

              {/* 2. Compilation Card (Selected Clips Sequential with Title Cards) */}
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
                  <span>顺序串联 {enabledClipCount} 个切片</span>
                  <span className="mono">{fmtTime(compilationTotalS)}</span>
                  <span
                    className={`tag ${
                      manifest.compilation?.output_file && manifest.compilation.render_hash === manifest.compilation.last_rendered_hash
                        ? "cached"
                        : "dirty"
                    }`}
                  >
                    {manifest.compilation?.output_file && manifest.compilation.render_hash === manifest.compilation.last_rendered_hash
                      ? "已渲染"
                      : "待渲染"}
                  </span>
                  {renderPubBadges(manifest.compilation?.publish_records)}
                </div>
              </div>

              {/* Standalone Clips */}
              {manifest.clips.length === 0 ? (
                <div className="empty-state" style={{ padding: 20 }}>
                  点击上方「一键出片」自动生成切片与封面。
                </div>
              ) : (
                manifest.clips.map((c, idx) => {
                  const isCached = Boolean(c.output_file && c.render_hash === c.last_rendered_hash);
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
                        <span className={`tag ${isCached ? "cached" : "dirty"}`}>
                          {isCached ? "已渲染" : "待切"}
                        </span>
                        {renderPubBadges(c.publish_records)}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </section>

          {/* CENTER PANE: Player + Cover Title & Publish Card + Collapsible Controls */}
          <section className="pane center-pane">
            <div className="player-wrap">
              <video
                ref={videoRef}
                key={mediaSrc}
                className="video-el"
                src={mediaSrc}
                controls
                onTimeUpdate={handleTimeUpdate}
              />
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

            {selectedId === "REMIX" ? (
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

                {/* Collapsed Remix Advanced Options */}
                <details className="fold">
                  <summary>
                    ▸ 高级编排选项（{remixStats.themeCount} 板块 / {remixStats.segmentCount} 段 · 预估 {fmtTime(remixStats.estTotalS)}）
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
                    <button
                      type="button"
                      className="sm"
                      disabled={jobRunning}
                      onClick={() => handleRun(["replan_remix"])}
                    >
                      重排脚本
                    </button>
                    <button
                      type="button"
                      className="sm primary"
                      disabled={jobRunning}
                      onClick={() => handleRun(["replan_remix", "render"])}
                    >
                      重排并渲染
                    </button>
                  </div>
                </details>

                <div className="subtitle-list scroller">
                  {[...manifest.remix.items]
                    .sort((a, b) => a.order - b.order)
                    .map((it: RemixItem, idx) => (
                      <div key={it.remix_id} className="remix-row" style={{ opacity: it.enabled ? 1 : 0.45 }}>
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
                        </div>
                      </div>
                    ))}
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
                        disabled={coverRefreshing || jobRunning || enabledClipCount === 0}
                        onClick={handleSaveCoverTitleAndRefresh}
                      >
                        {coverRefreshing ? "生成中..." : "更新封面"}
                      </button>
                    </div>
                    <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                      <button
                        type="button"
                        className="sm"
                        disabled={singleRendering || jobRunning || enabledClipCount === 0}
                        onClick={handleRenderCompilation}
                        title="按顺序串联所有勾选切片，并在每个切片前自动插入磨砂玻璃章节片头卡"
                      >
                        {singleRendering ? "渲染中..." : manifest.compilation?.output_file ? "重渲精选集" : "渲染精选集"}
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

                {/* Compilation Title Card Settings */}
                <details className="fold" open>
                  <summary>
                    ▸ 精选集串联与片头卡设置（顺序串联 {enabledClipCount} 个启用切片 · 预估总长 {fmtTime(compilationTotalS)}）
                  </summary>
                  <div className="fold-body" style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center" }}>
                    <label>
                      章节片头卡时长：
                      <select
                        value={String(manifest.compilation?.card_duration_s ?? 1.8)}
                        onChange={(e) => handlePatchCompilation({ card_duration_s: Number(e.target.value) })}
                        style={{ marginLeft: 4 }}
                      >
                        <option value="1.2">1.2s</option>
                        <option value="1.8">1.8s</option>
                        <option value="2.5">2.5s</option>
                      </select>
                    </label>
                    <span style={{ color: "var(--text-2)", fontSize: 12 }}>
                      每个启用切片前自动插入磨砂玻璃章节片头卡（Part 01, Part 02...），按时间顺序无缝串联。
                    </span>
                  </div>
                </details>

                {/* Compilation Sequential Clips List */}
                <div className="subtitle-list scroller">
                  {enabledClipCount === 0 ? (
                    <div className="empty-state" style={{ padding: 24 }}>
                      当前未勾选任何切片。请在左侧列表勾选切片，或点击「策划Top5」后返回此处渲染精选集。
                    </div>
                  ) : (
                    manifest.clips
                      .filter((c) => c.enabled)
                      .map((c, idx) => {
                        const partNum = String(idx + 1).padStart(2, "0");
                        const cardSec = (manifest.compilation?.card_duration_s ?? 1.8).toFixed(1);
                        const durSec = Math.max(0, c.end_s - c.start_s);
                        return (
                          <div
                            key={c.clip_id}
                            className="remix-row"
                            style={{ alignItems: "center", cursor: "pointer" }}
                            onClick={() => {
                              setPlayMode("source");
                              seekTo(c.start_s);
                            }}
                          >
                            <span
                              className="mono"
                              style={{
                                fontSize: 13,
                                fontWeight: 700,
                                color: "#60a5fa",
                                minWidth: 28,
                                textAlign: "center",
                              }}
                            >
                              {partNum}
                            </span>
                            <div style={{ minWidth: 0 }}>
                              <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
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
                                <span style={{ color: "var(--text-2)", fontSize: 12 }}>
                                  {c.cover_title || c.title}
                                </span>
                              </div>
                              <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 6, flexWrap: "wrap" }}>
                                <span style={{ fontWeight: 600 }}>
                                  #{String(c.index).padStart(2, "0")} {c.title}
                                </span>
                                <span className="utt-time">
                                  {fmtTime(c.start_s)}–{fmtTime(c.end_s)} ({fmtTime(durSec)})
                                </span>
                                {c.curation_tag && (
                                  <span
                                    className="tag highlight"
                                    style={{
                                      background: "rgba(245, 158, 11, 0.18)",
                                      color: "#f59e0b",
                                      borderColor: "rgba(245, 158, 11, 0.45)",
                                    }}
                                  >
                                    {c.curation_tag}
                                  </span>
                                )}
                              </div>
                            </div>
                            <div style={{ display: "flex", gap: 6 }}>
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
                              <button
                                type="button"
                                className="sm"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handlePatchClip(c.clip_id, { enabled: false });
                                }}
                                title="取消勾选（移出精选集）"
                              >
                                移出
                              </button>
                            </div>
                          </div>
                        );
                      })
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

                {/* Collapsed Clip Subtitle Trimming & Editorial Notes */}
                <details className="fold" style={{ flex: "1 1 auto", display: "flex", flexDirection: "column", minHeight: 0 }}>
                  <summary>▸ 微调字幕入出点与编审说明</summary>
                  <div className="fold-body scroller" style={{ flex: 1, overflowY: "auto", maxHeight: "42vh" }}>
                    {activeClip.reason && (
                      <div style={{ fontSize: 12, color: "var(--text-2)", marginBottom: 8 }}>
                        说明：{activeClip.reason}
                      </div>
                    )}
                    {contextUtterances.map((u) => {
                      const inRange = u.id >= activeClip.start_utt && u.id <= activeClip.end_utt;
                      const isPlaying =
                        playMode === "source" && currentTime >= u.start_s && currentTime <= u.end_s + 0.15;
                      return (
                        <div
                          key={u.id}
                          className={`utt-row ${inRange ? "in-range" : "out-range"} ${isPlaying ? "playing" : ""}`}
                        >
                          <span className="utt-time" onClick={() => seekTo(u.start_s)} title="点击跳转">
                            {fmtTime(u.start_s)}
                          </span>
                          <div>
                            <div className="utt-zh">{u.zh || u.en}</div>
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
              </>
            ) : (
              <div className="empty-state">在左侧选择一个切片或 Remix 进行预览与发布。</div>
            )}
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
                        <option value="analyze">analyze</option>
                        <option value="verify_and_plan">verify_and_plan</option>
                        <option value="render">render</option>
                        <option value="publish">publish</option>
                        <option value="human_edit">human_edit</option>
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

      {/* Settings Dialog (Manual path, models, analyze-only, YAML config) */}
      <dialog
        ref={settingsDialogRef}
        aria-labelledby="settings-title"
        {...({ closedby: "any" } as Record<string, string>)}
      >
        <div className="dialog-head">
          <h2 id="settings-title">高级设置与手动载入</h2>
          <form method="dialog">
            <button type="submit" className="sm">
              关闭
            </button>
          </form>
        </div>
        <div className="dialog-body scroller" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
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

          <div style={{ display: "flex", gap: 8, paddingTop: 6 }}>
            <button
              type="button"
              disabled={!manifest || jobRunning}
              onClick={() => {
                settingsDialogRef.current?.close();
                handleRun(["signals", "analyze", "verify_and_plan"]);
              }}
            >
              仅重新分析/复核（不渲染）
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
    </div>
  );
}
