import type {
  ContentFilterConfig,
  PlatformStatus,
  ProjectManifest,
  ProjectSummary,
  Receipt,
  RenderedAssetsInfo,
  ScanResult,
  SubtitleRepairItem,
  SubtitleRepairResponse,
} from "./types";

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const data = await res.json();
      msg = data.detail || msg;
    } catch {
      // ignore
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  pickPath: (kind: "file" | "folder") =>
    req<{ cancelled: boolean; path: string | null }>("/api/pick-path", {
      method: "POST",
      body: JSON.stringify({ kind }),
    }),

  scan: (path: string) =>
    req<ScanResult>("/api/scan", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),

  listProjects: () => req<{ projects: ProjectSummary[] }>("/api/projects"),

  openProject: (payload: {
    video_path: string;
    subtitle_path?: string | null;
    info_json_path?: string | null;
    source_url?: string | null;
    manual_title?: string | null;
    manual_description?: string | null;
  }) =>
    req<{ project_id: string; manifest: ProjectManifest }>("/api/projects/open", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getProject: (projectId: string) =>
    req<{
      manifest: ProjectManifest;
      job: { running: boolean; cancelling?: boolean; cancelled?: boolean; stages?: string[]; error?: string | null };
      assets?: RenderedAssetsInfo;
      can_mask?: boolean;
    }>(`/api/projects/${projectId}`),

  runStages: (projectId: string, stages: string[], forceRender = false) =>
    req<{ started: boolean; job_id: string }>(`/api/projects/${projectId}/run`, {
      method: "POST",
      body: JSON.stringify({ stages, force_render: forceRender }),
    }),

  cancelJob: (projectId: string) =>
    req<{ cancelled: boolean; job_id?: string; message: string }>(`/api/projects/${projectId}/cancel`, {
      method: "POST",
    }),

  getRenderedAssets: (projectId: string) =>
    req<RenderedAssetsInfo>(`/api/projects/${projectId}/rendered-assets`),

  replanStrategy: (
    projectId: string,
    payload: { delete_rendered_files: boolean; force_signals?: boolean }
  ) =>
    req<{
      started: boolean;
      job_id: string;
      stages: string[];
      deleted_files: string[];
      deleted_count: number;
      freed_bytes: number;
      manifest: ProjectManifest;
    }>(`/api/projects/${projectId}/replan-strategy`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  patchClip: (
    projectId: string,
    clipId: string,
    patch: { enabled?: boolean; title?: string; cover_title?: string; start_utt?: number; end_utt?: number }
  ) =>
    req<{ manifest: ProjectManifest }>(`/api/projects/${projectId}/clips/${clipId}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),

  batchClips: (projectId: string, action: "enable_all" | "disable_all" | "top_n", n = 5) =>
    req<{ manifest: ProjectManifest; enabled_count: number }>(`/api/projects/${projectId}/clips/batch`, {
      method: "POST",
      body: JSON.stringify({ action, n }),
    }),

  reorderClips: (projectId: string, clipIds: string[]) =>
    req<{ manifest: ProjectManifest }>(`/api/projects/${projectId}/clips/reorder`, {
      method: "POST",
      body: JSON.stringify({ clip_ids: clipIds }),
    }),

  resetClipOrder: (projectId: string) =>
    req<{ manifest: ProjectManifest }>(`/api/projects/${projectId}/clips/reset-order`, {
      method: "POST",
    }),

  renderClip: (projectId: string, clipId: string, force = false) =>
    req<{ ok: boolean; output_file: string; manifest: ProjectManifest }>(
      `/api/projects/${projectId}/clips/${clipId}/render?force=${force}`,
      {
        method: "POST",
      }
    ),

  repairClipSubtitles: (
    projectId: string,
    clipId: string,
    payload: {
      repairs: SubtitleRepairItem[];
      rebuild_full_subtitle?: boolean;
      render_clip?: boolean;
      render_full_video?: boolean;
    }
  ) =>
    req<SubtitleRepairResponse>(`/api/projects/${projectId}/clips/${clipId}/subtitle-repair`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  patchRemix: (projectId: string, patch: Record<string, unknown>) =>
    req<{ manifest: ProjectManifest }>(`/api/projects/${projectId}/remix`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),

  renderRemix: (projectId: string, force = true) =>
    req<{ started: boolean; job_id: string }>(
      `/api/projects/${projectId}/remix/render?force=${force}`,
      {
        method: "POST",
      }
    ),

  patchCompilation: (projectId: string, patch: Record<string, unknown>) =>
    req<{ manifest: ProjectManifest }>(`/api/projects/${projectId}/compilation`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),

  renderCompilation: (projectId: string, force = true) =>
    req<{ started: boolean; job_id: string }>(
      `/api/projects/${projectId}/compilation/render?force=${force}`,
      {
        method: "POST",
      }
    ),

  scanCleanCut: (
    projectId: string,
    payload: { cut_ads?: boolean; cut_political_sensitive?: boolean; sensitivity?: string } = {}
  ) =>
    req<{ started: boolean; job_id: string }>(`/api/projects/${projectId}/clean-cut/scan`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  patchCleanCut: (projectId: string, patch: Record<string, unknown>) =>
    req<{ manifest: ProjectManifest }>(`/api/projects/${projectId}/clean-cut`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),

  renderCleanCut: (projectId: string, force = false) =>
    req<{ started: boolean; job_id: string }>(
      `/api/projects/${projectId}/clean-cut/render?force=${force}`,
      {
        method: "POST",
      }
    ),

  refreshCover: (projectId: string, target: string) =>
    req<{ refreshed: string; manifest: ProjectManifest }>(`/api/projects/${projectId}/covers/refresh`, {
      method: "POST",
      body: JSON.stringify({ target }),
    }),

  getPublishStatus: () =>
    req<{ platforms: Record<string, PlatformStatus> }>("/api/publish/status"),

  launchPublishLogin: (platform: string) =>
    req<{ started: boolean; platform: string; pid: number; message: string }>(`/api/publish/login/${platform}`, {
      method: "POST",
    }),

  publishTarget: (
    projectId: string,
    payload: {
      target: string;
      platforms: string[];
      title?: string;
      cover_title?: string;
      desc?: string;
      tags?: string[];
      is_public?: boolean;
      collection?: string;
      ui_schema?: number;
    }
  ) =>
    req<{ started: boolean; job_id: string; target: string; platforms: string[] }>(
      `/api/projects/${projectId}/publish`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      }
    ),

  openOutputFolder: (projectId: string) =>
    req<{ opened: string }>(`/api/projects/${projectId}/open-output`, {
      method: "POST",
    }),

  undo: (projectId: string) =>
    req<{ restored: string; manifest: ProjectManifest }>(`/api/projects/${projectId}/undo`, {
      method: "POST",
    }),

  getReceipts: (projectId: string, stage?: string) => {
    const q = stage ? `?stage=${encodeURIComponent(stage)}` : "";
    return req<{ receipts: Receipt[]; pipeline_log: string[] }>(`/api/projects/${projectId}/receipts${q}`);
  },

  getModels: () =>
    req<{
      catalog: {
        text: { provider: string; model: string; label: string }[];
        multimodal: { provider: string; model: string; label: string }[];
      };
      selected: {
        text_provider: string;
        text_model: string;
        remix_model: string;
        multimodal_provider: string;
        multimodal_model: string;
      };
    }>("/api/models"),

  saveModels: (payload: {
    text_provider?: string;
    text_model?: string;
    multimodal_provider?: string;
    multimodal_model?: string;
  }) =>
    req<{
      saved: boolean;
      selected: {
        text_provider: string;
        text_model: string;
        remix_model: string;
        multimodal_provider: string;
        multimodal_model: string;
      };
    }>("/api/models", {
      method: "PUT",
      body: JSON.stringify(payload),
    }),

  getEditorialYaml: () => req<{ raw_yaml: string }>("/api/config/editorial"),

  saveEditorialYaml: (rawYaml: string) =>
    req<{ saved: boolean; categories: number }>("/api/config/editorial", {
      method: "PUT",
      body: JSON.stringify({ raw_yaml: rawYaml }),
    }),

  getContentFilterConfig: () => req<ContentFilterConfig>("/api/config/content-filter"),

  saveContentFilterConfig: (payload: Partial<ContentFilterConfig>) =>
    req<{ saved: boolean } & ContentFilterConfig>("/api/config/content-filter", {
      method: "PUT",
      body: JSON.stringify(payload),
    }),

  resetContentFilterConfig: (profile?: "strict" | "balanced" | "loose") =>
    req<{ reset: boolean } & ContentFilterConfig>("/api/config/content-filter/reset", {
      method: "POST",
      body: JSON.stringify({ profile: profile ?? null }),
    }),
};
