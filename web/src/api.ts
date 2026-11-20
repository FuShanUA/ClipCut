import type { PlatformStatus, ProjectManifest, ProjectSummary, Receipt, ScanResult } from "./types";

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
    req<{ manifest: ProjectManifest; job: { running: boolean; error?: string | null } }>(
      `/api/projects/${projectId}`
    ),

  runStages: (projectId: string, stages: string[], forceRender = false) =>
    req<{ started: boolean; job_id: string }>(`/api/projects/${projectId}/run`, {
      method: "POST",
      body: JSON.stringify({ stages, force_render: forceRender }),
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

  renderClip: (projectId: string, clipId: string, force = false) =>
    req<{ ok: boolean; output_file: string; manifest: ProjectManifest }>(
      `/api/projects/${projectId}/clips/${clipId}/render?force=${force}`,
      {
        method: "POST",
      }
    ),

  patchRemix: (projectId: string, patch: Record<string, unknown>) =>
    req<{ manifest: ProjectManifest }>(`/api/projects/${projectId}/remix`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),

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
};
