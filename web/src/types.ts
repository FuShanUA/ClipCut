export interface VideoFileCandidate {
  path: string;
  filename: string;
  size_bytes: number;
  duration_s: number;
  width: number;
  height: number;
  fps: number;
  is_hardsub: boolean;
  suggested_subtitle: string | null;
  suggested_info_json: string | null;
}

export interface ScanResult {
  input_path: string;
  is_directory: boolean;
  need_selection: boolean;
  videos: VideoFileCandidate[];
  subtitles: string[];
  info_jsons: string[];
}

export interface StageStatus {
  status: "idle" | "running" | "completed" | "error";
  started_at: string | null;
  finished_at: string | null;
  elapsed_ms: number;
  message: string;
  error: string | null;
}

export interface Utterance {
  id: number;
  start_s: number;
  end_s: number;
  en: string;
  zh: string;
  raw: string;
  silence_before_ms: number;
  silence_after_ms: number;
  audio_rms_z: number;
  audio_peak_z: number;
  speech_rate_z: number;
  motion_z: number;
  rapid_turn: boolean;
  signal_tags: string[];
}

export interface TopicMapItem {
  topic_id: string;
  start_utt: number;
  end_utt: number;
  start_s: number;
  end_s: number;
  chapter_title: string;
  title: string;
  summary: string;
  punchline_utt: number | null;
  is_filler: boolean;
  filler_reason: string;
  selected_clip_id: string | null;
  dropped_reason: string;
}

export interface SuppressedHotTopic {
  topic_id: string | null;
  start_utt: number;
  end_utt: number;
  matched_keyword: string;
  primary_topic_conflict: string;
  reason: string;
}

export interface PublishRecord {
  platform: "bilibili" | "wechat" | "xiaohongshu";
  status: "running" | "published" | "draft" | "error";
  mode: "draft" | "public";
  title: string;
  url_or_id: string;
  started_at?: string;
  updated_at: string;
  progress_pct?: number | null;
  progress_stage?: string;
  message: string;
  error: string | null;
  error_code?: string;
  solution?: string;
}

export interface PublishJobInfo {
  job_id: string;
  kind: "single" | "batch";
  running: boolean;
  cancelling?: boolean;
  cancelled?: boolean;
  targets: string[];
  current_target: string | null;
  completed_targets: string[];
  failed_targets: string[];
  skipped_targets?: string[];
  platforms: string[];
  collection: string;
  error?: string | null;
}

export interface SubtitleRepairItem {
  utterance_id: number;
  zh?: string | null;
  en?: string | null;
}

export interface SubtitleRepairResponse {
  ok: boolean;
  changed_utterance_ids: number[];
  bi_srt_path: string;
  cn_srt_path: string;
  en_srt_path: string;
  ass_path: string;
  clip_output_path: string | null;
  full_hardsub_path: string | null;
  manifest: ProjectManifest;
}

export interface PlatformStatus {
  platform: string;
  label: string;
  logged_in: boolean;
  python_ready: boolean;
  script: string;
  hint: string;
  login_running?: boolean;
  login_returncode?: number | null;
  login_updated_at?: string;
  login_message?: string;
}

export interface ClipItem {
  clip_id: string;
  index: number;
  enabled: boolean;
  categories: string[];
  matched_hot_topics: string[];
  chapter_title: string;
  title: string;
  cover_title?: string;
  title_en: string;
  slug: string;
  hook: string;
  reason: string;
  summary: string;
  key_quote: string;
  notes: string[];
  start_utt: number;
  end_utt: number;
  start_s: number;
  end_s: number;
  snap_delta_start_ms: number;
  snap_delta_end_ms: number;
  multimodal_status: string;
  score?: number;
  curation_tag?: string;
  render_hash: string;
  last_rendered_hash: string;
  output_file: string | null;
  cover_file: string | null;
  cover_hash?: string;
  rendered_filter_tag?: string;
  publish_records?: PublishRecord[];
}

export interface RemixItem {
  remix_id: string;
  order: number;
  enabled: boolean;
  source_clip_id: string | null;
  theme_id?: string;
  beat_type?: string;
  role: string;
  group_label: string;
  card_title: string;
  macro_intent?: string;
  start_utt: number;
  end_utt: number;
  start_s: number;
  end_s: number;
  quote: string;
  reason: string;
}

export type RemixSourceMode = "compilation" | "manual_clips" | "prompt" | "ai_auto";

export interface RemixConfig {
  enabled: boolean;
  title: string;
  cover_title?: string;
  publish_desc?: string;
  editorial_thesis?: string;
  card_mode: "group" | "every" | "none";
  card_duration_s: number;
  card_fade_s: number;
  enable_sfx: boolean;
  enable_bgm: boolean;
  bgm_volume: number;
  sfx_volume: number;
  source_mode?: RemixSourceMode;
  custom_prompt?: string;
  source_clip_ids?: string[] | null;
  manual_script_locked?: boolean;
  items: RemixItem[];
  render_hash: string;
  last_rendered_hash: string;
  output_file: string | null;
  cover_file?: string | null;
  cover_hash?: string;
  rendered_filter_tag?: string;
  publish_records?: PublishRecord[];
}

export interface CompilationConfig {
  enabled: boolean;
  title: string;
  cover_title?: string;
  publish_desc?: string;
  clip_ids?: string[] | null;
  card_duration_s: number;
  card_fade_s: number;
  render_hash: string;
  last_rendered_hash: string;
  output_file: string | null;
  cover_file?: string | null;
  cover_hash?: string;
  rendered_filter_tag?: string;
  publish_records?: PublishRecord[];
}

export type PoliticalAction = "auto" | "delete" | "mask" | "precise_mask";

export interface VisualMaskBox {
  x: number;
  y: number;
  w: number;
  h: number;
  text?: string;
}

export interface CutSegment {
  cut_id: string;
  kind: "ad" | "political_sensitive";
  subtype?: string;
  enabled: boolean;
  detector: "llm" | "visual" | "visual_ocr" | "manual";
  action?: "inherit" | "delete" | "mask" | "precise_mask";
  recommended_action?: "delete" | "mask";
  sensitive_terms_zh?: string[];
  sensitive_terms_en?: string[];
  precise_terms_zh?: string[];
  precise_terms_en?: string[];
  precise_terms_extracted?: boolean;
  visual_boxes?: VisualMaskBox[];
  start_utt?: number | null;
  end_utt?: number | null;
  start_s: number;
  end_s: number;
  quote?: string;
  reason?: string;
  severity?: "high" | "medium" | "low";
  chapter_title?: string;
}

export interface CleanCutConfig {
  enabled: boolean;
  title: string;
  cover_title?: string;
  publish_desc?: string;
  cut_ads: boolean;
  cut_political_sensitive: boolean;
  sensitivity: "strict" | "balanced" | "loose";
  political_action?: PoliticalAction;
  apply_to_clips: boolean;
  cut_mode: "auto" | "precise" | "reencode";
  scanned_at?: string | null;
  render_hash: string;
  last_rendered_hash: string;
  output_file: string | null;
  cover_file?: string | null;
  cover_hash?: string;
  rendered_filter_tag?: string;
  publish_records?: PublishRecord[];
}

export interface ProjectManifest {
  project_id: string;
  revision: number;
  updated_at: string;
  source: {
    render_video: string;
    analysis_video: string;
    subtitle_path: string | null;
    output_dir: string;
    duration_s: number;
    width: number;
    height: number;
    fps: number;
  };
  metadata: {
    original_title: string;
    description: string;
    channel: string;
    guest_zh: string;
    channel_zh: string;
    original_theme_zh: string;
    full_summary_zh?: string;
    published_date?: string;
    metadata_source: string;
    chapters: { index: number; title: string; start_s: number; end_s: number }[];
    chapter_audits: { chapter_index: number; title: string; decision: string; reason: string }[];
  };
  stages: Record<string, StageStatus>;
  primary_topics: string[];
  utterances: Utterance[];
  topic_map: TopicMapItem[];
  suppressed_hot_topics: SuppressedHotTopic[];
  clips: ClipItem[];
  remix_only_pool?: unknown[];
  remix: RemixConfig;
  compilation?: CompilationConfig;
  cut_segments?: CutSegment[];
  clean_cut?: CleanCutConfig;
}

export interface Receipt {
  receipt_id: string;
  run_id: string;
  timestamp: string;
  stage: string;
  actor: "ai" | "system" | "human";
  action: string;
  model: string;
  prompt_tokens: number;
  output_tokens: number;
  elapsed_ms: number;
  summary: string;
  details: Record<string, unknown>;
  diff: Record<string, unknown>;
  error: string | null;
}

export interface ProjectSummary {
  project_id: string;
  title: string;
  video_path: string;
  duration_s: number;
  clip_count: number;
  updated_at: string;
}

export interface RenderedAssetsInfo {
  has_strategy: boolean;
  has_rendered: boolean;
  output_dir: string;
  video_files: string[];
  cover_files: string[];
  doc_files: string[];
  total_files: number;
  total_size_bytes: number;
  rendered_clips_count: number;
  remix_rendered: boolean;
  compilation_rendered: boolean;
  clean_cut_rendered: boolean;
}

export interface PoliticalRule {
  id: string;
  name: string;
  severity: "high" | "medium" | "low";
  description: string;
}

export interface PoliticalProfile {
  label: string;
  description: string;
  trigger_severities: ("high" | "medium" | "low")[];
  rules: PoliticalRule[];
}

export interface ContentFilterConfig {
  default_sensitivity: "strict" | "balanced" | "loose";
  default_political_action?: PoliticalAction;
  political_profiles: Record<"strict" | "balanced" | "loose", PoliticalProfile>;
}

export interface NavColorPreset {
  id: string;
  label: string;
  hex: string;
}

export type NavStyleMode = "top_nav" | "both" | "title_card";
export type ScreenLayoutMode = "portrait" | "landscape";

export interface NavigationConfig {
  style: NavStyleMode;
  remix_style: NavStyleMode;
  compilation_style: NavStyleMode;
  screen_layout: ScreenLayoutMode;
  color_scheme: string;
  custom_accent: string;
  presets?: NavColorPreset[];
}
