from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VideoFileCandidate(BaseModel):
    path: str
    filename: str
    size_bytes: int
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    is_hardsub: bool = False
    suggested_subtitle: str | None = None
    suggested_info_json: str | None = None


class ScanResult(BaseModel):
    input_path: str
    is_directory: bool
    videos: list[VideoFileCandidate] = Field(default_factory=list)
    subtitles: list[str] = Field(default_factory=list)
    info_jsons: list[str] = Field(default_factory=list)


class Chapter(BaseModel):
    index: int
    title: str
    start_s: float
    end_s: float
    source: Literal["info_json", "description", "ai_inferred", "manual"] = "info_json"


class ChapterAudit(BaseModel):
    chapter_index: int
    title: str
    start_s: float
    end_s: float
    decision: Literal["keep", "skip"]
    reason: str


class VideoMetadata(BaseModel):
    original_title: str = ""
    description: str = ""
    channel: str = ""
    uploader: str = ""
    video_id: str | None = None
    webpage_url: str | None = None
    metadata_source: Literal["info_json", "yt_dlp", "manual", "filename_fallback"] = "filename_fallback"
    guest_zh: str = ""
    channel_zh: str = ""
    original_theme_zh: str = ""
    published_date: str = ""
    chapters: list[Chapter] = Field(default_factory=list)
    chapter_audits: list[ChapterAudit] = Field(default_factory=list)


class MediaSource(BaseModel):
    render_video: str
    analysis_video: str
    subtitle_path: str | None = None
    subtitle_format: str = "srt"
    subtitle_sha256: str = ""
    work_dir: str
    output_dir: str
    duration_s: float = 0.0
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    video_codec: str = ""
    audio_codec: str = ""


class Utterance(BaseModel):
    id: int
    start_s: float
    end_s: float
    en: str = ""
    zh: str = ""
    raw: str = ""
    silence_before_ms: int = 0
    silence_after_ms: int = 0
    audio_rms_z: float = 0.0
    audio_peak_z: float = 0.0
    speech_rate_z: float = 0.0
    motion_z: float = 0.0
    rapid_turn: bool = False
    signal_tags: list[str] = Field(default_factory=list)


class TopicMapItem(BaseModel):
    topic_id: str
    start_utt: int
    end_utt: int
    start_s: float = 0.0
    end_s: float = 0.0
    chapter_title: str = ""
    title: str
    summary: str
    punchline_utt: int | None = None
    is_filler: bool = False
    filler_reason: str = ""
    selected_clip_id: str | None = None
    dropped_reason: str = ""


class SuppressedHotTopic(BaseModel):
    topic_id: str | None = None
    start_utt: int
    end_utt: int
    matched_keyword: str
    primary_topic_conflict: str
    reason: str


class MultimodalVerdict(BaseModel):
    status: Literal["not_required", "pass", "downgrade", "reject", "unverified"] = "not_required"
    model: str = ""
    seen_motion: str = ""
    heard_emotion: str = ""
    intensity_score: float = 0.0
    evidence: str = ""
    clip_preview_path: str | None = None
    error: str | None = None


class Candidate(BaseModel):
    candidate_id: str
    topic_ids: list[str] = Field(default_factory=list)
    categories: list[Literal["key_dialogue", "exciting_dialogue", "hot_topic"]] = Field(default_factory=list)
    matched_hot_topics: list[str] = Field(default_factory=list)
    chapter_title: str = ""
    title: str
    title_en: str = ""
    slug: str = ""
    start_utt: int
    end_utt: int
    hook: str = ""
    reason: str = ""
    summary: str = ""
    key_quote: str = ""
    score: float = 0.0
    needs_multimodal_verify: bool = False
    multimodal: MultimodalVerdict = Field(default_factory=MultimodalVerdict)
    disposition: Literal["pending", "selected", "merged", "remix_only", "dropped"] = "pending"
    disposition_reason: str = ""


class PublishRecord(BaseModel):
    platform: Literal["bilibili", "wechat", "xiaohongshu"]
    status: Literal["running", "published", "draft", "error"] = "running"
    mode: Literal["draft", "public"] = "draft"
    title: str = ""
    url_or_id: str = ""
    started_at: str = ""
    updated_at: str = Field(default_factory=utc_now_iso)
    progress_pct: float | None = None
    progress_stage: str = ""
    message: str = ""
    error: str | None = None


class SubtitleRepairItem(BaseModel):
    utterance_id: int
    zh: str | None = None
    en: str | None = None


class ClipItem(BaseModel):
    clip_id: str
    index: int
    enabled: bool = True
    categories: list[str] = Field(default_factory=list)
    matched_hot_topics: list[str] = Field(default_factory=list)
    chapter_title: str = ""
    title: str
    cover_title: str = ""
    title_en: str = ""
    slug: str = ""
    hook: str = ""
    reason: str = ""
    summary: str = ""
    key_quote: str = ""
    notes: list[str] = Field(default_factory=list)
    start_utt: int
    end_utt: int
    start_s: float
    end_s: float
    snap_delta_start_ms: int = 0
    snap_delta_end_ms: int = 0
    multimodal_status: str = "not_required"
    score: float = 7.5
    curation_tag: str = ""
    render_hash: str = ""
    last_rendered_hash: str = ""
    output_file: str | None = None
    cover_file: str | None = None
    cover_hash: str = ""
    rendered_filter_tag: str = ""
    publish_records: list[PublishRecord] = Field(default_factory=list)


class RemixItem(BaseModel):
    remix_id: str
    order: int
    enabled: bool = True
    source_clip_id: str | None = None
    theme_id: str = ""
    beat_type: Literal["punchline", "qa_argument", "case_demo", ""] = ""
    role: Literal["hook", "core", "hot_topic", "exciting", "finale"] = "core"
    group_label: str = ""
    card_title: str = ""
    macro_intent: str = ""
    start_utt: int
    end_utt: int
    start_s: float
    end_s: float
    quote: str = ""
    reason: str = ""


class RemixConfig(BaseModel):
    enabled: bool = True
    title: str = "高光混剪集锦"
    cover_title: str = ""
    editorial_thesis: str = ""
    card_mode: Literal["group", "every", "none"] = "group"
    card_duration_s: float = 0.9
    card_fade_s: float = 0.15
    enable_sfx: bool = True
    enable_bgm: bool = True
    bgm_volume: float = 0.16
    sfx_volume: float = 0.35
    bgm_path: str | None = None
    sfx_path: str | None = None
    source_clip_ids: list[str] | None = None
    manual_script_locked: bool = False
    items: list[RemixItem] = Field(default_factory=list)
    planned_source_signature: str = ""
    render_hash: str = ""
    last_rendered_hash: str = ""
    output_file: str | None = None
    cover_file: str | None = None
    cover_hash: str = ""
    rendered_filter_tag: str = ""
    publish_records: list[PublishRecord] = Field(default_factory=list)


class CompilationConfig(BaseModel):
    enabled: bool = True
    title: str = "切片精选集"
    cover_title: str = ""
    clip_ids: list[str] | None = None
    card_duration_s: float = 1.8
    card_fade_s: float = 0.2
    render_hash: str = ""
    last_rendered_hash: str = ""
    output_file: str | None = None
    cover_file: str | None = None
    cover_hash: str = ""
    rendered_filter_tag: str = ""
    publish_records: list[PublishRecord] = Field(default_factory=list)


class CutSegment(BaseModel):
    cut_id: str
    kind: Literal["ad", "political_sensitive"] = "ad"
    subtype: str = ""
    enabled: bool = True
    detector: Literal["llm", "visual", "manual"] = "llm"
    action: Literal["inherit", "delete", "mask"] = "inherit"
    recommended_action: Literal["delete", "mask"] = "delete"
    sensitive_terms_zh: list[str] = Field(default_factory=list)
    sensitive_terms_en: list[str] = Field(default_factory=list)
    start_utt: int | None = None
    end_utt: int | None = None
    start_s: float
    end_s: float
    quote: str = ""
    reason: str = ""
    severity: Literal["high", "medium", "low"] = "high"
    chapter_title: str = ""


class CleanCutConfig(BaseModel):
    enabled: bool = False
    title: str = ""
    cover_title: str = ""
    cut_ads: bool = False
    cut_political_sensitive: bool = False
    sensitivity: Literal["strict", "balanced", "loose"] = "strict"
    political_action: Literal["auto", "delete", "mask"] = "delete"
    apply_to_clips: bool = True
    cut_mode: Literal["auto", "precise", "reencode"] = "auto"
    scanned_at: str | None = None
    render_hash: str = ""
    last_rendered_hash: str = ""
    output_file: str | None = None
    cover_file: str | None = None
    cover_hash: str = ""
    rendered_filter_tag: str = ""
    publish_records: list[PublishRecord] = Field(default_factory=list)


class StageStatus(BaseModel):
    status: Literal["idle", "running", "completed", "error"] = "idle"
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_ms: int = 0
    message: str = ""
    error: str | None = None


class ProjectManifest(BaseModel):
    schema_version: str = "1.0"
    project_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    revision: int = 1
    source: MediaSource
    metadata: VideoMetadata = Field(default_factory=VideoMetadata)
    stages: dict[str, StageStatus] = Field(
        default_factory=lambda: {
            "ingest": StageStatus(),
            "signals": StageStatus(),
            "scan_cuts": StageStatus(),
            "analyze": StageStatus(),
            "verify_and_plan": StageStatus(),
            "render": StageStatus(),
        }
    )
    primary_topics: list[str] = Field(default_factory=list)
    utterances: list[Utterance] = Field(default_factory=list)
    topic_map: list[TopicMapItem] = Field(default_factory=list)
    suppressed_hot_topics: list[SuppressedHotTopic] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    clips: list[ClipItem] = Field(default_factory=list)
    remix_only_pool: list[Candidate] = Field(default_factory=list)
    remix: RemixConfig = Field(default_factory=RemixConfig)
    compilation: CompilationConfig = Field(default_factory=CompilationConfig)
    cut_segments: list[CutSegment] = Field(default_factory=list)
    clean_cut: CleanCutConfig = Field(default_factory=CleanCutConfig)


class Receipt(BaseModel):
    receipt_id: str
    run_id: str
    timestamp: str = Field(default_factory=utc_now_iso)
    stage: str
    actor: Literal["ai", "system", "human"] = "system"
    action: str
    model: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: int = 0
    summary: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    diff: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
