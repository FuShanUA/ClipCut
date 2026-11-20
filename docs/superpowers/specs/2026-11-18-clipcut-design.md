# ClipCut Design Specification

Date: 2026-11-18
Project Directory: `./clipcut`

## 1. Objective & Scope

ClipCut is a local-first, automated serious video clipping and Remix production application (CLI + FastAPI backend + React/Vite web workbench). It targets fast, repeatable automatic production of 1–5 minute editorial video clips plus a deluxe highlight Remix from long-form interviews, podcasts, and talks, while providing BaoCut-inspired full decision receipts and minimal human review/adjustment.

## 2. Input & Source Resolution (Decoupled from AutoSub)

- Primary input: a single video file (`.mp4`, `.mov`, `.mkv`, `.webm`, `.m4v`).
- Secondary input: a folder path. When given a folder, ClipCut scans top-level video files (excluding `.clipcut/` and `clips/`). If exactly one video exists, it is selected automatically; if multiple videos exist (e.g., raw `.mp4` and `*_hardsub.mp4`), CLI/Web lists resolution, duration, and file size and asks the user to pick one.
- Sibling discovery (best-effort, optional override):
  - Subtitles: matches `<stem>.bi.srt` -> `<stem>.cn.srt` -> `<stem>.en.srt` -> `<stem>.srt` -> any `.srt`/`.vtt` in the same folder. Also supports explicit `--subtitle` / UI path input.
  - Metadata: matches `<stem>.info.json` -> `*.info.json` -> `metadata.json`. If missing and a URL or YouTube `[VIDEO_ID]` exists in filename/folder, runs `yt-dlp --dump-single-json --skip-download` to fetch original title, full description, and chapters. Falls back gracefully to manual metadata entry or filename.
  - Clean analysis video: if the selected render video is `*_hardsub.mp4` and a matching raw `.mp4` exists alongside it, uses raw `.mp4` for visual motion detection to avoid subtitle pixels skewing frame differences, while rendering from the user-selected video.

## 3. Declarative Project State & Receipts (BaoCut-Inspired)

Each processed video writes working state to `<video_dir>/.clipcut/<video_slug>/` and final deliverables to `<video_dir>/clips/`:

- `project.json` (Single Source of Truth):
  - `source`: selected render video, analysis video, subtitle file, metadata origin, duration, fps, resolution.
  - `metadata`: original title, description, channel/guest, chapters, chapter audit (`keep`/`skip` + reason).
  - `utterances[]`: normalized sentence list (`id`, `start_s`, `end_s`, `en`, `zh`, acoustic/motion z-scores, silence before/after ms).
  - `topic_map[]`: 100% contiguous chronological map covering the whole video (`topic_id`, `start_utt`, `end_utt`, `title`, `summary`, `punchline_utt`, `chapter_title`, `is_filler`).
  - `primary_topics[]`: main topics of the video used to suppress false hot-topic matches.
  - `candidates[]`: three-track candidates with scores, multimodal verification verdicts, and keep/drop decisions.
  - `clips[]`: finalized clips (`clip_id`, `enabled`, `categories`, `title`, `title_en`, `slug`, `hook`, `reason`, `notes`, `start_utt`, `end_utt`, `start_s`, `end_s`, `snap_delta_start_ms`, `snap_delta_end_ms`, `render_hash`, `output_file`, `cover_file`).
  - `remix`: ordered highlight items (`remix_id`, `enabled`, `source_clip_id`, `group_label`, `card_title`, `start_utt`, `end_utt`, `start_s`, `end_s`, `quote`) plus deluxe render config (`card_mode`, `card_duration_s`, `enable_sfx`, `enable_bgm`, `bgm_path`, `sfx_path`, `render_hash`, `output_file`).
- `receipts.jsonl` (Append-Only Audit Ledger):
  - Every automated stage, LLM call, multimodal check, ffmpeg render, and human edit appends a structured JSON receipt (`receipt_id`, `run_id`, `timestamp`, `stage`, `actor`, `action`, `model`, `tokens`, `elapsed_ms`, `summary`, `details`, `diff`, `error`).
  - Human edits store a reversible snapshot diff enabling single-step Undo.

## 4. Five-Stage Pipeline

1. **`ingest`**: Video probing (`ffprobe`), subtitle parsing (bilingual EN/ZH splitting), metadata & chapter loading (including regex parsing of description timestamps when official chapters are absent).
2. **`signals`** (Local, zero LLM cost):
   - Audio: 16kHz mono PCM extraction via `ffmpeg`; per-utterance RMS energy z-score, peak burst, speech rate z-score, rapid turn-taking flag, and acoustic silence valley detection around sentence boundaries.
   - Video: 1fps 160x90 grayscale frame extraction via `ffmpeg`; sustained motion z-score filtering out single-frame hard scene cuts.
3. **`analyze`** (Full-text LLM via Vertex AI / Gemini fallback):
   - Pass 1 (Topic Map & Profile): Extracts `primary_topics`, audits chapters (excluding short teasers/intros/smalltalk/sponsors/outros with explicit reasons), and builds a contiguous `topic_map` covering 100% of utterances.
   - Pass 2 (Three-Track Selection from `topic_map`):
     - `key_dialogue`: original title subject, description takeaways, and each valid chapter's core thesis.
     - `exciting_dialogue`: laughter, big gestures, heated debates, sharp pushback (guided by acoustic/motion tags).
     - `hot_topic`: configurable Chinese hot topics & editorial focus (`config/editorial.yaml`) with strict **Primary-Topic Deduplication** (keywords in `primary_topics` do not qualify as standalone hot topics unless accompanied by quantified business ROI, China angle, or secondary cross-topic). All suppressed hot-topic hits and unchosen topic-map entries are logged with explicit reasons.
4. **`verify_and_plan`**:
   - Targeted Multimodal Verification: extracts compact 480p 2fps + 64k audio clips for `exciting_dialogue` candidates and calls Gemini multimodal to verify visual gestures, laughter, and debate intensity. Failures mark `unverified` without halting the pipeline.
   - Overlap Merge: merges overlapping candidates (IoU > 0.35), preserving combined tags.
   - Completeness & Pause-Midpoint Snapping (BaoCut craft): prevents opening on dangling context references or cutting mid-sentence; snaps boundaries to inter-utterance pause midpoints with `<40ms` short-pause protection so adjacent words are never clipped. Enforces 60–300s bounds for standalone clips (high-impact <50s moments go into `remix_only_pool`).
   - Remix Planning: runs strictly after finalized clips are locked; selects 1–2 punchy sentences or 1–2 turns per clip (4–15s each, total 60–150s), sequenced as Hook -> Core/Hot/Clash -> Finale, with 4–10 char transition card titles.
5. **`render`** (Deluxe, Hash-Cached):
   - Standalone clips: frame-accurate re-encode via `h264_videotoolbox` (fallback `libx264`), `aac 192k`, `25ms` anti-click fades, editorial JPG covers, `clips_index.md`, `clips_detail.md`.
   - Remix: loudness-normalized (`loudnorm`) excerpts interleaved with Pillow-rendered dark editorial title cards (`0.9s` with `0.15s` fade), transition SFX at card entries, and optional BGM with `sidechaincompress` vocal ducking + `1.5s` outro fade, plus `remix_script.md`.
   - Incremental caching: only clips/Remix whose `render_hash` changed since last render are re-encoded.

## 5. Minimal Web Workbench & CLI

- CLI: `clipcut scan <path>`, `clipcut run <path>`, `clipcut render <project>`, `clipcut replan-remix <project>`, `clipcut serve`.
- Web UI (React + Vite + TypeScript, FastAPI API):
  - Left pane: video/folder loader (with multi-video picker), 5-stage runner & progress, clip/Remix list with enable switches, hash cache badges, incremental render trigger.
  - Center pane: HTTP Range video player with clip-loop preview + BaoCut-style contextual subtitle list (adjacent out-of-range lines struck-through/dimmed, click any line to set start/end) + Remix item toggles/order/title-card editor.
  - Right pane: Receipts & Debug Log inspector filtered by stage (skipped chapters, suppressed hot topics, multimodal verdicts, dropped topics, snap diffs, model token/cost/latency) + single-step Undo.
