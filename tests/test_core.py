from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from clipcut.adapters.ingest import (
    extract_published_date,
    extract_youtube_id,
    parse_chapters_from_description,
    parse_subtitle_file,
    scan_input_path,
)
from clipcut.assets_gen import ensure_default_assets
from clipcut.config import load_editorial_config
from clipcut.models import (
    Chapter,
    ClipItem,
    MediaSource,
    ProjectManifest,
    RemixConfig,
    RemixItem,
    TopicMapItem,
    Utterance,
    VideoMetadata,
)
from clipcut.pipeline.analyze import _is_primary_topic_match
from clipcut.pipeline.verify_and_plan import (
    _compute_clip_bridge_window,
    _match_quote_to_utterances,
    _parse_utt_int,
    compute_clip_render_hash,
    ensure_sentence_completeness,
    snap_bounds_with_acoustic_guard,
)
from clipcut.receipts import ProjectStore
from clipcut.render.frame_selector import (
    HERO_POOL_SCHEMA_VERSION,
    FaceBox,
    GuestProfile,
    SelectedFrame,
    compose_cinema_fit_width_frame,
    crop_frame_face_safe,
    score_candidate_frame,
)
from clipcut.render.renderer import (
    _load_autosub_cover_utils,
    build_clip_filename,
    build_remix_filename,
    compose_pip_card_image,
    compute_cover_hash,
    remix_item_group_key,
    render_natural_cover_image,
    should_insert_title_card,
    sync_platform_covers_for_target,
)


def test_remix_quote_matching_and_utt_parsing():
    assert _parse_utt_int("句号=48") == 48
    assert _parse_utt_int(12) == 12
    utts = [
        Utterance(id=0, start_s=0.0, end_s=4.0, zh="第一句闲聊。"),
        Utterance(id=1, start_s=231.0, end_s=236.0, zh="我们有前车之鉴，叫欧洲。"),
        Utterance(id=2, start_s=236.2, end_s=241.0, zh="欧洲把所有人挡在外面。"),
    ]
    su, eu = _match_quote_to_utterances("我们有前车之鉴，叫欧洲", utts, 0, 2)
    assert su == 1 and eu == 1


def test_editorial_config_loads():
    cfg = load_editorial_config()
    assert cfg["clip_duration"]["min_seconds"] == 60
    assert cfg["clip_duration"]["max_seconds"] == 300
    assert cfg["hot_topics"]["primary_topic_dedup"] is True
    assert len(cfg["hot_topics"]["categories"]) >= 5


def test_default_assets_generated():
    assets = ensure_default_assets()
    assert assets["sfx"].exists() and assets["sfx"].stat().st_size > 1024
    assert assets["bgm"].exists() and assets["bgm"].stat().st_size > 4096


def test_youtube_id_and_description_chapters():
    assert extract_youtube_id("Palmer Luckey [Wj00YlDuWXc] [1080p].mp4") == "Wj00YlDuWXc"
    assert extract_youtube_id("https://www.youtube.com/watch?v=zVp3gGEyqQE") == "zVp3gGEyqQE"
    desc = """
    00:00 Intro
    02:15 Autonomous Weapons
    15:40 US-China AI Race
    """
    chs = parse_chapters_from_description(desc, 1800.0)
    assert len(chs) == 3
    assert chs[0].title == "Intro" and chs[0].end_s == 135.0
    assert chs[2].start_s == 940.0 and chs[2].end_s == 1800.0


def test_primary_topic_dedup_matcher():
    assert _is_primary_topic_match("Palantir", ["Palantir", "AI Risks"]) == "Palantir"
    assert _is_primary_topic_match("英伟达", ["NVIDIA / 英伟达"]) == "NVIDIA / 英伟达"
    assert _is_primary_topic_match("Anduril", ["Palantir"]) is None


def test_sentence_completeness_and_short_pause_guard():
    utts = [
        Utterance(id=0, start_s=0.0, end_s=3.0, zh="主持人问为什么美国不能照搬欧洲模式？"),
        Utterance(id=1, start_s=3.02, end_s=6.0, zh="所以这就是核心分歧所在"),
        Utterance(id=2, start_s=6.5, end_s=10.0, zh="欧洲监管太重会拖慢创新。"),
        Utterance(id=3, start_s=10.01, end_s=13.0, zh="下一句开始。"),
    ]
    su, eu, notes = ensure_sentence_completeness(1, 1, utts)
    # Leading "所以..." pulls in utterance 0; trailing lacks terminal punctuation so pulls in utterance 2
    assert su == 0
    assert eu == 2
    assert len(notes) >= 2

    cfg = load_editorial_config()
    rms = np.ones(1000, dtype=np.float32) * 0.1
    # Gap between utt 2 (10.0) and utt 3 (10.01) is 10ms (<40ms short-pause guard)
    s_t, e_t, ds, de, guard_notes = snap_bounds_with_acoustic_guard(
        start_utt=1,
        end_utt=2,
        utterances=utts,
        rms=rms,
        hop_s=0.02,
        video_duration=20.0,
        cfg=cfg,
    )
    assert s_t >= 3.0 and s_t <= 3.02
    assert e_t >= 10.0 and e_t <= 10.01
    assert any("<40ms" in n for n in guard_notes)


def test_parse_bilingual_srt(tmp_path: Path):
    p = tmp_path / "sample.bi.srt"
    p.write_text(
        "1\n00:00:01,000 --> 00:00:04,000\nHello world.\n你好世界。\n\n"
        "2\n00:00:04,500 --> 00:00:07,200\nSecond line.\n第二句。\n",
        encoding="utf-8",
    )
    utts = parse_subtitle_file(p)
    assert len(utts) == 2
    assert utts[0].en == "Hello world."
    assert utts[0].zh == "你好世界。"
    assert utts[0].silence_after_ms == 500
    assert utts[1].silence_before_ms == 500


def test_scan_folder_multiple_videos(tmp_path: Path):
    (tmp_path / "a.mp4").write_bytes(b"0" * 100)
    (tmp_path / "a_hardsub.mp4").write_bytes(b"0" * 200)
    (tmp_path / "a.bi.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
    res = scan_input_path(tmp_path)
    assert res.is_directory is True
    assert len(res.videos) == 2
    assert res.videos[0].is_hardsub is True
    assert res.videos[0].suggested_subtitle is not None


def test_long_quote_and_ellipsis_anchor_matching():
    utts = [
        Utterance(id=i, start_s=i * 5.0, end_s=i * 5.0 + 4.5, zh=text)
        for i, text in enumerate(
            [
                "先聊聊今天的开场背景。",
                "市面上有很多聊天机器人。",
                "但我们内部的产品叫Genie。",
                "它底层连接了整个公司的本体图谱。",
                "CFO原本猜测是加班导致行动力下降。",
                "结果数据拆开后彻底推翻了直觉。",
            ]
        )
    ]
    su, eu = _match_quote_to_utterances("市面上有很多聊天机器人...结果数据拆开后彻底推翻了直觉", utts, 0, 5, max_span_s=45.0)
    assert su == 1 and eu == 5

    su2, eu2 = _match_quote_to_utterances(
        "但我们内部的产品叫Genie。它底层连接了整个公司的本体图谱。CFO原本猜测是加班导致行动力下降。",
        utts,
        0,
        5,
        max_span_s=45.0,
    )
    assert su2 == 2 and eu2 == 4


def test_clip_bridge_window_respects_chapters_disabled_clips_and_fillers():
    utts = [Utterance(id=i, start_s=i * 6.0, end_s=i * 6.0 + 5.0, zh=f"第{i}句") for i in range(20)]
    chapters = [
        Chapter(index=0, title="Intro", start_s=0.0, end_s=30.0),
        Chapter(index=1, title="Live Demo", start_s=30.0, end_s=120.0),
    ]
    # u0..u4 in Intro (0-30s); u5..u19 in Live Demo (30-120s)
    # Disabled clip ends at u6; filler ends at u7 -> bridge must start no earlier than u8
    all_clips = [
        ClipItem(clip_id="clip_01", index=1, enabled=False, title="Disabled", start_utt=5, end_utt=6, start_s=30.0, end_s=41.0),
        ClipItem(clip_id="clip_02", index=2, enabled=True, chapter_title="Live Demo", title="Main Demo", start_utt=12, end_utt=18, start_s=72.0, end_s=113.0),
    ]
    topics = [
        TopicMapItem(topic_id="T01", start_utt=7, end_utt=7, start_s=42.0, end_s=47.0, title="Sponsor", summary="Ad", is_filler=True),
        TopicMapItem(topic_id="T02", start_utt=8, end_utt=11, start_s=48.0, end_s=71.0, chapter_title="Live Demo", title="Tool Setup", summary="Genie intro", is_filler=False),
    ]
    bridge_lo = _compute_clip_bridge_window(all_clips[1], all_clips, topics, chapters, utts)
    assert bridge_lo == 8

    # Without disabled clip or filler, chapter boundary stops bridge_lo at u5 (>= 30.0s)
    bridge_lo_ch = _compute_clip_bridge_window(all_clips[1], [all_clips[1]], [], chapters, utts)
    assert bridge_lo_ch == 5


def test_group_card_mode_inserts_once_per_group_even_with_card_titles():
    items = [
        RemixItem(remix_id="rmx_01", order=1, theme_id="TH01", group_label="企业AI工程", card_title="超越问数的决策分析", start_utt=1, end_utt=5, start_s=10.0, end_s=40.0),
        RemixItem(remix_id="rmx_02", order=2, theme_id="TH01", group_label="企业AI工程", card_title="超越问数的决策分析", start_utt=6, end_utt=10, start_s=42.0, end_s=75.0),
        RemixItem(remix_id="rmx_03", order=3, theme_id="TH02", group_label="RSI证伪", card_title="当前无递归自改进飞轮", start_utt=20, end_utt=28, start_s=120.0, end_s=165.0),
    ]
    cards = []
    last_gk = None
    for idx, it in enumerate(items):
        gk = remix_item_group_key(it)
        cards.append(should_insert_title_card("group", idx, gk, last_gk))
        last_gk = gk
    assert cards == [True, False, True]


def test_extract_published_date_formats():
    assert extract_published_date({"upload_date": "20261117"}) == "2026-11-17"
    assert extract_published_date({"release_date": "2025-12-01", "upload_date": "20251202"}) == "2025-12-01"
    assert extract_published_date({"timestamp": 1700000000}) == "2023-11-14"
    assert extract_published_date(None) == ""


def test_frame_scoring_filters_tiny_pip_and_prefers_two_shot(tmp_path: Path):
    from PIL import Image, ImageDraw

    img_p = tmp_path / "bright.jpg"
    im = Image.new("RGB", (320, 180), (90, 100, 115))
    dr = ImageDraw.Draw(im)
    dr.rectangle([40, 20, 280, 150], fill=(170, 145, 130))
    im.save(img_p)

    # Tiny screen-share corner face (h=0.10, w=0.06) should be rejected (< 0.5 score)
    sc_pip, faces_pip, two_pip = score_candidate_frame(
        img_p,
        [FaceBox(x=0.85, y=0.25, w=0.06, h=0.10, quality=0.8)],
        prefer_two_shot=True,
    )
    assert faces_pip == []
    assert two_pip is False
    assert sc_pip < 0.5

    # Single crisp face vs two-shot face: both >= 1.2, two-shot wins when prefer_two_shot=True
    sc_single, f_single, t_single = score_candidate_frame(
        img_p,
        [FaceBox(x=0.35, y=0.18, w=0.24, h=0.45, quality=0.55)],
        prefer_two_shot=True,
    )
    sc_two, f_two, t_two = score_candidate_frame(
        img_p,
        [
            FaceBox(x=0.14, y=0.20, w=0.20, h=0.40, quality=0.55),
            FaceBox(x=0.62, y=0.20, w=0.20, h=0.40, quality=0.55),
        ],
        prefer_two_shot=True,
    )
    assert len(f_single) == 1 and not t_single and sc_single >= 1.2
    assert len(f_two) == 2 and t_two and sc_two > sc_single


def test_crop_frame_face_safe_lifts_low_chin():
    from PIL import Image

    im = Image.new("RGB", (1920, 1080), (60, 70, 80))
    # Face starts at y=0.22 and extends to bottom=0.80 (> safe_bottom=0.71)
    faces = [FaceBox(x=0.36, y=0.22, w=0.28, h=0.58, quality=0.6)]
    out_im, new_faces = crop_frame_face_safe(im, 1920, 1080, faces, safe_bottom_ratio=0.71)
    assert out_im.size == (1920, 1080)
    assert len(new_faces) == 1
    assert new_faces[0].bottom < faces[0].bottom
    assert new_faces[0].y >= 0.035


def test_cover_and_pip_card_image_composition(tmp_path: Path):
    from PIL import Image

    portrait_p = tmp_path / "portrait.jpg"
    Image.new("RGB", (1280, 720), (75, 95, 120)).save(portrait_p)
    sf = SelectedFrame(
        image_path=portrait_p,
        timestamp_s=15.0,
        faces=[FaceBox(x=0.38, y=0.18, w=0.24, h=0.46, quality=0.6)],
        is_two_shot=False,
        score=4.5,
    )

    with Image.open(portrait_p) as raw_im:
        nat = render_natural_cover_image(
            frame_img=raw_im,
            faces=sf.faces,
            width=1280,
            height=720,
            badge_text="#01  重点对话",
            dur_text="03:20",
            title="超越问数的企业级决策AI与组织洞察",
            meta_parts=["阿里·戈德西", "All-In Podcast", "2026-11-17"],
        )
    assert nat.size == (1280, 720)

    card = compose_pip_card_image(
        width=1280,
        height=720,
        index_text="01",
        group_label="决策智能重构",
        card_title="超越问数的企业级决策AI与组织洞察",
        meta_parts=["阿里·戈德西", "All-In Podcast", "2026-11-17"],
        left_meta_lines=["片段 01 · 案例演示", "2026-11-17"],
        portrait_frame=sf,
        enable_pip=True,
    )
    assert card.size == (1280, 720)


def test_standardized_filenames_and_cover_title_hash(tmp_path: Path):
    cfg = load_editorial_config()
    manifest = ProjectManifest(
        project_id="proj_test",
        source=MediaSource(
            render_video=str(tmp_path / "src.mp4"),
            analysis_video=str(tmp_path / "src.mp4"),
            work_dir=str(tmp_path / "work"),
            output_dir=str(tmp_path / "clips"),
            duration_s=3600.0,
        ),
        metadata=VideoMetadata(
            channel="All-In Podcast",
            channel_zh="All-In播客",
            published_date="2026-11-17",
            original_title="Databricks CEO Interview",
        ),
        remix=RemixConfig(
            enabled=True,
            title="AGI落地：企业AI的真实拐点",
            items=[
                RemixItem(
                    remix_id="rmx_01",
                    order=1,
                    enabled=True,
                    start_utt=1,
                    end_utt=5,
                    start_s=12.0,
                    end_s=92.0,
                )
            ],
        ),
    )
    clip = ClipItem(
        clip_id="clip_01",
        index=1,
        enabled=True,
        categories=["企业AI"],
        title="超越问数：Genie如何连接企业本体/图谱?",
        start_utt=10,
        end_utt=30,
        start_s=185.0,
        end_s=365.0,
    )

    clip_name = build_clip_filename(clip, manifest, published_date="2026-11-17")
    assert clip_name.startswith("01_") and clip_name.endswith(".mp4")
    assert "All-In" in clip_name
    assert "2026-11-17" in clip_name
    assert "03m05s-06m05s" in clip_name
    assert "超越问数" in clip_name
    assert ":" not in clip_name and "/" not in clip_name

    remix_name = build_remix_filename(manifest, published_date="2026-11-17")
    assert remix_name.startswith("00_Remix_") and remix_name.endswith(".mp4")
    assert "All-In" in remix_name
    assert "2026-11-17" in remix_name
    assert "01m" in remix_name
    assert ":" not in remix_name and "/" not in remix_name

    base_render_hash = compute_clip_render_hash(clip, manifest.source.render_video)
    base_cover_hash = compute_cover_hash(
        key="clip_01",
        title=(clip.cover_title or clip.title),
        categories=clip.categories,
        start_s=clip.start_s,
        end_s=clip.end_s,
        manifest=manifest,
        cfg=cfg,
        published_date="2026-11-17",
    )

    clip_edited = clip.model_copy(update={"cover_title": "Genie如何重塑企业数据决策"})
    edited_render_hash = compute_clip_render_hash(clip_edited, manifest.source.render_video)
    edited_cover_hash = compute_cover_hash(
        key="clip_01",
        title=(clip_edited.cover_title or clip_edited.title),
        categories=clip_edited.categories,
        start_s=clip_edited.start_s,
        end_s=clip_edited.end_s,
        manifest=manifest,
        cfg=cfg,
        published_date="2026-11-17",
    )

    assert edited_render_hash == base_render_hash
    assert edited_cover_hash != base_cover_hash


def test_crop_frame_face_safe_two_shot_portrait_picks_strongest_face():
    from PIL import Image

    im = Image.new("RGB", (1920, 1080), (60, 75, 95))
    faces = [
        FaceBox(x=0.14, y=0.20, w=0.20, h=0.40, quality=0.55),
        FaceBox(x=0.66, y=0.18, w=0.24, h=0.44, quality=0.80),
    ]
    cropped, new_faces = crop_frame_face_safe(im, 1080, 1440, faces, safe_bottom_ratio=0.68)
    assert cropped.size == (1080, 1440)
    assert len(new_faces) == 1
    assert 0.25 <= new_faces[0].cx <= 0.75


def test_cover_multi_aspect_long_chinese_title(tmp_path: Path):
    from PIL import Image

    raw_im = Image.new("RGB", (1920, 1080), (95, 115, 135))
    faces = [FaceBox(x=0.38, y=0.18, w=0.24, h=0.42, quality=0.75)]
    title = "超越传统问数：企业级决策智能如何重构全员数据分析"
    for w, h in ((1920, 1080), (1080, 810), (1080, 1440)):
        out = render_natural_cover_image(
            frame_img=raw_im,
            faces=faces,
            width=w,
            height=h,
            badge_text="REMIX  ·  高光集锦",
            dur_text="06:10",
            title=title,
            meta_parts=["阿里·戈德西", "All-In Podcast", "2026-11-17"],
        )
        assert out.size == (w, h)


def test_autosub_cover_cache_reuses_clipcut_platform_covers(tmp_path: Path):
    from PIL import Image

    cfg = load_editorial_config()
    work_dir = tmp_path / "work"
    out_dir = tmp_path / "clips"
    work_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    hero_dir = work_dir / "hero_frames"
    hero_dir.mkdir(parents=True)
    hero_jpg = hero_dir / "cand_00.jpg"
    Image.new("RGB", (1920, 1080), (85, 105, 130)).save(hero_jpg)
    (hero_dir / "index.json").write_text(
        json.dumps(
            {
                "schema": HERO_POOL_SCHEMA_VERSION,
                "source": str(tmp_path / "clean.mp4"),
                "frames": [
                    {
                        "image_path": str(hero_jpg),
                        "timestamp_s": 25.0,
                        "is_two_shot": False,
                        "score": 4.2,
                        "from_fallback": False,
                        "faces": [{"x": 0.38, "y": 0.18, "w": 0.22, "h": 0.42, "quality": 0.78, "yaw": 0.0}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dummy_mp4 = out_dir / "00_Remix_All-In-Podcast_AGI已至_2026-11-17_06m10s.mp4"
    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=s=640x360:d=1.2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(dummy_mp4),
        ],
        check=True,
    )
    clean_mp4 = tmp_path / "clean.mp4"
    clean_mp4.write_bytes(dummy_mp4.read_bytes())

    title = "AGI已至：企业AI落地困境与奇点证伪"
    manifest = ProjectManifest(
        project_id="proj_cover_sync",
        source=MediaSource(
            render_video=str(tmp_path / "clean.mp4"),
            analysis_video=str(tmp_path / "clean.mp4"),
            work_dir=str(work_dir),
            output_dir=str(out_dir),
            duration_s=1800.0,
        ),
        metadata=VideoMetadata(
            channel="All-In Podcast",
            channel_zh="All-In播客",
            guest_zh="阿里·戈德西",
            published_date="2026-11-17",
        ),
        remix=RemixConfig(
            enabled=True,
            title=title,
            cover_title=title,
            output_file=str(dummy_mp4),
            items=[RemixItem(remix_id="rmx_01", order=1, enabled=True, start_utt=1, end_utt=4, start_s=10.0, end_s=80.0)],
        ),
    )
    store = ProjectStore(work_dir)
    store.save(manifest)

    written = sync_platform_covers_for_target(
        manifest=manifest,
        store=store,
        cfg=cfg,
        target="remix",
        platforms=["bilibili", "wechat", "xiaohongshu"],
    )
    assert len(written) == 5

    bili_wide = out_dir / f"{dummy_mp4.stem}.cover.bili.jpg"
    bili_share = out_dir / f"{dummy_mp4.stem}.cover.bili.4x3.jpg"
    wx_port = out_dir / f"{dummy_mp4.stem}.cover.wechat.jpg"
    wx_share = out_dir / f"{dummy_mp4.stem}.cover.wechat.share.jpg"
    orig_bytes = {p: p.read_bytes() for p in (bili_wide, bili_share, wx_port, wx_share)}

    cu = _load_autosub_cover_utils()
    assert cu is not None
    assert cu.cover_style_matches(str(bili_wide), title, "wide")
    assert cu.cover_style_matches(str(bili_share), title, "share")
    assert cu.cover_style_matches(str(wx_port), title, "portrait")
    assert cu.cover_style_matches(str(wx_share), title, "share")

    ret_bili = cu.generate_platform_covers(str(dummy_mp4), "bilibili", title)
    ret_wx = cu.generate_platform_covers(str(dummy_mp4), "wechat", title)
    assert Path(ret_bili[0]).read_bytes() == orig_bytes[bili_wide]
    assert Path(ret_bili[1]).read_bytes() == orig_bytes[bili_share]
    assert Path(ret_wx[0]).read_bytes() == orig_bytes[wx_port]
    assert Path(ret_wx[1]).read_bytes() == orig_bytes[wx_share]


def test_crop_frame_face_safe_prioritizes_guest_over_larger_host():
    from PIL import Image

    im = Image.new("RGB", (1920, 1080), (60, 75, 95))
    # Left face is smaller Guest (is_guest=True); right face is larger Host (is_guest=False)
    faces = [
        FaceBox(x=0.14, y=0.22, w=0.18, h=0.36, quality=0.55, is_guest=True),
        FaceBox(x=0.66, y=0.18, w=0.26, h=0.48, quality=0.85, is_guest=False),
    ]
    cropped, new_faces = crop_frame_face_safe(im, 1080, 1440, faces, safe_bottom_ratio=0.68)
    assert cropped.size == (1080, 1440)
    assert len(new_faces) == 1
    assert new_faces[0].is_guest is True
    assert 0.25 <= new_faces[0].cx <= 0.75


def test_score_candidate_frame_rejects_host_and_boosts_guest(tmp_path: Path):
    from PIL import Image, ImageDraw

    img_p = tmp_path / "bright.jpg"
    im = Image.new("RGB", (320, 180), (90, 100, 115))
    ImageDraw.Draw(im).rectangle([40, 20, 280, 150], fill=(170, 145, 130))
    im.save(img_p)

    guest_emb = [1.0, 0.0, 0.0, 0.0]
    host_emb = [0.0, 1.0, 0.0, 0.0]
    prof = GuestProfile(
        source=str(tmp_path / "src.mp4"),
        guest_name="Ali Ghodsi",
        guest_embedding=guest_emb,
        host_embeddings=[host_emb],
    )

    host_face = [FaceBox(x=0.36, y=0.18, w=0.24, h=0.45, quality=0.75, embedding=host_emb)]
    sc_host, vf_host, _ = score_candidate_frame(img_p, host_face, prefer_two_shot=True, guest_profile=prof)
    assert sc_host < 1.0
    assert vf_host[0].is_guest is False

    guest_face = [FaceBox(x=0.36, y=0.18, w=0.24, h=0.45, quality=0.75, embedding=guest_emb)]
    sc_guest, vf_guest, _ = score_candidate_frame(img_p, guest_face, prefer_two_shot=True, guest_profile=prof)
    assert sc_guest > 3.5
    assert vf_guest[0].is_guest is True


def test_cinema_fit_width_preserves_two_shot_in_4x3():
    from PIL import Image

    im = Image.new("RGB", (1920, 1080), (90, 105, 125))
    faces = [
        FaceBox(x=0.12, y=0.32, w=0.22, h=0.40, quality=0.60, is_guest=True),
        FaceBox(x=0.66, y=0.34, w=0.22, h=0.40, quality=0.65, is_guest=False),
    ]
    out, mapped_faces = compose_cinema_fit_width_frame(im, 1080, 810, faces)
    assert out.size == (1080, 810)
    assert len(mapped_faces) == 2

    rendered = render_natural_cover_image(
        frame_img=im,
        faces=faces,
        width=1080,
        height=810,
        badge_text="REMIX  ·  高光集锦",
        dur_text="06:10",
        title="AGI已至：企业AI落地困境与奇点证伪",
        meta_parts=["阿里·戈德西", "All-In Podcast", "2026-11-17"],
    )
    assert rendered.size == (1080, 810)


def test_updated_model_catalog_and_fallback_chain():
    from clipcut.llm import MODEL_CATALOG

    cfg = load_editorial_config()
    m_cfg = cfg["models"]
    fb = m_cfg["text_fallback_models"]
    for required in ("qwen4-max", "glm-5.3", "kimi-k3.1-pro", "deepseek-v4-pro", "glm-5.2", "gemini:gemini-2.5-flash"):
        assert required in fb
    assert "qwen-max" not in fb
    assert "qwen3.7-max" not in fb

    text_models = {item["model"] for item in MODEL_CATALOG["text"]}
    for required in ("qwen4-max", "qwen3.8-max", "glm-5.3", "glm-5.2", "kimi-k3.1-pro", "deepseek-v4-pro", "MiniMax/MiniMax-M3"):
        assert required in text_models


def test_autosub_cover_utils_reloads_on_mtime_change(tmp_path: Path, monkeypatch):
    import os
    from clipcut.render import renderer as r_mod

    fake_cu = tmp_path / "cover_utils.py"
    fake_cu.write_text('_COVER_STYLE_VERSION = "test-v1"\n', encoding="utf-8")
    os.utime(fake_cu, (1700000100.0, 1700000100.0))

    monkeypatch.setattr(r_mod, "_AUTOSUB_COVER_UTILS_PATH", fake_cu)
    monkeypatch.setattr(r_mod, "_AUTOSUB_COVER_UTILS_CACHE", None)

    mod1 = r_mod._load_autosub_cover_utils()
    assert mod1 is not None
    assert mod1._COVER_STYLE_VERSION == "test-v1"

    fake_cu.write_text('_COVER_STYLE_VERSION = "test-v2"\n', encoding="utf-8")
    os.utime(fake_cu, (1700000200.0, 1700000200.0))

    mod2 = r_mod._load_autosub_cover_utils()
    assert mod2 is not None
    assert mod2._COVER_STYLE_VERSION == "test-v2"


def test_publish_default_metadata_separates_title_and_cover_title(tmp_path: Path):
    from clipcut.publish.publishers import _default_target_metadata

    src_mp4 = tmp_path / "src.mp4"
    src_mp4.write_bytes(b"0")
    out_dir = tmp_path / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    manifest = ProjectManifest(
        project_id="p_pub_titles",
        source=MediaSource(
            input_path=str(src_mp4),
            render_video=str(src_mp4),
            analysis_video=str(src_mp4),
            work_dir=str(work_dir),
            output_dir=str(out_dir),
            duration_s=300.0,
        ),
        metadata=VideoMetadata(
            channel="Sourcery",
            channel_zh="Sourcery",
            guest_zh="加文·贝克 (Gavin Baker)",
            published_date="2026-11-19",
        ),
        clips=[
            ClipItem(
                clip_id="clip_01",
                index=1,
                title="切片主标题A",
                cover_title="封面短标题A",
                start_utt=0,
                end_utt=1,
                start_s=10.0,
                end_s=70.0,
            ),
            ClipItem(
                clip_id="clip_02",
                index=2,
                title="切片主标题B",
                cover_title="",
                start_utt=2,
                end_utt=3,
                start_s=80.0,
                end_s=140.0,
            ),
        ],
        remix=RemixConfig(
            enabled=True,
            title="AI基建长周期与黄仁勋的供应链护城河",
            cover_title="黄仁勋在台湾是活神仙",
            items=[RemixItem(remix_id="rmx_01", order=1, enabled=True, start_utt=0, end_utt=1, start_s=10.0, end_s=70.0)],
        ),
    )

    _, rmx_title, rmx_cover, _, _, _ = _default_target_metadata(manifest, "remix")
    assert rmx_title == "AI基建长周期与黄仁勋的供应链护城河"
    assert rmx_cover == "黄仁勋在台湾是活神仙"

    _, c1_title, c1_cover, _, _, _ = _default_target_metadata(manifest, "clip_01")
    assert c1_title == "切片主标题A"
    assert c1_cover == "封面短标题A"

    _, c2_title, c2_cover, _, _, _ = _default_target_metadata(manifest, "clip_02")
    assert c2_title == "切片主标题B"
    assert c2_cover == "切片主标题B"

    manifest.remix.cover_title = ""
    _, rmx2_title, rmx2_cover, _, _, _ = _default_target_metadata(manifest, "remix")
    assert rmx2_title == "AI基建长周期与黄仁勋的供应链护城河"
    assert rmx2_cover == "AI基建长周期与黄仁勋的供应链护城河"


def test_publish_stale_ui_title_guard(tmp_path: Path, monkeypatch):
    import subprocess
    import sys
    from clipcut.publish import publishers as pub_mod

    clip_mp4 = tmp_path / "clip_01.mp4"
    clip_mp4.write_bytes(b"0")
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    fake_script = tmp_path / "fake_bili.py"
    fake_script.write_text("# stub\n", encoding="utf-8")

    manifest = ProjectManifest(
        project_id="p_stale_guard",
        source=MediaSource(
            input_path=str(clip_mp4),
            render_video=str(clip_mp4),
            analysis_video=str(clip_mp4),
            work_dir=str(work_dir),
            output_dir=str(tmp_path),
            duration_s=120.0,
        ),
        clips=[
            ClipItem(
                clip_id="clip_01",
                index=1,
                title="切片主标题A",
                cover_title="封面短标题A",
                output_file=str(clip_mp4),
                start_utt=0,
                end_utt=1,
                start_s=0.0,
                end_s=60.0,
            ),
        ],
    )
    store = ProjectStore(work_dir)
    store.save(manifest)

    captured_cmds: list[list[str]] = []

    monkeypatch.setattr(pub_mod, "sync_platform_covers_for_target", lambda **kwargs: [])
    monkeypatch.setattr(
        pub_mod,
        "_get_paths",
        lambda: {"bilibili": {"python": sys.executable, "script": fake_script}},
    )

    def _fake_run(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="BV1test123456", stderr="")

    monkeypatch.setattr(pub_mod.subprocess, "run", _fake_run)

    # Stale UI (ui_schema is None) accidentally submits cover_title as title
    pub_mod.publish_to_platforms(
        store=store,
        target="clip_01",
        platforms=["bilibili"],
        title="封面短标题A",
        cover_title="封面短标题A",
        ui_schema=None,
    )
    assert len(captured_cmds) == 1
    cmd1 = captured_cmds[0]
    script_idx1 = cmd1.index(str(fake_script))
    assert cmd1[script_idx1 + 2] == "切片主标题A"
    assert cmd1[cmd1.index("--cover-title") + 1] == "封面短标题A"

    # Updated UI (ui_schema=2) deliberately submits matching title and cover_title
    pub_mod.publish_to_platforms(
        store=store,
        target="clip_01",
        platforms=["bilibili"],
        title="封面短标题A",
        cover_title="封面短标题A",
        ui_schema=2,
    )
    assert len(captured_cmds) == 2
    cmd2 = captured_cmds[1]
    script_idx2 = cmd2.index(str(fake_script))
    assert cmd2[script_idx2 + 2] == "封面短标题A"
    assert cmd2[cmd2.index("--cover-title") + 1] == "封面短标题A"


def test_publishers_missing_local_config_is_safe(tmp_path: Path, monkeypatch):
    from clipcut.publish.publishers import get_platforms_status

    monkeypatch.setenv("CLIPCUT_LOCAL_CONFIG", str(tmp_path / "nonexistent.yaml"))
    for var in (
        "CLIPCUT_WECHAT_SCRIPT",
        "CLIPCUT_WECHAT_PYTHON",
        "WECHAT_BROWSER_DATA_DIR",
        "CLIPCUT_BILI_SCRIPT",
        "CLIPCUT_BILI_PYTHON",
        "CLIPCUT_BILI_LOGIN_SCRIPT",
        "CLIPCUT_BILI_COOKIES",
        "CLIPCUT_XHS_SCRIPT",
        "CLIPCUT_XHS_PYTHON",
        "CLIPCUT_XHS_LOGIN_SCRIPT",
        "CLIPCUT_XHS_COOKIES",
    ):
        monkeypatch.delenv(var, raising=False)

    status = get_platforms_status()
    assert set(status.keys()) == {"wechat", "bilibili", "xiaohongshu"}
    for plat in ("wechat", "bilibili", "xiaohongshu"):
        assert status[plat]["logged_in"] is False
        assert status[plat]["python_ready"] is False


def test_format_concat_file_entry_escapes_single_quotes_and_uses_relative_path(tmp_path: Path):
    from clipcut.render.renderer import _format_concat_file_entry

    base_dir = tmp_path / "Building Legal Agents When You Can't See the Data" / "remix_tmp"
    base_dir.mkdir(parents=True)
    seg_file = base_dir / "seg_'01'.mp4"
    entry = _format_concat_file_entry(seg_file, base_dir)
    assert entry == r"file 'seg_'\''01'\''.mp4'"

    outside_file = tmp_path / "other" / "it's_cool.mp4"
    outside_file.parent.mkdir(parents=True)
    outside_entry = _format_concat_file_entry(outside_file, base_dir)
    assert r"'\''" in outside_entry


def test_select_curated_top_clips_covers_editorial_pillars(tmp_path: Path):
    from clipcut.models import ClipItem, MediaSource, ProjectManifest, VideoMetadata
    from clipcut.pipeline.verify_and_plan import select_curated_top_clips

    clips = [
        ClipItem(clip_id="c01", index=1, enabled=True, title="开场介绍客套", summary="团队背景介绍", score=7.8, start_utt=0, end_utt=5, start_s=0, end_s=50),
        ClipItem(clip_id="c02", index=2, enabled=True, title="律师要可靠结果而非模型名", categories=["exciting_dialogue"], summary="激烈反驳模型选择器", score=8.9, start_utt=6, end_utt=10, start_s=50, end_s=150),
        ClipItem(clip_id="c03", index=3, enabled=True, title="初级律师不会简单消失", summary="按小时计费与合伙制危机辩论", score=8.9, start_utt=11, end_utt=15, start_s=150, end_s=250),
        ClipItem(clip_id="c04", index=4, enabled=True, title="看不到客户机密数据如何做AI", summary="无数据下的产品优化指标", score=9.2, start_utt=16, end_utt=20, start_s=250, end_s=350),
        ClipItem(clip_id="c05", index=5, enabled=True, title="前线部署工程师FDE落地", categories=["hot_topic"], summary="FDE入驻客户实操落地", score=9.0, start_utt=21, end_utt=25, start_s=350, end_s=450),
        ClipItem(clip_id="c06", index=6, enabled=True, title="常规文字校对流程", summary="转写校对小细节", score=7.6, start_utt=26, end_utt=30, start_s=450, end_s=500),
    ]

    manifest = ProjectManifest(
        project_id="curation-test",
        source=MediaSource(work_dir=str(tmp_path), render_video="v.mp4", analysis_video="v.mp4", output_dir=str(tmp_path / "out"), duration_s=500, width=1920, height=1080, fps=30),
        metadata=VideoMetadata(original_title="Building Legal Agents When You Can't See the Data", description="", channel="", guest_zh="", channel_zh="", original_theme_zh="", metadata_source="manual"),
        clips=clips,
    )

    curated = select_curated_top_clips(manifest, n=5)
    assert len(curated) == 5
    curated_ids = [c.clip_id for c, _ in curated]

    # Verify key editorial pillars are captured:
    # 1. c04 (Theme): 看不到客户机密数据
    assert "c04" in curated_ids
    # 2. c03 (Controversy): 初级律师不会消失/计费合伙制
    assert "c03" in curated_ids
    # 3. c02 (Exciting): 激烈交锋
    assert "c02" in curated_ids
    # 4. c05 (Practice/FDE): 前线部署
    assert "c05" in curated_ids
    # Not chronological c01-c05; low value c01 should NOT crowd out high value ones
    tags = {c.clip_id: tag for c, tag in curated}
    assert "核心主题" in tags["c04"]
    assert "行业争议" in tags["c03"]
    assert "精彩交锋" in tags["c02"]
    assert "硬核实操" in tags["c05"]


def test_compilation_render_hash_tracks_enabled_clips_and_cards(tmp_path: Path):
    from clipcut.models import ClipItem, CompilationConfig, MediaSource, ProjectManifest, VideoMetadata
    from clipcut.pipeline.verify_and_plan import compute_compilation_render_hash

    manifest = ProjectManifest(
        project_id="comp-hash-test",
        source=MediaSource(
            work_dir=str(tmp_path),
            render_video="v.mp4",
            analysis_video="v.mp4",
            output_dir=str(tmp_path / "out"),
            duration_s=100.0,
            width=1920,
            height=1080,
            fps=30.0,
        ),
        metadata=VideoMetadata(
            original_title="Test",
            description="",
            channel="",
            guest_zh="",
            channel_zh="",
            original_theme_zh="",
            metadata_source="manual",
        ),
        clips=[
            ClipItem(clip_id="c1", index=1, enabled=True, title="Part A", start_utt=0, end_utt=1, start_s=0.0, end_s=10.0),
            ClipItem(clip_id="c2", index=2, enabled=True, title="Part B", start_utt=1, end_utt=2, start_s=10.0, end_s=20.0),
        ],
        compilation=CompilationConfig(enabled=True, title="精选集", card_duration_s=1.8),
    )

    h1 = compute_compilation_render_hash(manifest)
    manifest.clips[1].enabled = False
    h2 = compute_compilation_render_hash(manifest)
    assert h1 != h2

    manifest.clips[1].enabled = True
    manifest.compilation.card_duration_s = 2.5
    h3 = compute_compilation_render_hash(manifest)
    assert h3 != h1

    manifest.compilation.card_duration_s = 1.8
    manifest.clips[0].title = "Part A Edited"
    h4 = compute_compilation_render_hash(manifest)
    assert h4 != h1








