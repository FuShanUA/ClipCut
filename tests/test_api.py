from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from clipcut.api.server import app

client = TestClient(app)


def test_health_and_editorial_yaml():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    cfg_r = client.get("/api/config/editorial")
    assert cfg_r.status_code == 200
    assert "Palantir" in cfg_r.json()["raw_yaml"]

    root_r = client.get("/")
    assert root_r.status_code == 200
    assert "ClipCut Studio" in root_r.text
    assert "no-store" in root_r.headers.get("cache-control", "").lower()


def test_models_catalog_and_selection():
    r = client.get("/api/models")
    assert r.status_code == 200
    data = r.json()
    assert data["selected"]["text_provider"] == "dashscope"
    assert data["selected"]["multimodal_provider"] == "dashscope"
    assert len(data["catalog"]["text"]) >= 5
    assert len(data["catalog"]["multimodal"]) >= 4


def test_scan_endpoint_multi_video(tmp_path: Path):
    (tmp_path / "demo.mp4").write_bytes(b"0" * 512)
    (tmp_path / "demo_hardsub.mp4").write_bytes(b"0" * 1024)
    (tmp_path / "demo.bi.srt").write_text("1\n00:00:00,000 --> 00:00:02,000\nHello\n你好\n", encoding="utf-8")

    r = client.post("/api/scan", json={"path": str(tmp_path)})
    assert r.status_code == 200
    data = r.json()
    assert data["is_directory"] is True
    assert data["need_selection"] is True
    assert len(data["videos"]) == 2


def test_publish_status_endpoint():
    r = client.get("/api/publish/status")
    assert r.status_code == 200
    data = r.json()
    platforms = data["platforms"]
    for key in ("wechat", "bilibili", "xiaohongshu"):
        assert key in platforms


def test_batch_clips_endpoint(tmp_path: Path, monkeypatch):
    from clipcut.models import ProjectManifest, MediaSource, VideoMetadata, ClipItem
    from clipcut.receipts import ProjectStore
    import clipcut.api.server as srv

    store = ProjectStore(tmp_path)
    m = ProjectManifest(
        project_id="test-batch-01",
        source=MediaSource(
            work_dir=str(tmp_path),
            render_video="fake.mp4",
            analysis_video="fake.mp4",
            output_dir=str(tmp_path / "out"),
            duration_s=100.0,
            width=1920,
            height=1080,
            fps=30.0,
        ),
        metadata=VideoMetadata(
            original_title="test",
            description="",
            channel="",
            guest_zh="",
            channel_zh="",
            original_theme_zh="",
            metadata_source="manual",
        ),
        clips=[
            ClipItem(clip_id="c1", index=1, enabled=True, title="t1", start_utt=0, end_utt=1, start_s=0.0, end_s=10.0, score=9.0),
            ClipItem(clip_id="c2", index=2, enabled=True, title="t2", start_utt=1, end_utt=2, start_s=10.0, end_s=20.0, score=6.0),
            ClipItem(clip_id="c3", index=3, enabled=True, title="t3", start_utt=2, end_utt=3, start_s=20.0, end_s=30.0, score=8.0),
        ],
    )
    store.save(m)
    monkeypatch.setattr(srv, "_get_store", lambda pid: store)

    # Disable all
    r = client.post("/api/projects/test-batch-01/clips/batch", json={"action": "disable_all"})
    assert r.status_code == 200
    assert r.json()["enabled_count"] == 0

    # Enable all
    r = client.post("/api/projects/test-batch-01/clips/batch", json={"action": "enable_all"})
    assert r.status_code == 200
    assert r.json()["enabled_count"] == 3

    # Top 2 (scores are 9.0 and 8.0 -> c1 and c3)
    r = client.post("/api/projects/test-batch-01/clips/batch", json={"action": "top_n", "n": 2})
    assert r.status_code == 200
    assert r.json()["enabled_count"] == 2
    clips = r.json()["manifest"]["clips"]
    assert clips[0]["enabled"] is True   # c1 (score 9.0)
    assert clips[1]["enabled"] is False  # c2 (score 6.0)
    assert clips[2]["enabled"] is True   # c3 (score 8.0)


def test_dual_remix_and_compilation_endpoints(tmp_path: Path, monkeypatch):
    from clipcut.models import ProjectManifest, MediaSource, VideoMetadata, ClipItem
    from clipcut.receipts import ProjectStore
    import clipcut.api.server as srv

    store = ProjectStore(tmp_path)
    m = ProjectManifest(
        project_id="test-dual-remix-01",
        source=MediaSource(
            work_dir=str(tmp_path),
            render_video="fake.mp4",
            analysis_video="fake.mp4",
            output_dir=str(tmp_path / "out"),
            duration_s=120.0,
            width=1920,
            height=1080,
            fps=30.0,
        ),
        metadata=VideoMetadata(
            original_title="test dual",
            description="",
            channel="",
            guest_zh="",
            channel_zh="",
            original_theme_zh="",
            metadata_source="manual",
        ),
        clips=[
            ClipItem(clip_id="c1", index=1, enabled=True, title="t1", start_utt=0, end_utt=1, start_s=0.0, end_s=10.0),
            ClipItem(clip_id="c2", index=2, enabled=True, title="t2", start_utt=1, end_utt=2, start_s=10.0, end_s=20.0),
        ],
    )
    store.save(m)
    monkeypatch.setattr(srv, "_get_store", lambda pid: store)

    # Toggle compilation config independently
    r = client.patch(
        "/api/projects/test-dual-remix-01/compilation",
        json={"enabled": True, "title": "我的Top2精选", "cover_title": "精选合辑", "card_duration_s": 2.5},
    )
    assert r.status_code == 200
    comp = r.json()["manifest"]["compilation"]
    assert comp["enabled"] is True
    assert comp["title"] == "我的Top2精选"
    assert comp["cover_title"] == "精选合辑"
    assert comp["card_duration_s"] == 2.5
    assert comp["render_hash"] != ""
    hash_before = comp["render_hash"]

    # Toggle remix enabled off while compilation remains enabled
    r2 = client.patch("/api/projects/test-dual-remix-01/remix", json={"enabled": False})
    assert r2.status_code == 200
    assert r2.json()["manifest"]["remix"]["enabled"] is False
    assert r2.json()["manifest"]["compilation"]["enabled"] is True

    # Toggle compilation enabled off and update card duration -> hash updates
    r3 = client.patch(
        "/api/projects/test-dual-remix-01/compilation",
        json={"enabled": False, "card_duration_s": 1.2},
    )
    assert r3.status_code == 200
    comp3 = r3.json()["manifest"]["compilation"]
    assert comp3["enabled"] is False
    assert comp3["card_duration_s"] == 1.2
    assert comp3["render_hash"] != hash_before


def test_stale_running_publish_record_recovered_on_get_project(tmp_path: Path, monkeypatch):
    from clipcut.models import ProjectManifest, MediaSource, VideoMetadata, ClipItem, PublishRecord
    from clipcut.receipts import ProjectStore
    import clipcut.api.server as srv

    store = ProjectStore(tmp_path)
    m = ProjectManifest(
        project_id="test-stale-pub-01",
        source=MediaSource(
            work_dir=str(tmp_path),
            render_video="fake.mp4",
            analysis_video="fake.mp4",
            output_dir=str(tmp_path / "out"),
            duration_s=60.0,
            width=1920,
            height=1080,
            fps=30.0,
        ),
        metadata=VideoMetadata(
            original_title="test stale",
            description="",
            channel="",
            guest_zh="",
            channel_zh="",
            original_theme_zh="",
            metadata_source="manual",
        ),
        clips=[
            ClipItem(
                clip_id="c1",
                index=1,
                enabled=True,
                title="t1",
                start_utt=0,
                end_utt=1,
                start_s=0.0,
                end_s=10.0,
                publish_records=[
                    PublishRecord(
                        platform="wechat",
                        status="running",
                        mode="draft",
                        title="t1",
                        updated_at="2020-01-01T00:00:00+00:00",
                        progress_pct=42.0,
                        progress_stage="上传中",
                        message="微信视频上传中 42%",
                    )
                ],
            )
        ],
    )
    store.save(m)
    monkeypatch.setattr(srv, "_get_store", lambda pid: store)

    r = client.get("/api/projects/test-stale-pub-01")
    assert r.status_code == 200
    rec = r.json()["manifest"]["clips"][0]["publish_records"][0]
    assert rec["status"] == "error"
    assert rec["progress_pct"] is None
    assert "服务重启" in rec["message"]


def test_reorder_and_reset_clips(tmp_path, monkeypatch):
    from clipcut.api import server as srv
    from clipcut.models import ClipItem, MediaSource, ProjectManifest
    from clipcut.receipts import ProjectStore
    store = ProjectStore(tmp_path / "work")
    m = ProjectManifest(
        project_id="test-reorder-01",
        source=MediaSource(
            analysis_video=str(tmp_path / "source.mp4"),
            render_video=str(tmp_path / "source.mp4"),
            duration_s=100.0,
            work_dir=str(tmp_path / "work"),
            output_dir=str(tmp_path / "clips"),
            metadata_source="manual",
        ),
        clips=[
            ClipItem(clip_id="c1", index=1, enabled=True, title="t1", start_utt=0, end_utt=5, start_s=0.0, end_s=10.0),
            ClipItem(clip_id="c2", index=2, enabled=True, title="t2", start_utt=6, end_utt=10, start_s=12.0, end_s=25.0),
            ClipItem(clip_id="c3", index=3, enabled=True, title="t3", start_utt=11, end_utt=15, start_s=30.0, end_s=50.0),
        ],
    )
    store.save(m)
    monkeypatch.setattr(srv, "_get_store", lambda pid: store)

    # Reorder c3 -> c1 -> c2
    r_reorder = client.post("/api/projects/test-reorder-01/clips/reorder", json={"clip_ids": ["c3", "c1", "c2"]})
    assert r_reorder.status_code == 200
    clips = r_reorder.json()["manifest"]["clips"]
    assert [c["clip_id"] for c in clips] == ["c3", "c1", "c2"]
    assert [c["index"] for c in clips] == [1, 2, 3]

    # Reset back to timeline order
    r_reset = client.post("/api/projects/test-reorder-01/clips/reset-order")
    assert r_reset.status_code == 200
    reset_clips = r_reset.json()["manifest"]["clips"]
    assert [c["clip_id"] for c in reset_clips] == ["c1", "c2", "c3"]
    assert [c["index"] for c in reset_clips] == [1, 2, 3]


def test_reorder_reset_and_batch_sync_remix_hashes(tmp_path, monkeypatch):
    from clipcut.api import server as srv
    from clipcut.models import ClipItem, MediaSource, ProjectManifest, RemixConfig, RemixItem
    from clipcut.pipeline.verify_and_plan import (
        compute_compilation_render_hash,
        compute_remix_render_hash,
        compute_remix_source_signature,
    )
    from clipcut.receipts import ProjectStore

    store = ProjectStore(tmp_path / "work")
    m = ProjectManifest(
        project_id="test-sync-hashes-01",
        source=MediaSource(
            analysis_video=str(tmp_path / "source.mp4"),
            render_video=str(tmp_path / "source.mp4"),
            duration_s=100.0,
            work_dir=str(tmp_path / "work"),
            output_dir=str(tmp_path / "clips"),
            metadata_source="manual",
        ),
        clips=[
            ClipItem(clip_id="c1", index=1, enabled=True, title="t1", start_utt=0, end_utt=5, start_s=0.0, end_s=10.0),
            ClipItem(clip_id="c2", index=2, enabled=True, title="t2", start_utt=6, end_utt=10, start_s=12.0, end_s=25.0),
            ClipItem(clip_id="c3", index=3, enabled=True, title="t3", start_utt=11, end_utt=15, start_s=30.0, end_s=50.0),
        ],
        remix=RemixConfig(
            enabled=True,
            items=[
                RemixItem(remix_id="rmx_01", order=1, source_clip_id="c1", card_title="段1", start_utt=0, end_utt=2, start_s=0.0, end_s=5.0),
                RemixItem(remix_id="rmx_02", order=2, source_clip_id="c2", card_title="段2", start_utt=6, end_utt=8, start_s=12.0, end_s=18.0),
                RemixItem(remix_id="rmx_03", order=3, source_clip_id="c3", card_title="段3", start_utt=11, end_utt=13, start_s=30.0, end_s=36.0),
            ],
        ),
    )
    m.remix.planned_source_signature = compute_remix_source_signature(m)
    m.remix.render_hash = compute_remix_render_hash(m)
    m.remix.last_rendered_hash = m.remix.render_hash
    m.compilation.render_hash = compute_compilation_render_hash(m)
    store.save(m)
    monkeypatch.setattr(srv, "_get_store", lambda pid: store)

    orig_rmx_hash = m.remix.render_hash
    orig_cmp_hash = m.compilation.render_hash

    # Reorder c3 -> c1 -> c2: remix.items should follow [c3, c1, c2]
    r_reorder = client.post("/api/projects/test-sync-hashes-01/clips/reorder", json={"clip_ids": ["c3", "c1", "c2"]})
    assert r_reorder.status_code == 200
    rm1 = r_reorder.json()["manifest"]
    assert [it["source_clip_id"] for it in rm1["remix"]["items"]] == ["c3", "c1", "c2"]
    assert [it["order"] for it in rm1["remix"]["items"]] == [1, 2, 3]
    assert rm1["remix"]["render_hash"] != orig_rmx_hash
    assert rm1["compilation"]["render_hash"] != orig_cmp_hash

    # Reset order: remix.items should restore to [c1, c2, c3]
    r_reset = client.post("/api/projects/test-sync-hashes-01/clips/reset-order")
    assert r_reset.status_code == 200
    rm2 = r_reset.json()["manifest"]
    assert [it["source_clip_id"] for it in rm2["remix"]["items"]] == ["c1", "c2", "c3"]
    assert [it["order"] for it in rm2["remix"]["items"]] == [1, 2, 3]

    # Simulate legacy project with empty planned_source_signature, then batch disable_all
    loaded = store.load()
    loaded.remix.planned_source_signature = ""
    loaded.remix.render_hash = compute_remix_render_hash(loaded)
    loaded.remix.last_rendered_hash = loaded.remix.render_hash
    old_comp_hash = loaded.compilation.render_hash
    store.save(loaded)

    r_batch = client.post("/api/projects/test-sync-hashes-01/clips/batch", json={"action": "disable_all"})
    assert r_batch.status_code == 200
    rm3 = r_batch.json()["manifest"]
    assert rm3["remix"]["planned_source_signature"] == "__stale__"
    assert rm3["remix"]["render_hash"] != rm3["remix"]["last_rendered_hash"]
    assert rm3["compilation"]["render_hash"] != old_comp_hash


def test_subtitle_repair_endpoint(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    from clipcut.api import server as srv
    from clipcut.models import ClipItem, MediaSource, ProjectManifest
    from clipcut.receipts import ProjectStore

    store = ProjectStore(tmp_path / "work")
    manifest = ProjectManifest(
        project_id="test-subtitle-repair-01",
        source=MediaSource(
            analysis_video=str(tmp_path / "source.mp4"),
            render_video=str(tmp_path / "source.mp4"),
            work_dir=str(tmp_path / "work"),
            output_dir=str(tmp_path / "clips"),
            metadata_source="manual",
        ),
        clips=[
            ClipItem(
                clip_id="c1",
                index=1,
                enabled=True,
                title="t1",
                start_utt=0,
                end_utt=1,
                start_s=0.0,
                end_s=10.0,
            )
        ],
    )
    store.save(manifest)
    monkeypatch.setattr(srv, "_get_store", lambda pid: store)

    outcome = SimpleNamespace(
        changed_utterance_ids=[1],
        bi_srt_path=tmp_path / "a.bi.srt",
        cn_srt_path=tmp_path / "a.cn.srt",
        en_srt_path=tmp_path / "a.srt",
        ass_path=tmp_path / "a.ass",
        clip_output_path=tmp_path / "clips" / "c1.mp4",
        full_hardsub_path=None,
    )

    def fake_apply(*args, **kwargs):
        return outcome

    monkeypatch.setattr(srv, "apply_subtitle_repairs", fake_apply)
    r = client.post(
        "/api/projects/test-subtitle-repair-01/clips/c1/subtitle-repair",
        json={
            "repairs": [{"utterance_id": 1, "zh": "修正", "en": "Fixed"}],
            "rebuild_full_subtitle": True,
            "render_clip": True,
            "render_full_video": False,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["changed_utterance_ids"] == [1]
    assert data["clip_output_path"].endswith("c1.mp4")
    assert data["full_hardsub_path"] is None


def test_rendered_assets_and_replan_strategy_endpoints(tmp_path, monkeypatch):
    from clipcut.api import server as srv
    from clipcut.models import (
        CleanCutConfig,
        ClipItem,
        MediaSource,
        ProjectManifest,
        RemixConfig,
        RemixItem,
        StageStatus,
        TopicMapItem,
    )
    from clipcut.pipeline.orchestrator import reset_project_strategy
    from clipcut.receipts import ProjectStore

    src_dir = tmp_path / "media"
    out_dir = src_dir / "clips"
    work_dir = tmp_path / "work"
    src_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    src_vid = src_dir / "source_hardsub.mp4"
    src_vid.write_bytes(b"SOURCE_VIDEO_DATA")
    src_srt = src_dir / "source.bi.srt"
    src_srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nhi\n你好\n", encoding="utf-8")

    # Populate work_dir cache & rendered outputs in out_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "audio_envelope.npz").write_bytes(b"npz")
    (work_dir / "cover_candidates").mkdir()
    (work_dir / "cover_candidates" / "c1.jpg").write_bytes(b"cand")
    (work_dir / "remix_tmp").mkdir()
    (work_dir / "remix_tmp" / "card_01.mp4").write_bytes(b"card")

    clip_mp4 = out_dir / "01_demo.mp4"
    clip_jpg = out_dir / "01_demo.jpg"
    clip_wx = out_dir / "01_demo.cover.wechat.jpg"
    remix_mp4 = out_dir / "00_Remix_demo.mp4"
    remix_jpg = out_dir / "00_Remix_demo.jpg"
    doc_md = out_dir / "clips_index.md"
    for f, data in [
        (clip_mp4, b"1" * 100),
        (clip_jpg, b"2" * 40),
        (clip_wx, b"3" * 30),
        (remix_mp4, b"4" * 200),
        (remix_jpg, b"5" * 50),
        (doc_md, b"# index"),
    ]:
        f.write_bytes(data)

    store = ProjectStore(work_dir)
    m = ProjectManifest(
        project_id="test-replan-01",
        source=MediaSource(
            analysis_video=str(src_vid),
            render_video=str(src_vid),
            subtitle_path=str(src_srt),
            duration_s=120.0,
            work_dir=str(work_dir),
            output_dir=str(out_dir),
        ),
        primary_topics=["AI"],
        topic_map=[TopicMapItem(topic_id="T01", start_utt=0, end_utt=1, title="话题1", summary="概述")],
        clips=[
            ClipItem(
                clip_id="clip_01",
                index=1,
                title="切片1",
                start_utt=0,
                end_utt=1,
                start_s=0.0,
                end_s=50.0,
                output_file=str(clip_mp4),
                cover_file=str(clip_jpg),
                render_hash="h1",
                last_rendered_hash="h1",
            )
        ],
        remix=RemixConfig(
            enabled=True,
            items=[RemixItem(remix_id="rmx_01", order=1, start_utt=0, end_utt=1, start_s=0.0, end_s=10.0)],
            output_file=str(remix_mp4),
            cover_file=str(remix_jpg),
            render_hash="rh1",
            last_rendered_hash="rh1",
        ),
        clean_cut=CleanCutConfig(output_file=str(out_dir / "00_clean.mp4"), last_rendered_hash="ch1"),
    )
    for st in ("ingest", "signals", "scan_cuts", "analyze", "verify_and_plan", "render"):
        m.stages[st] = StageStatus(status="completed", message="完成")
    store.save(m)
    monkeypatch.setattr(srv, "_get_store", lambda pid: store)

    # 1. Inspect rendered assets
    r_assets = client.get("/api/projects/test-replan-01/rendered-assets")
    assert r_assets.status_code == 200
    assets = r_assets.json()
    assert assets["has_strategy"] is True
    assert assets["has_rendered"] is True
    assert assets["video_files"] == ["00_Remix_demo.mp4", "01_demo.mp4"]
    assert assets["cover_files"] == ["00_Remix_demo.jpg", "01_demo.cover.wechat.jpg", "01_demo.jpg"]
    assert assets["doc_files"] == ["clips_index.md"]
    assert assets["rendered_clips_count"] == 1
    assert assets["remix_rendered"] is True

    # 2. Stub background worker & run replan while keeping rendered files
    captured_calls: list[list[str]] = []

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    monkeypatch.setattr(srv.threading, "Thread", FakeThread)
    monkeypatch.setattr(
        srv,
        "run_existing_project_stages",
        lambda wd, stages, **kw: captured_calls.append(list(stages)),
    )

    r_keep = client.post(
        "/api/projects/test-replan-01/replan-strategy",
        json={"delete_rendered_files": False},
    )
    assert r_keep.status_code == 200
    d_keep = r_keep.json()
    assert d_keep["started"] is True
    assert d_keep["deleted_count"] == 0
    # Signals already completed + audio_envelope.npz exists -> skipped for speed
    assert d_keep["stages"] == ["scan_cuts", "analyze", "verify_and_plan"]
    assert captured_calls == [["scan_cuts", "analyze", "verify_and_plan"]]
    assert clip_mp4.exists() and remix_mp4.exists()
    assert d_keep["manifest"]["clips"] == []
    assert d_keep["manifest"]["stages"]["render"]["status"] == "completed"

    # Undo back to restore rendered files metadata, then test delete_rendered_files=True
    r_undo = client.post("/api/projects/test-replan-01/undo")
    assert r_undo.status_code == 200
    assert len(r_undo.json()["manifest"]["clips"]) == 1

    r_del = client.post(
        "/api/projects/test-replan-01/replan-strategy",
        json={"delete_rendered_files": True},
    )
    assert r_del.status_code == 200
    d_del = r_del.json()
    assert d_del["deleted_count"] == 6
    assert not clip_mp4.exists()
    assert not clip_wx.exists()
    assert not remix_mp4.exists()
    assert not doc_md.exists()
    assert not (work_dir / "cover_candidates").exists()
    assert not (work_dir / "remix_tmp").exists()
    # Original source video, subtitle & audio envelope MUST remain untouched
    assert src_vid.exists() and src_srt.exists() and (work_dir / "audio_envelope.npz").exists()
    assert d_del["manifest"]["stages"]["render"]["status"] == "idle"
    assert d_del["manifest"]["clean_cut"]["output_file"] is None

    # 3. Safety guard: if output_dir somehow equals render_video parent, NEVER delete files there
    m_bad = store.load()
    m_bad.source.output_dir = str(src_dir)
    stray = src_dir / "other.mp4"
    stray.write_bytes(b"OTHER")
    res_safe = reset_project_strategy(m_bad, store, delete_rendered_files=True, run_id="safe_test")
    assert res_safe["deleted_count"] == 0
    assert src_vid.exists() and stray.exists()


def test_cancel_project_job_endpoint_idempotent_and_interrupts_running_job(tmp_path, monkeypatch):
    import threading
    import time
    from clipcut.api import server as srv
    from clipcut.cancel import check_cancelled
    from clipcut.models import MediaSource, ProjectManifest, Utterance
    import clipcut.pipeline.orchestrator as orch
    from clipcut.receipts import ProjectStore

    work_dir = tmp_path / "work"
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"v")
    store = ProjectStore(work_dir)
    m = ProjectManifest(
        project_id="test-cancel-01",
        source=MediaSource(
            analysis_video=str(vid),
            render_video=str(vid),
            duration_s=100.0,
            work_dir=str(work_dir),
            output_dir=str(tmp_path / "clips"),
        ),
        utterances=[Utterance(id=0, start_s=0.0, end_s=10.0, zh="测试中止。")],
    )
    store.save(m)
    monkeypatch.setattr(srv, "_get_store", lambda pid: store)

    # 1. Idle cancel is idempotent
    r_idle = client.post("/api/projects/test-cancel-01/cancel")
    assert r_idle.status_code == 200
    assert r_idle.json()["cancelled"] is False

    # 2. Launch background analyze job that blocks until cancelled
    entered = threading.Event()

    def slow_analyze(manifest, st, cfg, rid):
        entered.set()
        for _ in range(100):
            check_cancelled()
            time.sleep(0.02)
        return manifest

    monkeypatch.setattr(orch, "run_analyze_stage", slow_analyze)
    r_run = client.post("/api/projects/test-cancel-01/run", json={"stages": ["analyze", "verify_and_plan"]})
    assert r_run.status_code == 200
    assert entered.wait(timeout=2.0)

    r_cancel = client.post("/api/projects/test-cancel-01/cancel")
    assert r_cancel.status_code == 200
    assert r_cancel.json()["cancelled"] is True

    # Poll until worker finishes cleanup
    final_data = None
    for _ in range(50):
        g = client.get("/api/projects/test-cancel-01")
        assert g.status_code == 200
        d = g.json()
        if not d["job"]["running"]:
            final_data = d
            break
        time.sleep(0.02)

    assert final_data is not None
    assert final_data["job"]["running"] is False
    assert final_data["job"]["cancelled"] is True
    assert final_data["job"]["error"] is None
    assert final_data["manifest"]["stages"]["analyze"]["status"] == "idle"
    assert final_data["manifest"]["stages"]["analyze"]["message"] == "已手动中止"


def test_stale_stage_recovery_and_full_from_scratch_replan(tmp_path: Path, monkeypatch):
    from clipcut.api import server as srv
    from clipcut.models import CutSegment, MediaSource, ProjectManifest, StageStatus
    from clipcut.receipts import ProjectStore

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    env_file = work_dir / "audio_envelope.npz"
    env_file.write_bytes(b"npz")

    src_vid = tmp_path / "src.mp4"
    src_vid.write_bytes(b"vid")
    store = ProjectStore(work_dir)
    m = ProjectManifest(
        project_id="test-dhh-like-01",
        source=MediaSource(
            analysis_video=str(src_vid),
            render_video=str(src_vid),
            duration_s=100.0,
            work_dir=str(work_dir),
            output_dir=str(tmp_path / "clips"),
        ),
        cut_segments=[
            CutSegment(cut_id="cut_01", kind="political_sensitive", detector="llm", start_s=10.0, end_s=20.0, reason="ai"),
            CutSegment(cut_id="cut_m", kind="ad", detector="manual", start_s=30.0, end_s=35.0, reason="manual"),
        ],
    )
    m.clean_cut.scanned_at = "2020-01-01T00:00:00+00:00"
    m.clean_cut.cut_ads = True
    m.clean_cut.cut_political_sensitive = True
    m.stages["signals"] = StageStatus(status="completed", message="完成", finished_at="2020-01-01T00:00:00+00:00")
    m.stages["scan_cuts"] = StageStatus(status="completed", message="完成", finished_at="2020-01-01T00:00:01+00:00")
    m.stages["analyze"] = StageStatus(status="running", started_at="2020-01-01T00:00:02+00:00", message="执行中...")
    store.save(m)
    monkeypatch.setattr(srv, "_get_store", lambda pid: store)

    # 1. GET /rendered-assets must treat cut_segments/prior stages (even with clips==[]) as has_strategy=True
    r_assets = client.get("/api/projects/test-dhh-like-01/rendered-assets")
    assert r_assets.status_code == 200
    assert r_assets.json()["has_strategy"] is True

    # 2. GET /api/projects/{pid} auto-recovers stale running analyze stage & includes assets
    r_get = client.get("/api/projects/test-dhh-like-01")
    assert r_get.status_code == 200
    d_get = r_get.json()
    assert d_get["assets"]["has_strategy"] is True
    assert d_get["manifest"]["stages"]["analyze"]["status"] == "idle"
    assert "服务重启中断" in d_get["manifest"]["stages"]["analyze"]["message"]

    # 3. POST /replan-strategy with force_signals=True removes AI cut_segments & audio_envelope.npz, preserves manual cut, runs from signals
    captured: list[list[str]] = []

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    monkeypatch.setattr(srv.threading, "Thread", FakeThread)
    monkeypatch.setattr(srv, "run_existing_project_stages", lambda wd, stages, **kw: captured.append(list(stages)))

    r_replan = client.post(
        "/api/projects/test-dhh-like-01/replan-strategy",
        json={"delete_rendered_files": True, "force_signals": True},
    )
    assert r_replan.status_code == 200
    d_rep = r_replan.json()
    assert d_rep["stages"] == ["signals", "scan_cuts", "analyze", "verify_and_plan"]
    assert captured == [["signals", "scan_cuts", "analyze", "verify_and_plan"]]
    assert not env_file.exists()
    assert d_rep["manifest"]["cut_segments"] == []
    cc = d_rep["manifest"]["clean_cut"]
    assert cc["enabled"] is False
    assert cc["title"] == ""
    assert cc["cover_title"] == ""
    assert cc["render_hash"] == ""
    assert cc["scanned_at"] is None
    assert cc["cut_political_sensitive"] is False
    assert cc["cut_ads"] is False
    assert d_rep["manifest"]["stages"]["signals"]["status"] == "idle"


def test_compilation_independent_clip_ids_and_render_without_standalone_clips(tmp_path: Path, monkeypatch):
    from clipcut.api import server as srv
    from clipcut.models import ClipItem, CompilationConfig, MediaSource, ProjectManifest, RemixConfig, Utterance
    from clipcut.receipts import ProjectStore
    from clipcut.render import renderer

    vid = tmp_path / "src.mp4"
    vid.write_bytes(b"dummy")
    store = ProjectStore(tmp_path / "work")
    m = ProjectManifest(
        project_id="test-comp-indep-01",
        source=MediaSource(
            analysis_video=str(vid),
            render_video=str(vid),
            duration_s=60.0,
            work_dir=str(tmp_path / "work"),
            output_dir=str(tmp_path / "clips"),
            metadata_source="manual",
        ),
        utterances=[Utterance(id=0, start_s=0.0, end_s=10.0, zh="第一句"), Utterance(id=1, start_s=10.0, end_s=20.0, zh="第二句")],
        clips=[
            ClipItem(clip_id="c1", index=1, enabled=False, title="片段一", start_utt=0, end_utt=0, start_s=0.0, end_s=10.0),
            ClipItem(clip_id="c2", index=2, enabled=False, title="片段二", start_utt=1, end_utt=1, start_s=10.0, end_s=20.0),
        ],
        remix=RemixConfig(enabled=False, items=[]),
        compilation=CompilationConfig(enabled=True, clip_ids=["c2"]),
    )
    store.save(m)
    monkeypatch.setattr(srv, "_get_store", lambda pid: store)
    monkeypatch.setattr(renderer, "_has_videotoolbox", lambda: False)
    monkeypatch.setattr(renderer, "_build_hero_pool_for_manifest", lambda *a, **k: [])

    single_clip_calls: list[str] = []
    comp_rendered_ids: list[list[str]] = []

    monkeypatch.setattr(renderer, "_render_single_clip", lambda src, clip, *a, **k: single_clip_calls.append(clip.clip_id) or (True, "libx264", ""))

    def fake_render_comp(manifest, store, cfg, use_vt, run_id, **kw):
        from clipcut.pipeline.verify_and_plan import get_compilation_clips
        ids = [c.clip_id for c in get_compilation_clips(manifest)]
        comp_rendered_ids.append(ids)
        out_p = Path(manifest.source.output_dir) / "00_Compilation_test.mp4"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(b"comp")
        manifest.compilation.output_file = str(out_p)
        manifest.compilation.last_rendered_hash = manifest.compilation.render_hash
        store.save(manifest)
        return out_p

    monkeypatch.setattr(renderer, "_render_compilation", fake_render_comp)

    # All standalone clips are unchecked, yet Compilation renders only its independent clip_ids=["c2"]
    res = renderer.run_render_stage(store.load(), store, {}, "run_comp_only")
    assert single_clip_calls == []
    assert comp_rendered_ids == [["c2"]]
    assert res.compilation.output_file and Path(res.compilation.output_file).exists()

    # API: clearing compilation clip_ids rejects /compilation/render with 400
    r_clr = client.patch("/api/projects/test-comp-indep-01/compilation", json={"clear_clip_ids": True})
    assert r_clr.status_code == 200
    r_bad = client.post("/api/projects/test-comp-indep-01/compilation/render")
    assert r_bad.status_code == 400

    # API: restoring clip_ids=["c1", "c2"] allows background /compilation/render
    r_set = client.patch("/api/projects/test-comp-indep-01/compilation", json={"clip_ids": ["c1", "c2"]})
    assert r_set.status_code == 200

    orig_thread = srv.threading.Thread

    class FakeThread:
        def __init__(self, *args, target=None, daemon=None, **kwargs):
            self._real = None if (args == () and not kwargs and callable(target)) else orig_thread(*args, target=target, daemon=daemon, **kwargs)
            self._target = target

        def start(self):
            if self._real is not None:
                return self._real.start()
            if self._target:
                self._target()

    monkeypatch.setattr(srv.threading, "Thread", FakeThread)
    r_ok = client.post("/api/projects/test-comp-indep-01/compilation/render?force=true")
    assert r_ok.status_code == 200
    assert r_ok.json()["started"] is True
    assert store.load().stages["render"].status == "completed"
    assert store.load().stages["render"].error is None
    assert comp_rendered_ids[-1] == ["c1", "c2"]


def test_remix_manual_lock_source_clip_ids_and_standalone_render(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace
    from clipcut.api import server as srv
    from clipcut.models import ClipItem, CompilationConfig, MediaSource, ProjectManifest, RemixConfig, RemixItem, Utterance
    from clipcut.pipeline import verify_and_plan as vp
    from clipcut.receipts import ProjectStore
    from clipcut.render import renderer

    vid = tmp_path / "src.mp4"
    vid.write_bytes(b"dummy")
    store = ProjectStore(tmp_path / "work")
    m = ProjectManifest(
        project_id="test-remix-ctrl-01",
        source=MediaSource(
            analysis_video=str(vid),
            render_video=str(vid),
            duration_s=60.0,
            work_dir=str(tmp_path / "work"),
            output_dir=str(tmp_path / "clips"),
            metadata_source="manual",
        ),
        utterances=[
            Utterance(id=0, start_s=0.0, end_s=10.0, zh="第一句"),
            Utterance(id=1, start_s=12.0, end_s=22.0, zh="第二句"),
        ],
        clips=[
            ClipItem(clip_id="c1", index=1, enabled=True, title="片段一", start_utt=0, end_utt=0, start_s=0.0, end_s=10.0),
            ClipItem(clip_id="c2", index=2, enabled=True, title="片段二", start_utt=1, end_utt=1, start_s=12.0, end_s=22.0),
        ],
        remix=RemixConfig(
            enabled=True,
            planned_source_signature="old_sig",
            items=[
                RemixItem(remix_id="rmx_01", order=1, enabled=True, source_clip_id="c1", card_title="段1", start_utt=0, end_utt=0, start_s=0.0, end_s=10.0),
            ],
        ),
        compilation=CompilationConfig(enabled=False, clip_ids=[]),
    )
    store.save(m)
    monkeypatch.setattr(srv, "_get_store", lambda pid: store)
    monkeypatch.setattr(renderer, "_has_videotoolbox", lambda: False)
    monkeypatch.setattr(renderer, "_build_hero_pool_for_manifest", lambda *a, **k: [])

    # 1. Manually adding c2 sets manual_script_locked=True
    r_add = client.patch("/api/projects/test-remix-ctrl-01/remix", json={"add_clip_id": "c2"})
    assert r_add.status_code == 200
    rmx = r_add.json()["manifest"]["remix"]
    assert rmx["manual_script_locked"] is True
    assert [it["source_clip_id"] for it in rmx["items"]] == ["c1", "c2"]

    # Uncheck first item & uncheck all standalone clips -> manual_script_locked prevents AI auto-replan!
    first_id = rmx["items"][0]["remix_id"]
    r_tog = client.patch("/api/projects/test-remix-ctrl-01/remix", json={"items": [{"remix_id": first_id, "enabled": False}]})
    assert r_tog.status_code == 200
    assert r_tog.json()["manifest"]["remix"]["manual_script_locked"] is True
    client.post("/api/projects/test-remix-ctrl-01/clips/batch", json={"action": "disable_all"})

    plan_called: list[int] = []
    monkeypatch.setattr(renderer, "plan_remix_from_clips", lambda *a, **k: plan_called.append(1) or store.load())

    rendered_remix_ids: list[list[str]] = []

    def fake_render_remix(manifest, store, cfg, use_vt, run_id, **kw):
        enabled_ids = [it.source_clip_id for it in manifest.remix.items if it.enabled]
        rendered_remix_ids.append(enabled_ids)
        out_p = Path(manifest.source.output_dir) / "00_Remix_manual.mp4"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(b"rmx")
        manifest.remix.output_file = str(out_p)
        manifest.remix.last_rendered_hash = manifest.remix.render_hash
        store.save(manifest)
        return out_p

    monkeypatch.setattr(renderer, "_render_remix", fake_render_remix)

    orig_thread = srv.threading.Thread

    class FakeThread:
        def __init__(self, *args, target=None, daemon=None, **kwargs):
            self._real = None if (args == () and not kwargs and callable(target)) else orig_thread(*args, target=target, daemon=daemon, **kwargs)
            self._target = target

        def start(self):
            if self._real is not None:
                return self._real.start()
            if self._target:
                self._target()

    monkeypatch.setattr(srv.threading, "Thread", FakeThread)

    r_rmx_render = client.post("/api/projects/test-remix-ctrl-01/remix/render?force=true")
    assert r_rmx_render.status_code == 200
    assert store.load().stages["render"].status == "completed"
    assert store.load().stages["render"].error is None
    assert plan_called == []
    assert rendered_remix_ids == [["c2"]]

    # 2. If user disables all remix items, /remix/render returns 400
    second_id = store.load().remix.items[1].remix_id
    client.patch("/api/projects/test-remix-ctrl-01/remix", json={"items": [{"remix_id": second_id, "enabled": False}]})
    r_empty = client.post("/api/projects/test-remix-ctrl-01/remix/render")
    assert r_empty.status_code == 400

    # 3. Explicit source_clip_ids drives plan_remix_from_clips even when all standalone clip.enabled == False
    client.patch("/api/projects/test-remix-ctrl-01/remix", json={"source_clip_ids": ["c2"]})
    seen_clip_ids: list[str] = []

    def fake_llm(self, prompt, **kw):
        if "片段一" in prompt:
            seen_clip_ids.append("c1")
        if "片段二" in prompt:
            seen_clip_ids.append("c2")
        return SimpleNamespace(
            parsed={
                "remix_title": "指定范围混剪",
                "themes": [
                    {
                        "theme_id": "TH01",
                        "group_label": "测试组",
                        "card_title": "仅c2高光",
                        "segments": [
                            {"source_clip_id": "c2", "start_utt": 1, "end_utt": 1, "quote": "第二句", "beat_type": "punchline"}
                        ],
                    }
                ],
            },
            provider="dashscope",
            model="qwen3.8-max",
            prompt_tokens=1,
            output_tokens=1,
            elapsed_ms=1,
        )

    monkeypatch.setattr(vp.ClipCutLLM, "generate_json", fake_llm)
    planned = vp.plan_remix_from_clips(store.load(), store, {}, "run_src_filter")
    assert seen_clip_ids == ["c2"]
    assert [it.source_clip_id for it in planned.remix.items] == ["c2"]
    assert planned.remix.manual_script_locked is False









