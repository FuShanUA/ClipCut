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





