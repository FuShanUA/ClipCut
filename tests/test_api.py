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



