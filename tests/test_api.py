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

