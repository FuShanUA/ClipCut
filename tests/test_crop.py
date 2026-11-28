from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from clipcut.api.server import app
import clipcut.api.server as srv
from clipcut.crop import (
    crop_output_size,
    crop_rect_to_pixels,
    crop_vf_expr,
    resolve_effective_crop,
    sanitize_crop_rect,
)
from clipcut.models import (
    ClipCropOverride,
    ClipItem,
    CropConfig,
    CropRect,
    MediaSource,
    ProjectManifest,
    RemixConfig,
    RemixItem,
    Utterance,
    VideoMetadata,
)
from clipcut.pipeline.content_filter import compute_clean_cut_render_hash, sync_all_render_hashes
from clipcut.pipeline.verify_and_plan import (
    compute_clip_render_hash,
    compute_compilation_render_hash,
    compute_remix_render_hash,
)
from clipcut.receipts import ProjectStore
from clipcut.render.frame_selector import FaceBox
from clipcut.render.renderer import (
    _crop_pil_image_and_faces,
    _probe_video_wh,
    _render_clean_cut,
    _render_compilation,
    _render_remix,
    _render_single_clip,
)

client = TestClient(app)


def _make_manifest(tmp_path: Path, video_path: Path | None = None) -> ProjectManifest:
    vpath = str(video_path) if video_path is not None else "fake.mp4"
    return ProjectManifest(
        project_id="test-crop-01",
        source=MediaSource(
            work_dir=str(tmp_path),
            render_video=vpath,
            analysis_video=vpath,
            output_dir=str(tmp_path / "out"),
            duration_s=2.5,
            width=640,
            height=360,
            fps=30.0,
        ),
        metadata=VideoMetadata(
            original_title="Crop Test Video",
            channel="TestCh",
            channel_zh="测试频道",
            guest_zh="嘉宾",
            original_theme_zh="画面裁剪测试",
            published_date="2026-11-28",
            metadata_source="manual",
        ),
        utterances=[
            Utterance(id=0, start_s=0.0, end_s=0.8, zh="第一句", en="First"),
            Utterance(id=1, start_s=0.8, end_s=1.6, zh="第二句", en="Second"),
            Utterance(id=2, start_s=1.6, end_s=2.4, zh="第三句", en="Third"),
        ],
        clips=[
            ClipItem(
                clip_id="c1",
                index=1,
                enabled=True,
                title="切片一",
                slug="clip1",
                start_utt=0,
                end_utt=0,
                start_s=0.0,
                end_s=0.8,
            ),
            ClipItem(
                clip_id="c2",
                index=2,
                enabled=True,
                title="切片二",
                slug="clip2",
                start_utt=1,
                end_utt=1,
                start_s=0.8,
                end_s=1.6,
            ),
            ClipItem(
                clip_id="c3",
                index=3,
                enabled=True,
                title="切片三",
                slug="clip3",
                start_utt=2,
                end_utt=2,
                start_s=1.6,
                end_s=2.4,
            ),
        ],
        remix=RemixConfig(
            enabled=True,
            title="裁剪混剪",
            card_mode="none",
            enable_bgm=False,
            enable_sfx=False,
            items=[
                RemixItem(
                    remix_id="r1",
                    order=1,
                    enabled=True,
                    source_clip_id="c1",
                    start_utt=0,
                    end_utt=0,
                    start_s=0.0,
                    end_s=0.8,
                )
            ],
        ),
    )


def _synth_mp4(path: Path, width: int = 640, height: int = 360, duration: float = 2.5) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={width}x{height}:rate=30:duration={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate=44100:duration={duration}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_crop_utils_and_precedence(tmp_path: Path):
    sx, sy, sw, sh = sanitize_crop_rect(-0.2, 0.98, 2.0, 0.01)
    assert 0.0 <= sx <= 0.95
    assert 0.0 <= sy <= 0.95
    assert sw >= 0.05 and sx + sw <= 1.0001
    assert sh >= 0.05 and sy + sh <= 1.0001

    assert crop_rect_to_pixels(CropRect(x=0.0, y=0.0, w=1.0, h=1.0), 1920, 1080) is None
    px, py, pw, ph = crop_rect_to_pixels(CropRect(x=0.01, y=0.0, w=0.503, h=0.8), 1921, 1081)  # type: ignore[misc]
    assert px % 2 == 0 and py % 2 == 0 and pw % 2 == 0 and ph % 2 == 0
    assert pw == 966 and ph == 864
    assert crop_vf_expr(CropRect(x=0.0, y=0.0, w=1.0, h=0.8), 640, 360) == "crop=640:288:0:0"
    assert crop_output_size(CropRect(x=0.0, y=0.0, w=1.0, h=0.8), 640, 360) == (640, 288)

    m = _make_manifest(tmp_path)
    assert resolve_effective_crop(m) is None
    assert resolve_effective_crop(m, m.clips[0]) is None

    m.crop = CropConfig(enabled=True, x=0.0, y=0.0, w=1.0, h=0.8)
    assert resolve_effective_crop(m) == CropRect(x=0.0, y=0.0, w=1.0, h=0.8)
    assert resolve_effective_crop(m, m.clips[0]) == CropRect(x=0.0, y=0.0, w=1.0, h=0.8)

    m.clips[1].crop_override = ClipCropOverride(mode="none")
    assert resolve_effective_crop(m, m.clips[1]) is None

    m.clips[2].crop_override = ClipCropOverride(mode="custom", x=0.1, y=0.1, w=0.75, h=0.6667)
    eff2 = resolve_effective_crop(m, m.clips[2])
    assert eff2 is not None and eff2.w == 0.75


def test_render_hashes_backward_compatible_and_reactive(tmp_path: Path):
    m = _make_manifest(tmp_path)
    legacy_c1 = compute_clip_render_hash(m.clips[0], m.source.render_video, [], "")
    legacy_c2 = compute_clip_render_hash(m.clips[1], m.source.render_video, [], "")
    legacy_comp = compute_compilation_render_hash(m)
    legacy_rmx = compute_remix_render_hash(m)
    legacy_cc = compute_clean_cut_render_hash(m)

    assert compute_clip_render_hash(m.clips[0], m.source.render_video, [], "", crop_rect=None) == legacy_c1

    # Give c2 custom override and c3 none before enabling global crop
    m.clips[1].crop_override = ClipCropOverride(mode="custom", x=0.1, y=0.0, w=0.75, h=0.6667)
    m.clips[2].crop_override = ClipCropOverride(mode="none")
    sync_all_render_hashes(m)
    c2_custom_hash = m.clips[1].render_hash
    c3_none_hash = m.clips[2].render_hash
    assert c3_none_hash == compute_clip_render_hash(m.clips[2], m.source.render_video, [], "")

    m.crop = CropConfig(enabled=True, x=0.0, y=0.0, w=1.0, h=0.8)
    sync_all_render_hashes(m)
    assert m.clips[0].render_hash != legacy_c1
    assert m.clips[1].render_hash == c2_custom_hash
    assert m.clips[2].render_hash == c3_none_hash
    assert m.compilation.render_hash != legacy_comp
    assert m.remix.render_hash != legacy_rmx
    assert m.clean_cut.render_hash != legacy_cc

    # Changing c2 override invalidates c2 + compilation without touching c1 or clean_cut
    prev_c1 = m.clips[0].render_hash
    prev_cc = m.clean_cut.render_hash
    prev_comp = m.compilation.render_hash
    m.clips[1].crop_override = ClipCropOverride(mode="custom", x=0.05, y=0.05, w=0.7, h=0.6)
    sync_all_render_hashes(m)
    assert m.clips[1].render_hash != c2_custom_hash
    assert m.clips[0].render_hash == prev_c1
    assert m.clean_cut.render_hash == prev_cc
    assert m.compilation.render_hash != prev_comp
    assert legacy_c2 != c2_custom_hash


def test_ffmpeg_render_crop_single_clean_cut_compilation_and_remix(tmp_path: Path, monkeypatch):
    video = tmp_path / "source.mp4"
    _synth_mp4(video, 640, 360, 2.5)
    m = _make_manifest(tmp_path, video)
    m.crop = CropConfig(enabled=True, x=0.0, y=0.0, w=1.0, h=0.8)
    m.clips[1].crop_override = ClipCropOverride(mode="none")
    m.clips[2].crop_override = ClipCropOverride(mode="custom", x=0.125, y=0.1667, w=0.75, h=0.6667)

    store = ProjectStore(tmp_path)
    store.save(m)

    # 1. Single clip (inherits 640x288)
    out_c1 = tmp_path / "out" / "c1.mp4"
    ok, _, err = _render_single_clip(
        video,
        m.clips[0],
        out_c1,
        fade_ms=25,
        use_vt=False,
        crop_vf=crop_vf_expr(resolve_effective_crop(m, m.clips[0]), 640, 360),
    )
    assert ok, err
    assert _probe_video_wh(str(out_c1), 0, 0) == (640, 288)

    # 2. Clean Cut with 0 cut segments & project crop (640x288)
    m.clean_cut.enabled = True
    out_cc = _render_clean_cut(m, store, {"boundary": {"audio_fade_ms": 25}}, use_vt=False, run_id="t_cc", hero_pool=[])
    assert out_cc is not None and out_cc.exists()
    assert _probe_video_wh(str(out_cc), 0, 0) == (640, 288)

    # 3. Compilation landscape with mixed clip crop resolutions (640x288, 640x360, 480x240)
    monkeypatch.setattr(
        "clipcut.render.renderer.resolve_navigation_settings",
        lambda _cfg=None: {
            "style": "none",
            "remix_style": "none",
            "compilation_style": "none",
            "screen_layout": "landscape",
            "color_scheme": "tiffany",
            "custom_accent": "#0ABAB5",
        },
    )
    out_comp = _render_compilation(m, store, {"boundary": {"audio_fade_ms": 25}}, use_vt=False, run_id="t_comp", hero_pool=[])
    assert out_comp is not None and out_comp.exists()
    assert _probe_video_wh(str(out_comp), 0, 0) == (640, 288)

    # 4. Remix landscape with project crop (640x288)
    out_rmx = _render_remix(m, store, {"boundary": {"audio_fade_ms": 25}}, use_vt=False, run_id="t_rmx", hero_pool=[])
    assert out_rmx is not None and out_rmx.exists()
    assert _probe_video_wh(str(out_rmx), 0, 0) == (640, 288)


def test_crop_pil_image_and_faces():
    im = Image.new("RGBA", (1000, 500), (20, 30, 40, 255))
    faces = [
        FaceBox(x=0.2, y=0.2, w=0.1, h=0.2, quality=0.9, is_guest=True),
        FaceBox(x=0.8, y=0.85, w=0.1, h=0.1, quality=0.8),
    ]
    cropped, remapped = _crop_pil_image_and_faces(im, faces, CropRect(x=0.0, y=0.0, w=1.0, h=0.8))
    assert cropped.size == (1000, 400)
    assert len(remapped) == 1
    assert abs(remapped[0].y - 0.25) < 0.01
    assert abs(remapped[0].h - 0.25) < 0.01
    assert remapped[0].is_guest is True


def test_crop_api_endpoints_and_undo(tmp_path: Path, monkeypatch):
    store = ProjectStore(tmp_path)
    m = _make_manifest(tmp_path)
    store.save(m)
    monkeypatch.setattr(srv, "_get_store", lambda _pid: store)

    r1 = client.patch(
        f"/api/projects/{m.project_id}/crop",
        json={"enabled": True, "x": 0.0, "y": 0.0, "w": 1.0, "h": 0.8},
    )
    assert r1.status_code == 200, r1.text
    m1 = r1.json()["manifest"]
    assert m1["crop"]["enabled"] is True
    assert m1["crop"]["h"] == 0.8

    r2 = client.patch(
        f"/api/projects/{m.project_id}/clips/c1",
        json={"crop_override": {"mode": "custom", "x": 0.1, "y": 0.05, "w": 0.8, "h": 0.7}},
    )
    assert r2.status_code == 200, r2.text
    c1 = next(c for c in r2.json()["manifest"]["clips"] if c["clip_id"] == "c1")
    assert c1["crop_override"]["mode"] == "custom"
    assert c1["crop_override"]["w"] == 0.8

    r3 = client.post(f"/api/projects/{m.project_id}/undo")
    assert r3.status_code == 200, r3.text
    c1_undo = next(c for c in r3.json()["manifest"]["clips"] if c["clip_id"] == "c1")
    assert c1_undo["crop_override"] is None

    r4 = client.patch(
        f"/api/projects/{m.project_id}/clips/c2",
        json={"crop_override": {"mode": "none", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}},
    )
    assert r4.status_code == 200
    c2 = next(c for c in r4.json()["manifest"]["clips"] if c["clip_id"] == "c2")
    assert c2["crop_override"]["mode"] == "none"

    r5 = client.patch(
        f"/api/projects/{m.project_id}/clips/c2",
        json={"crop_override": {"mode": "inherit", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}},
    )
    assert r5.status_code == 200
    c2_inh = next(c for c in r5.json()["manifest"]["clips"] if c["clip_id"] == "c2")
    assert c2_inh["crop_override"] is None


def test_subtitle_less_open_crop_and_render(tmp_path: Path, monkeypatch):
    import pytest
    from clipcut.pipeline.analyze import run_analyze_stage
    from clipcut.pipeline.signals import run_signals_stage
    from clipcut.receipts import ProjectStore
    from clipcut.render.renderer import render_clean_cut_and_cover

    media_dir = tmp_path / "nosub_dir"
    media_dir.mkdir(parents=True, exist_ok=True)
    video = media_dir / "raw_video.mp4"
    _synth_mp4(video, 640, 360, 2.0)

    # Isolate registry to tmp_path so test does not mutate workspace .clipcut_registry.json
    monkeypatch.setattr(srv, "REGISTRY_PATH", tmp_path / ".registry.json")

    # 1. Scan directory with no subtitles
    r_scan = client.post("/api/scan", json={"path": str(media_dir)})
    assert r_scan.status_code == 200, r_scan.text
    scan_data = r_scan.json()
    assert len(scan_data["videos"]) == 1
    assert scan_data["videos"][0]["suggested_subtitle"] is None

    # 2. Explicit missing subtitle_path still raises 400
    r_bad = client.post(
        "/api/projects/open",
        json={"video_path": str(video), "subtitle_path": str(media_dir / "missing.srt"), "fetch_online": False},
    )
    assert r_bad.status_code == 400

    # 3. Open video without any subtitle succeeds
    r_open = client.post(
        "/api/projects/open",
        json={"video_path": str(video), "subtitle_path": None, "fetch_online": False},
    )
    assert r_open.status_code == 200, r_open.text
    open_data = r_open.json()
    pid = open_data["project_id"]
    m_open = open_data["manifest"]
    assert m_open["utterances"] == []
    assert m_open["source"]["subtitle_path"] is None
    assert m_open["stages"]["ingest"]["status"] == "completed"

    # 4. Global crop auto-enables clean_cut when clips == []
    r_crop = client.patch(
        f"/api/projects/{pid}/crop",
        json={"enabled": True, "x": 0.0, "y": 0.0, "w": 1.0, "h": 0.8},
    )
    assert r_crop.status_code == 200, r_crop.text
    m_crop = r_crop.json()["manifest"]
    assert m_crop["crop"]["enabled"] is True
    assert m_crop["clean_cut"]["enabled"] is True

    # 5. Render cropped video and cover without subtitles
    store = srv._get_store(pid)
    manifest = store.load()
    monkeypatch.setattr("clipcut.render.renderer._has_videotoolbox", lambda: False)
    ok, res_path = render_clean_cut_and_cover(manifest, store, {"boundary": {"audio_fade_ms": 25}})
    assert ok, res_path
    out_mp4 = Path(res_path)
    assert out_mp4.exists()
    assert _probe_video_wh(str(out_mp4), 0, 0) == (640, 288)
    updated_manifest = store.load()
    assert updated_manifest.clean_cut.cover_file is not None
    assert Path(updated_manifest.clean_cut.cover_file).exists()

    # 6. Subtitle-dependent stages raise clear ValueError on subtitle-less manifest
    with pytest.raises(ValueError, match="未找到字幕文件"):
        run_signals_stage(updated_manifest, store, "t_sig")
    with pytest.raises(ValueError, match="未找到字幕文件"):
        run_analyze_stage(updated_manifest, store, {}, "t_ana")
