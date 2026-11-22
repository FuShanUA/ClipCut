from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from clipcut.api.server import app
import clipcut.api.server as srv
from clipcut.config import load_editorial_config
from clipcut.models import (
    ClipItem,
    CutSegment,
    MediaSource,
    ProjectManifest,
    RemixItem,
    Utterance,
    VideoMetadata,
)
from clipcut.pipeline.content_filter import (
    compute_clean_cut_render_hash,
    get_active_cut_segments,
    get_active_cut_spans,
    scan_cut_segments,
    should_use_precise_copy,
    subtract_cut_intervals,
)
from clipcut.pipeline.verify_and_plan import (
    compute_clip_render_hash,
    plan_remix_from_clips,
)
from clipcut.publish.publishers import _default_target_metadata
from clipcut.receipts import ProjectStore
from clipcut.render.renderer import (
    _render_clean_cut,
    _render_single_clip,
)


def _ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def _make_manifest(tmp_path: Path, video_path: str = "demo.mp4", duration_s: float = 30.0) -> tuple[ProjectManifest, ProjectStore]:
    store = ProjectStore(tmp_path)
    utts = [
        Utterance(id=0, start_s=0.0, end_s=2.0, en="Intro talk.", zh="开场介绍。"),
        Utterance(id=1, start_s=2.2, end_s=5.0, en="Sponsored by FooBar.", zh="本期由某某品牌赞助。"),
        Utterance(id=2, start_s=5.2, end_s=8.0, en="AI enterprise ontology discussion.", zh="企业本体架构核心讨论。"),
        Utterance(id=3, start_s=8.2, end_s=11.0, en="Sensitive geopolitical deterrence.", zh="涉政红线与军事威慑讨论。"),
        Utterance(id=4, start_s=11.2, end_s=15.0, en="Agent deployment practices.", zh="企业智能体落地实践经验。"),
    ]
    m = ProjectManifest(
        project_id="test-clean-01",
        source=MediaSource(
            render_video=video_path,
            analysis_video=video_path,
            output_dir=str(tmp_path / "out"),
            work_dir=str(tmp_path),
            duration_s=duration_s,
            width=640,
            height=360,
            fps=24.0,
        ),
        metadata=VideoMetadata(
            original_title="Enterprise AI Keynote",
            original_theme_zh="企业智能体实战",
            channel="DemoPod",
            channel_zh="科技播客",
            guest_zh="张三",
            metadata_source="manual",
        ),
        utterances=utts,
    )
    store.save(m)
    return m, store


def test_interval_helpers_and_mode_selection(tmp_path: Path):
    m, _ = _make_manifest(tmp_path, duration_s=30.0)
    # Default state before scanning: both unchecked
    assert m.clean_cut.cut_ads is False
    assert m.clean_cut.cut_political_sensitive is False

    m.cut_segments = [
        CutSegment(cut_id="c1", kind="ad", enabled=True, start_s=2.0, end_s=5.0),
        CutSegment(cut_id="c2", kind="political_sensitive", enabled=True, start_s=8.0, end_s=11.0),
        CutSegment(cut_id="c3", kind="ad", enabled=False, start_s=20.0, end_s=22.0),
    ]
    m.clean_cut.cut_ads = True
    m.clean_cut.cut_political_sensitive = True

    # Auto mode: precise copy disabled whenever an active political_sensitive cut exists
    assert should_use_precise_copy(m) is False
    assert len(get_active_cut_segments(m)) == 2
    assert get_active_cut_spans(m) == [(2.0, 5.0), (8.0, 11.0)]

    # Toggle political_sensitive off -> only ad stays active -> auto switches to fast precise copy
    m.clean_cut.cut_political_sensitive = False
    assert should_use_precise_copy(m) is True
    assert get_active_cut_spans(m) == [(2.0, 5.0)]

    kept = subtract_cut_intervals(0.0, 15.0, [(2.0, 5.0), (8.0, 11.0)], min_keep_s=0.4)
    assert kept == [(0.0, 2.0), (5.0, 8.0), (11.0, 15.0)]

    h1 = compute_clean_cut_render_hash(m)
    m.clean_cut.cut_political_sensitive = True
    h2 = compute_clean_cut_render_hash(m)
    assert h1 != h2


def test_scan_cut_segments_preserves_manual_and_snaps_tightly(tmp_path: Path, monkeypatch):
    m, store = _make_manifest(tmp_path, duration_s=16.0)
    m.cut_segments = [
        CutSegment(cut_id="m1", kind="ad", enabled=True, detector="manual", start_s=14.0, end_s=15.0, reason="手动保留")
    ]
    store.save(m)
    cfg = load_editorial_config()

    class FakeLLM:
        def __init__(self, _cfg):
            pass

        def generate_json(self, **kwargs):
            return SimpleNamespace(
                parsed={
                    "segments": [
                        {
                            "kind": "ad",
                            "subtype": "sponsor",
                            "start_utt": 1,
                            "end_utt": 1,
                            "severity": "high",
                            "reason": "赞助商口播",
                        },
                        {
                            "kind": "political_sensitive",
                            "subtype": "hard_redline",
                            "start_utt": 3,
                            "end_utt": 3,
                            "severity": "high",
                            "reason": "红线涉政",
                        },
                    ]
                },
                provider="dashscope",
                model="qwen3.8-max",
                prompt_tokens=10,
                output_tokens=10,
                elapsed_ms=5,
            )

    import clipcut.pipeline.content_filter as cf
    monkeypatch.setattr(cf, "ClipCutLLM", FakeLLM)
    monkeypatch.setattr(cf, "detect_visual_ad_spans", lambda _p: [(12.5, 13.5)])

    out = scan_cut_segments(m, store, cfg, "run_scan_1", scan_ads=True, scan_political=True, sensitivity="strict")
    assert len(out.cut_segments) == 4
    assert out.clean_cut.cut_ads is True
    assert out.clean_cut.cut_political_sensitive is True
    kinds = [(s.detector, s.kind) for s in out.cut_segments]
    assert ("llm", "ad") in kinds
    assert ("llm", "political_sensitive") in kinds
    assert ("visual", "ad") in kinds
    assert ("manual", "ad") in kinds
    # Political sensitive utterance 3 is [8.2, 11.0], previous ends 8.0, next starts 11.2 -> strictly within [8.0, 11.2]
    pol = next(s for s in out.cut_segments if s.kind == "political_sensitive")
    assert 8.0 <= pol.start_s <= 8.2
    assert 11.0 <= pol.end_s <= 11.2
    assert "（净版）" in out.clean_cut.title
    assert out.clean_cut.scanned_at is not None


def test_linked_clip_hash_and_remix_exclusion(tmp_path: Path, monkeypatch):
    m, store = _make_manifest(tmp_path, duration_s=16.0)
    clip = ClipItem(clip_id="clip_01", index=1, enabled=True, title="Main", start_utt=2, end_utt=4, start_s=5.1, end_s=15.0)
    m.clips = [clip]
    m.cut_segments = [
        CutSegment(cut_id="cut_01", kind="political_sensitive", enabled=True, start_utt=3, end_utt=3, start_s=8.2, end_s=11.0)
    ]
    m.clean_cut.cut_political_sensitive = True
    store.save(m)

    h_no_cuts = compute_clip_render_hash(clip, m.source.render_video, None)
    h_with_cuts = compute_clip_render_hash(clip, m.source.render_video, get_active_cut_spans(m))
    assert h_no_cuts != h_with_cuts

    captured_prompts = []

    class FakeLLM:
        def __init__(self, _cfg):
            pass

        def generate_json(self, **kwargs):
            captured_prompts.append(kwargs["prompt"])
            return SimpleNamespace(
                parsed={
                    "remix_title": "合规混剪",
                    "editorial_thesis": "跳过敏感区间",
                    "themes": [
                        {
                            "theme_id": "TH01",
                            "group_label": "实战架构",
                            "card_title": "智能体落地",
                            "segments": [
                                {"source_clip_id": "clip_01", "role": "core", "beat_type": "punchline", "start_utt": 3, "end_utt": 3, "quote": ""},
                                {"source_clip_id": "clip_01", "role": "core", "beat_type": "punchline", "start_utt": 4, "end_utt": 4, "quote": "企业智能体落地实践经验。"},
                            ],
                        }
                    ],
                },
                provider="dashscope",
                model="qwen3.8-max",
                prompt_tokens=10,
                output_tokens=10,
                elapsed_ms=5,
            )

    import clipcut.pipeline.verify_and_plan as vp
    monkeypatch.setattr(vp, "ClipCutLLM", FakeLLM)
    cfg = load_editorial_config()
    out = plan_remix_from_clips(m, store, cfg, "run_rmx_test")
    assert len(captured_prompts) == 1
    assert "[已屏蔽广告/敏感区间]" in captured_prompts[0]
    assert "涉政红线" not in captured_prompts[0]
    # Segment referencing blocked utterance 3 must be dropped; utterance 4 survives
    assert len(out.remix.items) == 1
    assert out.remix.items[0].start_utt == 4


def test_real_ffmpeg_clean_cut_and_linked_clip_excision(tmp_path: Path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return
    src = tmp_path / "src.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=navy:s=640x360:r=24:d=12",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=12",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src),
        ],
        check=True,
    )

    m, store = _make_manifest(tmp_path, video_path=str(src), duration_s=12.0)
    m.cut_segments = [
        CutSegment(cut_id="cut_01", kind="ad", enabled=True, start_s=1.0, end_s=3.0),
        CutSegment(cut_id="cut_02", kind="political_sensitive", enabled=True, start_s=7.0, end_s=9.0),
    ]
    m.clean_cut.cut_ads = True
    m.clean_cut.cut_political_sensitive = True
    store.save(m)
    cfg = load_editorial_config()

    # 1. Single clip [5.0, 11.0] overlaps [7.0, 9.0] -> kept 4.0s
    clip = ClipItem(clip_id="clip_01", index=1, enabled=True, title="DemoClip", start_utt=2, end_utt=4, start_s=5.0, end_s=11.0)
    clip_mp4 = tmp_path / "out" / "clip.mp4"
    ok, enc, err = _render_single_clip(src, clip, clip_mp4, fade_ms=25, use_vt=False, cut_spans=get_active_cut_spans(m), tmp_dir=tmp_path / "tmp")
    assert ok, err
    assert abs(_ffprobe_duration(clip_mp4) - 4.0) < 0.45

    # 2. Full Clean Cut excises [1,3] + [7,9] from 12s -> ~8.0s + cover JPG
    clean_mp4 = _render_clean_cut(m, store, cfg, use_vt=False, run_id="run_cc_test", published_date="2026-11-22")
    assert clean_mp4 is not None and clean_mp4.exists()
    assert abs(_ffprobe_duration(clean_mp4) - 8.0) < 0.55
    assert m.clean_cut.cover_file and Path(m.clean_cut.cover_file).exists()


def test_api_clean_cut_endpoints_and_publish_target(tmp_path: Path, monkeypatch):
    client = TestClient(app)
    m, store = _make_manifest(tmp_path, duration_s=30.0)
    clean_out = tmp_path / "out" / "clean.mp4"
    clean_out.parent.mkdir(parents=True, exist_ok=True)
    clean_out.write_bytes(b"cleanmp4" * 200)
    cover_out = clean_out.with_suffix(".jpg")
    cover_out.write_bytes(b"jpgdata")
    m.clean_cut.output_file = str(clean_out)
    m.clean_cut.cover_file = str(cover_out)
    m.clean_cut.title = "企业智能体净版"
    store.save(m)

    monkeypatch.setattr(srv, "_get_store", lambda pid: store)

    # PATCH: add manual political_sensitive segment; cut_ads requested True with 0 ads must stay False!
    r = client.patch(
        "/api/projects/test-clean-01/clean-cut",
        json={
            "enabled": True,
            "cut_ads": True,
            "cut_political_sensitive": True,
            "sensitivity": "strict",
            "apply_to_clips": True,
            "add_segment": {"kind": "political_sensitive", "start_s": 8.0, "end_s": 11.0, "reason": "测试标注"},
        },
    )
    assert r.status_code == 200
    body = r.json()["manifest"]
    assert body["clean_cut"]["enabled"] is True
    assert body["clean_cut"]["cut_ads"] is False
    assert body["clean_cut"]["cut_political_sensitive"] is True
    assert len(body["cut_segments"]) == 1
    seg_id = body["cut_segments"][0]["cut_id"]

    # Edit segment enabled state & delete
    r2 = client.patch(
        "/api/projects/test-clean-01/clean-cut",
        json={"segments": [{"cut_id": seg_id, "enabled": False, "start_s": 8.2, "end_s": 10.8}]},
    )
    assert r2.status_code == 200
    assert r2.json()["manifest"]["cut_segments"][0]["enabled"] is False

    # Media routes for clean_cut and clean_cut_cover
    r_vid = client.get("/api/projects/test-clean-01/media/clean_cut")
    assert r_vid.status_code == 200
    r_cov = client.get("/api/projects/test-clean-01/media/clean_cut_cover")
    assert r_cov.status_code == 200

    # Publish metadata lookup for clean_cut
    vpath, title, cover_title, desc, tags, recs = _default_target_metadata(store.load(), "clean_cut")
    assert vpath == clean_out
    assert title == "企业智能体净版"
    assert recs is not None

    # Delete segment -> no political_sensitive left -> cut_political_sensitive resets to False
    r3 = client.patch("/api/projects/test-clean-01/clean-cut", json={"delete_ids": [seg_id]})
    assert r3.status_code == 200
    assert r3.json()["manifest"]["cut_segments"] == []
    assert r3.json()["manifest"]["clean_cut"]["cut_political_sensitive"] is False

    # Edit utterance bounds via PATCH
    r4 = client.patch(
        "/api/projects/test-clean-01/clean-cut",
        json={"add_segment": {"kind": "political_sensitive", "start_s": 8.0, "end_s": 11.0, "reason": "tmp"}},
    )
    assert r4.status_code == 200
    sid2 = r4.json()["manifest"]["cut_segments"][0]["cut_id"]
    r5 = client.patch(
        "/api/projects/test-clean-01/clean-cut",
        json={"segments": [{"cut_id": sid2, "start_utt": 2, "end_utt": 3}]},
    )
    assert r5.status_code == 200
    assert r5.json()["manifest"]["cut_segments"][0]["start_utt"] == 2
    assert r5.json()["manifest"]["cut_segments"][0]["end_utt"] == 3

    # Trigger background scan & render endpoints
    import time
    called = {"scan": 0, "render": 0}

    def fake_scan(manifest, s, cfg, jid, **kw):
        called["scan"] += 1
        return manifest

    def fake_render(manifest, s, cfg, *, force=False):
        called["render"] += 1
        return True, str(clean_out)

    monkeypatch.setattr(srv, "scan_cut_segments", fake_scan)
    monkeypatch.setattr(srv, "render_clean_cut_and_cover", fake_render)

    r_scan = client.post(
        "/api/projects/test-clean-01/clean-cut/scan",
        json={"cut_ads": True, "cut_political_sensitive": True, "sensitivity": "strict"},
    )
    assert r_scan.status_code == 200 and r_scan.json()["started"] is True
    for _ in range(20):
        if not client.get("/api/projects/test-clean-01").json()["job"].get("running"):
            break
        time.sleep(0.05)

    r_ren = client.post("/api/projects/test-clean-01/clean-cut/render?force=true")
    assert r_ren.status_code == 200 and r_ren.json()["started"] is True
    for _ in range(20):
        if not client.get("/api/projects/test-clean-01").json()["job"].get("running"):
            break
        time.sleep(0.05)

    assert called["scan"] == 1
    assert called["render"] == 1


def test_default_scan_applicability_and_strategy_stage_order(tmp_path: Path, monkeypatch):
    import clipcut.pipeline.content_filter as cf
    import clipcut.pipeline.orchestrator as orch
    from clipcut.models import Candidate

    cfg = load_editorial_config()
    monkeypatch.setattr(cf, "detect_visual_ad_spans", lambda _p: [])

    # 1. Ads-only -> cut_ads=True, cut_political_sensitive=False (even when starting both False)
    m1, s1 = _make_manifest(tmp_path / "ads_only", duration_s=16.0)
    class FakeLLM_AdsOnly:
        def __init__(self, _cfg): pass
        def generate_json(self, **kw):
            return SimpleNamespace(
                parsed={"segments": [{"kind": "ad", "subtype": "sponsor", "start_utt": 1, "end_utt": 1, "reason": "sponsor"}]},
                provider="dashscope", model="qwen3.8-max", prompt_tokens=1, output_tokens=1, elapsed_ms=1,
            )
    monkeypatch.setattr(cf, "ClipCutLLM", FakeLLM_AdsOnly)
    out1 = scan_cut_segments(m1, s1, cfg, "r1")
    assert out1.clean_cut.cut_ads is True
    assert out1.clean_cut.cut_political_sensitive is False

    # 2. Clean video (0 ads, 0 sensitive) -> both remain False
    m2, s2 = _make_manifest(tmp_path / "clean", duration_s=16.0)
    class FakeLLM_None:
        def __init__(self, _cfg): pass
        def generate_json(self, **kw):
            return SimpleNamespace(
                parsed={"segments": []},
                provider="dashscope", model="qwen3.8-max", prompt_tokens=1, output_tokens=1, elapsed_ms=1,
            )
    monkeypatch.setattr(cf, "ClipCutLLM", FakeLLM_None)
    out2 = scan_cut_segments(m2, s2, cfg, "r2")
    assert out2.clean_cut.cut_ads is False
    assert out2.clean_cut.cut_political_sensitive is False

    # 3. Orchestrator strategy run executes scan_cuts -> analyze -> verify_and_plan and drops candidates inside cuts
    m3, s3 = _make_manifest(tmp_path / "orch", duration_s=120.0)
    executed: list[str] = []

    def fake_signals(m, store, rid):
        executed.append("signals")
        return m

    def fake_scan(m, store, _cfg, rid, **kw):
        executed.append("scan_cuts")
        m.cut_segments = [
            CutSegment(cut_id="cut_01", kind="political_sensitive", enabled=True, start_utt=3, end_utt=3, start_s=8.0, end_s=11.5)
        ]
        m.clean_cut.cut_ads = False
        m.clean_cut.cut_political_sensitive = True
        store.save(m)
        return m

    def fake_analyze(m, store, _cfg, rid):
        executed.append("analyze")
        # One candidate wholly inside the cut [8.0..11.5], one clean candidate across long utterances
        m.utterances = [
            Utterance(id=0, start_s=0.0, end_s=8.0, en="Safe intro", zh="安全开场段落。"),
            Utterance(id=1, start_s=8.0, end_s=11.5, en="Sensitive segment", zh="涉政敏感区间。"),
            Utterance(id=2, start_s=12.0, end_s=75.0, en="Long safe tech discussion", zh="企业AI工程深入实践总结。"),
        ]
        m.candidates = [
            Candidate(
                candidate_id="cand_bad", categories=["key_dialogue"], title="SensitiveOnly",
                start_utt=1, end_utt=1, reason="bad", summary="bad", score=8.0,
            ),
            Candidate(
                candidate_id="cand_good", categories=["key_dialogue"], title="SafeTech",
                start_utt=2, end_utt=2, reason="good", summary="good", score=8.5,
            ),
        ]
        store.save(m)
        return m

    monkeypatch.setattr(orch, "run_signals_stage", fake_signals)
    monkeypatch.setattr(orch, "scan_cut_segments", fake_scan)
    monkeypatch.setattr(orch, "run_analyze_stage", fake_analyze)
    import clipcut.pipeline.verify_and_plan as vp
    monkeypatch.setattr(vp, "plan_remix_from_clips", lambda m, store, _cfg, rid: m)

    res_m = orch.run_existing_project_stages(s3.work_dir, ["signals", "scan_cuts", "analyze", "verify_and_plan"], config_path=None)
    assert executed == ["signals", "scan_cuts", "analyze"]
    assert res_m.stages["scan_cuts"].status == "completed"
    assert [c.title for c in res_m.clips] == ["SafeTech"]
    bad_cand = next(c for c in res_m.candidates if c.candidate_id == "cand_bad")
    assert bad_cand.disposition == "dropped"


def test_sensitivity_profile_linkage_and_config_crud(tmp_path: Path, monkeypatch):
    import clipcut.config as cfg_mod
    from clipcut.pipeline.content_filter import (
        DEFAULT_POLITICAL_PROFILES,
        _build_filter_prompt,
    )

    real_yaml = cfg_mod.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    temp_yaml = tmp_path / "editorial.yaml"
    temp_yaml.write_text(real_yaml, encoding="utf-8")
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_PATH", temp_yaml)

    cfg = load_editorial_config()
    m, store = _make_manifest(tmp_path, duration_s=30.0)
    m.cut_segments = [
        CutSegment(cut_id="cut_low", kind="political_sensitive", detector="llm", severity="low", start_s=2.0, end_s=5.0, reason="low"),
        CutSegment(cut_id="cut_med", kind="political_sensitive", detector="llm", severity="medium", start_s=8.0, end_s=11.0, reason="medium"),
        CutSegment(cut_id="cut_man", kind="political_sensitive", detector="manual", severity="low", start_s=12.0, end_s=13.0, reason="manual"),
    ]
    m.clean_cut.cut_political_sensitive = True
    m.clean_cut.sensitivity = "strict"
    store.save(m)

    # 1. Active segment filtering follows sensitivity profile; manual segments always stay active
    assert [s.cut_id for s in get_active_cut_segments(m, cfg)] == ["cut_low", "cut_med", "cut_man"]
    m.clean_cut.sensitivity = "balanced"
    assert [s.cut_id for s in get_active_cut_segments(m, cfg)] == ["cut_med", "cut_man"]
    m.clean_cut.sensitivity = "loose"
    assert [s.cut_id for s in get_active_cut_segments(m, cfg)] == ["cut_man"]

    # 2. Switching sensitivity without manual segments auto-toggles cut_political_sensitive & render_hash
    m.cut_segments = [s for s in m.cut_segments if s.detector != "manual"]
    m.clean_cut.sensitivity = "strict"
    m.clean_cut.cut_political_sensitive = True
    store.save(m)

    client = TestClient(app)
    monkeypatch.setattr(srv, "_get_store", lambda pid: store)

    r_bal = client.patch("/api/projects/test-clean-01/clean-cut", json={"sensitivity": "balanced"})
    assert r_bal.status_code == 200
    assert r_bal.json()["manifest"]["clean_cut"]["cut_political_sensitive"] is True
    hash_bal = r_bal.json()["manifest"]["clean_cut"]["render_hash"]

    r_loose = client.patch("/api/projects/test-clean-01/clean-cut", json={"sensitivity": "loose"})
    assert r_loose.status_code == 200
    cc_loose = r_loose.json()["manifest"]["clean_cut"]
    assert cc_loose["cut_political_sensitive"] is False
    assert cc_loose["render_hash"] != hash_bal

    r_back = client.patch("/api/projects/test-clean-01/clean-cut", json={"sensitivity": "strict"})
    assert r_back.status_code == 200
    assert r_back.json()["manifest"]["clean_cut"]["cut_political_sensitive"] is True

    # Updating individual segment severity from medium -> low drops it under balanced
    r_sev = client.patch(
        "/api/projects/test-clean-01/clean-cut",
        json={"sensitivity": "balanced", "segments": [{"cut_id": "cut_med", "severity": "low"}]},
    )
    assert r_sev.status_code == 200
    assert r_sev.json()["manifest"]["clean_cut"]["cut_political_sensitive"] is False

    # 3. Config CRUD & reset on /api/config/content-filter (stored in isolated temp_yaml)
    g_cfg = client.get("/api/config/content-filter")
    assert g_cfg.status_code == 200
    payload = g_cfg.json()
    assert set(payload["political_profiles"].keys()) == {"strict", "balanced", "loose"}

    # Add a rule to balanced, delete a rule from strict, change description in loose
    payload["default_sensitivity"] = "balanced"
    payload["political_profiles"]["balanced"]["rules"].append(
        {"id": "custom_chip", "name": "自定义规则", "severity": "medium", "description": "自定义测试拦截词条XYZ"}
    )
    payload["political_profiles"]["strict"]["rules"] = payload["political_profiles"]["strict"]["rules"][:-1]
    p_put = client.put("/api/config/content-filter", json=payload)
    assert p_put.status_code == 200
    assert p_put.json()["default_sensitivity"] == "balanced"
    assert any(r["id"] == "custom_chip" for r in p_put.json()["political_profiles"]["balanced"]["rules"])

    # Prompt uses updated config rules dynamically
    new_cfg = load_editorial_config()
    prompt_txt = _build_filter_prompt(
        m,
        m.utterances,
        scan_ads=True,
        scan_political=True,
        sensitivity="balanced",
        cfg=new_cfg,
    )
    assert "自定义测试拦截词条XYZ" in prompt_txt

    # Reset single profile restores defaults cleanly
    p_rst = client.post("/api/config/content-filter/reset", json={"profile": "balanced"})
    assert p_rst.status_code == 200
    bal_ids = [r["id"] for r in p_rst.json()["political_profiles"]["balanced"]["rules"]]
    assert "custom_chip" not in bal_ids
    assert bal_ids == [r["id"] for r in DEFAULT_POLITICAL_PROFILES["balanced"]["rules"]]


def test_resolve_segment_action_empty_disable_and_overlap_hashes(tmp_path: Path, monkeypatch):
    from clipcut.pipeline.content_filter import (
        describe_applied_filter,
        get_overlapping_mask_payload,
        redact_ass_dialogue_text,
        resolve_mask_audio_spans,
        resolve_segment_action,
        sync_all_render_hashes,
    )
    from clipcut.pipeline.verify_and_plan import (
        compute_compilation_render_hash,
        compute_remix_render_hash,
    )

    clean_mp4 = tmp_path / "clean.mp4"
    hs_mp4 = tmp_path / "clean_hardsub.mp4"
    clean_mp4.write_bytes(b"clean")
    hs_mp4.write_bytes(b"hardsub")

    m, store = _make_manifest(tmp_path, video_path=str(hs_mp4), duration_s=40.0)
    m.source.analysis_video = str(clean_mp4)
    seg_mask = CutSegment(
        cut_id="c1",
        kind="political_sensitive",
        enabled=True,
        recommended_action="mask",
        severity="low",
        sensitive_terms_zh=["中国"],
        sensitive_terms_en=["Chinese"],
        start_s=5.0,
        end_s=8.0,
        reason="顺带提及",
    )
    seg_del = CutSegment(
        cut_id="c2",
        kind="political_sensitive",
        enabled=True,
        recommended_action="delete",
        severity="high",
        start_s=20.0,
        end_s=24.0,
        reason="整段政论",
    )
    m.cut_segments = [seg_mask, seg_del]
    m.clean_cut.enabled = True
    m.clean_cut.cut_political_sensitive = True
    m.clean_cut.sensitivity = "strict"
    m.clean_cut.political_action = "auto"

    # auto honors recommended_action; per-segment action overrides; missing clean video falls back to delete
    assert resolve_segment_action(seg_mask, m) == "mask"
    assert resolve_segment_action(seg_del, m) == "delete"
    seg_del.action = "mask"
    assert resolve_segment_action(seg_del, m) == "mask"
    seg_del.action = "inherit"
    m.clean_cut.political_action = "mask"
    assert resolve_segment_action(seg_del, m) == "mask"
    m.clean_cut.political_action = "delete"
    assert resolve_segment_action(seg_mask, m) == "delete"
    m.clean_cut.political_action = "auto"
    assert resolve_segment_action(seg_mask, m, allow_mask=False) == "delete"

    # Overlapping hashes: c_safe [0..4] unaffected by [5..8] or [20..24], c_mask [4..10] changes when masked
    c_safe = ClipItem(clip_id="c1", index=1, enabled=True, title="Safe", start_utt=0, end_utt=0, start_s=0.0, end_s=4.0)
    c_mask = ClipItem(clip_id="c2", index=2, enabled=True, title="Masked", start_utt=1, end_utt=2, start_s=4.0, end_s=10.0)
    m.clips = [c_safe, c_mask]
    m.compilation.enabled = True
    m.remix.enabled = True
    m.remix.items = [RemixItem(remix_id="r1", order=1, enabled=True, group_label="g", card_title="c", start_utt=0, end_utt=0, start_s=0.0, end_s=4.0)]

    sync_all_render_hashes(m)
    safe_h_strict = c_safe.render_hash
    mask_h_strict = c_mask.render_hash
    rmx_h_strict = compute_remix_render_hash(m)
    comp_h_strict = compute_compilation_render_hash(m)
    assert get_overlapping_mask_payload(0.0, 4.0, m) == []
    assert len(get_overlapping_mask_payload(4.0, 10.0, m)) == 1
    assert "遮蔽" in describe_applied_filter(m, [(4.0, 10.0)])

    # Switch to loose -> passes seg_mask (severity=low): c_safe & rmx stay unchanged, c_mask & comp update!
    m.clean_cut.sensitivity = "loose"
    sync_all_render_hashes(m)
    assert c_safe.render_hash == safe_h_strict
    assert c_safe.render_hash != ""
    assert c_mask.render_hash != mask_h_strict
    assert compute_remix_render_hash(m) == rmx_h_strict
    assert compute_compilation_render_hash(m) != comp_h_strict

    # Empty active segments auto-disable clean_cut & reject manual render endpoint
    client = TestClient(app)
    monkeypatch.setattr(srv, "_get_store", lambda pid: store)
    m.cut_segments = [seg_mask]
    m.clean_cut.sensitivity = "strict"
    m.clean_cut.enabled = True
    store.save(m)
    r_patch = client.patch("/api/projects/test-clean-01/clean-cut", json={"sensitivity": "loose"})
    assert r_patch.status_code == 200
    assert r_patch.json()["manifest"]["clean_cut"]["enabled"] is False
    r_render_empty = client.post("/api/projects/test-clean-01/clean-cut/render")
    assert r_render_empty.status_code == 400
    cfg = load_editorial_config()
    assert _render_clean_cut(m, store, cfg, use_vt=False, run_id="r_empty") is None

    # ASS redaction preserves tags and newlines while masking Chinese & English terms
    raw_ass = (
        r"分析中国驱逐舰电磁辐射的{\fnCalibri}AI{\r}，你不可能\N"
        r"{\fnCalibri\fs36\c&H00FFFFFF}analyzing a Chinese destroyer's China signal,\Nlike China."
    )
    redacted = redact_ass_dialogue_text(raw_ass, ["中国"], ["Chinese", "China"])
    assert r"{\fnCalibri}AI{\r}" in redacted
    assert r"\N{\fnCalibri\fs36\c&H00FFFFFF}" in redacted
    assert "中国" not in redacted
    assert "Chinese" not in redacted
    assert "China" not in redacted
    assert "＊＊驱逐舰" in redacted

    # Word-level audio span lookup from transcribe_state.json + fallback interpolation
    ts_path = tmp_path / "transcribe_state.json"
    ts_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 5.0,
                        "end": 8.0,
                        "text": "analyzing a Chinese destroyer",
                        "words": [
                            {"start": 5.1, "end": 5.6, "word": "analyzing"},
                            {"start": 5.6, "end": 5.7, "word": "a"},
                            {"start": 5.8, "end": 6.25, "word": "Chinese"},
                            {"start": 6.25, "end": 6.9, "word": "destroyer"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    spans = resolve_mask_audio_spans(m, [seg_mask])
    assert len(spans) == 1
    assert abs(spans[0][0] - 5.76) < 0.05
    assert abs(spans[0][1] - 6.30) < 0.05
    ts_path.unlink()
    m.utterances = [Utterance(id=0, start_s=5.0, end_s=8.0, en="analyzing a Chinese destroyer", zh="分析中国驱逐舰")]
    spans_fb = resolve_mask_audio_spans(m, [seg_mask])
    assert len(spans_fb) == 1
    assert 5.0 <= spans_fb[0][0] < spans_fb[0][1] <= 8.5


def test_real_ffmpeg_mask_render_preserves_duration_and_bleeps(tmp_path: Path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return
    import numpy as np
    from clipcut.render.renderer import render_single_clip_and_cover

    clean_vid = tmp_path / "clean.mp4"
    hs_vid = tmp_path / "clean_hardsub.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=navy:s=640x360:r=24:d=6",
            "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=44100:duration=6",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(clean_vid),
        ],
        check=True,
    )
    shutil.copyfile(clean_vid, hs_vid)

    ass_path = tmp_path / "clean.bi.ass"
    ass_path.write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 640\nPlayResY: 360\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        "Style: TextTop,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,1,0,2,20,20,20,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:04.00,TextTop,,0,0,0,,分析中国驱逐舰技术\\N{\\fs18}analyzing a Chinese destroyer\n",
        encoding="utf-8",
    )
    (tmp_path / "transcribe_state.json").write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 1.0,
                        "end": 4.0,
                        "text": "analyzing a Chinese destroyer",
                        "words": [
                            {"start": 1.2, "end": 1.8, "word": "analyzing"},
                            {"start": 1.8, "end": 1.9, "word": "a"},
                            {"start": 2.0, "end": 2.6, "word": "Chinese"},
                            {"start": 2.6, "end": 3.5, "word": "destroyer"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    m, store = _make_manifest(tmp_path, video_path=str(hs_vid), duration_s=6.0)
    m.source.analysis_video = str(clean_vid)
    m.source.subtitle_path = str(ass_path)
    m.source.subtitle_format = "ass"
    m.source.width = 640
    m.source.height = 360
    m.source.fps = 24.0
    m.utterances = [
        Utterance(id=0, start_s=0.0, end_s=1.0, en="Intro safe.", zh="安全开场。"),
        Utterance(id=1, start_s=1.0, end_s=4.0, en="analyzing a Chinese destroyer", zh="分析中国驱逐舰技术"),
        Utterance(id=2, start_s=4.0, end_s=6.0, en="Outro safe.", zh="安全结尾。"),
    ]
    m.cut_segments = [
        CutSegment(
            cut_id="cut_01",
            kind="political_sensitive",
            enabled=True,
            detector="llm",
            action="inherit",
            recommended_action="mask",
            sensitive_terms_zh=["中国"],
            sensitive_terms_en=["Chinese"],
            start_utt=1,
            end_utt=1,
            start_s=1.0,
            end_s=4.0,
            severity="high",
            reason="顺带提及敏感词",
        )
    ]
    m.clean_cut.cut_political_sensitive = True
    m.clean_cut.political_action = "auto"
    m.clean_cut.scanned_at = "2026-11-22T00:00:00+00:00"
    clip = ClipItem(clip_id="clip_01", index=1, enabled=True, title="MaskedClip", start_utt=0, end_utt=2, start_s=0.0, end_s=6.0)
    m.clips = [clip]
    store.save(m)

    cfg = load_editorial_config()
    ok, res = render_single_clip_and_cover(m, store, "clip_01", cfg, force=True)
    assert ok, res
    out_mp4 = Path(res)
    assert out_mp4.exists()
    assert abs(_ffprobe_duration(out_mp4) - 6.0) < 0.35
    assert "脱敏(遮蔽" in store.load().clips[0].rendered_filter_tag

    # Masked ASS should contain redacted Chinese/English and no raw "Chinese"
    masked_files = list((store.work_dir / "masked_subs").glob("masked_*.ass"))
    assert masked_files
    ass_txt = masked_files[0].read_text(encoding="utf-8")
    assert "＊＊驱逐舰" in ass_txt
    assert "******* destroyer" in ass_txt

    def _dom_freq(start_t: float, dur_t: float) -> float:
        pcm = subprocess.check_output(
            [
                "ffmpeg", "-v", "error", "-ss", f"{start_t:.3f}", "-i", str(out_mp4),
                "-t", f"{dur_t:.3f}", "-f", "s16le", "-ac", "1", "-ar", "44100", "-",
            ]
        )
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        spec = np.abs(np.fft.rfft(arr))
        freqs = np.fft.rfftfreq(len(arr), 1.0 / 44100.0)
        return float(freqs[int(np.argmax(spec))])

    f_safe = _dom_freq(0.3, 0.5)
    f_bleep = _dom_freq(2.05, 0.45)
    assert abs(f_safe - 220.0) < 25.0, f"expected 220Hz outside mask, got {f_safe}"
    assert abs(f_bleep - 880.0) < 25.0, f"expected 880Hz bleep inside mask, got {f_bleep}"


def test_llm_data_inspection_short_circuits_dashscope(monkeypatch):
    from clipcut.llm import ClipCutLLM, LLMCallResult

    llm = ClipCutLLM({})
    called: list[str] = []

    def fake_ds(*, model, **kw):
        called.append(f"dashscope:{model}")
        raise RuntimeError('HTTP 400: {"error":{"code":"data_inspection_failed","message":"inappropriate content"}}')

    def fake_google(*, provider, model, **kw):
        called.append(f"{provider}:{model}")
        return LLMCallResult(
            text='{"segments": []}',
            parsed={"segments": []},
            model=model,
            provider=provider,
            prompt_tokens=1,
            output_tokens=1,
            elapsed_ms=1,
        )

    monkeypatch.setattr(llm, "_call_dashscope", fake_ds)
    monkeypatch.setattr(llm, "_call_google", fake_google)
    res = llm.generate_json(
        prompt="test",
        model="qwen3.8-max",
        provider="dashscope",
        fallback_models=["qwen4-max", "glm-5.3", "kimi-k3.1-pro", "gemini:gemini-2.5-flash"],
    )
    assert res.provider == "gemini"
    assert called == ["dashscope:qwen3.8-max", "gemini:gemini-2.5-flash"]


def test_reanchor_drifted_utterance_ids():
    from clipcut.models import Utterance
    from clipcut.pipeline.content_filter import _reanchor_hit_utterances, _term_in_utterance

    utts = [
        Utterance(id=0, start_s=0.0, end_s=3.0, zh="欢迎收看本期节目", en="Welcome to the show"),
        Utterance(id=1, start_s=3.0, end_s=7.0, zh="今天讨论国防工业软件", en="Today we discuss defense software"),
        Utterance(id=2, start_s=7.0, end_s=11.0, zh="这些系统需要长期验证", en="These systems exist across fleets"),
        Utterance(
            id=3,
            start_s=11.0,
            end_s=16.0,
            zh="比如分析中国驱逐舰的电磁信号",
            en="Like analyzing a Chinese destroyer's electromagnetic emissions",
        ),
        Utterance(id=4, start_s=16.0, end_s=20.0, zh="这能大幅提升态势感知能力", en="That improves situational awareness"),
    ]

    # Word-boundary guard: "xi" must not false-match "exist"
    assert _term_in_utterance("xi", utts[2]) is False

    # Drifted start_utt=0, end_utt=0 snaps back to u0003 by quote
    anchored = _reanchor_hit_utterances(
        utts,
        su=0,
        eu=0,
        chunk_lo=0,
        chunk_hi=4,
        quote="analyzing a Chinese destroyer's electromagnetic emissions",
        terms_zh=["中国"],
        terms_en=["Chinese"],
    )
    assert anchored == (3, 3)

    # Hallucinated quote + terms nowhere in chunk returns None
    dropped = _reanchor_hit_utterances(
        utts,
        su=0,
        eu=0,
        chunk_lo=0,
        chunk_hi=4,
        quote="Subscribe via NYT Audio app link in bio",
        terms_zh=[],
        terms_en=[],
    )
    assert dropped is None


def test_api_content_filter_default_political_action(monkeypatch):
    import copy
    from fastapi.testclient import TestClient
    import clipcut.api.server as srv

    mem_cfg: dict = {
        "content_filter": {
            "sensitivity": "strict",
            "political_action": "auto",
            "political_profiles": copy.deepcopy(srv.DEFAULT_POLITICAL_PROFILES),
        }
    }
    monkeypatch.setattr(srv, "load_editorial_config", lambda: copy.deepcopy(mem_cfg))

    def fake_save(cfg: dict):
        mem_cfg.clear()
        mem_cfg.update(copy.deepcopy(cfg))

    monkeypatch.setattr(srv, "save_editorial_config", fake_save)

    client = TestClient(srv.app)
    r0 = client.get("/api/config/content-filter")
    assert r0.status_code == 200
    d0 = r0.json()
    assert d0["default_political_action"] == "auto"
    assert set(d0["political_profiles"].keys()) == {"strict", "balanced", "loose"}

    # Partial PUT updates only default_political_action while keeping political_profiles intact
    r1 = client.put("/api/config/content-filter", json={"default_political_action": "mask"})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["default_political_action"] == "mask"
    assert d1["default_sensitivity"] == "strict"
    assert len(d1["political_profiles"]["strict"]["rules"]) == len(srv.DEFAULT_POLITICAL_PROFILES["strict"]["rules"])

    # Invalid value returns 400
    r_bad = client.put("/api/config/content-filter", json={"default_political_action": "invalid_mode"})
    assert r_bad.status_code == 400


def test_prompt_enforces_brand_ad_and_target_country_rules(tmp_path: Path):
    from clipcut.pipeline.content_filter import _build_filter_prompt

    m, _ = _make_manifest(tmp_path, duration_s=30.0)
    prompt_txt = _build_filter_prompt(
        m,
        m.utterances,
        scan_ads=True,
        scan_political=True,
        sensitivity="balanced",
        cfg={"content_filter": {}},
    )
    assert "具体可识别的品牌名" in prompt_txt
    assert "商业赞助" in prompt_txt
    assert "俄罗斯" in prompt_txt and "朝鲜" in prompt_txt and "伊斯兰" in prompt_txt
    assert "Apple" in prompt_txt and "宇树科技" in prompt_txt


def test_filter_drops_generic_commercial_ad_and_empty_reason_low_political(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace
    import clipcut.pipeline.content_filter as cf

    assert (
        cf._is_valid_ai_ad_hit(
            {"kind": "ad", "subtype": "sponsor", "brand": "商业软件"},
            "这间办公室，是商业赞助给我的。商业软件。",
            "提及商业赞助",
        )
        is False
    )
    assert (
        cf._is_valid_ai_ad_hit(
            {"kind": "ad", "subtype": "outro", "brand": ""},
            "这间办公室，是商业赞助给我的。商业软件。",
            "提及商业赞助和商业软件",
        )
        is False
    )
    assert (
        cf._is_valid_ai_ad_hit(
            {"kind": "ad", "subtype": "sponsor", "brand": "NordVPN"},
            "Use code DHH at nordvpn.com",
            "NordVPN 赞助口播",
        )
        is True
    )
    assert (
        cf._is_valid_ai_ad_hit(
            {"kind": "ad", "subtype": "sponsor"},
            "",
            "sponsor",
        )
        is True
    )

    m, store = _make_manifest(tmp_path, duration_s=16.0)
    monkeypatch.setattr(cf, "detect_visual_ad_spans", lambda _p: [])

    class FakeLLM_Noise:
        def __init__(self, _cfg):
            pass

        def generate_json(self, **kw):
            return SimpleNamespace(
                parsed={
                    "segments": [
                        {
                            "kind": "ad",
                            "subtype": "sponsor",
                            "brand": "商业软件",
                            "quote": "这间办公室",
                            "start_utt": 1,
                            "end_utt": 1,
                            "reason": "明确提及商业赞助和商业软件",
                        },
                        {
                            "kind": "political_sensitive",
                            "subtype": "borderline_stance",
                            "brand": "",
                            "quote": "welcome to today",
                            "start_utt": 0,
                            "end_utt": 0,
                            "severity": "low",
                            "reason": "",
                        },
                    ]
                },
                provider="gemini",
                model="gemini-2.5-flash",
                prompt_tokens=1,
                output_tokens=1,
                elapsed_ms=1,
            )

    monkeypatch.setattr(cf, "ClipCutLLM", FakeLLM_Noise)
    out = cf.scan_cut_segments(m, store, {"content_filter": {}}, "r_noise")
    assert out.cut_segments == []
    assert out.clean_cut.cut_ads is False
    assert out.clean_cut.cut_political_sensitive is False
    assert out.clean_cut.enabled is False


def test_filter_drops_speculative_non_target_political_hits():
    from clipcut.models import Utterance
    import clipcut.pipeline.content_filter as cf

    utts = [
        Utterance(id=0, start_s=0.0, end_s=4.0, zh="也许政府就该接管AI，避免两家公司垄断。", en="Maybe the government should just take it over."),
        Utterance(id=1, start_s=4.0, end_s=8.0, zh="Google 和 Meta 垄断市场非常反乌托邦。", en="Google and Meta duopoly is dystopian."),
        Utterance(id=2, start_s=8.0, end_s=12.0, zh="我两周前去中国参观了宇树科技，聊了AGI进度。", en="I went to China to visit Unitree and talked about AGI timelines."),
        Utterance(id=3, start_s=20.0, end_s=24.0, zh="习主席清洗军方并在台海周边制造危机。", en="Xi Jinping purged generals around the Taiwan Strait."),
        Utterance(id=4, start_s=30.0, end_s=34.0, zh="古巴政府的网络封锁政策遭到了批评。", en="Cuba's internet censorship policy was criticized."),
    ]

    # 1. Speculative "government" hit even if LLM labels target_country="中国" (no China/target in +/-3 utts -> wait: use separated indices or isolated slice)
    iso_utts = [utts[0], utts[1]]
    assert cf._is_valid_political_hit(
        {"kind": "political_sensitive", "target_country": "中国", "is_explicit_criticism": True},
        iso_utts,
        0,
        0,
        "Maybe the government should just take it over.",
        "可能引发对中国政府监管的联想",
    ) is False

    # 2. Non-target commercial critique
    assert cf._is_valid_political_hit(
        {"kind": "political_sensitive", "target_country": "非目标", "is_explicit_criticism": True},
        iso_utts,
        1,
        1,
        "Google and Meta duopoly is dystopian.",
        "垄断批评",
    ) is False

    # 3. Neutral China/Unitree visit where is_explicit_criticism is False
    assert cf._is_valid_political_hit(
        {"kind": "political_sensitive", "target_country": "中国", "is_explicit_criticism": False},
        [utts[2]],
        2,
        2,
        "我两周前去中国参观了宇树科技",
        "中性提及中国企业",
    ) is False

    # 4. Real China/Taiwan redline accepted
    assert cf._is_valid_political_hit(
        {"kind": "political_sensitive", "target_country": "中国 / 台海", "is_explicit_criticism": True},
        utts,
        3,
        3,
        "Xi Jinping purged generals around the Taiwan Strait.",
        "直接涉及中国领导人与台海危机",
    ) is True

    # 5. Another friendly country explicitly named in speech & target_country accepted (not hardcoded to a closed list)
    iso_cuba = [Utterance(id=0, start_s=0.0, end_s=4.0, zh="圣保罗和安道尔当地政府的财政预算引发了地方争议。", en="Andorra local fiscal budget controversy.")]
    assert cf._is_valid_political_hit(
        {"kind": "political_sensitive", "target_country": "安道尔", "is_explicit_criticism": True},
        iso_cuba,
        0,
        0,
        "安道尔当地政府的财政预算引发了地方争议",
        "对安道尔政府政策的批评",
    ) is True

    # 6. Legacy mock without grounding fields stays backwards-compatible
    assert cf._is_valid_political_hit(
        {"kind": "political_sensitive", "reason": "legacy mock"},
        iso_utts,
        0,
        0,
        "",
        "legacy mock",
    ) is True








