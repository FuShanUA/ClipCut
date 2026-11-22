from __future__ import annotations

from pathlib import Path

from clipcut.models import ClipItem, MediaSource, ProjectManifest, SubtitleRepairItem, VideoMetadata
from clipcut.adapters.ingest import parse_subtitle_file
from clipcut.receipts import ProjectStore
import clipcut.subtitle_repair as repair_mod
from clipcut.subtitle_repair import apply_subtitle_repairs


def _make_project(tmp_path: Path) -> tuple[ProjectManifest, ProjectStore, Path]:
    sub = tmp_path / "video.bi.srt"
    sub.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n你好\nHello\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\n世界\nWorld\n",
        encoding="utf-8",
    )
    store = ProjectStore(tmp_path / ".work")
    manifest = ProjectManifest(
        project_id="repair-test",
        source=MediaSource(
            render_video=str(tmp_path / "video.mp4"),
            analysis_video=str(tmp_path / "video.mp4"),
            subtitle_path=str(sub),
            work_dir=str(store.work_dir),
            output_dir=str(tmp_path / "clips"),
            duration_s=4.0,
            width=1920,
            height=1080,
            fps=30.0,
        ),
        metadata=VideoMetadata(original_title="test", metadata_source="manual"),
        clips=[
            ClipItem(
                clip_id="clip_01",
                index=1,
                title="test clip",
                start_utt=0,
                end_utt=1,
                start_s=0.0,
                end_s=4.0,
                output_file=str(tmp_path / "clips" / "clip_01.mp4"),
            )
        ],
    )
    manifest.utterances = parse_subtitle_file(sub)
    store.save(manifest)
    return manifest, store, sub


def test_apply_subtitle_repairs_updates_files_and_manifest(tmp_path: Path, monkeypatch):
    manifest, store, sub = _make_project(tmp_path)

    def fake_generate_ass(bi_srt: Path, ass_path: Path, width: int, height: int) -> None:
        ass_path.write_text("[Script Info]\n", encoding="utf-8")

    def fake_render_clip(manifest, start_s, end_s, ass_path, output_path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"clip")

    monkeypatch.setattr(repair_mod, "_generate_ass", fake_generate_ass)
    monkeypatch.setattr(repair_mod, "_render_clip_with_subtitles", fake_render_clip)

    outcome = apply_subtitle_repairs(
        manifest,
        store,
        "clip_01",
        [SubtitleRepairItem(utterance_id=1, zh="世界（修正）", en="World (fixed)")],
        rebuild_full_subtitle=True,
        render_clip=True,
        render_full_video=False,
    )

    assert outcome.changed_utterance_ids == [1]
    assert "世界（修正）" in sub.read_text(encoding="utf-8")
    assert outcome.cn_srt_path.exists()
    assert outcome.en_srt_path.exists()
    assert outcome.ass_path.exists()
    assert outcome.clip_output_path is not None and outcome.clip_output_path.exists()

    updated = store.load()
    assert updated.utterances[1].zh == "世界（修正）"
    assert updated.utterances[1].en == "World (fixed)"
    assert updated.source.subtitle_sha256 != ""


def test_apply_subtitle_repairs_rejects_out_of_range(tmp_path: Path):
    manifest, store, _ = _make_project(tmp_path)
    try:
        apply_subtitle_repairs(
            manifest,
            store,
            "clip_01",
            [SubtitleRepairItem(utterance_id=99, zh="x")],
            render_clip=False,
        )
    except ValueError as exc:
        assert "不在切片范围内" in str(exc)
    else:
        raise AssertionError("expected ValueError")
