from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from .models import ProjectManifest, Receipt, utc_now_iso


_STORE_LOCKS_GUARD = threading.Lock()
_STORE_LOCKS: dict[Path, threading.RLock] = {}


def _get_store_lock(work_dir: Path) -> threading.RLock:
    resolved = Path(work_dir).resolve()
    with _STORE_LOCKS_GUARD:
        lock = _STORE_LOCKS.get(resolved)
        if lock is None:
            lock = threading.RLock()
            _STORE_LOCKS[resolved] = lock
        return lock


class ProjectStore:
    """Manages project.json, append-only receipts.jsonl, undo stack, and pipeline.log."""

    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._lock = _get_store_lock(self.work_dir)
        self.manifest_path = self.work_dir / "project.json"
        self.receipts_path = self.work_dir / "receipts.jsonl"
        self.log_path = self.work_dir / "pipeline.log"
        self.undo_dir = self.work_dir / "undo"
        self.undo_dir.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.manifest_path.exists()

    def load(self) -> ProjectManifest:
        with self._lock:
            raw = self.manifest_path.read_text(encoding="utf-8")
            manifest = ProjectManifest.model_validate_json(raw)
            # Self-healing: if clips have flat default scores, backfill from candidates
            if manifest.candidates and manifest.clips:
                flat_scores = all(c.score == 7.5 for c in manifest.clips)
                if flat_scores:
                    cands_by_title = {c.title: c for c in manifest.candidates}
                    cands_by_id = {c.candidate_id: c for c in manifest.candidates}
                    modified = False
                    for cl in manifest.clips:
                        matched = cands_by_title.get(cl.title) or cands_by_id.get(cl.clip_id.replace("clip_", "C"))
                        if matched and matched.score and matched.score != 7.5:
                            cl.score = float(matched.score)
                            modified = True
                    if modified:
                        self.save(manifest)
            return manifest

    def save(self, manifest: ProjectManifest) -> None:
        with self._lock:
            manifest.updated_at = utc_now_iso()
            manifest.revision += 1
            tmp = self.work_dir / f".project.{uuid.uuid4().hex[:8]}.json.tmp"
            tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            tmp.replace(self.manifest_path)

    def update_manifest(self, mutator: Callable[[ProjectManifest], None]) -> ProjectManifest:
        """Atomically read-modify-write project.json under per-project RLock."""
        with self._lock:
            manifest = self.load()
            mutator(manifest)
            self.save(manifest)
            return manifest

    def log(self, message: str) -> None:
        line = f"[{utc_now_iso()}] {message}\n"
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line)

    def read_logs(self, tail: int = 200) -> list[str]:
        with self._lock:
            if not self.log_path.exists():
                return []
            lines = self.log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            return lines[-tail:]

    def append_receipt(
        self,
        *,
        run_id: str,
        stage: str,
        action: str,
        summary: str,
        actor: str = "system",
        model: str = "",
        prompt_tokens: int = 0,
        output_tokens: int = 0,
        elapsed_ms: int = 0,
        details: dict[str, Any] | None = None,
        diff: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> Receipt:
        receipt = Receipt(
            receipt_id=f"rcpt_{uuid.uuid4().hex[:10]}",
            run_id=run_id,
            stage=stage,
            actor=actor,  # type: ignore[arg-type]
            action=action,
            model=model,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            elapsed_ms=elapsed_ms,
            summary=summary,
            details=details or {},
            diff=diff or {},
            error=error,
        )
        with self._lock:
            with self.receipts_path.open("a", encoding="utf-8") as f:
                f.write(receipt.model_dump_json() + "\n")
            self.log(f"[{stage}/{actor}] {action}: {summary}" + (f" | ERROR: {error}" if error else ""))
        return receipt

    def list_receipts(self, stage: str | None = None, limit: int = 200) -> list[Receipt]:
        if not self.receipts_path.exists():
            return []
        items: list[Receipt] = []
        for line in self.receipts_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = Receipt.model_validate_json(line)
                if stage and rec.stage != stage:
                    continue
                items.append(rec)
            except Exception:
                continue
        return items[-limit:]

    def save_undo_snapshot(self, manifest: ProjectManifest, label: str) -> str:
        snap_id = f"{manifest.revision:05d}_{uuid.uuid4().hex[:6]}"
        snap_path = self.undo_dir / f"{snap_id}.json"
        payload = {
            "snapshot_id": snap_id,
            "label": label,
            "saved_at": utc_now_iso(),
            "clips": [c.model_dump() for c in manifest.clips],
            "remix": manifest.remix.model_dump(),
            "metadata": manifest.metadata.model_dump(),
            "source": manifest.source.model_dump(),
            "utterances": [u.model_dump() for u in manifest.utterances],
        }
        snap_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        # Keep latest 30 snapshots
        snaps = sorted(self.undo_dir.glob("*.json"))
        for old in snaps[:-30]:
            old.unlink(missing_ok=True)
        return snap_id

    def undo_last(self, run_id: str) -> tuple[ProjectManifest, dict[str, Any]] | None:
        snaps = sorted(self.undo_dir.glob("*.json"))
        if not snaps:
            return None
        latest = snaps[-1]
        data = json.loads(latest.read_text(encoding="utf-8"))
        manifest = self.load()
        from .models import ClipItem, MediaSource, RemixConfig, Utterance, VideoMetadata

        manifest.clips = [ClipItem.model_validate(c) for c in data["clips"]]
        manifest.remix = RemixConfig.model_validate(data["remix"])
        manifest.metadata = VideoMetadata.model_validate(data["metadata"])
        if "source" in data:
            manifest.source = MediaSource.model_validate(data["source"])
        if "utterances" in data:
            manifest.utterances = [Utterance.model_validate(u) for u in data["utterances"]]
        self.save(manifest)
        latest.unlink(missing_ok=True)
        self.append_receipt(
            run_id=run_id,
            stage="human_edit",
            actor="human",
            action="undo",
            summary=f"撤销人工修改: {data.get('label', data['snapshot_id'])}",
            diff={"restored_snapshot": data["snapshot_id"]},
        )
        return manifest, data
