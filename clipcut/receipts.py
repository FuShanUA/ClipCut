from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .models import ProjectManifest, Receipt, utc_now_iso


class ProjectStore:
    """Manages project.json, append-only receipts.jsonl, undo stack, and pipeline.log."""

    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.work_dir / "project.json"
        self.receipts_path = self.work_dir / "receipts.jsonl"
        self.log_path = self.work_dir / "pipeline.log"
        self.undo_dir = self.work_dir / "undo"
        self.undo_dir.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.manifest_path.exists()

    def load(self) -> ProjectManifest:
        raw = self.manifest_path.read_text(encoding="utf-8")
        return ProjectManifest.model_validate_json(raw)

    def save(self, manifest: ProjectManifest) -> None:
        manifest.updated_at = utc_now_iso()
        manifest.revision += 1
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def log(self, message: str) -> None:
        line = f"[{utc_now_iso()}] {message}\n"
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def read_logs(self, tail: int = 200) -> list[str]:
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
        from .models import ClipItem, RemixConfig, VideoMetadata

        manifest.clips = [ClipItem.model_validate(c) for c in data["clips"]]
        manifest.remix = RemixConfig.model_validate(data["remix"])
        manifest.metadata = VideoMetadata.model_validate(data["metadata"])
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
