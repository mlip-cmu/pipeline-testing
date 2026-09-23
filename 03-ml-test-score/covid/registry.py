"""File-based model registry: validates models before serving, supports canaries and rollback."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib

from covid.model import TrainedModel

TRACKED_DEPENDENCIES = ["numpy", "pandas", "scikit-learn", "joblib"]


class ModelRejected(Exception):
    pass


def dependency_versions() -> dict[str, str]:
    return {name: version(name) for name in TRACKED_DEPENDENCIES}


class ModelRegistry:
    def __init__(self, root: Path, min_auc: float = 0.8, min_recall: float = 0.5, max_auc_drop: float = 0.01):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.min_auc = min_auc
        self.min_recall = min_recall
        self.max_auc_drop = max_auc_drop

    def versions(self) -> list[int]:
        return sorted(int(p.name[1:]) for p in self.root.glob("v*") if p.is_dir())

    def current_version(self) -> int | None:
        pointer = self.root / "CURRENT"
        return int(pointer.read_text()) if pointer.exists() else None

    def metadata(self, v: int | None = None) -> dict:
        v = v or self.current_version()
        return json.loads((self.root / f"v{v}" / "metadata.json").read_text())

    def load(self, v: int | None = None) -> TrainedModel:
        v = v or self.current_version()
        if v is None:
            raise LookupError("No model deployed")
        return joblib.load(self.root / f"v{v}" / "model.joblib")

    def publish(self, model: TrainedModel, metrics: dict, canary=None, trained_at: datetime | None = None) -> int:
        self._validate(metrics)
        if canary is not None:
            canary(model)
        v = max(self.versions(), default=0) + 1
        directory = self.root / f"v{v}"
        directory.mkdir()
        joblib.dump(model, directory / "model.joblib")
        metadata = {
            "version": v,
            "trained_at": (trained_at or datetime.now(timezone.utc)).isoformat(),
            "metrics": metrics,
            "spec": {k: val for k, val in asdict(model.spec).items() if k != "notes"},
            "dependencies": dependency_versions(),
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
        self._activate(v)
        return v

    def rollback(self) -> int:
        current = self.current_version()
        older = [v for v in self.versions() if current is None or v < current]
        if not older:
            raise LookupError("No earlier model to roll back to")
        self._activate(older[-1])
        return older[-1]

    def _validate(self, metrics: dict) -> None:
        if metrics.get("auc", 0) < self.min_auc:
            raise ModelRejected(f"AUC {metrics.get('auc')} below minimum {self.min_auc}")
        if metrics.get("recall", 0) < self.min_recall:
            raise ModelRejected(f"Recall {metrics.get('recall')} below minimum {self.min_recall}")
        if self.current_version() is not None:
            current_auc = self.metadata()["metrics"]["auc"]
            if metrics["auc"] < current_auc - self.max_auc_drop:
                raise ModelRejected(f"AUC {metrics['auc']:.3f} worse than deployed model ({current_auc:.3f})")

    def _activate(self, v: int) -> None:
        (self.root / "CURRENT").write_text(str(v))
