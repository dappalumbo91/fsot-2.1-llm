"""
FSOT checkpoint version registry — circle back on poison/failure.

Philosophy (Fluid Spacetime Omni-Theory):
  Weights are a local observer amplitude. Versioning preserves *fold purity*
  history so suction–poof failures can be rolled back without parameter bloat.

Large .safetensors stay on disk (gitignored). Git tracks code + manifests +
registry pointers. Optional git tags mark code states that pair with a fold.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fsot_llm.paths import workspace_root


def full_ft_root() -> Path:
    return workspace_root() / "llm" / "models" / "full_ft"


def registry_path() -> Path:
    return full_ft_root() / "registry.json"


def load_registry() -> dict[str, Any]:
    p = registry_path()
    if not p.is_file():
        return {
            "schema": "fsot_full_ft_registry_v1",
            "ontology": "FSOT 2.1",
            "rule": "Rollback on anti-poison failure; never grow N-params as the fix",
            "active": None,
            "versions": [],
        }
    return json.loads(p.read_text(encoding="utf-8"))


def save_registry(reg: dict[str, Any]) -> None:
    full_ft_root().mkdir(parents=True, exist_ok=True)
    registry_path().write_text(json.dumps(reg, indent=2), encoding="utf-8")


def _git_head() -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(workspace_root()),
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def register_version(
    *,
    version_id: str,
    fold: str,
    path: Path,
    parent: Optional[str] = None,
    scores: Optional[dict[str, float]] = None,
    notes: str = "",
    train_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    reg = load_registry()
    entry = {
        "version_id": version_id,
        "fold": fold,
        "path": str(path.resolve()),
        "parent": parent or reg.get("active"),
        "scores": scores or {},
        "notes": notes,
        "train_meta": train_meta or {},
        "git_head": _git_head(),
        "written_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate",
    }
    # drop duplicate id
    reg["versions"] = [v for v in reg["versions"] if v.get("version_id") != version_id]
    reg["versions"].append(entry)
    save_registry(reg)
    # per-checkpoint manifest (tracked by git if small)
    man = path / "fsot_version_manifest.json"
    path.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    return entry


def promote(version_id: str) -> dict[str, Any]:
    reg = load_registry()
    found = None
    for v in reg["versions"]:
        if v.get("version_id") == version_id:
            v["status"] = "promoted"
            found = v
        elif v.get("status") == "promoted":
            v["status"] = "superseded"
    if found is None:
        raise KeyError(version_id)
    reg["active"] = version_id
    # pointer file for loaders
    (full_ft_root() / "latest.txt").write_text(
        found["path"], encoding="utf-8"
    )
    save_registry(reg)
    return found


def reject(version_id: str, reason: str = "anti_poison_fail") -> dict[str, Any]:
    reg = load_registry()
    found = None
    for v in reg["versions"]:
        if v.get("version_id") == version_id:
            v["status"] = "rejected"
            v["reject_reason"] = reason
            found = v
    if found is None:
        raise KeyError(version_id)
    save_registry(reg)
    return found


def rollback(to_version_id: Optional[str] = None) -> dict[str, Any]:
    """Activate previous promoted parent or explicit version_id."""
    reg = load_registry()
    if to_version_id is None:
        active = reg.get("active")
        parent = None
        for v in reg["versions"]:
            if v.get("version_id") == active:
                parent = v.get("parent")
                break
        if not parent:
            # last promoted excluding active
            promoted = [
                v
                for v in reg["versions"]
                if v.get("status") in ("promoted", "superseded")
                and v.get("version_id") != active
            ]
            if not promoted:
                raise RuntimeError("no parent version to roll back to")
            to_version_id = promoted[-1]["version_id"]
        else:
            to_version_id = parent
    return promote(to_version_id)


def get_active() -> Optional[dict[str, Any]]:
    reg = load_registry()
    aid = reg.get("active")
    if not aid:
        return None
    for v in reg["versions"]:
        if v.get("version_id") == aid:
            return v
    return None


def list_versions() -> list[dict[str, Any]]:
    return load_registry().get("versions") or []
