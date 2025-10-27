# local_wandb.py
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "init",
    "log",
    "finish",
    "Video",
]

_ACTIVE_RUN: "Run | None" = None


def _slugify(s: str) -> str:
    out = []
    for ch in s.strip():
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch.lower())
        else:
            out.append("-")
    # collapse consecutive '-'
    slug = []
    prev_dash = False
    for ch in out:
        if ch == "-" and prev_dash:
            continue
        prev_dash = (ch == "-")
        slug.append(ch)
    return "".join(slug).strip("-") or "unnamed"


class Video:
    """Simple wrapper to signal a video file for logging."""
    def __init__(self, path: str, format: str = "mp4"):
        self.path = path
        self.format = format


@dataclass
class Artifact:
    path: str    # absolute path to copied artifact file
    name: str
    type: str
    uri: str     # run-relative path to the artifact file


class Run:
    def __init__(self, project: str, name: Optional[str], entity: Optional[str], base_dir: Optional[str] = None):
        self.project = project
        self.name = name or f"run-{uuid.uuid4().hex[:8]}"
        self.entity = entity
        self.started_at = time.time()
        self.id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

        base_dir = base_dir or os.environ.get("LOCAL_WANDB_DIR", "./local_wandb")
        self.root = os.path.join(base_dir, _slugify(project), f"{self.id}_{_slugify(self.name)}")

        self.artifacts_dir = os.path.join(self.root, "artifacts")
        self.registry_dir = os.path.join(self.root, "registry")
        self.media_dir = os.path.join(self.root, "media")
        self.logs_path = os.path.join(self.root, "logs.jsonl")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.registry_dir, exist_ok=True)
        os.makedirs(self.media_dir, exist_ok=True)

        # Persist basic run metadata
        with open(os.path.join(self.root, "config.json"), "w") as f:
            json.dump(
                {
                    "project": project,
                    "name": self.name,
                    "entity": entity,
                    "run_id": self.id,
                    "created_at": self.started_at,
                },
                f,
                indent=2,
            )

        print(f"[local-wandb] Run directory: {self.root}")

    # --- Public API (subset of wandb.Run) -----------------------------------

    def log_artifact(self, artifact_or_path: str, name: str, type: str) -> Artifact:
        """Copy an artifact file into this run's artifact store."""
        src = os.path.abspath(artifact_or_path)
        type_dir = os.path.join(self.artifacts_dir, _slugify(type))
        dest_dir = os.path.join(type_dir, _slugify(name))
        os.makedirs(dest_dir, exist_ok=True)
        dst = os.path.join(dest_dir, os.path.basename(src))
        shutil.copy2(src, dst)

        meta = {
            "name": name,
            "type": type,
            "src": src,
            "dst": os.path.abspath(dst),
            "logged_at": time.time(),
        }
        with open(os.path.join(dest_dir, "artifact.json"), "w") as f:
            json.dump(meta, f, indent=2)

        art = Artifact(
            path=os.path.abspath(dst),
            name=name,
            type=type,
            uri=os.path.relpath(dst, self.root),
        )
        print(f"[local-wandb] Logged artifact '{name}' (type '{type}') -> {art.path}")
        return art

    def link_artifact(self, artifact: Artifact, target_path: str) -> None:
        """Create a symlink or copy under run/registry/<target_path>/"""
        target_dir = os.path.join(self.registry_dir, _slugify(target_path))
        os.makedirs(target_dir, exist_ok=True)
        target = os.path.join(target_dir, os.path.basename(artifact.path))

        # Try symlink; fall back to copy if not permitted (e.g., Windows without admin)
        try:
            if os.path.lexists(target):
                os.remove(target)
            os.symlink(artifact.path, target)
            method = "symlinked"
        except Exception:
            shutil.copy2(artifact.path, target)
            method = "copied"

        with open(os.path.join(target_dir, "link.json"), "w") as f:
            json.dump({"artifact_uri": artifact.uri, "linked_at": time.time()}, f, indent=2)

        print(f"[local-wandb] {method} artifact -> {target}")

    def log(self, data: dict[str, Any]) -> None:
        """Append a JSONL log row; copy any Video payloads into media/."""
        row: dict[str, Any] = {"_time": time.time()}
        for k, v in data.items():
            if isinstance(v, Video):
                vids_dir = os.path.join(self.media_dir, "videos")
                os.makedirs(vids_dir, exist_ok=True)
                dst = os.path.join(vids_dir, os.path.basename(v.path))
                shutil.copy2(v.path, dst)
                row[k] = {"_type": "video", "path": os.path.relpath(dst, self.root), "format": v.format}
                print(f"[local-wandb] Logged video -> {dst}")
            else:
                # Must be JSON-serializable
                row[k] = v
        with open(self.logs_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def finish(self) -> None:
        runtime = time.time() - self.started_at
        with open(os.path.join(self.root, "summary.json"), "w") as f:
            json.dump({"runtime_sec": runtime}, f, indent=2)
        print(f"[local-wandb] Finished run. Runtime: {runtime:.2f}s")
        # nothing else to close


# --- Module-level helpers (to mimic wandb.<fn>) --------------------------------

def init(project: str, name: Optional[str] = None, entity: Optional[str] = None, **_: Any) -> Run:
    """Start a local run."""
    global _ACTIVE_RUN
    _ACTIVE_RUN = Run(project=project, name=name, entity=entity)
    return _ACTIVE_RUN


def log(data: dict[str, Any]) -> None:
    if _ACTIVE_RUN is None:
        raise RuntimeError("local_wandb.log() called before init()")
    _ACTIVE_RUN.log(data)


def finish() -> None:
    global _ACTIVE_RUN
    if _ACTIVE_RUN is not None:
        _ACTIVE_RUN.finish()
        _ACTIVE_RUN = None
