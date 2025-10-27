"""Script to train RL agent with RSL-RL."""


from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import tyro

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class TrainConfig:
  env: Any
  agent: RslRlOnPolicyRunnerCfg
  registry_name: str | None = None
  device: str = "cuda:0"
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False



def _slugify_local_wandb(s: str) -> str:
    """Mirror local_wandb's slugify logic (must match for path lookups)."""
    out = []
    for ch in s.strip():
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch.lower())
        else:
            out.append("-")
    slug = []
    prev_dash = False
    for ch in out:
        if ch == "-" and prev_dash:
            continue
        prev_dash = (ch == "-")
        slug.append(ch)
    return "".join(slug).strip("-") or "unnamed"


def _parse_registry_name(registry_name: str) -> Tuple[str, str, str]:
    """
    Parse strings like:
      - "motions/my_collection:latest"
      - "wandb-registry-motions/my_collection:latest"
      - "entity/project/motions/my_collection:alias"
    Returns (type, name, alias).
    """
    # Ensure alias component exists for uniformity
    if ":" not in registry_name:
        registry_name = registry_name + ":latest"

    # Keep only the last two path segments before the alias, in case entity/project are present
    # Example: "entity/project/motions/foo:bar" -> "motions/foo:bar"
    m = re.search(r"([^/:]+)/([^/:]+):([^/:]+)$", registry_name)
    if not m:
        raise ValueError(f"Unrecognized registry name format: {registry_name}")
    type_or_regprefix, name, alias = m.group(1), m.group(2), m.group(3)

    # Accept either "motions" or "wandb-registry-motions"
    if type_or_regprefix.startswith("wandb-registry-"):
        art_type = type_or_regprefix.replace("wandb-registry-", "", 1)
    else:
        art_type = type_or_regprefix

    return _slugify_local_wandb(art_type), _slugify_local_wandb(name), alias


def _find_latest_linked_artifact(art_type: str, art_name: str, base_dir: Path) -> Optional[Path]:
    """
    Search all runs under base_dir for a registry link pointing to
    artifacts/<art_type>/<art_name>/...npz. Prefer the newest linked_at.
    """
    best_path: Optional[Path] = None
    best_ts: float = -1.0

    # Search pattern: <base>/<project>/<run>/registry/**/link.json
    for project_dir in base_dir.iterdir():
        if not project_dir.is_dir():
            continue
        for run_dir in project_dir.iterdir():
            if not run_dir.is_dir():
                continue
            reg_dir = run_dir / "registry"
            if not reg_dir.is_dir():
                continue

            # Look at all link.json files
            for link in reg_dir.rglob("link.json"):
                try:
                    with open(link, "r") as f:
                        meta = json.load(f)
                except Exception:
                    continue
                uri = str(meta.get("artifact_uri", "")).replace("\\", "/")
                linked_at = float(meta.get("linked_at", 0.0))

                # Expect something like: artifacts/<type>/<name>/motion.npz
                needle = f"artifacts/{art_type}/{art_name}/"
                if needle in uri:
                    run_root = run_dir
                    artifact_file = (run_root / uri).resolve()
                    # Prefer the registry file itself if it exists (copy/symlink target),
                    # else fall back to original artifact file path.
                    registry_dir = link.parent
                    # try to pick the .npz right next to link.json (copy/symlink destination)
                    candidates = list(registry_dir.glob("*.npz"))
                    candidate_path = candidates[0] if candidates else artifact_file
                    if candidate_path.exists() and linked_at >= best_ts:
                        best_ts = linked_at
                        best_path = candidate_path

    return best_path


def _fallback_find_artifact_file(art_type: str, art_name: str, base_dir: Path) -> Optional[Path]:
    """
    If no registry link exists, try raw artifact store:
    <base>/<project>/<run>/artifacts/<type>/<name>/*.npz (newest mtime wins).
    """
    best_path: Optional[Path] = None
    best_mtime: float = -1.0

    for project_dir in base_dir.iterdir():
        if not project_dir.is_dir():
            continue
        for run_dir in project_dir.iterdir():
            if not run_dir.is_dir():
                continue
            art_dir = run_dir / "artifacts" / art_type / art_name
            if not art_dir.is_dir():
                continue
            for f in art_dir.glob("*.npz"):
                try:
                    mt = f.stat().st_mtime
                except Exception:
                    continue
                if mt >= best_mtime:
                    best_mtime = mt
                    best_path = f.resolve()
    return best_path


def resolve_local_wandb_artifact_path(registry_name: str, *, env_var: str = "LOCAL_WANDB_DIR") -> Path:
    """
    Resolve a local_wandb artifact file (.npz) given a W&B-like registry name.
    Also accepts a direct filesystem path (file or directory) for convenience.
    """
    # 1) If user passed a direct path, use it.
    p = Path(registry_name)
    if p.exists():
        if p.is_dir():
            # e.g., a directory containing "motion.npz"
            candidate = p / "motion.npz"
            if candidate.exists():
                return candidate.resolve()
            # fall back to the first npz in the directory
            npzs = sorted(p.glob("*.npz"))
            if npzs:
                return npzs[0].resolve()
            raise FileNotFoundError(f"No .npz found in directory: {p}")
        else:
            # direct file
            return p.resolve()

    # 2) Parse W&B-like name and search the local_wandb store
    art_type, art_name, _alias = _parse_registry_name(registry_name)
    base_dir = Path(os.getenv(env_var, "./local_wandb")).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Local W&B directory not found: {base_dir} (set {env_var} to override)"
        )

    path = _find_latest_linked_artifact(art_type, art_name, base_dir)
    if path is None:
        path = _fallback_find_artifact_file(art_type, art_name, base_dir)
    if path is None:
        raise FileNotFoundError(
            f"Could not locate artifact '{art_type}/{art_name}' in {base_dir}. "
            f"Make sure it was logged with local_wandb."
        )
    return path


def run_train(task: str, cfg: TrainConfig) -> None:
  configure_torch_backends()

  registry_name: str | None = None


  if isinstance(cfg.env, TrackingEnvCfg):
      if not cfg.registry_name:
          raise ValueError("Must provide --registry-name for tracking tasks.")

      registry_name = cast(str, cfg.registry_name)
      if ":" not in registry_name:
          registry_name = registry_name + ":latest"

      # Resolve from local_wandb store (or direct path)
      motion_npz_path = resolve_local_wandb_artifact_path(registry_name)
      cfg.env.commands.motion.motion_file = str(motion_npz_path)

  # Enable NaN guard if requested
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  # Specify directory for logging experiments.
  log_root_path = Path("logs") / "rsl_rl" / cfg.agent.experiment_name
  log_root_path.resolve()
  print(f"[INFO] Logging experiment in directory: {log_root_path}")
  log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if cfg.agent.run_name:
    log_dir += f"_{cfg.agent.run_name}"
  log_dir = log_root_path / log_dir

  env = gym.make(
    task, cfg=cfg.env, device=cfg.device, render_mode="rgb_array" if cfg.video else None
  )

  resume_path = (
    get_checkpoint_path(log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint)
    if cfg.agent.resume
    else None
  )

  if cfg.video:
    env = gym.wrappers.RecordVideo(
      env,
      video_folder=os.path.join(log_dir, "videos", "train"),
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  if isinstance(cfg.env, TrackingEnvCfg):
    runner = MotionTrackingOnPolicyRunner(
      env, agent_cfg, str(log_dir), cfg.device, registry_name
    )
  else:
    runner = VelocityOnPolicyRunner(env, agent_cfg, str(log_dir), cfg.device)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
  dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def main():
  # Parse first argument to choose the task.
  task_prefix = "Mjlab-"
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(
      [k for k in gym.registry.keys() if k.startswith(task_prefix)]
    ),
    add_help=False,
    return_unknown_args=True,
  )
  del task_prefix

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  env_cfg = load_cfg_from_registry(chosen_task, "env_cfg_entry_point")
  agent_cfg = load_cfg_from_registry(chosen_task, "rl_cfg_entry_point")
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig(env=env_cfg, agent=agent_cfg),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del env_cfg, agent_cfg, remaining_args

  run_train(chosen_task, args)


if __name__ == "__main__":
  main()
