"""Script to train RL agent with RSL-RL."""

<<<<<<< HEAD

from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple
=======
import logging
>>>>>>> 18764564cdf3b77fe4719b26e986ab099b93c6ba
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlOnPolicyRunnerCfg
  registry_name: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  wandb_run_path: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


<<<<<<< HEAD

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
=======
def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

>>>>>>> 18764564cdf3b77fe4719b26e986ab099b93c6ba
  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  registry_name: str | None = None

<<<<<<< HEAD

  if isinstance(cfg.env, TrackingEnvCfg):
      if not cfg.registry_name:
          raise ValueError("Must provide --registry-name for tracking tasks.")
=======
  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    if not cfg.registry_name:
      raise ValueError("Must provide --registry-name for tracking tasks.")
>>>>>>> 18764564cdf3b77fe4719b26e986ab099b93c6ba

      registry_name = cast(str, cfg.registry_name)
      if ":" not in registry_name:
          registry_name = registry_name + ":latest"

<<<<<<< HEAD
      # Resolve from local_wandb store (or direct path)
      motion_npz_path = resolve_local_wandb_artifact_path(registry_name)
      cfg.env.commands.motion.motion_file = str(motion_npz_path)
=======
    api = wandb.Api()
    artifact = api.artifact(registry_name)
>>>>>>> 18764564cdf3b77fe4719b26e986ab099b93c6ba

    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      # Load checkpoint from W&B.
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = OnPolicyRunner

  runner_kwargs = {}
  if is_tracking_task:
    runner_kwargs["registry_name"] = registry_name

  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
