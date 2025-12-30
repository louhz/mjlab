"""Script to play RL agent with RSL-RL.

Local-only checkpoint resolution:
- --wandb-run-path is treated as a local *hint* to find a run directory under:
      logs/rsl_rl/<experiment_name>/
  It never calls wandb.Api().
- The newest checkpoint (*.pt|*.pth|*.ckpt) inside that run directory is used.
- For tracking tasks, pass --motion-file explicitly (or extend this to auto-read params).

Example:
  play Mjlab-Tracking-Flat-Unitree-G1-Play \
    --wandb-run-path wandb/run-20251024_050948-gfg4luh3 \
    --motion-file /abs/path/to/motion.npz
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
<<<<<<< HEAD
from typing import Literal, Optional, cast, List
=======
from typing import Literal
>>>>>>> 18764564cdf3b77fe4719b26e986ab099b93c6ba

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

<<<<<<< HEAD
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
    load_cfg_from_registry,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserViewer
from mjlab.viewer.base import EnvProtocol


ViewerChoice = Literal["auto", "native", "viser"]
ResolvedViewer = Literal["native", "viser"]
=======
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
>>>>>>> 18764564cdf3b77fe4719b26e986ab099b93c6ba


@dataclass(frozen=True)
class PlayConfig:
<<<<<<< HEAD
    agent: Literal["zero", "random", "trained"] = "trained"

    # Use this as a *local* hint to find a run directory under logs/rsl_rl/<experiment>/
    wandb_run_path: str | None = None

    # Optional direct overrides
    checkpoint_file: str | None = None
    motion_file: str | None = None

    # Runtime
    num_envs: int | None = None
    device: str | None = None

    # Video / viewer
    video: bool = False
    video_length: int = 200
    video_height: int | None = None
    video_width: int | None = None
    camera: int | str | None = None
    viewer: ViewerChoice = "auto"


def _resolve_viewer_choice(choice: ViewerChoice) -> ResolvedViewer:
    if choice != "auto":
        return cast(ResolvedViewer, choice)
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved: ResolvedViewer = "native" if has_display else "viser"
    print(f"[INFO]: Auto-selected viewer: {resolved} (display detected: {has_display})")
    return resolved


def _collect_run_dirs(log_root_path: Path) -> List[Path]:
    if not log_root_path.exists():
        return []
    return [p for p in log_root_path.iterdir() if p.is_dir()]


def _find_run_dir_by_hint(log_root_path: Path, hint: str) -> Optional[Path]:
    """
    Try to find a run directory inside log_root_path that matches 'hint'.
    Matching strategy:
      1) exact folder name == basename(hint)
      2) substring match in folder name (e.g., trailing run id like gfg4luh3)
      3) if nothing matches, return None
    """
    hint = hint.strip()
    base = Path(hint).name  # e.g., "wandb/run-20251024_050948-gfg4luh3"
    token_candidates = {hint, base}

    # If the last 8-10 chars look like a run id, add it as a token candidate
    m = re.search(r'([a-z0-9]{8,10})$', base)
    if m:
        token_candidates.add(m.group(1))

    # Also add the full "run-YYYYMMDD_HHMMSS-<id>" if present
    m2 = re.search(r'(run-\d{8}_\d{6}-[a-z0-9]+)', base)
    if m2:
        token_candidates.add(m2.group(1))

    run_dirs = _collect_run_dirs(log_root_path)

    # 1) Exact match
    for d in run_dirs:
        if d.name == base:
            return d

    # 2) Substring match (any token)
    for tok in token_candidates:
        for d in run_dirs:
            if tok in d.name:
                return d

    return None


def _find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    """Search recursively for newest *.pt|*.pth|*.ckpt inside run_dir."""
    if not run_dir or not run_dir.exists():
        return None
    patterns = ["**/*.pt", "**/*.pth", "**/*.ckpt"]
    best: Optional[Path] = None
    best_mtime: float = -1.0
    for pat in patterns:
        for f in run_dir.glob(pat):
            if f.is_file():
                try:
                    mt = f.stat().st_mtime
                except Exception:
                    continue
                if mt >= best_mtime:
                    best_mtime = mt
                    best = f.resolve()
    return best


def _find_most_recent_run(log_root_path: Path) -> Optional[Path]:
    run_dirs = _collect_run_dirs(log_root_path)
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def run_play(task: str, cfg: PlayConfig):
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]: Using device: {device}")

    env_cfg = cast(
        ManagerBasedRlEnvCfg, load_cfg_from_registry(task, "env_cfg_entry_point")
    )
    agent_cfg = cast(
        RslRlOnPolicyRunnerCfg, load_cfg_from_registry(task, "rl_cfg_entry_point")
=======
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = (
    env_cfg.commands is not None
    and "motion" in env_cfg.commands
    and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    assert env_cfg.commands is not None
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    assert env_cfg.commands is not None
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    if DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require `registry_name` when using dummy agents."
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
>>>>>>> 18764564cdf3b77fe4719b26e986ab099b93c6ba
    )

    DUMMY_MODE = cfg.agent in {"zero", "random"}
    TRAINED_MODE = not DUMMY_MODE

<<<<<<< HEAD
    # -------------------------
    # Resolve motion file (Tracking tasks)
    # -------------------------
    if isinstance(env_cfg, TrackingEnvCfg):
        if cfg.motion_file is None:
            raise ValueError(
                "Tracking task requires a motion source. Please pass --motion-file <path to motion.npz>."
            )
        if not Path(cfg.motion_file).expanduser().exists():
            raise FileNotFoundError(f"Motion file not found: {cfg.motion_file}")
        env_cfg.commands.motion.motion_file = str(Path(cfg.motion_file).expanduser().resolve())
        print(f"[INFO]: Using motion file: {env_cfg.commands.motion.motion_file}")

    # -------------------------
    # Resolve checkpoint (trained mode) locally
    # -------------------------
    resume_path: Optional[Path] = None
    if TRAINED_MODE:
        if cfg.checkpoint_file is not None:
            resume_path = Path(cfg.checkpoint_file).expanduser().resolve()
            if not resume_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
            print(f"[INFO]: Using checkpoint file: {resume_path}")
        else:
            # Search under logs/rsl_rl/<experiment_name>/
            log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
            print(f"[INFO]: Looking for run directory in: {log_root_path}")

            run_dir: Optional[Path] = None
            if cfg.wandb_run_path:
                run_dir = _find_run_dir_by_hint(log_root_path, cfg.wandb_run_path)
                if run_dir:
                    print(f"[INFO]: Matched run directory: {run_dir}")
                else:
                    print("[WARN]: Could not match a run directory from --wandb-run-path hint; "
                          "falling back to most recent run.")
            if run_dir is None:
                run_dir = _find_most_recent_run(log_root_path)
                if run_dir:
                    print(f"[INFO]: Using most recent run directory: {run_dir}")
                else:
                    raise FileNotFoundError(
                        f"No run directories found under: {log_root_path}. "
                        "Provide --checkpoint-file explicitly."
                    )

            resume_path = _find_latest_checkpoint(run_dir)
            if not resume_path:
                raise FileNotFoundError(
                    f"No checkpoints (*.pt|*.pth|*.ckpt) found under: {run_dir}. "
                    "Provide --checkpoint-file explicitly."
                )
            print(f"[INFO]: Resolved checkpoint: {resume_path}")

    # Optional runtime overrides
    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width
    if cfg.camera is not None and hasattr(env_cfg.viewer, "camera"):
        env_cfg.viewer.camera = cfg.camera  # best-effort

    # Environment
    render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
    if cfg.video and DUMMY_MODE:
        print("[WARN] Video recording with dummy agents is disabled (no checkpoint).")

    env = gym.make(task, cfg=env_cfg, device=device, render_mode=render_mode)

    if TRAINED_MODE and cfg.video:
        print("[INFO] Recording videos during play")
        video_folder = Path("logs") / "rsl_rl" / agent_cfg.experiment_name / "videos" / "play"
        video_folder.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Policy
    if DUMMY_MODE:
        action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore

        if cfg.agent == "zero":
            class PolicyZero:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return torch.zeros(action_shape, device=env.unwrapped.device)
            policy = PolicyZero()
        else:
            class PolicyRandom:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1
            policy = PolicyRandom()
    else:
        # Use a dedicated play log dir
        play_log_dir = Path("logs") / "rsl_rl" / agent_cfg.experiment_name / "_play"
        play_log_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(env_cfg, TrackingEnvCfg):
            runner = MotionTrackingOnPolicyRunner(
                env, asdict(agent_cfg), log_dir=str(play_log_dir), device=device
            )
        else:
            runner = OnPolicyRunner(
                env, asdict(agent_cfg), log_dir=str(play_log_dir), device=device
            )
        runner.load(str(resume_path), map_location=device)
        policy = runner.get_inference_policy(device=device)

    # Viewer
    resolved_viewer = _resolve_viewer_choice(cfg.viewer)
    if resolved_viewer == "native":
        NativeMujocoViewer(cast(EnvProtocol, env), policy).run()
    elif resolved_viewer == "viser":
        ViserViewer(cast(EnvProtocol, env), policy).run()
    else:
        raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
=======
      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
>>>>>>> 18764564cdf3b77fe4719b26e986ab099b93c6ba

    env.close()


def main():
<<<<<<< HEAD
    # Pick the task from gym registry
    task_prefix = "Mjlab-"
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(
            [k for k in gym.registry.keys() if k.startswith(task_prefix)]
        ),
        add_help=False,
        return_unknown_args=True,
    )
    del task_prefix

    # Load default cfgs (we keep this to detect experiment_name)
    env_cfg = load_cfg_from_registry(chosen_task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(chosen_task, "rl_cfg_entry_point")
    assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

    args = tyro.cli(
        PlayConfig,
        args=remaining_args,
        default=PlayConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=(
            tyro.conf.AvoidSubcommands,
            tyro.conf.FlagConversionOff,
        ),
    )
    del env_cfg, agent_cfg, remaining_args
=======
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del remaining_args, agent_cfg
>>>>>>> 18764564cdf3b77fe4719b26e986ab099b93c6ba

    run_play(chosen_task, args)


if __name__ == "__main__":
    # Allow flat repo usage without installation
    try:
        import mjlab  # noqa: F401
    except ImportError:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))
    main()
