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
from typing import Literal, Optional, cast, List

import gymnasium as gym
import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

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


@dataclass(frozen=True)
class PlayConfig:
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
    )

    DUMMY_MODE = cfg.agent in {"zero", "random"}
    TRAINED_MODE = not DUMMY_MODE

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

    env.close()


def main():
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
