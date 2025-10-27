import os
import wandb
from rsl_rl.env.vec_env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.rl.exporter import (
    attach_onnx_metadata,
    export_motion_policy_as_onnx,
)

def _wandb_mode() -> str:
    """Return 'online' | 'offline' | 'disabled' | '' (unknown)."""
    env_mode = os.getenv("WANDB_MODE", "").strip().lower()
    if env_mode:
        return env_mode  # env wins
    try:
        import wandb  # type: ignore
        run = getattr(wandb, "run", None)
        if run is None:
            return ""
        settings = getattr(run, "settings", None)
        mode = getattr(settings, "mode", "") if settings is not None else ""
        return str(mode).strip().lower()
    except Exception:
        return ""

def _wandb_use_artifact_if_online(registry_name: str) -> None:
    """
    Call wandb.run.use_artifact(registry_name) only when online.
    In offline/disabled/unknown modes, do nothing.
    """
    mode = _wandb_mode()
    if mode in ("offline", "disabled", ""):
        # Skip artifact usage in offline or disabled modes
        return

    try:
        if hasattr(wandb.run, "use_artifact"):
            wandb.run.use_artifact(registry_name)  # type: ignore
    except Exception as e:
        # Handle errors gracefully (skip artifact if issues arise)
        print(f"[Warning] Failed to use artifact: {e}")

class MotionTrackingOnPolicyRunner(OnPolicyRunner):
    env: RslRlVecEnvWrapper

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        registry_name: str | None = None,
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            if self.alg.policy.actor_obs_normalization:
                normalizer = self.alg.policy.actor_obs_normalizer
            else:
                normalizer = None
            export_motion_policy_as_onnx(
                self.env.unwrapped,
                self.alg.policy,
                normalizer=normalizer,
                path=policy_path,
                filename=filename,
            )
            attach_onnx_metadata(
                self.env.unwrapped,
                wandb.run.name,  # type: ignore
                path=policy_path,
                filename=filename,
            )
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run only if W&B is online
            if self.registry_name is not None:
                _wandb_use_artifact_if_online(self.registry_name)
                self.registry_name = None
