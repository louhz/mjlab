#!/usr/bin/env python3
"""
Batch-train policies for all motion folders that contain a `motion.npz`.

Per-folder command:
  MUJOCO_GL=egl train Mjlab-Tracking-Flat-Unitree-G1 \
      --registry-name <motion_dir> \
      --env.scene.num-envs 4096

Key points:
- No log files are created.
- Live streaming to the console (stdout) only.
- Preflight check that the 'train' launcher (or 'python') is runnable; or wrap with `--conda-env`.
- Two discovery modes: --parent (immediate subfolders) or --scan-root (recurse all runs).
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Set


def discover_from_parent(parent: Path, npz_pattern: str, include_hidden: bool = False) -> List[Path]:
    parent = parent.expanduser().resolve()
    if not parent.is_dir():
        raise FileNotFoundError(f"Parent directory does not exist or is not a directory: {parent}")

    motion_dirs = []
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        if not include_hidden and child.name.startswith("."):
            continue
        if any(child.rglob(npz_pattern)):
            motion_dirs.append(child.resolve())
    return motion_dirs


def discover_from_scanroot(scan_root: Path, npz_pattern: str) -> List[Path]:
    scan_root = scan_root.expanduser().resolve()
    if not scan_root.is_dir():
        raise FileNotFoundError(f"Scan root does not exist or is not a directory: {scan_root}")

    matches: Iterable[Path] = scan_root.glob(f"**/artifacts/motions/*/{npz_pattern}")
    motion_dirs: Set[Path] = set()
    for f in matches:
        if f.is_file():
            motion_dirs.add(f.parent.resolve())
    return sorted(motion_dirs)


def format_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def build_launch_cmd(
    base_train_cmd: List[str],
    model_name: str,
    motion_dir: Path,
    num_envs: int,
    extra_args: List[str],
    conda_env: Optional[str],
) -> List[str]:
    cmd = base_train_cmd + [
        model_name,
        "--registry-name",
        str(motion_dir),
        "--env.scene.num-envs",
        str(num_envs),
    ] + (extra_args or [])

    if conda_env:
        cmd = ["conda", "run", "-n", conda_env, "--no-capture-output"] + cmd
    return cmd


def preflight_or_die(train_cmd_tokens: List[str], conda_env: Optional[str]) -> None:
    """
    Ensure the launcher is runnable from this shell (or via `conda run`).
    We check only the first token of the *base* train command.
    """
    launcher = train_cmd_tokens[0]

    if conda_env:
        if shutil.which("conda") is None:
            print("ERROR: 'conda' not found on PATH but --conda-env was provided.", file=sys.stderr)
            sys.exit(127)
        return

    if launcher in ("python", "python3", sys.executable):
        return

    if shutil.which(launcher) is None:
        print(f"ERROR: '{launcher}' not found on PATH.", file=sys.stderr)
        print("Hint: run inside the environment where 'train' works,")
        print("      or use: --conda-env <env_name> (wraps with `conda run`),")
        print("      or set: --train-cmd 'python -m yourpkg.train'")
        sys.exit(127)


def run_one_training(
    base_train_cmd: List[str],
    model_name: str,
    motion_dir: Path,
    num_envs: int,
    extra_args: List[str],
    env_vars: dict,
    prefix_lines: bool,
    conda_env: Optional[str],
    workdir: Optional[Path],
    dry_run: bool,
    print_lock: threading.Lock,
) -> int:
    label = motion_dir.name
    cmd = build_launch_cmd(base_train_cmd, model_name, motion_dir, num_envs, extra_args, conda_env)

    with print_lock:
        print(f"\n=== Training: {label} ===")
        print(f"Dir: {motion_dir}")
        print(f"CMD: {format_cmd(cmd)}")
        if workdir:
            print(f"CWD: {workdir}")
        print(f"Env: MUJOCO_GL={env_vars.get('MUJOCO_GL','')}")
        sys.stdout.flush()

    if dry_run:
        return 0

    env = os.environ.copy()
    env.update(env_vars)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(workdir) if workdir else None,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    # Stream lines live to console
    for raw in proc.stdout:
        with print_lock:
            if prefix_lines:
                sys.stdout.write(f"[{label}] {raw}")
            else:
                sys.stdout.write(raw)
            sys.stdout.flush()

    rc = proc.wait()
    with print_lock:
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"=== Done: {label} -> {status} ===")
        sys.stdout.flush()
    return rc


def main():
    default_parent = Path(
        "local_wandb/csv_to_npz/20251025-024937-4931d0_dance1_subject1/artifacts/motions"
    )

    parser = argparse.ArgumentParser(description="Train across motion folders that contain a given .npz file (no logs).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--parent",
        type=Path,
        default=default_parent,
        help="Parent directory containing motion subfolders (typically .../artifacts/motions).",
    )
    group.add_argument(
        "--scan-root",
        type=Path,
        help="Scan all runs under this root (recursively finds **/artifacts/motions/*/<npz-pattern>).",
    )

    parser.add_argument("--npz-pattern", default="motion.npz", help='NPZ filename pattern (default: "motion.npz").')
    parser.add_argument("--model", default="Mjlab-Tracking-Flat-Unitree-G1", help="Train task/model name.")
    parser.add_argument("--num-envs", type=int, default=4096, help="--env.scene.num-envs value (default: 4096).")
    parser.add_argument(
        "--train-cmd",
        default="train",
        help='Training command (default: "train"). Examples: "train" | "python -m yourpkg.train".',
    )
    parser.add_argument("--mujoco-gl", default=os.environ.get("MUJOCO_GL", "egl"), help='MUJOCO_GL value (default: "egl").')
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel trainings (default: 1).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only (no execution).")
    parser.add_argument(
        "--prefix-lines",
        action="store_true",
        help="Prefix each output line with the motion folder name (enabled automatically when concurrency > 1).",
    )
    parser.add_argument(
        "--conda-env",
        default=None,
        help="If provided, wrap launches with `conda run -n <env>` (ensures the correct env for 'train').",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="Working directory for the training process (e.g., repository root for configs).",
    )
    parser.add_argument("remainder", nargs=argparse.REMAINDER, help="Extra args passed directly to `train` (prefix with --).")

    args = parser.parse_args()

    base_train_cmd = shlex.split(args.train_cmd)

    # Pass-through args after '--'
    extra_args = list(args.remainder or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    # Discover motion directories
    try:
        if args.scan_root:
            motion_dirs = discover_from_scanroot(args.scan_root, args.npz_pattern)
        else:
            motion_dirs = discover_from_parent(args.parent, args.npz_pattern)
    except Exception as e:
        print(f"Error discovering motion directories: {e}", file=sys.stderr)
        sys.exit(2)

    if not motion_dirs:
        target = str(args.scan_root or args.parent)
        print(f"No motion folders with pattern '{args.npz_pattern}' found under: {target}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(motion_dirs)} motion folder(s):")
    for d in motion_dirs:
        print(f"  - {d}")
    print(f"Using PATH: {os.environ.get('PATH','')}")
    sys.stdout.flush()

    # Preflight: ensure launcher is runnable (or we use conda run)
    preflight_or_die(base_train_cmd, args.conda_env)

    env_vars = {"MUJOCO_GL": args.mujoco_gl}

    # Auto-enable prefixing for readability when running in parallel
    prefix_lines = args.prefix_lines or (args.concurrency > 1)

    results = {}
    print_lock = threading.Lock()

    if args.concurrency <= 1:
        for d in motion_dirs:
            rc = run_one_training(
                base_train_cmd,
                args.model,
                d,
                args.num_envs,
                extra_args,
                env_vars,
                prefix_lines=prefix_lines,
                conda_env=args.conda_env,
                workdir=args.cwd,
                dry_run=args.dry_run,
                print_lock=print_lock,
            )
            results[d] = rc
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            fut_to_dir = {
                pool.submit(
                    run_one_training,
                    base_train_cmd,
                    args.model,
                    d,
                    args.num_envs,
                    extra_args,
                    env_vars,
                    prefix_lines,
                    args.conda_env,
                    args.cwd,
                    args.dry_run,
                    print_lock,
                ): d
                for d in motion_dirs
            }
            for fut in as_completed(fut_to_dir):
                d = fut_to_dir[fut]
                try:
                    rc = fut.result()
                except Exception as e:
                    with print_lock:
                        print(f"[{d.name}] crashed: {e}", file=sys.stderr)
                    rc = 999
                results[d] = rc

    # Summary
    ok = [d for d, rc in results.items() if rc == 0]
    bad = [(d, rc) for d, rc in results.items() if rc != 0]

    print("\n=== Summary ===")
    print(f"Success: {len(ok)}")
    for d in ok:
        print(f"  ✓ {d.name}")
    print(f"Failed: {len(bad)}")
    for d, rc in bad:
        print(f"  ✗ {d.name} (exit code {rc})")

    if not args.dry_run and bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
