#!/usr/bin/env python3
"""
Batch runner for csv_to_npz.py that streams per-file logs.

Example:
  python process_folder.py \
    --input-dir /path/to/folder \
    --script src/mjlab/scripts/csv_to_npz.py \
    --input-fps 30 --output-fps 50 --render --jobs 1 -v
"""

import argparse
import os
import sys
import subprocess
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Thread-safe printing so logs from parallel jobs don't interleave mid-line
_print_lock = threading.Lock()
def log(msg: str, *, end: str = "\n"):
    with _print_lock:
        print(msg, end=end, flush=True)

def find_project_root(script_path: Path) -> Path:
    """
    Walk up from the script path and return the first directory that looks like a project root.
    If none found, fall back to the script's directory.
    """
    cur = script_path.resolve()
    if cur.is_file():
        cur = cur.parent
    markers = {"pyproject.toml", "poetry.lock", "requirements.txt", ".git"}
    while True:
        if any((cur / m).exists() for m in markers):
            return cur
        if cur.parent == cur:
            return script_path.resolve().parent
        cur = cur.parent

def build_cmd(script_path: Path, csv_path: Path, output_name: str,
              in_fps: int, out_fps: int, render: bool):
    # Use the same Python interpreter running this script, with -u for unbuffered output
    cmd = [
        sys.executable, "-u", str(script_path),
        "--input-file", str(csv_path),
        "--output-name", output_name,
        "--input-fps", str(in_fps),
        "--output-fps", str(out_fps),
    ]
    if render:
        cmd.append("--render")
    return cmd

def run_one(idx: int, total: int, csv_path: Path, args, run_cwd: Path):
    """
    Run one file; stream output if requested; print a one-line result.
    Returns one of: 'ok', 'fail', 'skipped', 'dry'
    """
    prefix = f"[{idx}/{total}] [{csv_path.name}]"
    output_name = csv_path.stem if args.output_name_mode == "stem" else csv_path.name

    # Optional skip based on a plausible output location
    if args.skip_existing:
        probe_dir = args.output_dir or csv_path.parent
        candidate = probe_dir / f"{csv_path.stem}.npz"
        if candidate.exists():
            log(f"{prefix} SKIP — output exists: {candidate}")
            return "skipped"

    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    # Encourage unbuffered Python in the child so logs appear promptly
    env["PYTHONUNBUFFERED"] = "1"

    cmd = build_cmd(args.script_abs, csv_path, output_name,
                    args.input_fps, args.output_fps, args.render)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)  # just ensure exists

    if args.dry_run:
        log(f"{prefix} DRY — cwd={run_cwd} → {' '.join(cmd)}")
        return "dry"

    log(f"{prefix} ▶ start (cwd={run_cwd})")
    if args.verbose:
        log(f"{prefix} cmd: {' '.join(cmd)}")

    try:
        if args.stream:
            # Live stream combined stdout/stderr from the child
            with subprocess.Popen(
                cmd,
                cwd=str(run_cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as p:
                assert p.stdout is not None
                for line in p.stdout:
                    log(f"{prefix} {line.rstrip()}")
                rc = p.wait()
        else:
            completed = subprocess.run(
                cmd,
                cwd=str(run_cwd),
                env=env,
                capture_output=True,
                text=True,
            )
            rc = completed.returncode
            if args.verbose and completed.stdout:
                log(f"{prefix} STDOUT:\n{completed.stdout.rstrip()}")
            if completed.stderr:
                log(f"{prefix} STDERR:\n{completed.stderr.rstrip()}")

        if rc == 0:
            log(f"{prefix} ✅ done")
            return "ok"
        else:
            log(f"{prefix} ❌ failed (code {rc})")
            return "fail"

    except FileNotFoundError as e:
        log(f"{prefix} ❌ error: {e}. Tip: ensure the --script path is correct.")
        return "fail"
    except Exception as e:
        log(f"{prefix} ❌ error: {e}")
        return "fail"

def main():
    p = argparse.ArgumentParser(description="Process all CSVs in a folder with csv_to_npz.py (streaming logs).")
    p.add_argument("--input-dir", required=True, type=Path, help="Folder containing .csv files")
    p.add_argument("--glob", default="*.csv", help="Filename pattern to match (default: *.csv)")
    p.add_argument("--script", type=Path, default=Path("src/mjlab/scripts/csv_to_npz.py"),
                   help="Path to csv_to_npz.py (run with the current Python)")
    p.add_argument("--run-cwd", type=Path,
                   help="Directory to run the command from. Default: auto-detected project root near --script.")
    p.add_argument("--input-fps", type=int, default=30)
    p.add_argument("--output-fps", type=int, default=50)
    p.add_argument("--render", action="store_true", help="Pass --render to the script")
    p.add_argument("--jobs", type=int, default=1, help="Number of files to process in parallel (MuJoCo/EGL often prefers 1)")
    p.add_argument("--output-dir", type=Path, help="If the converter writes relative outputs, ensure this dir exists")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip if <stem>.npz already exists in output-dir (or alongside the CSV)")
    p.add_argument("--dry-run", action="store_true", help="Print commands instead of running")
    p.add_argument("--output-name-mode", choices=["stem", "filename"], default="stem",
                   help="Use CSV stem (no suffix) or full filename for --output-name")
    p.add_argument("--no-stream", action="store_false", dest="stream",
                   help="Do not stream child output; print only after completion")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)")
    args = p.parse_args()
    args.verbose = int(args.verbose or 0)

    if not args.input_dir.is_dir():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(2)

    # Resolve the script to an absolute file, then auto-detect a sensible run CWD
    args.script_abs = args.script.resolve()
    if not args.script_abs.exists():
        print(f"--script not found: {args.script_abs}", file=sys.stderr)
        sys.exit(2)

    run_cwd = args.run_cwd or find_project_root(args.script_abs)

    csv_files = sorted(args.input_dir.rglob(args.glob))
    if not csv_files:
        print(f"No files matching {args.glob} under {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Run CWD: {run_cwd}")
    print(f"Found {len(csv_files)} file(s). Starting…", flush=True)

    total = len(csv_files)
    counters = {"ok": 0, "skipped": 0, "fail": 0, "dry": 0}

    if args.jobs <= 1:
        for i, f in enumerate(csv_files, start=1):
            status = run_one(i, total, f, args, run_cwd)
            counters[status] = counters.get(status, 0) + 1
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            fut_map = {ex.submit(run_one, i, total, f, args, run_cwd): f
                       for i, f in enumerate(csv_files, start=1)}
            for fut in as_completed(fut_map):
                status = fut.result()
                counters[status] = counters.get(status, 0) + 1

    # Summary
    print("\nSummary:")
    print(f"  OK:      {counters.get('ok', 0)}")
    print(f"  Skipped: {counters.get('skipped', 0)}")
    print(f"  Failed:  {counters.get('fail', 0)}")
    print(f"  Dry-run: {counters.get('dry', 0)}")
    sys.exit(0 if counters.get('fail', 0) == 0 else 1)

if __name__ == "__main__":
    main()
