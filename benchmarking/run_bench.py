#!/usr/bin/env python3

import argparse
import os
import socket
import platform
import json
import subprocess
from pathlib import Path
from datetime import datetime

BENCH_DIR = Path("benchmarking")
RUNS_DIR = BENCH_DIR / "runs"
RESULT_DIR_NAME_FORMAT = "%Y_%m_%d_%H_%M_%S"


def run_bench(executable_path, executable_args):
    os.chdir(git_root())
    executable_path = Path(executable_path).absolute()
    if not executable_path.is_file():
        raise FileNotFoundError(f"Executable not found at path: {executable_path}")
    now = datetime.now()
    dir = create_run_dir(now)
    preamble(dir)  # TODO: Rename
    results_file = dir / "results.json"
    min_proc, max_proc = 1, 8
    benched_command = f"mpiexec -n {{nproc}} {executable_path} {executable_args}"

    command = [
        "hyperfine",
        "--export-json",
        f"{results_file}",
        "--warmup=1",
        "--parameter-scan",
        "nproc",
        f"{min_proc}",
        f"{max_proc}",
        benched_command,
    ]
    print(f"Running benchmark with command: {' '.join(command)}")
    # We're not actually using this for true timing, but rather for bounding the benchmark for other data gathering.
    start = datetime.now() 
    subprocess.run(command, cwd=dir, check=True)
    end = datetime.now()
    elapsed = end - start


def git_root():
    command = ["git", "rev-parse", "--show-toplevel"]
    output = subprocess.check_output(command, encoding="utf-8").strip()
    return Path(output)


def preamble(dir: Path):
    # Create a unique directory for this benchmark run
    context = gather_context()
    with open(dir / "context.json", "w") as f:
        json.dump(context, f, indent=4)
    write_dirty_changes(dir / "dirty_changes.patch")


def create_run_dir(date: datetime):
    timestamp = date.strftime(RESULT_DIR_NAME_FORMAT)
    result_dir = RUNS_DIR / f"benchmark_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir.absolute()


def write_dirty_changes(patch_name: Path):
    with open(patch_name, "w") as f:
        try:
            dirty_files = subprocess.check_output(
                ["git", "ls-files", "--others", "--exclude-standard"], encoding="utf-8"
            ).splitlines()
            for file in dirty_files:
                if not file.strip():
                    continue
                command = ["git", "--no-pager", "diff", "/dev/null", file]
                patch = subprocess.run(
                    command,
                    encoding="utf-8",
                    capture_output=True,
                )
                # By default, git will exit 1 if there are differences
                if patch.returncode != 1:
                    patch.check_returncode()
                f.write(patch.stdout)
        except Exception as e:
            print(f"Error gathering dirty changes: {e}")


def gather_context():
    # Gather context for the benchmark results for reference later
    context = {}
    # Get git information
    try:
        context["git"] = {
            "commit_hash": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                encoding="utf-8",
            ).strip(),
            "branch": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                encoding="utf-8",
            ).strip(),
            "tag": None,
        }
        ret = subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            encoding="utf-8",
            capture_output=True,
        )
        if ret.returncode == 0:
            context["git"]["tag"] = ret.stdout.strip()
    except Exception as e:
        print(f"Error gathering git information: {e}")
    # Get system information
    try:
        context["system"] = platform.uname()._asdict()
    except Exception as e:
        print(f"Error gathering system information: {e}")
    return context


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmarks and save results.")
    return parser.parse_args()


def main():
    args = parse_args()
    executable_path = "./target/release/find-dump"
    executable_args = Path(".").absolute()

    run_bench(executable_path, executable_args)


if __name__ == "__main__":
    main()
