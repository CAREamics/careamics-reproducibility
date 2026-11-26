import shutil
import subprocess
from importlib.metadata import distributions
from pathlib import Path


def save_run_files(run_dir, script):
    shutil.copytree(
        "src", run_dir / "code" / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copy(script, run_dir / "code" / Path(script).name)

    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions())
    (run_dir / "environment.txt").write_text("\n".join(packages) + "\n")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "HEAD"], capture_output=True, text=True, check=True
    ).stdout
    (run_dir / "git.diff").write_text(diff)

    return {"git_commit": commit, "git_dirty": bool(status)}
