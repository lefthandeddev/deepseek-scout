"""Git awareness: ephemeral worktrees, repo stamping, citation vintage.

Cleanliness contract: scout NEVER writes inside a target repo. All working
artifacts (worktrees of explored refs) live under the scout repo's own
gitignored .work/ folder and are removed after each call; a stale sweep at
server start clears leftovers from crashed runs.
"""

import shutil
import subprocess
import uuid
from pathlib import Path

from .config import WORKTREES_DIR

GIT_TIMEOUT = 30


def _git(repo: str | Path, *args: str) -> str | None:
    """Run a git command; return stripped stdout, or None on any failure."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=GIT_TIMEOUT,
            encoding="utf-8", errors="replace",
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def get_repo_info(root: str | Path) -> dict | None:
    """HEAD sha, branch, and dirty-file count — None if not a git repo."""
    sha = _git(root, "rev-parse", "HEAD")
    if not sha:
        return None
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD") or "detached"
    status = _git(root, "status", "--porcelain")
    dirty = len(status.splitlines()) if status else 0
    return {"sha": sha, "branch": branch, "dirty_files": dirty}


def resolve_exploration_root(root_dir: str, git_ref: str = ""):
    """Resolve where to explore.

    Returns (path_to_explore, repo_info_dict_or_None, cleanup_callable).

    No git_ref  -> explore the working tree as-is (repo info stamped if git).
    With git_ref -> create an ephemeral detached worktree of that ref under
    the scout repo's .work/worktrees/ and explore that snapshot. The target
    repo itself is never written to; cleanup removes the worktree.
    """
    root = Path(root_dir).resolve()
    if not git_ref:
        return str(root), get_repo_info(root), lambda: None

    sha = _git(root, "rev-parse", "--verify", f"{git_ref}^{{commit}}")
    if not sha:
        raise ValueError(
            f"git_ref '{git_ref}' could not be resolved in {root} "
            "(not a git repo, or unknown ref)"
        )

    wt_path = WORKTREES_DIR / f"{root.name}-{sha[:7]}-{uuid.uuid4().hex[:8]}"
    add = subprocess.run(
        ["git", "-C", str(root), "worktree", "add", "--detach", str(wt_path), sha],
        capture_output=True, text=True, timeout=120,
        encoding="utf-8", errors="replace",
    )
    if add.returncode != 0:
        raise RuntimeError(f"worktree add failed: {add.stderr.strip()[:300]}")

    def cleanup() -> None:
        removed = _git(root, "worktree", "remove", "--force", str(wt_path))
        if removed is None and wt_path.exists():
            shutil.rmtree(wt_path, ignore_errors=True)
        _git(root, "worktree", "prune")

    info = {"sha": sha, "branch": f"worktree@{git_ref}", "dirty_files": 0}
    return str(wt_path), info, cleanup


def file_last_commit_date(repo_root: str | Path, rel_path: str) -> str | None:
    """Date (YYYY-MM-DD) of the last commit touching rel_path, if known."""
    return _git(repo_root, "log", "-1", "--format=%cs", "--", rel_path) or None


def annotate_citation_dates(repo_root: str | Path, locations: list[dict]) -> None:
    """Stamp each key_location with its file's last-change date (in place)."""
    cache: dict[str, str | None] = {}
    for loc in locations or []:
        path = loc.get("path")
        if not path:
            continue
        if path not in cache:
            cache[path] = file_last_commit_date(repo_root, path)
        if cache[path]:
            loc["last_change"] = cache[path]


def sweep_stale_worktrees() -> int:
    """Best-effort removal of leftover worktrees from crashed runs."""
    removed = 0
    if not WORKTREES_DIR.exists():
        return 0
    for wt in WORKTREES_DIR.iterdir():
        if not wt.is_dir():
            continue
        parent = None
        gitfile = wt / ".git"
        try:
            # ".git" file content: "gitdir: <parent>/.git/worktrees/<name>"
            content = gitfile.read_text(encoding="utf-8", errors="replace").strip()
            if content.startswith("gitdir:"):
                meta = Path(content.split(":", 1)[1].strip())
                # <parent>/.git/worktrees/<name> -> <parent>
                parent = meta.parent.parent.parent
        except OSError:
            pass
        if parent and _git(parent, "worktree", "remove", "--force", str(wt)) is not None:
            _git(parent, "worktree", "prune")
        else:
            shutil.rmtree(wt, ignore_errors=True)
        removed += 1
    return removed
