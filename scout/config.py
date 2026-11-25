"""Configuration: env vars, optional .env file, provider client factory."""

import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env_path = PKG_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

from functools import lru_cache
from .providers import Settings, ProviderClient, LEVELS

SETTINGS = Settings.from_env()
MODEL = SETTINGS.model
CHECK_MODEL = SETTINGS.check_model
THINKING_LEVELS = LEVELS
THINKING_DEFAULT = os.environ.get("SCOUT_THINKING", os.environ.get("DEEPSEEK_THINKING", "auto") if SETTINGS.provider == "deepseek" else "auto")

# Designated scratch area for ALL scout working artifacts (ephemeral
# worktrees, caches). Lives inside the scout repo and is gitignored —
# target repos are never written to.
WORK_DIR = PKG_ROOT / ".work"
WORKTREES_DIR = WORK_DIR / "worktrees"
WORKTREES_DIR.mkdir(parents=True, exist_ok=True)

MAX_CLAIMS_PER_CALL = 20

# Iteration caps for the find() agent loop, per thoroughness level.
ITERATION_CAPS = {"quick": 12, "medium": 30, "thorough": 60}

# Output caps so tool results never blow up the model context.
MAX_GREP_CHARS = 12_000
MAX_READ_LINES = 400
MAX_GLOB_RESULTS = 200
MAX_EDGES_RENDERED = 150


@lru_cache(maxsize=1)
def get_client():
    return ProviderClient(SETTINGS)
