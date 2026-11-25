"""Configuration: env vars, optional .env file, DeepSeek client factory."""

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

API_KEY_ENV = "DEEPSEEK_API_KEY"
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# Current DeepSeek model IDs: deepseek-flash (fast) and deepseek-v4-pro.
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")

# Thinking level, sent as OpenAI-style reasoning_effort. "none" disables
# reasoning; "auto" omits the parameter and uses DeepSeek's default effort.
# Pin globally with DEEPSEEK_THINKING;
# the find() tool's `thinking` argument overrides per call.
THINKING_LEVELS = ("auto", "none", "low", "medium", "high", "max")
THINKING_DEFAULT = os.environ.get("DEEPSEEK_THINKING", "auto")

# Model for sanity_check() — the gate before acting on a trusted
# recommendation, so it defaults to the heavier model. Exploration speed
# matters; verdict quality matters more.
CHECK_MODEL = os.environ.get("DEEPSEEK_CHECK_MODEL", "deepseek-v4-pro")

# Designated scratch area for ALL scout working artifacts (ephemeral
# worktrees, caches). Lives inside the scout repo and is gitignored —
# target repos are never written to.
WORK_DIR = PKG_ROOT / ".work"
WORKTREES_DIR = WORK_DIR / "worktrees"
WORKTREES_DIR.mkdir(parents=True, exist_ok=True)

MAX_CLAIMS_PER_CALL = 20

# Iteration caps for the find() agent loop, per thoroughness level.
ITERATION_CAPS = {"quick": 12, "medium": 30, "thorough": 60}

# Output caps so tool results never blow up the DeepSeek context.
MAX_GREP_CHARS = 12_000
MAX_READ_LINES = 400
MAX_GLOB_RESULTS = 200
MAX_EDGES_RENDERED = 150


def get_client():
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV} is not set. Set it in the environment or in "
            f"{PKG_ROOT / '.env'} (see .env.example)."
        )
    from openai import OpenAI

    return OpenAI(api_key=key, base_url=BASE_URL)
