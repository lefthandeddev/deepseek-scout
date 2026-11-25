"""Read-only filesystem tools the scout agent can call.

Every path is resolved against the exploration root and rejected if it
escapes — the scout can never read outside the repo it was pointed at.
"""

import fnmatch
import re
from pathlib import Path

from .config import MAX_GLOB_RESULTS, MAX_GREP_CHARS, MAX_READ_LINES

IGNORE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".next", ".nuxt", "target", ".idea", ".vscode",
    ".tox", ".eggs", "coverage", ".coverage_html",
}

SENSITIVE_DIRS = {".aws", ".azure", ".gnupg", ".kube", ".secrets", ".ssh", "secrets"}
SENSITIVE_FILE_NAMES = {
    ".dockerconfigjson", ".dockercfg", ".git-credentials", ".netrc",
    ".npmrc", ".pypirc", "authorized_keys", "credentials",
    "credentials.json", "id_dsa", "id_ecdsa", "id_ed25519", "id_rsa",
    "known_hosts", "service-account.json",
}
SENSITIVE_FILE_SUFFIXES = {".jks", ".key", ".keystore", ".p12", ".pem", ".pfx"}
RG_SENSITIVE_GLOBS = (
    "!**/.env*", "!**/.dockerconfigjson", "!**/.dockercfg",
    "!**/.git-credentials", "!**/.netrc", "!**/.npmrc", "!**/.pypirc",
    "!**/authorized_keys", "!**/credentials", "!**/credentials.json",
    "!**/id_dsa", "!**/id_ecdsa", "!**/id_ed25519", "!**/id_rsa",
    "!**/known_hosts", "!**/service-account.json", "!**/*.jks",
    "!**/*.key", "!**/*.keystore", "!**/*.p12", "!**/*.pem", "!**/*.pfx",
    "!**/.aws/**", "!**/.azure/**", "!**/.gnupg/**", "!**/.kube/**",
    "!**/.secrets/**", "!**/.ssh/**", "!**/secrets/**",
)
SECRET_VALUE_PATTERNS = (
    re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
    re.compile(
        r"(?i)\b(?:sk-[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        r"gh[pousr]_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|"
        r"AIza[0-9A-Za-z_-]{30,}|xox[baprs]-[A-Za-z0-9-]{20,})\b"
    ),
    re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/=-]{20,}"),
    re.compile(
        r"(?i)([\"']?\b(?:api[_-]?key|client[_-]?secret|access[_-]?token|"
        r"refresh[_-]?token|password)\b[\"']?\s*[:=]\s*[\"'])[^\"']{12,}([\"'])"
    ),
)

TEXT_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".json", ".md",
    ".txt", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".html", ".css",
    ".scss", ".sql", ".sh", ".ps1", ".bat", ".go", ".rs", ".java", ".c",
    ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".vue", ".svelte", ".xml",
    ".env", ".gitignore", ".dockerfile",
}


def _is_ignored(path: Path) -> bool:
    return any(part in IGNORE_DIRS for part in path.parts)


def _is_sensitive(path: Path) -> bool:
    parts = tuple(part.lower() for part in path.parts)
    name = path.name.lower()
    return (
        any(part in SENSITIVE_DIRS for part in parts[:-1])
        or name.startswith(".env")
        or name in SENSITIVE_FILE_NAMES
        or Path(name).suffix in SENSITIVE_FILE_SUFFIXES
    )


def _redact_secrets(text: str) -> str:
    private_key_pattern, token_pattern, bearer_pattern, assignment_pattern = SECRET_VALUE_PATTERNS
    text = private_key_pattern.sub(
        lambda match: "[REDACTED PRIVATE KEY]" + "\n" * match.group(0).count("\n"),
        text,
    )
    text = token_pattern.sub("[REDACTED]", text)
    text = bearer_pattern.sub(r"\1[REDACTED]", text)
    text = assignment_pattern.sub(r"\1[REDACTED]\2", text)
    return text


class LocalTools:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Exploration root does not exist: {self.root}")
        self.files_read: set[str] = set()
        self.call_count = 0

    # ---------- helpers ----------

    def _resolve(self, rel_path: str) -> Path:
        p = (self.root / rel_path).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"Path escapes exploration root: {rel_path}")
        if _is_sensitive(p.relative_to(self.root)) or _is_ignored(p.relative_to(self.root)):
            raise ValueError("Path is excluded from Scout exploration")
        # Reject symlinks and sensitive aliases as well as escaped targets.
        raw = self.root / rel_path
        try:
            lexical = raw.relative_to(self.root)
        except ValueError:
            raise ValueError("Path escapes exploration root") from None
        if ".." in lexical.parts or _is_sensitive(lexical) or _is_ignored(lexical):
            raise ValueError("Path is excluded from Scout exploration")
        if any((self.root.joinpath(*lexical.parts[:i])).is_symlink() for i in range(1, len(lexical.parts) + 1)):
            raise ValueError("Symlinks are excluded from Scout exploration")
        return p

    def _rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix()

    def _walk_files(self):
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and not p.is_symlink():
                rel = p.relative_to(self.root)
                if not _is_ignored(rel) and not _is_sensitive(rel):
                    try:
                        self._resolve(rel.as_posix())
                    except ValueError:
                        continue
                    yield p

    # ---------- tools ----------

    def glob(self, pattern: str) -> str:
        """List files matching a glob pattern like **/*.py."""
        self.call_count += 1
        matches = []
        for p in self._walk_files():
            rel = self._rel(p)
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(p.name, pattern):
                matches.append(rel)
                if len(matches) >= MAX_GLOB_RESULTS:
                    matches.append(f"... truncated at {MAX_GLOB_RESULTS} results")
                    break
        return "\n".join(matches) if matches else f"No files match: {pattern}"

    def list_dir(self, rel_path: str = ".", depth: int = 2) -> str:
        """Depth-limited directory tree."""
        self.call_count += 1
        base = self._resolve(rel_path)
        if not base.is_dir():
            return f"Not a directory: {rel_path}"
        lines: list[str] = []

        def walk(d: Path, level: int):
            if level > depth or len(lines) > 300:
                return
            entries = sorted(d.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
            for e in entries:
                rel = e.relative_to(self.root)
                if e.is_symlink() or e.name in IGNORE_DIRS or _is_sensitive(rel):
                    continue
                indent = "  " * level
                if e.is_dir():
                    lines.append(f"{indent}{e.name}/")
                    walk(e, level + 1)
                else:
                    lines.append(f"{indent}{e.name}")

        walk(base, 0)
        if len(lines) > 300:
            lines = lines[:300] + ["... truncated"]
        return "\n".join(lines) if lines else "(empty)"

    def grep(self, pattern: str, glob: str = "", max_results: int = 60) -> str:
        """Regex search across the repo. Returns path:line:text matches."""
        self.call_count += 1
        # Scan and redact whole file contents before extracting matching lines.
        # This avoids leaking fragments of multiline credentials via grep.
        return self._python_grep(pattern, glob, max_results)

    def _python_grep(self, pattern: str, glob: str, max_results: int) -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex: {e}"
        results: list[str] = []
        for p in self._walk_files():
            rel = self._rel(p)
            if glob and not (fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(p.name, glob)):
                continue
            if p.suffix.lower() not in TEXT_EXTENSIONS and p.suffix:
                continue
            if p.stat().st_size > 2_000_000:
                continue
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            per_file = 0
            raw_lines = content.splitlines()
            safe_lines = _redact_secrets(content).splitlines()
            for i, line in enumerate(raw_lines, 1):
                if rx.search(line):
                    safe = safe_lines[i - 1] if i <= len(safe_lines) else "[REDACTED]"
                    results.append(f"{rel}:{i}:{safe.strip()[:300]}")
                    per_file += 1
                    if per_file >= 5:
                        break
            if len(results) >= max_results:
                results.append(f"... truncated at {max_results} matches")
                break
        if not results:
            return f"No matches for: {pattern}"
        return _redact_secrets("\n".join(results)[:MAX_GREP_CHARS])

    def read_file(self, rel_path: str, start_line: int = 1, end_line: int = 0) -> str:
        """Read a file slice with line numbers. end_line=0 means start+MAX lines."""
        self.call_count += 1
        p = self._resolve(rel_path)
        if _is_sensitive(p.relative_to(self.root)):
            return f"Refusing to read sensitive path: {rel_path}"
        if not p.is_file():
            return f"Not a file: {rel_path}"
        if p.stat().st_size > 5_000_000:
            return f"File too large to read: {rel_path} ({p.stat().st_size} bytes)"
        try:
            lines = _redact_secrets(p.read_text(encoding="utf-8", errors="replace")).splitlines()
        except OSError as e:
            return f"Cannot read {rel_path}: {e}"
        total = len(lines)
        start = max(1, start_line)
        end = min(total, end_line if end_line > 0 else start + MAX_READ_LINES - 1)
        end = min(end, start + MAX_READ_LINES - 1)
        self.files_read.add(self._rel(p))
        header = f"# {self._rel(p)} — lines {start}-{end} of {total}\n"
        body = "\n".join(f"{n:5d}| {lines[n - 1]}" for n in range(start, end + 1))
        if end < total:
            body += f"\n... ({total - end} more lines; call read_file again with start_line={end + 1})"
        return _redact_secrets(header + body)
