"""Deterministic import-graph builder for Python and JS/TS codebases.

The graph is built by parsing imports (Python: ast, JS/TS: regex), not by
asking an LLM to crawl — so edges are facts, not guesses. The LLM only
narrates the subgraph afterwards.
"""

import ast
import re
import time
from pathlib import Path

from .localtools import LocalTools

PY_EXT = {".py"}
JS_EXT = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
MAX_FILES = 8000
CACHE_TTL_SECONDS = 300

_JS_IMPORT_RX = re.compile(
    r"""(?:import\s+(?:[^'"]*?\s+from\s+)?|export\s+[^'"]*?\s+from\s+|require\(\s*)['"]([^'"]+)['"]""",
)

# root -> (built_at, deps, reverse)
_cache: dict[str, tuple[float, dict, dict]] = {}


def _collect_files(root: Path) -> list[Path]:
    files = []
    for p in LocalTools(root)._walk_files():
        if p.stat().st_size > 5_000_000:
            continue
        if p.suffix in PY_EXT | JS_EXT:
            files.append(p)
            if len(files) >= MAX_FILES:
                break
    return files


def _rel(p: Path, root: Path) -> str:
    return p.relative_to(root).as_posix()


def _resolve_py(module: str, level: int, cur_file: Path, root: Path, known: set[str]) -> str | None:
    """Resolve a Python import to a repo-relative file path, if it's local."""
    if level > 0:  # relative import
        base = cur_file.parent
        for _ in range(level - 1):
            base = base.parent
        parts = module.split(".") if module else []
        candidates_base = [base]
    else:
        parts = module.split(".")
        candidates_base = [root, root / "src"]
        # also try one level down for src-layout monorepos
    for cb in candidates_base:
        target = cb.joinpath(*parts) if parts else cb
        for candidate in (target.with_suffix(".py"), target / "__init__.py"):
            try:
                rel = candidate.resolve().relative_to(root).as_posix()
            except ValueError:
                continue
            if rel in known:
                return rel
        # fall back: strip trailing attribute (from a.b import name -> a/b.py)
        if parts:
            target2 = cb.joinpath(*parts[:-1]) if len(parts) > 1 else cb
            for candidate in (target2.with_suffix(".py"), target2 / "__init__.py"):
                try:
                    rel = candidate.resolve().relative_to(root).as_posix()
                except ValueError:
                    continue
                if rel in known:
                    return rel
    return None


def _resolve_js(spec: str, cur_file: Path, root: Path, known: set[str]) -> str | None:
    if not spec.startswith("."):
        return None  # external package
    base = (cur_file.parent / spec).resolve()
    candidates = [base]
    candidates += [base.with_suffix(base.suffix + ext) if base.suffix else base.with_suffix(ext)
                   for ext in JS_EXT]
    candidates += [base / f"index{ext}" for ext in JS_EXT]
    for c in candidates:
        try:
            rel = c.relative_to(root).as_posix()
        except ValueError:
            continue
        if rel in known:
            return rel
    return None


def build_graph(root: str | Path, force: bool = False) -> tuple[dict, dict]:
    """Return (deps, reverse): deps[file] = set of repo files it imports."""
    root = Path(root).resolve()
    key = str(root)
    now = time.time()
    if not force and key in _cache and now - _cache[key][0] < CACHE_TTL_SECONDS:
        return _cache[key][1], _cache[key][2]

    files = _collect_files(root)
    known = {_rel(p, root) for p in files}
    deps: dict[str, set[str]] = {}

    for p in files:
        rel = _rel(p, root)
        edges: set[str] = set()
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if p.suffix in PY_EXT:
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        r = _resolve_py(alias.name, 0, p, root, known)
                        if r and r != rel:
                            edges.add(r)
                elif isinstance(node, ast.ImportFrom):
                    r = _resolve_py(node.module or "", node.level, p, root, known)
                    if r and r != rel:
                        edges.add(r)
        else:
            for m in _JS_IMPORT_RX.finditer(text):
                r = _resolve_js(m.group(1), p, root, known)
                if r and r != rel:
                    edges.add(r)
        if edges:
            deps[rel] = edges

    reverse: dict[str, set[str]] = {}
    for src, targets in deps.items():
        for t in targets:
            reverse.setdefault(t, set()).add(src)

    _cache[key] = (now, deps, reverse)
    return deps, reverse


def match_target(target: str, deps: dict, reverse: dict) -> list[str]:
    """Match a user-supplied target string to known files."""
    target = target.replace("\\", "/").strip()
    known = sorted(set(deps) | set(reverse) | {f for s in deps.values() for f in s})
    exact = [f for f in known if f == target]
    if exact:
        return exact
    suffix = [f for f in known if f.endswith("/" + target) or f.endswith(target)]
    if suffix:
        return suffix
    return [f for f in known if target.lower() in f.lower()]


def bfs_tree(start: str, graph: dict, depth: int) -> list[tuple[int, str]]:
    """Depth-limited BFS returning (level, node) rows in discovery order."""
    seen = {start}
    rows: list[tuple[int, str]] = []
    frontier = [start]
    for level in range(1, depth + 1):
        nxt = []
        for node in frontier:
            for n in sorted(graph.get(node, ())):
                if n not in seen:
                    seen.add(n)
                    rows.append((level, n))
                    nxt.append(n)
        frontier = nxt
        if not frontier:
            break
    return rows


def render_tree(title: str, start: str, rows: list[tuple[int, str]], cap: int) -> str:
    lines = [f"### {title}", f"{start}"]
    for level, node in rows[:cap]:
        lines.append(f"{'  ' * level}└─ {node}")
    if len(rows) > cap:
        lines.append(f"... {len(rows) - cap} more files omitted")
    if not rows:
        lines.append("  (none found within the repo)")
    return "\n".join(lines)
