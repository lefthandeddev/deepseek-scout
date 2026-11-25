"""Mechanical verification of file:line citations in a scout handoff.

This is the quality gate native Claude subagents don't have: every location
the model claims must actually exist before the handoff is returned.
"""

from pathlib import Path


def verify_locations(root: str | Path, locations: list[dict]) -> list[str]:
    """Return a list of human-readable citation errors (empty = all valid)."""
    root = Path(root).resolve()
    errors: list[str] = []
    for i, loc in enumerate(locations or []):
        path = str(loc.get("path", "")).strip().replace("\\", "/")
        line = loc.get("line")
        if not path:
            errors.append(f"key_locations[{i}]: missing path")
            continue
        p = (root / path).resolve()
        if p != root and root not in p.parents:
            errors.append(f"key_locations[{i}]: path escapes root: {path}")
            continue
        if not p.is_file():
            errors.append(f"key_locations[{i}]: file does not exist: {path}")
            continue
        if line is not None:
            try:
                line_no = int(line)
            except (TypeError, ValueError):
                errors.append(f"key_locations[{i}]: non-integer line for {path}: {line!r}")
                continue
            try:
                total = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
            except OSError:
                errors.append(f"key_locations[{i}]: unreadable file: {path}")
                continue
            if not (1 <= line_no <= total):
                errors.append(
                    f"key_locations[{i}]: {path}:{line_no} is out of range (file has {total} lines)"
                )
    return errors
