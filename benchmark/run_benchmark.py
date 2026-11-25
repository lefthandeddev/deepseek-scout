"""Benchmark harness: run scout find() on a question set and save handoffs.

Usage:
    .venv\\Scripts\\python.exe benchmark\\run_benchmark.py [questions.json]

questions.json format (see questions.example.json):
    [{"question": "...", "root": "G:/some-repo", "thoroughness": "medium"}, ...]

Results land in benchmark/results/<timestamp>.md. To compare against native
Claude, ask Claude Code to run its built-in Explore agent on the same
questions and diff the handoffs by hand — tune scout/agent.py SYSTEM_PROMPT
until scout wins.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scout.agent import render_handoff, run_find  # noqa: E402


def main() -> None:
    questions_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "questions.json"
    if not questions_path.exists():
        print(f"No question set at {questions_path}. Copy questions.example.json to questions.json and edit.")
        sys.exit(1)

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.md"

    sections = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['question'][:70]}...")
        started = time.time()
        try:
            handoff = run_find(q["question"], q["root"], q.get("thoroughness", "medium"))
            body = render_handoff(handoff)
        except Exception as e:
            body = f"ERROR: {e}"
        sections.append(
            f"# Q{i}: {q['question']}\n\n_root: {q['root']} — {round(time.time() - started, 1)}s_\n\n{body}"
        )

    out_path.write_text("\n\n---\n\n".join(sections), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
