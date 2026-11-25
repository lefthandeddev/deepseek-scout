"""deepseek-scout MCP server.

Two tools that offload codebase exploration from Claude to a fast DeepSeek
agent, returning compact, citation-verified handoffs:

  find             — Explore-agent replacement (question in, verified brief out)
  map_dependencies — deterministic import graph + blast-radius narrative

Register with:
  claude mcp add --scope user deepseek-scout -- <venv-python> <path-to-server.py>
"""

import json

from mcp.server import MCPServer

from scout.agent import (
    narrate_subgraph,
    render_claim_report,
    render_handoff,
    render_verdict,
    run_find,
    run_sanity_check,
    run_verify_claims,
)
from scout.config import MAX_EDGES_RENDERED
from scout.depgraph import bfs_tree, build_graph, match_target, render_tree
from scout.gitutils import resolve_exploration_root, sweep_stale_worktrees

mcp = MCPServer("deepseek-scout")


@mcp.tool()
def find(
    question: str,
    root_dir: str,
    thoroughness: str = "medium",
    thinking: str = "auto",
    git_ref: str = "",
    max_seconds: int = 0,
    format: str = "markdown",
) -> str:
    """Delegate a codebase exploration question to a fast scout agent.

    Use this INSTEAD of the built-in Explore agent or manual Glob/Grep sweeps
    whenever you need to locate code, understand how something works, or find
    where a change belongs. The scout runs its own search loop in a separate
    context and returns a compact handoff: a direct answer, key file:line
    locations (mechanically verified to exist), gotchas, and unexplored areas.
    Trust the citations — read only the cited files you need next.

    Args:
        question: Natural-language question about the codebase. Include intent
            (why you're asking) for a sharper handoff.
        root_dir: Absolute path of the repo/directory to explore.
        thoroughness: "quick" (first solid answer), "medium" (answer plus
            context, default), or "thorough" (exhaustive sweep). Controls how
            many search iterations the scout gets.
        thinking: Reasoning depth of the scout model — pick per question:
            "none" for simple where-is-X lookups (fastest), "low" for routine
            questions, "medium"/"high" for tangled control flow, cross-cutting
            architecture questions, or suspected-bug hunts. "max" is the
            highest effort. "auto" (default, or the DEEPSEEK_THINKING env
            pin) uses the provider's default effort.
        git_ref: Explore a clean snapshot of this commit/branch/tag (e.g.
            "HEAD", "main", a sha) instead of the working tree. Use whenever
            other agents may be editing concurrently, or you need citations
            pinned to a known commit. The target repo is never written to —
            the snapshot is an ephemeral worktree that is removed afterwards.
        max_seconds: Soft wall-clock budget. On breach the scout hands off
            what it has, marked PARTIAL. 0 (default) = no time cap.
        format: "markdown" (default) or "json" for the raw structured handoff
            (feed citations straight into reports/ADRs).

    Every handoff is stamped with the explored commit sha and dirty-file
    count, and each citation carries its file's last-change date.
    """
    handoff = run_find(question, root_dir, thoroughness, thinking, git_ref, max_seconds)
    return json.dumps(handoff, indent=2) if format == "json" else render_handoff(handoff)


@mcp.tool()
def map_dependencies(
    target: str,
    root_dir: str,
    depth: int = 3,
    question: str = "",
    narrative: bool = True,
    thinking: str = "auto",
    git_ref: str = "",
) -> str:
    """Map the verified import graph around a file: dependencies and dependents.

    Use this BEFORE refactoring or editing a shared file, or when asked what
    depends on something / what breaks if a file changes. Edges are built by
    parsing imports (Python ast, JS/TS import syntax) — they are facts, not
    LLM guesses. Optionally adds a short blast-radius narrative.

    Args:
        target: Repo-relative file path or filename to map (e.g. "scout/agent.py"
            or just "agent.py").
        root_dir: Absolute path of the repo to analyze.
        depth: Transitive depth to walk in each direction (default 3).
        question: Optional specific question to focus the narrative on.
        narrative: Set false to skip the LLM narrative and get just the graph.
        thinking: Reasoning depth for the narrative pass — "none"/"low" for a
            quick brief (default "auto"), "medium"/"high"/"max" when the
            blast radius is large or the question is subtle.
        git_ref: Map a clean snapshot of this commit/branch instead of the
            working tree (ephemeral worktree; target repo never written to).
    """
    explore_root, _repo_info, cleanup = resolve_exploration_root(root_dir, git_ref)
    try:
        return _map_dependencies_inner(target, explore_root, depth, question, narrative, thinking)
    finally:
        cleanup()


def _map_dependencies_inner(
    target: str, root_dir: str, depth: int, question: str, narrative: bool, thinking: str
) -> str:
    deps, reverse = build_graph(root_dir)
    matches = match_target(target, deps, reverse)

    if not matches:
        return (
            f"No Python/JS/TS file matching '{target}' found in the import graph of {root_dir}. "
            "It may have no imports/importers, or check the spelling."
        )
    if len(matches) > 6:
        listing = "\n".join(f"- {m}" for m in matches[:20])
        return f"'{target}' is ambiguous — {len(matches)} matches:\n{listing}\n\nCall again with a more specific path."

    sections = []
    for node in matches:
        down = bfs_tree(node, deps, depth)
        up = bfs_tree(node, reverse, depth)
        deps_text = render_tree(f"{node} depends on (depth {depth})", node, down, MAX_EDGES_RENDERED)
        dependents_text = render_tree(f"Files that depend on {node} (depth {depth})", node, up, MAX_EDGES_RENDERED)
        section = f"{deps_text}\n\n{dependents_text}"
        if narrative:
            try:
                brief = narrate_subgraph(node, question, deps_text, dependents_text, thinking)
                section += f"\n\n### Blast-radius brief\n{brief}"
            except Exception as e:
                section += f"\n\n(Narrative skipped: {e})"
        sections.append(section)

    return "\n\n---\n\n".join(sections)


@mcp.tool()
def sanity_check(
    recommendation: str,
    context: str = "",
    root_dir: str = "",
    thinking: str = "auto",
) -> str:
    """Adversarial second opinion on a recommendation before acting on it.

    Call this ONLY when the user has pre-authorized acting on trusted
    recommendations (e.g. said "trust your recommendation", "proceed if the
    checker agrees", or enabled a similar standing arrangement). Protocol:

    - Verdict NO OBJECTION → the recommendation is validated; proceed under
      the pre-authorization without asking the user again.
    - Verdict OBJECTION → do NOT proceed on the pre-authorization. Present
      the objections to the user and reconfirm, or revise the recommendation
      and re-check.

    The checker is a separate model prompted to find substantive flaws only
    (factual errors, broken logic, unflagged destructive steps, missed
    better alternatives) — not style nits.

    Args:
        recommendation: The full recommendation being validated — what you
            propose to do and why, in enough detail to be judged.
        context: The user's goal, constraints, and relevant background.
        root_dir: STRONGLY RECOMMENDED when the recommendation makes claims
            about code — the checker gets read-only repo access and verifies
            those claims against the actual files before ruling.
        thinking: Reasoning depth; "auto" resolves to "high" for checks.
    """
    verdict = run_sanity_check(recommendation, context, root_dir, thinking)
    return render_verdict(verdict)


@mcp.tool()
def verify_claims(
    claims: list[dict],
    root_dir: str,
    git_ref: str = "",
    thinking: str = "auto",
    format: str = "markdown",
) -> str:
    """Re-measure reported facts against the actual code, claim by claim.

    Use this whenever an agent (or a document/ADR/report) states facts about
    code with file:line citations and you need them checked before accepting
    them — the "trust but re-measure" loop, batched. Much cheaper than a full
    find() or sanity_check(): mechanical existence checks first, then one
    model pass judging each claim strictly against the code at its citation.

    Verdicts per claim:
    - CONFIRMED — the code at the citation does what the claim says
    - REFUTED — the code contradicts the claim (evidence says how)
    - NOT_FOUND — bad path/line, or nothing relevant at the citation

    This is fact-checking, not judgment: it never rules on whether the code
    is good, only whether the stated fact is true at the stated location.

    Args:
        claims: List of {"claim": str, "path": str, "line": int?, "search": str?}.
            path is repo-relative; line optional (file head is used if absent);
            search optionally adds repo-wide grep hits for that term as extra
            evidence. Max 20 claims per call.
        root_dir: Absolute path of the repo the claims are about.
        git_ref: Verify against a clean snapshot of this commit/branch (e.g.
            the commit the report was written at) instead of the working tree.
        thinking: Reasoning depth; "auto" resolves to "medium".
        format: "markdown" (default) or "json" for structured results.
    """
    report = run_verify_claims(claims, root_dir, git_ref, thinking)
    return json.dumps(report, indent=2) if format == "json" else render_claim_report(report)


if __name__ == "__main__":
    sweep_stale_worktrees()
    mcp.run()
