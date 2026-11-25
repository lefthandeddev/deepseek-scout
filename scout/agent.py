"""DeepSeek agent loop for find(), plus the narrative pass for map_dependencies().

The loop gives DeepSeek read-only local tools and requires it to finish by
calling submit_handoff. Handoffs with invalid file:line citations are
rejected back into the loop and retried — Claude only ever sees verified
citations.
"""

import json
import time

from .citations import verify_locations
from .config import (
    CHECK_MODEL,
    ITERATION_CAPS,
    MAX_CLAIMS_PER_CALL,
    MODEL,
    THINKING_DEFAULT,
    THINKING_LEVELS,
    get_client,
)
from .gitutils import annotate_citation_dates, resolve_exploration_root
from .localtools import LocalTools

# Static on purpose: DeepSeek's automatic prefix cache is keyed on exact
# bytes, so a constant system prompt + constant tool schemas means every
# call after the first hits the cache. Per-call detail (root, question,
# thoroughness) goes in the user message.
SYSTEM_PROMPT = """You are a code exploration scout. You are given a question about a codebase and read-only tools to investigate it. The exploration root and thoroughness are stated in the user message.

Your job is to return a precise, evidence-backed handoff to a senior engineer (an AI coding agent) who will act on it WITHOUT redoing your search. Rules:

- Every claim must be backed by a file:line citation you actually saw via grep or read_file. Citations are mechanically verified; invented ones are rejected and waste your turn.
- Strategy: orient with list_dir/glob, locate with grep, confirm with read_file. Read excerpts, not whole files.
- Batch independent lookups: when your next steps don't depend on each other's results (several greps, reads of different files), issue them as multiple tool calls in a single turn.
- Return conclusions, not file dumps. The engineer needs to know WHERE things are and HOW they connect, not see the code again.
- Note things that look relevant but are NOT (gotchas), so the engineer doesn't chase them.
- Be honest about what you did not check (unexplored) and your confidence.
- When you have enough evidence, call submit_handoff. Do not keep exploring past the point of diminishing returns.

Thoroughness levels: quick = first solid answer; medium = answer plus immediate context; thorough = exhaustive sweep across naming conventions and locations."""

# All tool schemas are strict (DeepSeek v4 supports strict function calling,
# probed): every property required, additionalProperties false. Optional
# semantics are encoded as sentinel values (0 / "") instead of omitted keys.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "List files matching a glob pattern (e.g. **/*.py, src/**/*.ts).",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "Show a depth-limited directory tree to orient yourself.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path; '.' for the root"},
                    "depth": {"type": "integer", "description": "Tree depth; 2 is a good default"},
                },
                "required": ["path", "depth"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Regex search across the repo. Returns path:line:text matches.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression"},
                    "glob": {"type": "string", "description": "Filename filter, e.g. *.py — pass \"\" for no filter"},
                },
                "required": ["pattern", "glob"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a slice of a file with line numbers. Use after grep to confirm context.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "description": "1-based; pass 1 for start"},
                    "end_line": {"type": "integer", "description": "Pass 0 for start + 400 lines"},
                },
                "required": ["path", "start_line", "end_line"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_handoff",
            "description": "Deliver your final findings. Call this exactly once, when you have enough evidence.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Direct answer to the question, 2-6 sentences.",
                    },
                    "key_locations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "Repo-relative path"},
                                "line": {"type": "integer"},
                                "role": {"type": "string", "description": "One line: what this location is/does"},
                            },
                            "required": ["path", "line", "role"],
                            "additionalProperties": False,
                        },
                    },
                    "gotchas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Things that look relevant but aren't, or traps for the implementer. [] if none.",
                    },
                    "unexplored": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What you did not check. [] only if truly exhaustive.",
                    },
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["answer", "key_locations", "gotchas", "unexplored", "confidence"],
                "additionalProperties": False,
            },
        },
    },
]


def _resolve_thinking(level: str | None) -> str:
    level = (level or THINKING_DEFAULT or "auto").lower().strip()
    return level if level in THINKING_LEVELS else "auto"


def _create(client, thinking: str, *, force_tool: str | None = None,
            deterministic: bool = False, **kwargs):
    """DeepSeek-native completion call.

    - Forced tool_choice is incompatible with thinking mode (hard 400), so a
      forced call always runs with reasoning_effort="none" — forcing is only
      used for submit-now moments where structure matters more than depth.
    - deterministic=True sets temperature to 0.0 for verdict/verifier calls.
      DeepSeek ignores temperature in thinking mode, so this only affects
      non-thinking calls.
    - "auto" omits reasoning_effort and uses DeepSeek's default effort
      (currently high).
    """
    if force_tool:
        kwargs["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
        kwargs["reasoning_effort"] = "none"
    elif thinking != "auto":
        kwargs["reasoning_effort"] = thinking
    if deterministic:
        kwargs["temperature"] = 0.0
    return client.chat.completions.create(**kwargs)


def _assistant_message(message, tool_calls: list) -> dict:
    """Serialize an assistant turn for the next DeepSeek tool-call request."""
    assistant_msg: dict = {"role": "assistant", "content": message.content or ""}
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content is not None:
        # DeepSeek requires this field to be carried into later tool-call turns.
        assistant_msg["reasoning_content"] = reasoning_content
    if tool_calls:
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ]
    return assistant_msg


def _add_cache_usage(resp, stats: dict) -> None:
    """Accumulate DeepSeek prefix-cache usage (prompt_cache_hit/miss_tokens)."""
    u = getattr(resp, "usage", None)
    if not u:
        return
    stats["cache_hit_tokens"] = stats.get("cache_hit_tokens", 0) + (
        getattr(u, "prompt_cache_hit_tokens", 0) or 0
    )
    stats["cache_miss_tokens"] = stats.get("cache_miss_tokens", 0) + (
        getattr(u, "prompt_cache_miss_tokens", 0) or 0
    )


def _dispatch(tools: LocalTools, name: str, args: dict) -> str:
    try:
        if name == "glob":
            return tools.glob(args.get("pattern", "*"))
        if name == "list_dir":
            return tools.list_dir(args.get("path", "."), int(args.get("depth", 2)))
        if name == "grep":
            return tools.grep(args.get("pattern", ""), args.get("glob", ""))
        if name == "read_file":
            return tools.read_file(
                args.get("path", ""),
                int(args.get("start_line", 1)),
                int(args.get("end_line", 0)),
            )
        return f"Unknown tool: {name}"
    except Exception as e:  # tool errors go back to the model, not up the stack
        return f"Tool error: {e}"


def run_find(
    question: str,
    root: str,
    thoroughness: str = "medium",
    thinking: str | None = None,
    git_ref: str = "",
    max_seconds: int = 0,
) -> dict:
    """Run the exploration loop. Returns the validated handoff dict plus stats.

    git_ref explores an ephemeral worktree snapshot of that ref (target repo
    untouched); max_seconds is a soft wall-clock budget — on breach the scout
    is forced to hand off what it has, marked partial.
    """
    explore_root, repo_info, cleanup = resolve_exploration_root(root, git_ref)
    try:
        return _run_find_inner(
            question, explore_root, thoroughness, thinking, git_ref, max_seconds, repo_info
        )
    finally:
        cleanup()


def _run_find_inner(
    question: str,
    root: str,
    thoroughness: str,
    thinking: str | None,
    git_ref: str,
    max_seconds: int,
    repo_info: dict | None,
) -> dict:
    thoroughness = thoroughness if thoroughness in ITERATION_CAPS else "medium"
    level = _resolve_thinking(thinking)
    client = get_client()
    tools = LocalTools(root)
    cap = ITERATION_CAPS[thoroughness]
    started = time.time()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question: {question}\n\n"
                       f"Exploration root: {tools.root}\n"
                       f"Thoroughness: {thoroughness}",
        },
    ]
    submit_rejections = 0
    forced_attempts = 0
    out_of_time = False
    cache_stats: dict = {}
    deadline = started + max_seconds if max_seconds > 0 else None

    for iteration in range(cap + 4):
        out_of_time = deadline is not None and time.time() > deadline
        force_submit = iteration >= cap or out_of_time
        if force_submit:
            forced_attempts += 1
            if forced_attempts > 3:
                break
            messages.append({
                "role": "user",
                "content": "BUDGET EXHAUSTED. Call submit_handoff NOW with your findings "
                           "so far. Do not call any other tool.",
            })
        resp = _create(
            client,
            level,
            force_tool="submit_handoff" if force_submit else None,
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        _add_cache_usage(resp, cache_stats)
        msg = resp.choices[0].message
        tool_calls = [tc for tc in (msg.tool_calls or []) if getattr(tc, "function", None)]

        messages.append(_assistant_message(msg, tool_calls))

        if not tool_calls:
            messages.append(
                {"role": "user", "content": "Continue investigating, or call submit_handoff with your findings."}
            )
            continue

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if tc.function.name == "submit_handoff":
                errors = verify_locations(tools.root, args.get("key_locations", []))
                if errors and submit_rejections < 2 and not force_submit:
                    submit_rejections += 1
                    result = (
                        "HANDOFF REJECTED — invalid citations:\n"
                        + "\n".join(errors)
                        + "\nVerify with grep/read_file, fix the citations, and call submit_handoff again."
                    )
                else:
                    if errors:
                        # Final attempt: drop the bad citations rather than fail.
                        bad = {e.split(":")[0] for e in errors}
                        args["key_locations"] = [
                            loc
                            for i, loc in enumerate(args.get("key_locations", []))
                            if f"key_locations[{i}]" not in bad
                        ]
                        args.setdefault("gotchas", []).append(
                            "Some citations failed mechanical verification and were removed."
                        )
                    annotate_citation_dates(tools.root, args.get("key_locations", []))
                    args["_stats"] = {
                        "model": MODEL,
                        "thinking": level,
                        "tool_calls": tools.call_count,
                        "files_read": len(tools.files_read),
                        "citation_rejections": submit_rejections,
                        "elapsed_s": round(time.time() - started, 1),
                        "git_ref": git_ref or None,
                        "explored_sha": (repo_info or {}).get("sha"),
                        "branch": (repo_info or {}).get("branch"),
                        "dirty_files": (repo_info or {}).get("dirty_files"),
                        "partial": "time_budget_exhausted" if out_of_time else None,
                        **cache_stats,
                    }
                    return args
            else:
                result = _dispatch(tools, tc.function.name, args)

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # The model never produced a structured handoff despite forced attempts —
    # degrade to a low-confidence partial rather than failing the call.
    last_text = next(
        (m.get("content") for m in reversed(messages)
         if m.get("role") == "assistant" and m.get("content")),
        "",
    )
    return {
        "answer": (
            "Exploration budget exhausted before a structured handoff could be produced. "
            + (f"Last scout notes: {last_text[:1500]}" if last_text else "No findings were structured.")
        ),
        "key_locations": [],
        "gotchas": [
            "Degraded handoff — no verified citations. Re-run with a larger "
            "max_seconds / higher thoroughness, or narrow the question."
        ],
        "unexplored": ["most of the question"],
        "confidence": "low",
        "_stats": {
            "model": MODEL,
            "thinking": level,
            "tool_calls": tools.call_count,
            "files_read": len(tools.files_read),
            "citation_rejections": submit_rejections,
            "elapsed_s": round(time.time() - started, 1),
            "git_ref": git_ref or None,
            "explored_sha": (repo_info or {}).get("sha"),
            "branch": (repo_info or {}).get("branch"),
            "dirty_files": (repo_info or {}).get("dirty_files"),
            "partial": "budget_exhausted_unstructured",
            **cache_stats,
        },
    }


def render_handoff(handoff: dict) -> str:
    stats = handoff.get("_stats", {})
    header = f"## Scout handoff — confidence: {handoff.get('confidence', 'unknown')}"
    if stats.get("partial"):
        header += " (PARTIAL — time budget exhausted)"
    lines = [header, ""]
    lines += [handoff.get("answer", "(no answer)"), "", "**Key locations (citations verified):**"]
    for loc in handoff.get("key_locations", []):
        vintage = f" (last change {loc['last_change']})" if loc.get("last_change") else ""
        lines.append(f"- `{loc.get('path')}:{loc.get('line')}` — {loc.get('role', '')}{vintage}")
    if handoff.get("gotchas"):
        lines += ["", "**Gotchas:**"] + [f"- {g}" for g in handoff["gotchas"]]
    if handoff.get("unexplored"):
        lines += ["", "**Not explored:**"] + [f"- {u}" for u in handoff["unexplored"]]
    if stats:
        sha = stats.get("explored_sha")
        if sha:
            dirty = stats.get("dirty_files") or 0
            ref_bit = f"explored at {sha[:9]} ({stats.get('branch')})"
            ref_bit += f", dirty files: {dirty}" if not stats.get("git_ref") else " [clean snapshot]"
        else:
            ref_bit = "not a git repo — citations valid for current tree only"
        hit = stats.get("cache_hit_tokens", 0)
        miss = stats.get("cache_miss_tokens", 0)
        cache_bit = f", cache hit {hit * 100 // (hit + miss)}%" if hit + miss else ""
        lines += [
            "",
            f"_{stats.get('model')} (thinking: {stats.get('thinking', 'auto')}): "
            f"{stats.get('tool_calls')} tool calls, "
            f"{stats.get('files_read')} files read, {stats.get('elapsed_s')}s{cache_bit} — {ref_bit}_",
        ]
    return "\n".join(lines)


def narrate_subgraph(
    target: str,
    question: str,
    deps_text: str,
    dependents_text: str,
    thinking: str | None = None,
) -> str:
    """One-shot DeepSeek pass turning a deterministic subgraph into a risk narrative."""
    client = get_client()
    level = _resolve_thinking(thinking)
    prompt = (
        f"Below is the verified import graph around `{target}` in a codebase "
        "(built by parsing imports — these edges are facts).\n\n"
        f"{deps_text}\n\n{dependents_text}\n\n"
        "Write a brief for an engineer about to work on this file: "
        "(1) what this file depends on and why that matters, "
        "(2) the blast radius — what could break if it changes, ordered by risk, "
        "(3) anything structurally notable (cycles, hub files, surprisingly wide reach). "
        "Be concise and concrete; reference file paths from the graph only."
    )
    if question:
        prompt += f"\n\nThe engineer's specific question: {question}"
    resp = _create(
        client,
        level,
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------
# Sanity check: adversarial second opinion on a recommendation before acting
# --------------------------------------------------------------------------

CHECK_SYSTEM_PROMPT = """You are a blunt, skeptical senior engineer performing a pre-flight sanity check on a recommendation made by another AI (Claude) before it is executed under a user's standing "trust the recommendation" authorization. Your verdict is the tripwire: no objection means the action proceeds WITHOUT the user being asked again.

Your job is NOT to rubber-stamp and NOT to nitpick. Object only to substantive problems:
- factual errors (when repo tools are available, VERIFY the recommendation's factual claims against the actual code before ruling — do not take them on faith)
- logic that doesn't hold, missing failure modes, or steps that are destructive/irreversible without being flagged
- the recommendation not actually achieving the user's stated goal
- a materially better or safer alternative that was ignored

Do NOT object over style, naming, minor inefficiencies, or "I'd have done it differently" — record those as notes, not objections.

Finish by calling submit_verdict exactly once. An objection with severity "blocking" means acting on this would be wrong or harmful; "serious" means the user should weigh in first."""

SUBMIT_VERDICT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_verdict",
        "description": "Deliver your final verdict. Call exactly once, after any verification.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["no_objection", "objection"]},
                "summary": {
                    "type": "string",
                    "description": "One paragraph: your reasoning and what you verified.",
                },
                "objections": {
                    "type": "array",
                    "description": "Substantive problems only. [] unless verdict is objection.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "severity": {"type": "string", "enum": ["blocking", "serious"]},
                            "point": {"type": "string"},
                            "evidence": {
                                "type": "string",
                                "description": "Why — cite file:line when repo-verified. \"\" if none.",
                            },
                        },
                        "required": ["severity", "point", "evidence"],
                        "additionalProperties": False,
                    },
                },
                "notes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Non-blocking observations worth passing along. [] if none.",
                },
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["verdict", "summary", "objections", "notes", "confidence"],
            "additionalProperties": False,
        },
    },
}

CHECK_ITERATION_CAP = 20


def run_sanity_check(
    recommendation: str,
    context: str = "",
    root_dir: str = "",
    thinking: str | None = None,
) -> dict:
    """Adversarial review of a recommendation. Returns the verdict dict + stats."""
    level = _resolve_thinking(thinking)
    if level == "auto":
        level = "high"  # verdicts gate actions — think hard by default
    client = get_client()
    started = time.time()

    tools_obj = LocalTools(root_dir) if root_dir else None
    repo_tools = [t for t in TOOLS if t["function"]["name"] != "submit_handoff"] if tools_obj else []
    tool_list = repo_tools + [SUBMIT_VERDICT_TOOL]

    user_msg = ""
    if context:
        user_msg += f"User's goal / context:\n{context}\n\n"
    user_msg += f"Claude's recommendation to check:\n{recommendation}"
    if tools_obj:
        user_msg += f"\n\nRepo available for verification at root: {tools_obj.root}"

    messages: list[dict] = [
        {"role": "system", "content": CHECK_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    cap = CHECK_ITERATION_CAP if tools_obj else 1
    for iteration in range(cap + 1):
        # Not forced on the first pass even without repo tools — an auto call
        # keeps thinking enabled (forcing requires thinking off on DeepSeek v4).
        force_submit = iteration == cap
        resp = _create(
            client,
            level,
            force_tool="submit_verdict" if force_submit else None,
            deterministic=True,
            model=CHECK_MODEL,
            messages=messages,
            tools=tool_list,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        tool_calls = [tc for tc in (msg.tool_calls or []) if getattr(tc, "function", None)]

        messages.append(_assistant_message(msg, tool_calls))

        if not tool_calls:
            messages.append(
                {"role": "user", "content": "Continue verifying, or call submit_verdict."}
            )
            continue

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if tc.function.name == "submit_verdict":
                verdict = _normalize_verdict(args)
                verdict["_stats"] = {
                    "model": CHECK_MODEL,
                    "thinking": level,
                    "tool_calls": tools_obj.call_count if tools_obj else 0,
                    "repo_verified": bool(tools_obj),
                    "elapsed_s": round(time.time() - started, 1),
                }
                return verdict
            result = _dispatch(tools_obj, tc.function.name, args) if tools_obj else "No repo tools available."
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Checker never produced a verdict — a gate that fails must escalate,
    # never silently pass.
    return {
        "verdict": "objection",
        "summary": "The sanity checker failed to produce a structured verdict. "
        "Treat this as an objection: do not proceed on pre-authorization.",
        "objections": [{"severity": "serious", "point": "Checker produced no verdict"}],
        "notes": [],
        "confidence": "low",
        "_stats": {
            "model": CHECK_MODEL,
            "thinking": level,
            "tool_calls": tools_obj.call_count if tools_obj else 0,
            "repo_verified": bool(tools_obj),
            "elapsed_s": round(time.time() - started, 1),
        },
    }


def _normalize_verdict(args: dict) -> dict:
    """Resolve contradictory model output conservatively (toward escalation)."""
    objections = args.get("objections") or []
    verdict = args.get("verdict", "objection")
    if verdict == "objection" and not objections:
        objections = [{"severity": "serious", "point": args.get("summary", "Unspecified objection")}]
    if verdict == "no_objection" and any(
        o.get("severity") in ("blocking", "serious") for o in objections
    ):
        verdict = "objection"  # contradictory output escalates, never auto-passes
    args["verdict"] = verdict
    args["objections"] = objections
    return args


def render_verdict(v: dict) -> str:
    stats = v.get("_stats", {})
    ok = v.get("verdict") == "no_objection"
    lines = [
        f"## Sanity check — {'NO OBJECTION' if ok else 'OBJECTION'} "
        f"(confidence: {v.get('confidence', 'unknown')})",
        "",
        v.get("summary", ""),
    ]
    if v.get("objections"):
        lines += ["", "**Objections:**"]
        for o in v["objections"]:
            ev = f" — {o['evidence']}" if o.get("evidence") else ""
            lines.append(f"- [{o.get('severity', 'serious')}] {o.get('point', '')}{ev}")
    if v.get("notes"):
        lines += ["", "**Notes (non-blocking):**"] + [f"- {n}" for n in v["notes"]]
    lines += [
        "",
        (
            "**Protocol:** No objection — the recommendation stands; proceed under the "
            "user's pre-authorization."
            if ok
            else "**Protocol:** Objection raised — do NOT proceed on pre-authorization alone. "
            "Present the objections to the user and reconfirm, or revise and re-check."
        ),
        "",
        f"_{stats.get('model')} (thinking: {stats.get('thinking')}, "
        f"repo-verified: {stats.get('repo_verified')}): "
        f"{stats.get('tool_calls')} tool calls, {stats.get('elapsed_s')}s_",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Claim verification: re-measure reported facts at their cited locations
# --------------------------------------------------------------------------

CLAIMS_SYSTEM_PROMPT = """You verify factual claims about code. For each claim you are given the exact code at its cited location. Judge ONLY whether the stated fact is true at that location:

- CONFIRMED: the code at the citation states/does what the claim says
- REFUTED: the code contradicts the claim (say exactly how, citing the line)
- NOT_FOUND: the cited location contains nothing relevant to the claim

You are NOT judging design quality, style, or whether the code is good — only whether the claim is factually accurate at the cited location. Base your judgment strictly on the code shown. Call submit_claim_verdicts exactly once, with a verdict for every claim index you were given."""

SUBMIT_CLAIMS_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_claim_verdicts",
        "description": "Deliver a verdict for every claim. Call exactly once.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "verdicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "status": {
                                "type": "string",
                                "enum": ["CONFIRMED", "REFUTED", "NOT_FOUND"],
                            },
                            "evidence": {
                                "type": "string",
                                "description": "Line-cited justification, one or two sentences.",
                            },
                        },
                        "required": ["index", "status", "evidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["verdicts"],
            "additionalProperties": False,
        },
    },
}


def run_verify_claims(
    claims: list[dict],
    root: str,
    git_ref: str = "",
    thinking: str | None = None,
) -> dict:
    """Check each {claim, path, line?, search?} against the actual code."""
    explore_root, repo_info, cleanup = resolve_exploration_root(root, git_ref)
    try:
        return _verify_claims_inner(claims, explore_root, thinking, git_ref, repo_info)
    finally:
        cleanup()


def _verify_claims_inner(
    claims: list[dict], root: str, thinking: str | None, git_ref: str, repo_info: dict | None
) -> dict:
    started = time.time()
    level = _resolve_thinking(thinking)
    if level == "auto":
        level = "medium"
    lt = LocalTools(root)
    results: list[dict | None] = [None] * len(claims)
    pending: list[tuple[int, str]] = []  # (index, evidence block for the model)

    for i, c in enumerate(claims):
        claim_text = str(c.get("claim", "")).strip()
        path = str(c.get("path", "")).strip().replace("\\", "/")
        line = c.get("line")
        if i >= MAX_CLAIMS_PER_CALL:
            results[i] = {"index": i, "status": "SKIPPED",
                          "evidence": f"exceeds {MAX_CLAIMS_PER_CALL} claims per call — split the batch"}
            continue
        if not claim_text or not path:
            results[i] = {"index": i, "status": "NOT_FOUND", "evidence": "claim or path missing"}
            continue
        try:
            p = lt._resolve(path)
        except ValueError as e:
            results[i] = {"index": i, "status": "NOT_FOUND", "evidence": str(e)}
            continue
        if not p.is_file():
            results[i] = {"index": i, "status": "NOT_FOUND", "evidence": f"file does not exist: {path}"}
            continue
        total = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
        if line is not None and not (1 <= int(line) <= total):
            results[i] = {"index": i, "status": "NOT_FOUND",
                          "evidence": f"{path}:{line} out of range (file has {total} lines)"}
            continue
        if line is not None:
            window = lt.read_file(path, max(1, int(line) - 25), int(line) + 25)
        else:
            window = lt.read_file(path, 1, 150)
        if c.get("search"):
            window += "\n\nRepo-wide search hits for the claim's search term:\n"
            window += lt.grep(str(c["search"]))[:4000]
        block = f"### Claim {i}\nClaim: {claim_text}\nCited location: {path}"
        block += f":{line}\n" if line is not None else " (no line given)\n"
        block += f"Code at citation:\n{window}"
        pending.append((i, block))

    if pending:
        client = get_client()
        user_msg = "\n\n".join(b for _, b in pending)
        messages = [
            {"role": "system", "content": CLAIMS_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        def _collect(resp) -> dict[int, dict]:
            found: dict[int, dict] = {}
            m = resp.choices[0].message
            for tc in [t for t in (m.tool_calls or []) if getattr(t, "function", None)]:
                try:
                    payload = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    payload = {}
                for v in payload.get("verdicts", []):
                    if isinstance(v.get("index"), int):
                        found[v["index"]] = v
            return found

        # First pass keeps thinking on (tool_choice auto — forcing requires
        # thinking off on DeepSeek v4); force only if the model didn't comply.
        verdicts = _collect(
            _create(client, level, deterministic=True, model=MODEL,
                    messages=messages, tools=[SUBMIT_CLAIMS_TOOL], tool_choice="auto")
        )
        if not verdicts:
            verdicts = _collect(
                _create(client, level, force_tool="submit_claim_verdicts",
                        deterministic=True, model=MODEL, messages=messages,
                        tools=[SUBMIT_CLAIMS_TOOL])
            )
        for i, _ in pending:
            v = verdicts.get(i)
            results[i] = (
                {"index": i, "status": v.get("status", "NOT_FOUND"), "evidence": v.get("evidence", "")}
                if v
                else {"index": i, "status": "NOT_FOUND", "evidence": "verifier returned no verdict for this claim"}
            )

    for i, c in enumerate(claims):
        r = results[i]
        if r is not None:
            r["claim"] = str(c.get("claim", ""))
            r["path"] = str(c.get("path", ""))
            r["line"] = c.get("line")

    final = [r for r in results if r is not None]
    counts = {s: sum(1 for r in final if r["status"] == s)
              for s in ("CONFIRMED", "REFUTED", "NOT_FOUND", "SKIPPED")}
    return {
        "results": final,
        "counts": counts,
        "_stats": {
            "model": MODEL,
            "thinking": level,
            "elapsed_s": round(time.time() - started, 1),
            "git_ref": git_ref or None,
            "explored_sha": (repo_info or {}).get("sha"),
            "dirty_files": (repo_info or {}).get("dirty_files"),
        },
    }


_STATUS_MARKS = {"CONFIRMED": "[OK]", "REFUTED": "[X]", "NOT_FOUND": "[?]", "SKIPPED": "[skip]"}


def render_claim_report(report: dict) -> str:
    stats = report.get("_stats", {})
    c = report.get("counts", {})
    lines = [
        f"## Claim verification — {c.get('CONFIRMED', 0)} confirmed, "
        f"{c.get('REFUTED', 0)} refuted, {c.get('NOT_FOUND', 0)} not found"
        + (f", {c['SKIPPED']} skipped" if c.get("SKIPPED") else ""),
        "",
    ]
    for r in report.get("results", []):
        loc = f"{r.get('path')}:{r.get('line')}" if r.get("line") is not None else r.get("path")
        lines.append(f"- {_STATUS_MARKS.get(r['status'], '[?]')} **{r['status']}** — {r.get('claim')}")
        lines.append(f"  `{loc}` — {r.get('evidence', '')}")
    sha = stats.get("explored_sha")
    ref_bit = (
        f"verified at {sha[:9]}" + (" [clean snapshot]" if stats.get("git_ref") else f", dirty files: {stats.get('dirty_files') or 0}")
        if sha
        else "not a git repo"
    )
    lines += ["", f"_{stats.get('model')} (thinking: {stats.get('thinking')}): {stats.get('elapsed_s')}s — {ref_bit}_"]
    return "\n".join(lines)
