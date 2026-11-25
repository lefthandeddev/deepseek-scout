# DeepSeek Scout

DeepSeek Scout is a stdio MCP server for read-only codebase exploration. It
searches a local project, asks DeepSeek to analyze the relevant results, and
returns a structured handoff with citations checked against the files.

## Why it was built

Scout was built for Claude Code to give its user more control over Claude token
usage. It delegates exploration to a fast, lower-cost DeepSeek model and returns
a compact handoff to Claude, aiming to save time and reduce Claude usage on
search-heavy work. The original motivation included a preference for DeepSeek
over Haiku for this role. Actual speed and savings depend on the models,
pricing, task, and amount of exploration.

A Claude subagent can also call Scout, adding a DeepSeek exploration agent
within the subagent's workflow. This provides another layer of delegation while
letting the Claude agent continue with the returned findings.

The earliest surviving commit found for this version is dated August 7, 2026;
the original repository was created on August 9, 2026. An earlier prototype may
have existed, but its date has not been confirmed. This project is not currently
in active use; it is published as a foundation for further development.

## Tools

| Tool | Purpose |
|---|---|
| `find(question, root_dir, thoroughness)` | Explore a codebase and return an answer, verified `file:line` locations, gotchas, and areas not explored. |
| `map_dependencies(target, root_dir, depth, question)` | Build a deterministic Python/JavaScript/TypeScript import graph around a file, with an optional DeepSeek summary of its blast radius. |
| `verify_claims(claims, root_dir, git_ref)` | Check cited claims against the referenced code and return `CONFIRMED`, `REFUTED`, or `NOT_FOUND` with evidence. |
| `sanity_check(recommendation, context, root_dir)` | Provide an adversarial second opinion. Use only when the user has explicitly pre-authorized acting on a trusted recommendation; an objection requires reconfirmation. |

`verify_claims` checks paths and line ranges locally before asking DeepSeek to
judge the claims. It verifies factual statements; it does not assess design
quality.

## Safety and data handling

- Scout does not write source files in the target repository. With `git_ref`,
  it uses a temporary worktree and removes it after the call.
- The question and the search/read results selected for analysis are sent to
  the configured DeepSeek API. Scout does not upload the repository as an
  archive. Do not use it on code that you are not allowed to send to that API.
- Search and direct-read tools skip common credential locations and files,
  including `.env*`, cloud credential directories, private keys, and
  credential files such as `.netrc` and `.npmrc`.
- Common key/token formats and credential assignments are redacted from
  search and read results before they are sent to DeepSeek.
- `DEEPSEEK_API_KEY` is read from the environment or a local `.env` file.
  `.env` is ignored by Git; never commit a real key.

## Install

Python 3.10 or newer is required.

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Set `DEEPSEEK_API_KEY` in the environment or copy `.env.example` to `.env` and
add your key. For an MCP client, run the virtual environment's Python
executable with the absolute path to `server.py`. Example configuration:

```json
{
  "mcpServers": {
    "deepseek-scout": {
      "command": "/absolute/path/to/deepseek-scout/.venv/bin/python",
      "args": ["/absolute/path/to/deepseek-scout/server.py"]
    }
  }
}
```

On Windows, use `.venv\Scripts\python.exe` for `command` and escaped
absolute paths in `args`. If your client does not inherit the environment,
set `DEEPSEEK_API_KEY` in that client's MCP configuration.

For Claude Code, register the server with:

```sh
claude mcp add --scope user deepseek-scout -- "<venv-python>" "<path-to-server.py>"
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | Required API key. |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek API endpoint. |
| `DEEPSEEK_MODEL` | `deepseek-flash` | Model for `find`, `map_dependencies`, and `verify_claims`. |
| `DEEPSEEK_CHECK_MODEL` | `deepseek-v4-pro` | Model for `sanity_check`. |
| `DEEPSEEK_THINKING` | `auto` | Default reasoning effort; per-call `thinking` values override it. |

Model IDs and prices can change. See DeepSeek's [current Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/).

## Reasoning and tool calls

Model-calling tools accept `thinking`: `auto`, `none`, `low`, `medium`, `high`,
or `max`. `auto` leaves the effort at the provider default; `none` disables
reasoning. Invalid values fall back to `auto`.

DeepSeek does not allow forced function calls while thinking is enabled, so
forced submissions use reasoning off. Normal calls use automatic tool
selection. Assistant reasoning required for subsequent tool-call requests is
passed back to DeepSeek and is not included in MCP results.

Tool schemas are strict. Prompts and schemas are kept stable to support
DeepSeek's prefix cache, and independent local search calls may run in
parallel.

## Git snapshots and output

- `git_ref` on `find`, `map_dependencies`, or `verify_claims` explores a clean
  snapshot of a branch, tag, or commit using a temporary worktree.
- Handoffs include the explored commit, branch, and dirty-file count. Citation
  entries include the file's last-change date when available.
- `max_seconds` on `find` sets a soft time budget. When reached, Scout returns
  the findings so far as a partial handoff.
- `format="json"` on `find` and `verify_claims` returns structured JSON.

## Claude Code guidance

To prefer Scout for exploration in Claude Code, add this to a global
`~/.claude/CLAUDE.md` or a project `CLAUDE.md`:

```markdown
## DeepSeek Scout
When mcp__deepseek-scout__* tools are available:
- Use `find` for codebase exploration questions. Pass the repo root as
  `root_dir` and include the reason for the question.
- Use `map_dependencies` before refactoring a shared file or when asked what
  depends on it.
- Treat returned citations as checked locations; read the cited files needed
  for the task.
```

## Benchmark

Copy `benchmark/questions.example.json` to `benchmark/questions.json`, edit
the questions, then run:

```powershell
.venv\Scripts\python.exe benchmark\run_benchmark.py
```

Results are written to `benchmark/results/`, both ignored by Git.

## License

MIT. See [LICENSE](LICENSE).
