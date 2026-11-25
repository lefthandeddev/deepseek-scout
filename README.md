# DeepSeek Scout — multi-provider edition

DeepSeek Scout is a stdio MCP server for read-only codebase exploration. It
searches a local project, asks the selected model to analyze the relevant results, and
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
| `map_dependencies(target, root_dir, depth, question)` | Build a deterministic Python/JavaScript/TypeScript import graph around a file, with an optional model summary of its blast radius. |
| `verify_claims(claims, root_dir, git_ref)` | Check cited claims against the referenced code and return `CONFIRMED`, `REFUTED`, or `NOT_FOUND` with evidence. |
| `sanity_check(recommendation, context, root_dir)` | Provide an adversarial second opinion. Use only when the user has explicitly pre-authorized acting on a trusted recommendation; an objection requires reconfirmation. |

`verify_claims` checks paths and line ranges locally before asking the model to
judge the claims. It verifies factual statements; it does not assess design
quality.

## Safety and data handling

- Scout does not write source files in the target repository. With `git_ref`,
  it uses a temporary worktree and removes it after the call.
- The question and the search/read results selected for analysis are sent to
  the explicitly configured inference endpoint (cloud or local). Scout does not upload the repository as an
  archive. Do not use it on code that you are not allowed to send to that endpoint.
- Search and direct-read tools skip common credential locations and files,
  including `.env*`, cloud credential directories, private keys, and
  credential files such as `.netrc` and `.npmrc`.
- Common key/token formats and credential assignments are redacted from
  search and read results before they are sent to the inference endpoint.
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

Select a provider and set its key/model using the configuration below. Copy `.env.example` to `.env` if desired. For an MCP client, run the virtual environment's Python
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
set the selected provider variables in that client's MCP configuration.

For Claude Code, register the server with:

```sh
claude mcp add --scope user deepseek-scout -- "<venv-python>" "<path-to-server.py>"
```

## Configuration

DeepSeek remains the default; all five legacy `DEEPSEEK_*` variables retain their meanings when DeepSeek is selected. The four MCP tool names and the JSON/Markdown result formats are unchanged. Set `SCOUT_PROVIDER` explicitly for another provider. The registry is [scout/providers.json](scout/providers.json).

| Provider | Authentication | Transport |
|---|---|---|
| `deepseek` | `DEEPSEEK_API_KEY` | OpenAI-compatible chat |
| `openai` | `OPENAI_API_KEY` | OpenAI-compatible chat |
| `anthropic` | `ANTHROPIC_API_KEY` | Native Messages API |
| `gemini` | `GEMINI_API_KEY` | OpenAI-compatible chat |
| `openrouter` | `OPENROUTER_API_KEY` | OpenAI-compatible chat |
| `groq` | `GROQ_API_KEY` | OpenAI-compatible chat |
| `xai` | `XAI_API_KEY` | OpenAI-compatible chat |
| `mistral` | `MISTRAL_API_KEY` | OpenAI-compatible chat |
| `together` | `TOGETHER_API_KEY` | OpenAI-compatible chat |
| `fireworks` | `FIREWORKS_API_KEY` | OpenAI-compatible chat |
| `cerebras` | `CEREBRAS_API_KEY` | OpenAI-compatible chat |
| `deepinfra` | `DEEPINFRA_API_KEY` | OpenAI-compatible chat |
| `huggingface` | `HF_TOKEN` | OpenAI-compatible chat |
| `dashscope` | `DASHSCOPE_API_KEY` | OpenAI-compatible chat |
| `azure` | `AZURE_OPENAI_API_KEY` | Azure OpenAI v1 |
| `compatible` | Explicit key or no-key opt-in | OpenAI-compatible chat |
| `ollama` | None by default | OpenAI-compatible chat |
| `lmstudio` | None by default | OpenAI-compatible chat |
| `llamacpp` | None by default | OpenAI-compatible chat |
| `vllm` | None by default | OpenAI-compatible chat |
| `sglang` | None by default | OpenAI-compatible chat |
| `localai` | None by default | OpenAI-compatible chat |

Except DeepSeek, supply a tool-capable model ID explicitly. `SCOUT_CHECK_MODEL` defaults to the selected model (DeepSeek retains its heavier checker default). API/model support is not interchangeable: the selected model must return function calls and accept tool results; the generic/local transport cannot make a text-only model tool-capable. Azure requires an explicit deployment model and base URL ending in `/openai/v1`; Gemini uses Google's OpenAI-compatible endpoint, not the native Google SDK. Model-specific reasoning controls and vendor API availability need live validation.

| Variable | Meaning |
|---|---|
| `SCOUT_PROVIDER` | Registry ID; default `deepseek`. |
| `SCOUT_MODEL` / `SCOUT_CHECK_MODEL` | Exploration and checker model IDs. |
| `SCOUT_BASE_URL` | HTTP(S) base path before `/chat/completions` or `/messages`. |
| `SCOUT_API_KEY` | Explicit key override for the selected endpoint only; never commit credentials. |
| `SCOUT_TOOL_CHOICE` | `named` for native forced selection; `prompt` restricts forced requests to one tool and prompts the model, for endpoints without forced selection. |
| `SCOUT_THINKING` | Requested level: auto, none, low, medium, high, max. |
| `SCOUT_REASONING_MODE` | `deepseek` (DeepSeek default), `openai` (opt-in on compatible reasoning models), or `off` (other defaults). |
| `SCOUT_TIMEOUT_SECONDS` | Per-request timeout; default 120. |
| `SCOUT_MAX_OUTPUT_TOKENS` | Optional output cap; native Anthropic defaults to 8192. |
| `SCOUT_LOCAL_ONLY=1` | Reject non-loopback endpoints, no proxy, no fallback. |
| `SCOUT_ALLOW_NO_KEY=1` | Explicitly permit an unauthenticated compatible endpoint. |
| `SCOUT_ALLOW_INSECURE_HTTP=1` | Explicit opt-in for non-loopback plaintext HTTP; prefer HTTPS. |

### Examples

```sh
# OpenAI
SCOUT_PROVIDER=openai SCOUT_MODEL=<model-id> OPENAI_API_KEY=<key> python server.py
# Local Ollama; the model must already be served locally
SCOUT_PROVIDER=ollama SCOUT_MODEL=qwen3:8b SCOUT_LOCAL_ONLY=1 python server.py
# Arbitrary OpenAI-compatible server
SCOUT_PROVIDER=compatible SCOUT_BASE_URL=http://127.0.0.1:8000/v1 SCOUT_MODEL=<model-id> SCOUT_ALLOW_NO_KEY=1 SCOUT_LOCAL_ONLY=1 python server.py
```

Replace placeholder values; for Windows or MCP clients, set the same variables in the environment. Local-only is transport isolation, not a guarantee about the behavior of the separately installed local model server. Scout never silently switches providers. No HTTP redirects or ambient HTTP proxies are used. The legacy DeepSeek base URL override retains its historical key behavior; when using a custom host for another preset, explicitly supply `SCOUT_API_KEY`. The model can see selected source excerpts and prompts. Redaction is best-effort, not a DLP guarantee. Read-only tools reject sensitive paths and symlinks. Git worktree snapshots write Git metadata in the source repository's Git directory but do not modify its checked-out source files.

## Reasoning and tool calls

DeepSeek's existing reasoning and forced-tool handling is retained. Other endpoints omit unsupported reasoning parameters by default, so a requested thinking level is not a guarantee of actual reasoning. Native Anthropic uses its standard Messages API without extended thinking. OpenAI-style reasoning controls are opt-in and model dependent. Forced submissions restrict the available tool list; `prompt` mode does not guarantee compliance. Provider errors do not trigger cloud fallback. All tool arguments are validated against the declared schemas before local execution or acceptance; a checker that cannot produce a valid verdict cannot authorize an action. Prefix-cache counters are provider-dependent.

Run offline tests with `python -m unittest discover -s tests -v`. These exercise protocol adapters with mocked HTTP, not paid/live providers. The benchmark harness remains available for real-model comparisons.
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
