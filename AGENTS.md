# gymrat

If `AGENTS.local.md` exists at the repo root, its instructions take precedence over this file.

## Design

`VISION.md` states the ranked pillars and vetoes that decide design questions. Read it before
designing a change that touches the session loop, the supervisor, or a contract, and check the
design against it. Rules that follow from it:

- Liveness is the advisory file lock. Never probe process IDs. A file that gates behavior carries
  its own deadline; never infer staleness from a file's age.
- The turn protocol (a turn ended, a follow-up is accepted) is the only backend seam. Abstract
  nothing else ahead of a second harness, and keep the Claude Agent SDK's transport names inside
  its driver.
- A tool exposed to the agent is a thin host over the CLI command: spawn it, relay its output, kill
  the child when the call ends.
- Every timing constant is a parameter a test can override.
- Import the agent SDK, and anything else with a startup cost, inside the function that needs it.

## Commands

`task --list` shows the workflows. Use `task` targets instead of calling `pytest`, `ruff`, or
`python` directly. Exceptions:

- `task test -- -k name` disables coverage; a subset run would fail the global threshold.
- `task check` before committing. One hook: `uv run prek run <hook-id>`; IDs are in
  `.pre-commit-config.yaml`.
- `uvx pymaxlines --show-sizes [path]` prints code-line sizes per file and function, largest first,
  counted the way the `check-max-lines` hook counts.

## Checks

- Never skip the prek hooks (`--no-verify`, `-n`).
- Fix a failing check at its source. Never edit a test to make it pass.
- Lint and type-check config is fixed: never add an ignore, disable a rule, lower a severity, or
  exclude a file. Suppress only a genuine false positive, inline, with a reason. A config-level
  ignore needs the user's explicit approval.
- A cspell failure is a prompt to reword. A word earns a `cspell.json` entry only when it comes
  from outside the project and cannot be renamed. In tests, use a real word (`banana`) for filler,
  never gibberish; `# cspell:disable-line` only where the gibberish is the behavior under test.
