# AGENTS.md

## Local overrides

If `AGENTS.local.md` exists at the repo root, read it and let its instructions take precedence over
this file. It is gitignored for personal, machine-specific preferences and never committed.

## Commands

`Taskfile.yml` wraps the common workflows — run `task --list` to see them. The recipes below are the
preferred entrypoints; each shells out to the underlying `uv` command shown after it.

- `task install` — sync the project and dev dependencies from the lockfile (`uv sync --locked`),
  then install the prek hooks.
- `task test` — `uv run pytest` (append args after `--`, e.g. `task test -- -k name`).
- `task test:matrix` — run the suite on every supported Python version.
- `task check` — `uv run prek run -a` (all hooks). Run before committing.
- `task check:fix` — auto-fix everything that supports it: `uv run ruff check --fix`,
  `uv run ruff format`, `dprint fmt`, markdownlint (`uv run prek run -a markdownlint-cli2`).
- `task clean` — remove build artifacts, caches, and virtualenvs.

## Git hygiene

- Never run `git commit --no-verify`, `git commit -n`, or anything else that skips the prek hooks —
  the hooks are the gate, not an obstacle.
- Fix a failing check at its source. Never edit a test to make it pass; never widen a lint ignore to
  silence a real finding.
