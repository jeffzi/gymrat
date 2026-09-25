# AGENTS.md

## Local overrides

If `AGENTS.local.md` exists at the repo root, read it and let its instructions take precedence over
this file. It is gitignored for personal, machine-specific preferences and never committed.

## Commands

`Taskfile.yml` wraps the common workflows — run `task --list` to see them. These are the only
entrypoints; never bypass them by calling scripts, tools, or `python` directly — the task and prek
layers manage the virtualenv, file selection, and flags.

- `task install` — sync the project and dev dependencies from the lockfile, then install the prek
  hooks.
- `task test` — run the test suite (append args after `--`, e.g. `task test -- -k name`; passing
  args disables coverage, since a subset run would fail the global coverage threshold).
- `task test:matrix` — run the suite on every supported Python version.
- `task check` — run every prek hook over all tracked and untracked files. Hooks that can fix
  (Ruff, dprint, markdownlint) rewrite files in place. Run before committing.
- To run a single hook: `uv run prek run <hook-id>` (e.g. `uv run prek run check-max-lines`). Hook
  IDs are in `.pre-commit-config.yaml`.
- `uvx pymaxlines --show-sizes` — print a code-line breakdown of every file (largest first) instead
  of checking limits. Pass paths to scope it: `uvx pymaxlines --show-sizes src/heavy_module.py`.
  Use it to find the biggest files and functions before deciding where to split. Counts exclude
  blanks, comments, and docstrings — matching what the `check-max-lines` hook enforces.
- `task schemas` — regenerate `schemas/` and `docs/event-reference.md` from the pydantic record and
  event models. Run it after any change to a model field or docstring and commit the output; the
  drift test in `tests/event_docs/test_drift.py` fails on stale artifacts.
- `task clean` — remove build artifacts, caches, and virtualenvs.

## Git hygiene

- Never run `git commit --no-verify`, `git commit -n`, or anything else that skips the prek hooks —
  the hooks are the gate, not an obstacle.
- Fix a failing check at its source. Never edit a test to make it pass; never widen a lint ignore to
  silence a real finding.

## Linter and type-checker configuration

Treat lint and type-check config as fixed. Never add to an ignore list, disable a rule, lower a
severity, or exclude a file to get a check passing — fix the code instead. A suppression is
warranted only when the finding is a genuine false positive or the rule cannot apply (e.g. a
generated file, a documented upstream bug); then suppress at the narrowest scope — an inline
directive with a reason — not in the shared config. When the same inline directive keeps recurring
for the same rule, that is a signal the rule may deserve a config-level ignore — propose it to the
user and wait for explicit approval; never promote a suppression into config on your own.

## Module size

`check-max-lines` caps a file at 400 code lines and a function at 60. The cap signals a file with
more than one concern; it is not a budget.

- **New code lives in its caller's module**, unless two or more modules import it today, or it is a
  distinct concern of 50+ code lines — then it gets one flat module named after the concern.
  "Reusable later", "keeps the caller small" and "matches the existing small modules" are not
  reasons: the tiny modules and packages already in `src/` are debt, not precedent.
- **No packages of small modules** behind a re-exporting `__init__.py`. A package is justified only
  when the flat module would exceed the cap; planned features don't count.
- **At the cap, move one whole concern** — the code least tied to the rest that changes together for
  one reason — into a module named for what it does (never "helpers", "utils", "misc"): a new
  module, or an existing one you have read that already owns that concern. A module carved off its
  only importer, such as `supervisor/tasks.py`, owns nothing; never add to it. Never move just
  enough to pass, and never compress code. One concern, one move: the file should land at 350 or
  below, and if it doesn't, pick a different, larger concern — never top up the move with other
  code.

## Docstrings

Google style. Private functions need none. A one-line docstring has no sections; a longer one has
every section that applies — `Args:` whenever the function takes parameters. Beyond what it raises
directly, `Raises:` lists every error that propagates from a callee and that its callers are
expected to handle. An error a callee may raise that no caller handles stays out (an `OSError` from
a file call).

- Prose never restates what a section says.
- A function you edit gets its docstring brought to this shape, even if you did not write it.
- Tests: no docstrings on test functions; every fixture gets a one-line docstring.

## Spelling (cspell)

Treat a cspell failure as a prompt to reword, not to grow the dictionary. Prefer plain words in
prose and identifiers. A word earns a `cspell.json` entry only when it comes from outside the
project and cannot be renamed — command names, API identifiers, file formats, proper nouns, domain
vocabulary (e.g. `addopts`, `conftest`, `pyrefly`). In tests, never invent gibberish that needs a
suppression — any real word works for an unknown command, a bogus flag, or filler data, so pick one
(`banana`, not an invented pseudo-word). `# cspell:disable-line` is reserved for fixtures where the
gibberish itself is the behavior under test, never a dictionary entry.
