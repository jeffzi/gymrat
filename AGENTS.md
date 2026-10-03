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
  drift test in `tests/event_docs/test_event_docs_drift.py` fails on stale artifacts.
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

`check-max-lines` caps a file at 500 code lines and a function at 60. The cap signals a file with
more than one concern; it is not a budget. A fold, merge or move is made only when its target lands
at 450 code lines or below, summing every file that goes into it. A file may then grow past 450:
only the 500 cap forces a move.

- **Code lives in a module that already exists**, the first of these that applies:
  1. One module uses it: that module, unless the one-importer list below gives it its own. An entry
     point (`__main__.py`, a console script) is neither a user nor an importer.
  2. Two or more modules import it today (`TYPE_CHECKING` counts), it is under 50 code lines, and
     every importer already imports its owner, or is it: the owner. Judging the code as one unit,
     the owner is the module of the one function its types are passed to or its functions' results
     go to; failing that, for module-level functions (methods do not count), the one module that
     defines every project type in their signatures, parameters and returns, besides the code's own
     — none left, no owner. Error types and a module the importers merely share never make an owner.
  3. Otherwise: a module of its own, named after the concern, whatever its size.

  "Reusable later", "keeps the caller small" and "matches the existing small modules" are not
  reasons: the tiny modules and packages already in `src/` are debt, not precedent.
- **No fold crosses a boundary.** `cli/` holds only code about the command line itself: flag
  parsing, exit routing, terminal display, the wiring and guarding of commands. Code about anything
  else — the loop, a session, a report, the supervisor, telemetry — stays out even when commands are
  its only callers: it goes to the owner the tests in 2 give it on its own side (under 50 lines
  only), else to a module of its own there. An import seam is what a test protects: code that loads
  lazily, or without Rich, the agent SDK, OpenTelemetry or `cli/`. It blocks only a fold after which
  that code would no longer load that way.
- **A module with one importer needs a reason from this list**; "it is a distinct concern" is not
  one.
  - Folding it into its importer would land above 450. When several modules with no other reason
    share an importer and not all fit, fold the smallest first and stop before the one that does not
    fit.
  - The boundary or a seam keeps it out of its importer.
  - It is 50+ code lines of pure logic beside an importer that does I/O or renders. Pure means no
    file, process, network or terminal I/O (a clock read is not I/O) and no Rich objects or markup,
    directly or through what it calls.
  - It is 50+ code lines and one of two or more peers: modules of one role that the importer only
    registers or chooses between — the command modules under `cli/app.py`, the report kinds under
    `report/text/render.py`. A helper the importer calls as part of its own work, on some paths or
    all, is not a peer.

  An existing module with no reason is debt: report it, and fold it into its importer when asked.
- **No packages of small modules** behind a re-exporting `__init__.py`. A package is justified only
  when its modules merged into one would land above 450; planned features don't count.
- **One import path per name.** Import a name from the module that defines it. A package
  `__init__.py` re-exports nothing, and a module's `__all__` lists only names it defines.
- **Never pool unrelated code to cut the file count.** A module's name describes every name in it; a
  name that fits only because it is wide ("common", "shared", "core") is "utils" renamed. Asked to
  consolidate, fold only what this section folds and report the rest as compliant: a module with a
  reason to exist is never merged into another.
- **At the cap, move one whole concern** — the code least tied to the rest that changes together for
  one reason — into a module named for what it does (never "helpers", "utils", "misc"): a new
  module, or an existing one you have read that already owns that concern — never one carved off its
  only importer, which owns nothing. Never move just enough to pass, and never compress code. One
  concern, one move: the file should land at 450 or below, and if it doesn't, pick a different,
  larger concern — never top up the move with other code.

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
