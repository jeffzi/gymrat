# AGENTS.md

## Commands

- `npm run build` — compile with `tsc -p tsconfig.build.json`
- `npm test` — `vitest run --coverage`; coverage thresholds fail the run, and untested `src/` files count against them
- `npm run check` — all static checks: lint, markdown, formatting, workflow lint/audit, type-aware lint, dead code, spelling, types. Run before committing.
- `npm run fix` — auto-fix: lint, markdown, formatting
- `npm run fix:quality` — remove unused exports, dependencies, and enum members (`fallow fix`). Deletes code — review the diff before committing.
- `npm run check:security` — surface security candidates (`fallow security`). Candidates need verification; they are not confirmed vulnerabilities. Not part of `check`.
- `npm run check:package` — publish hygiene (`publint` + `@arethetypeswrong/cli`) for packages that ship to npm. Not part of `check`.

## Conventions

- **Errors:** all custom errors extend `GymratError` from `src/errors.ts` — never extend bare `Error`.
