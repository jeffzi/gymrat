import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { GymratError } from "./errors.js";

/** Relative path from the compiled `dist/bundled-skill.js` to the bundled skill. */
const SKILL_RELATIVE_PATH = "../skills/gymrat/SKILL.md";

/**
 * Read the bundled gymrat skill from disk, resolving relative to `importMetaUrl`.
 *
 * Callers pass their own `import.meta.url` so tests can supply a mock URL
 * pointing at a temp directory layout without depending on the real install.
 */
export function readBundledSkill(importMetaUrl: string): string {
  const callerDir = fileURLToPath(new URL(".", importMetaUrl));
  const skillPath = resolve(callerDir, SKILL_RELATIVE_PATH);

  try {
    return readFileSync(skillPath, "utf8");
  } catch (error) {
    throw new GymratError(
      `Bundled skill not found at ${skillPath} — the gymrat installation may be broken.`,
      "Reinstall the package to restore the bundled skill file.",
      { cause: error },
    );
  }
}
