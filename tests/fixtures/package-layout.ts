import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { freshRoot } from "./scratch-repo.js";

const SKILL_CONTENT = "# Test Skill\n\nDo the optimization.\n";

/**
 * Build a temp directory that mirrors the installed package layout from the
 * perspective of `dist/bundled-skill.js` (one level deep):
 *
 *     <root>/dist/bundled-skill.js   (mock import.meta.url target)
 *     <root>/skills/gymrat/SKILL.md  (bundled skill)
 *
 * The relative path inside `readBundledSkill` is `../skills/gymrat/SKILL.md`,
 * which resolves correctly from `dist/` depth.
 */
export function createPackageLayout(
  prefix = "gymrat-pkg-",
  options?: { skipSkill?: boolean },
): { root: string; importMetaUrl: string } {
  const root = freshRoot(prefix);
  const distDir = join(root, "dist");
  mkdirSync(distDir, { recursive: true });

  if (!options?.skipSkill) {
    const skillDir = join(root, "skills", "gymrat");
    mkdirSync(skillDir, { recursive: true });
    writeFileSync(join(skillDir, "SKILL.md"), SKILL_CONTENT);
  }

  const importMetaUrl = pathToFileURL(join(distDir, "bundled-skill.js")).href;
  return { root, importMetaUrl };
}
