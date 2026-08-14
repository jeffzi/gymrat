import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import type { BenchlessConfig } from "../config.js";
import { GymratError } from "../errors.js";

/** Relative path from the compiled `dist/supervisor/kickoff.js` to the bundled skill. */
const SKILL_RELATIVE_PATH = "../../skills/gymrat/SKILL.md";

const DEFAULT_KICKOFF =
  "Drive the optimization session. Follow the skill instructions and the runbook to guide your work.";

interface KickoffResult {
  readonly systemPromptAppend: string;
  readonly kickoff: string;
}

/**
 * Compose the system-prompt append and kickoff message for a supervised session.
 *
 * `importMetaUrl` is the caller's `import.meta.url` — tests supply a mock URL
 * pointing at a temp directory so the bundled-skill resolution is exercised
 * without depending on the real installed layout.
 */
export function composeKickoff(
  config: BenchlessConfig,
  importMetaUrl: string,
  prompt?: string,
): KickoffResult {
  const skillContent = readBundledSkill(importMetaUrl);

  if (config.runbook === undefined) {
    throw new GymratError(
      "No runbook configured — set `runbook` in gymrat.json.",
      "A supervised session has no human to answer the skill's fallback; a runbook is required.",
    );
  }

  let runbookContent: string;
  try {
    runbookContent = readFileSync(config.runbook, "utf8");
  } catch (error) {
    throw new GymratError(
      `Runbook not found at ${config.runbook}.`,
      "Verify the file exists at the path configured for `runbook` in gymrat.json.",
      { cause: error },
    );
  }
  const systemPromptAppend = `${skillContent}\n## Runbook: ${config.runbook}\n\n${runbookContent}`;

  return {
    systemPromptAppend,
    kickoff: prompt ?? DEFAULT_KICKOFF,
  };
}

function readBundledSkill(importMetaUrl: string): string {
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
