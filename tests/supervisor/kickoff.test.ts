import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { describe, expect, it } from "vitest";

import type { BenchlessConfig } from "../../src/config.js";
import { GymratError } from "../../src/errors.js";
import { composeKickoff } from "../../src/supervisor/kickoff.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const SKILL_CONTENT = "# Test Skill\n\nDo the optimization.\n";
const RUNBOOK_CONTENT = "# My Runbook\n\nStep 1: run benchmarks.\n";

/**
 * Build a temp directory that mirrors the installed package layout:
 *
 *     <root>/dist/supervisor/kickoff.js   (mock import.meta.url target)
 *     <root>/skills/gymrat/SKILL.md       (bundled skill)
 *
 * Returns the file URL pointing at the mock `kickoff.js` location, which
 * the function under test uses to resolve the skill path relative to itself.
 */
function createPackageLayout(options?: { skipSkill?: boolean }): {
  root: string;
  importMetaUrl: string;
} {
  const root = mkdtempSync(join(tmpdir(), "kickoff-"));
  const distDir = join(root, "dist", "supervisor");
  mkdirSync(distDir, { recursive: true });

  if (!options?.skipSkill) {
    const skillDir = join(root, "skills", "gymrat");
    mkdirSync(skillDir, { recursive: true });
    writeFileSync(join(skillDir, "SKILL.md"), SKILL_CONTENT);
  }

  const importMetaUrl = pathToFileURL(join(distDir, "kickoff.js")).href;
  return { root, importMetaUrl };
}

function createRunbook(root: string, content = RUNBOOK_CONTENT): string {
  const runbookPath = join(root, "runbook.md");
  writeFileSync(runbookPath, content);
  return runbookPath;
}

function makeConfig(overrides: Partial<BenchlessConfig> = {}): BenchlessConfig {
  return {
    adapter: "mitata",
    samples: 30,
    timeoutSeconds: 60,
    unstableNoisePct: 5,
    primary: "geomean",
    ...overrides,
  };
}

/** Build a package layout with a bundled skill and a configured runbook. */
function setupHappyPath(): { config: BenchlessConfig; importMetaUrl: string } {
  const { importMetaUrl, root } = createPackageLayout();
  const runbookPath = createRunbook(root);
  const config = makeConfig({ runbook: runbookPath });
  return { config, importMetaUrl };
}

// ---------------------------------------------------------------------------
// composeKickoff
// ---------------------------------------------------------------------------

describe("composeKickoff", () => {
  describe("when the bundled skill file exists", () => {
    it("includes the skill content in systemPromptAppend", () => {
      const { config, importMetaUrl } = setupHappyPath();

      const result = composeKickoff(config, importMetaUrl);

      expect(result.systemPromptAppend).toContain(SKILL_CONTENT);
    });
  });

  describe("when the bundled skill file is missing", () => {
    it("throws a GymratError indicating a broken installation", () => {
      const { importMetaUrl, root } = createPackageLayout({ skipSkill: true });
      const runbookPath = createRunbook(root);
      const config = makeConfig({ runbook: runbookPath });

      expect(() => composeKickoff(config, importMetaUrl)).toThrow(GymratError);
    });
  });

  describe("when config has a runbook", () => {
    it("appends runbook content after skill content with a heading naming the path", () => {
      const { config, importMetaUrl } = setupHappyPath();
      const runbookPath = config.runbook;

      const result = composeKickoff(config, importMetaUrl);

      expect(result.systemPromptAppend).toContain(RUNBOOK_CONTENT);
      expect(result.systemPromptAppend).toContain(`## Runbook: ${runbookPath}`);
    });

    it("places skill content before the runbook heading", () => {
      const { config, importMetaUrl } = setupHappyPath();

      const result = composeKickoff(config, importMetaUrl);

      const skillIdx = result.systemPromptAppend.indexOf(SKILL_CONTENT);
      const runbookIdx = result.systemPromptAppend.indexOf(`## Runbook:`);
      expect(skillIdx).toBeLessThan(runbookIdx);
    });
  });

  describe("when config has no runbook", () => {
    it("throws a GymratError mentioning both runbook and gymrat.json", () => {
      const { importMetaUrl } = createPackageLayout();
      const config = makeConfig();

      expect(() => composeKickoff(config, importMetaUrl)).toThrow(GymratError);
      expect(() => composeKickoff(config, importMetaUrl)).toThrow(/runbook/i);
      expect(() => composeKickoff(config, importMetaUrl)).toThrow(/gymrat\.json/);
    });
  });

  describe("when no prompt is provided", () => {
    it("returns a default kickoff containing an identifying substring", () => {
      const { config, importMetaUrl } = setupHappyPath();

      const result = composeKickoff(config, importMetaUrl);

      expect(result.kickoff).toBeTruthy();
      expect(typeof result.kickoff).toBe("string");
      expect(result.kickoff).toContain("optimization");
    });
  });

  describe("when a prompt is provided", () => {
    it("passes the caller-supplied prompt through unchanged", () => {
      const { config, importMetaUrl } = setupHappyPath();
      const customPrompt = "optimize the decoder loop";

      const result = composeKickoff(config, importMetaUrl, customPrompt);

      expect(result.kickoff).toBe(customPrompt);
    });
  });

  describe("return value", () => {
    it("contains systemPromptAppend and kickoff as strings", () => {
      const { config, importMetaUrl } = setupHappyPath();

      const result = composeKickoff(config, importMetaUrl);

      expect(result).toHaveProperty("systemPromptAppend");
      expect(result).toHaveProperty("kickoff");
      expect(typeof result.systemPromptAppend).toBe("string");
      expect(typeof result.kickoff).toBe("string");
    });
  });
});
