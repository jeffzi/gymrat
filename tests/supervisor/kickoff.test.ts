import { readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { describe, expect, it } from "vitest";

import { GymratError } from "../../src/errors.js";
import { composeKickoff } from "../../src/supervisor/kickoff.js";
import { createPackageLayout } from "../fixtures/package-layout.js";
import { freshRoot } from "../fixtures/scratch-repo.js";
import { benchlessConfig } from "../fixtures/session-records.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const RUNBOOK_CONTENT = "# My Runbook\n\nStep 1: run benchmarks.\n";

function createRunbook(root: string, content = RUNBOOK_CONTENT): string {
  const runbookPath = join(root, "runbook.md");
  writeFileSync(runbookPath, content);
  return runbookPath;
}

/** Build a package layout with a bundled skill and a configured runbook. */
function setupHappyPath(): {
  config: ReturnType<typeof benchlessConfig>;
  importMetaUrl: string;
} {
  const { importMetaUrl, root } = createPackageLayout("kickoff-");
  const runbookPath = createRunbook(root);
  const config = benchlessConfig({
    adapter: "mitata",
    samples: 30,
    timeoutSeconds: 60,
    unstableNoisePct: 5,
    runbook: runbookPath,
  });
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

      expect(result.systemPromptAppend).toContain("# Test Skill");
    });
  });

  describe("when the bundled skill file is missing", () => {
    it("throws a GymratError indicating a broken installation", () => {
      const { importMetaUrl, root } = createPackageLayout("kickoff-", { skipSkill: true });
      const runbookPath = createRunbook(root);
      const config = benchlessConfig({
        adapter: "mitata",
        samples: 30,
        timeoutSeconds: 60,
        unstableNoisePct: 5,
        runbook: runbookPath,
      });

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

      const skillIdx = result.systemPromptAppend.indexOf("# Test Skill");
      const runbookIdx = result.systemPromptAppend.indexOf(`## Runbook:`);
      expect(skillIdx).toBeLessThan(runbookIdx);
    });
  });

  describe("when config has no runbook", () => {
    it("throws a GymratError mentioning both runbook and gymrat.json", () => {
      const { importMetaUrl } = createPackageLayout("kickoff-");
      const config = benchlessConfig({
        adapter: "mitata",
        samples: 30,
        timeoutSeconds: 60,
        unstableNoisePct: 5,
      });

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

  describe("when called with the real installed layout", () => {
    it("resolves SKILL_RELATIVE_PATH to the real skills/gymrat/SKILL.md", () => {
      const projectRoot = resolve(fileURLToPath(import.meta.url), "..", "..", "..");
      const realDistEntry = join(projectRoot, "dist", "bundled-skill.js");
      const importMetaUrl = pathToFileURL(realDistEntry).href;

      const runbookDir = freshRoot("kickoff-runbook-");
      const runbookPath = join(runbookDir, "runbook.md");
      writeFileSync(runbookPath, "# Runbook\n");
      const config = benchlessConfig({
        adapter: "mitata",
        samples: 30,
        timeoutSeconds: 60,
        unstableNoisePct: 5,
        runbook: runbookPath,
      });

      const result = composeKickoff(config, importMetaUrl);

      const realSkillContent = readFileSync(
        join(projectRoot, "skills", "gymrat", "SKILL.md"),
        "utf-8",
      );
      expect(realSkillContent.length).toBeGreaterThan(0);
      expect(result.systemPromptAppend).toContain(realSkillContent);
    });
  });
});
