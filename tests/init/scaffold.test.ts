import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { beforeEach, describe, expect, it } from "vitest";

import { GymratError } from "../../src/errors.js";
import { scaffold } from "../../src/init/scaffold.js";
import type { WizardResult } from "../../src/init/wizard.js";
import { createPackageLayout } from "../fixtures/package-layout.js";
import { freshRoot } from "../fixtures/scratch-repo.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Read and parse the scaffolded config file, narrowed from JSON.parse's `any`. */
function readJsonConfig(baseDir: string): Record<string, unknown> {
  const raw: unknown = JSON.parse(readFileSync(join(baseDir, "gymrat.json"), "utf-8"));
  if (raw == null || typeof raw !== "object" || Array.isArray(raw)) {
    throw new Error("expected gymrat.json to contain a JSON object");
  }
  // oxlint-disable-next-line typescript/no-unsafe-type-assertion -- narrowed by typeof guard above
  return raw as Record<string, unknown>;
}

/** Build a minimal WizardResult with only required fields. */
function makeWizardResult(overrides: Partial<WizardResult> = {}): WizardResult {
  return {
    bench: "npm run bench",
    runbook: false,
    installSkill: false,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Shared arrange
// ---------------------------------------------------------------------------

let baseDir: string;
let importMetaUrl: string;

beforeEach(() => {
  baseDir = freshRoot("scaffold-base-");
  ({ importMetaUrl } = createPackageLayout("scaffold-pkg-"));
});

// ---------------------------------------------------------------------------
// scaffold — config object construction
// ---------------------------------------------------------------------------

describe("scaffold", () => {
  describe("when the wizard result has only bench", () => {
    it("writes a config with only bench", () => {
      scaffold(baseDir, makeWizardResult(), importMetaUrl);

      const config = readJsonConfig(baseDir);
      expect(config).toStrictEqual({ bench: "npm run bench" });
    });
  });

  describe("when the wizard result includes adapter", () => {
    it("includes adapter in the written config", () => {
      scaffold(baseDir, makeWizardResult({ adapter: "mitata" }), importMetaUrl);

      const config = readJsonConfig(baseDir);
      expect(config.adapter).toBe("mitata");
    });
  });

  describe("when adapter is undefined (default metric-lines)", () => {
    it("omits adapter from the written config", () => {
      scaffold(baseDir, makeWizardResult(), importMetaUrl);

      const config = readJsonConfig(baseDir);
      expect(config).not.toHaveProperty("adapter");
    });
  });

  describe("when the wizard result includes checks", () => {
    it("includes checks in the written config", () => {
      scaffold(baseDir, makeWizardResult({ checks: "npm test" }), importMetaUrl);

      const config = readJsonConfig(baseDir);
      expect(config.checks).toBe("npm test");
    });
  });

  describe("when the wizard result includes primary", () => {
    it("includes primary in the written config", () => {
      scaffold(baseDir, makeWizardResult({ primary: "latency" }), importMetaUrl);

      const config = readJsonConfig(baseDir);
      expect(config.primary).toBe("latency");
    });
  });

  describe("when the wizard result includes stop fields", () => {
    it("writes a stop sub-object with only the set fields", () => {
      scaffold(baseDir, makeWizardResult({ stopTarget: 1.5, primary: "latency" }), importMetaUrl);

      const config = readJsonConfig(baseDir);
      expect(config.stop).toStrictEqual({ targetValue: 1.5 });
    });

    it("includes maxIterations when set", () => {
      scaffold(baseDir, makeWizardResult({ stopMaxIterations: 10 }), importMetaUrl);

      const config = readJsonConfig(baseDir);
      expect(config.stop).toStrictEqual({ maxIterations: 10 });
    });

    it("includes both targetValue and maxIterations when both are set", () => {
      scaffold(
        baseDir,
        makeWizardResult({ stopTarget: 2.0, stopMaxIterations: 20, primary: "latency" }),
        importMetaUrl,
      );

      const config = readJsonConfig(baseDir);
      expect(config.stop).toStrictEqual({ targetValue: 2.0, maxIterations: 20 });
    });
  });

  describe("when the wizard result has no stop fields", () => {
    it("omits stop from the written config", () => {
      scaffold(baseDir, makeWizardResult(), importMetaUrl);

      const config = readJsonConfig(baseDir);
      expect(config).not.toHaveProperty("stop");
    });
  });

  describe("when the wizard result includes a runbook path", () => {
    it("includes the runbook key in the written config", () => {
      scaffold(
        baseDir,
        makeWizardResult({ runbook: { path: "gymrat-runbook.md" } }),
        importMetaUrl,
      );

      const config = readJsonConfig(baseDir);
      expect(config.runbook).toBe("gymrat-runbook.md");
    });
  });

  // ---------------------------------------------------------------------------
  // scaffold — validation
  // ---------------------------------------------------------------------------

  describe("when the built config would fail schema validation", () => {
    it("throws a GymratError before writing any files", () => {
      const wizardResult = makeWizardResult({ bench: "" });

      expect(() => scaffold(baseDir, wizardResult, importMetaUrl)).toThrow(GymratError);
      expect(existsSync(join(baseDir, "gymrat.json"))).toBe(false);
    });
  });

  describe("when the built config would fail cross-field validation", () => {
    it("throws a GymratError for stop.targetValue with geomean primary", () => {
      const wizardResult = makeWizardResult({ stopTarget: 1.5 });

      expect(() => scaffold(baseDir, wizardResult, importMetaUrl)).toThrow(GymratError);
      expect(existsSync(join(baseDir, "gymrat.json"))).toBe(false);
    });
  });

  // ---------------------------------------------------------------------------
  // scaffold — runbook stub
  // ---------------------------------------------------------------------------

  describe("when a runbook stub is requested and the path does not exist", () => {
    it("creates the runbook file with the expected sections", () => {
      scaffold(
        baseDir,
        makeWizardResult({ runbook: { path: "gymrat-runbook.md" } }),
        importMetaUrl,
      );

      const content = readFileSync(join(baseDir, "gymrat-runbook.md"), "utf-8");
      expect.soft(content).toContain("# Optimization Runbook");
      expect.soft(content).toContain("## Goal");
      expect.soft(content).toContain("## Gating metrics");
      expect.soft(content).toContain("## Constraints");
      expect.soft(content).toContain("## Approaches to try");
      expect(content).toContain("gymrat supervise");
    });
  });

  describe("when a runbook stub is requested in a nested directory", () => {
    it("creates parent directories as needed", () => {
      scaffold(
        baseDir,
        makeWizardResult({ runbook: { path: "docs/runbooks/bench.md" } }),
        importMetaUrl,
      );

      expect(existsSync(join(baseDir, "docs", "runbooks", "bench.md"))).toBe(true);
    });
  });

  describe("when a runbook path already exists", () => {
    it("leaves the existing file untouched", () => {
      const existingContent = "# My Custom Runbook\n";
      writeFileSync(join(baseDir, "my-runbook.md"), existingContent);

      scaffold(baseDir, makeWizardResult({ runbook: { path: "my-runbook.md" } }), importMetaUrl);

      const content = readFileSync(join(baseDir, "my-runbook.md"), "utf-8");
      expect(content).toBe(existingContent);
    });

    it("still includes the runbook key in the config", () => {
      writeFileSync(join(baseDir, "my-runbook.md"), "# Existing\n");

      scaffold(baseDir, makeWizardResult({ runbook: { path: "my-runbook.md" } }), importMetaUrl);

      const config = readJsonConfig(baseDir);
      expect(config.runbook).toBe("my-runbook.md");
    });
  });

  // ---------------------------------------------------------------------------
  // scaffold — absolute runbook path handling
  // ---------------------------------------------------------------------------

  describe("when the runbook path is absolute", () => {
    it("creates the file at the absolute location, not joined with baseDir", () => {
      const absDir = freshRoot("scaffold-abs-rb-");
      const absRunbookPath = join(absDir, "rb.md");

      scaffold(baseDir, makeWizardResult({ runbook: { path: absRunbookPath } }), importMetaUrl);

      expect(existsSync(absRunbookPath)).toBe(true);
      const content = readFileSync(absRunbookPath, "utf-8");
      expect(content).toContain("# Optimization Runbook");
      // The file should NOT exist at baseDir + absRunbookPath
      expect(existsSync(join(baseDir, absRunbookPath))).toBe(false);
    });

    it("records the absolute path verbatim in gymrat.json", () => {
      const absDir = freshRoot("scaffold-abs-rb-cfg-");
      const absRunbookPath = join(absDir, "rb.md");

      scaffold(baseDir, makeWizardResult({ runbook: { path: absRunbookPath } }), importMetaUrl);

      const config = readJsonConfig(baseDir);
      expect(config.runbook).toBe(absRunbookPath);

      // The file must exist at the absolute location, not relative to baseDir.
      expect(existsSync(absRunbookPath)).toBe(true);
    });
  });

  describe("when an absolute runbook path already exists", () => {
    it("reports runbook status as exists and preserves the file", () => {
      const absDir = freshRoot("scaffold-abs-rb-exist-");
      const absRunbookPath = join(absDir, "existing-rb.md");
      const existingContent = "# My Absolute Runbook\n";
      writeFileSync(absRunbookPath, existingContent);

      const result = scaffold(
        baseDir,
        makeWizardResult({ runbook: { path: absRunbookPath } }),
        importMetaUrl,
      );

      expect(result.runbook).toStrictEqual({ path: absRunbookPath, status: "exists" });
      const content = readFileSync(absRunbookPath, "utf-8");
      expect(content).toBe(existingContent);
    });
  });

  describe("when the wizard declined a runbook", () => {
    it("does not create a runbook file", () => {
      scaffold(baseDir, makeWizardResult({ runbook: false }), importMetaUrl);

      expect(existsSync(join(baseDir, "gymrat-runbook.md"))).toBe(false);
    });

    it("omits the runbook key from the config", () => {
      scaffold(baseDir, makeWizardResult({ runbook: false }), importMetaUrl);

      const config = readJsonConfig(baseDir);
      expect(config).not.toHaveProperty("runbook");
    });
  });

  // ---------------------------------------------------------------------------
  // scaffold — skill install
  // ---------------------------------------------------------------------------

  describe("when installSkill is true and the skill does not exist", () => {
    it("copies the bundled skill to .claude/skills/gymrat/SKILL.md", () => {
      scaffold(baseDir, makeWizardResult({ installSkill: true }), importMetaUrl);

      const skillPath = join(baseDir, ".claude", "skills", "gymrat", "SKILL.md");
      expect(existsSync(skillPath)).toBe(true);
      expect(readFileSync(skillPath, "utf-8")).toContain("# Test Skill");
    });
  });

  describe("when installSkill is true and the skill already exists", () => {
    it("leaves the existing skill untouched", () => {
      const existingSkill = "# Custom Skill\n";
      const skillDir = join(baseDir, ".claude", "skills", "gymrat");
      mkdirSync(skillDir, { recursive: true });
      writeFileSync(join(skillDir, "SKILL.md"), existingSkill);

      scaffold(baseDir, makeWizardResult({ installSkill: true }), importMetaUrl);

      expect(readFileSync(join(skillDir, "SKILL.md"), "utf-8")).toBe(existingSkill);
    });
  });

  describe("when installSkill is false", () => {
    it("does not create the skill file", () => {
      scaffold(baseDir, makeWizardResult({ installSkill: false }), importMetaUrl);

      expect(existsSync(join(baseDir, ".claude", "skills", "gymrat", "SKILL.md"))).toBe(false);
    });
  });

  // ---------------------------------------------------------------------------
  // scaffold — config write format
  // ---------------------------------------------------------------------------

  describe("when writing the config file", () => {
    it("writes pretty-printed JSON with a trailing newline", () => {
      scaffold(baseDir, makeWizardResult(), importMetaUrl);

      const raw = readFileSync(join(baseDir, "gymrat.json"), "utf-8");
      const expected = JSON.stringify({ bench: "npm run bench" }, null, 2) + "\n";
      expect(raw).toBe(expected);
    });
  });

  // ---------------------------------------------------------------------------
  // scaffold — write ordering (config last)
  // ---------------------------------------------------------------------------

  describe("when skill install throws an error", () => {
    it("does not leave a gymrat.json behind", () => {
      const { importMetaUrl: brokenUrl } = createPackageLayout("scaffold-broken-", {
        skipSkill: true,
      });
      const wizardResult = makeWizardResult({ installSkill: true });

      expect(() => scaffold(baseDir, wizardResult, brokenUrl)).toThrow(GymratError);
      expect(existsSync(join(baseDir, "gymrat.json"))).toBe(false);
    });
  });

  // ---------------------------------------------------------------------------
  // scaffold — return value
  // ---------------------------------------------------------------------------

  describe("when scaffold succeeds with all artifacts", () => {
    it("returns a result describing what happened to each artifact", () => {
      const result = scaffold(
        baseDir,
        makeWizardResult({ runbook: { path: "gymrat-runbook.md" }, installSkill: true }),
        importMetaUrl,
      );

      expect(result).toStrictEqual({
        config: { path: "gymrat.json", status: "created" },
        runbook: { path: "gymrat-runbook.md", status: "created" },
        skill: { path: ".claude/skills/gymrat/SKILL.md", status: "created" },
      });
    });
  });

  describe("when runbook was declined", () => {
    it("reports runbook status as declined", () => {
      const result = scaffold(
        baseDir,
        makeWizardResult({ runbook: false, installSkill: true }),
        importMetaUrl,
      );

      expect(result.runbook.status).toBe("declined");
    });
  });

  describe("when runbook already exists", () => {
    it("reports runbook status as exists", () => {
      writeFileSync(join(baseDir, "my-runbook.md"), "# Existing\n");

      const result = scaffold(
        baseDir,
        makeWizardResult({ runbook: { path: "my-runbook.md" } }),
        importMetaUrl,
      );

      expect(result.runbook).toStrictEqual({ path: "my-runbook.md", status: "exists" });
    });
  });

  describe("when skill install was declined", () => {
    it("reports skill status as declined", () => {
      const result = scaffold(baseDir, makeWizardResult({ installSkill: false }), importMetaUrl);

      expect(result.skill.status).toBe("declined");
    });
  });

  describe("when skill already exists", () => {
    it("reports skill status as exists", () => {
      const skillDir = join(baseDir, ".claude", "skills", "gymrat");
      mkdirSync(skillDir, { recursive: true });
      writeFileSync(join(skillDir, "SKILL.md"), "# Custom\n");

      const result = scaffold(baseDir, makeWizardResult({ installSkill: true }), importMetaUrl);

      expect(result.skill).toStrictEqual({
        path: ".claude/skills/gymrat/SKILL.md",
        status: "exists",
      });
    });
  });
});
