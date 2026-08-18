import { describe, expect, it } from "vitest";

import type { ConfigInspection } from "../../src/config-inspect.js";
import {
  buildConfigSection,
  buildEnvironmentSection,
  buildWorkflowSection,
  createDoctorReport,
  type CheckSection,
} from "../../src/doctor/checks.js";
import { createCheck } from "../fixtures/doctor.js";
import { benchlessConfig } from "../fixtures/session-records.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function defined<T>(value: T | undefined | null): T {
  expect(value).toBeDefined();
  if (value == null) throw new Error("expected defined");
  return value;
}

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------

function createInspection(overrides: Partial<ConfigInspection> = {}): ConfigInspection {
  return {
    configPath: "/project/gymrat.json",
    configExists: true,
    problems: [],
    config: benchlessConfig(),
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Report model
// ---------------------------------------------------------------------------

describe("DoctorReport", () => {
  describe("when all checks pass", () => {
    it("reports zero warnings, zero failures, and has-failures false", () => {
      const sections: CheckSection[] = [
        { title: "Environment", checks: [createCheck({ status: "ok" })] },
        { title: "Config", checks: [createCheck({ status: "ok" })] },
      ];

      const report = createDoctorReport(
        { gymratVersion: "0.5.0", nodeVersion: "22.12.0", platform: "darwin" },
        sections,
      );

      expect(report.okCount).toBe(2);
      expect(report.warnCount).toBe(0);
      expect(report.failCount).toBe(0);
      expect(report.hasFailures).toBe(false);
    });
  });

  describe("when checks include warnings and failures", () => {
    it("counts each status correctly and flags has-failures", () => {
      const sections: CheckSection[] = [
        {
          title: "Mixed",
          checks: [
            createCheck({ status: "ok" }),
            createCheck({ status: "warn" }),
            createCheck({ status: "fail" }),
            createCheck({ status: "fail" }),
          ],
        },
      ];

      const report = createDoctorReport(
        { gymratVersion: "0.5.0", nodeVersion: "22.12.0", platform: "linux" },
        sections,
      );

      expect(report.okCount).toBe(1);
      expect(report.warnCount).toBe(1);
      expect(report.failCount).toBe(2);
      expect(report.hasFailures).toBe(true);
    });
  });

  describe("when counts span multiple sections", () => {
    it("aggregates across all sections", () => {
      const sections: CheckSection[] = [
        { title: "A", checks: [createCheck({ status: "ok" })] },
        { title: "B", checks: [createCheck({ status: "warn" })] },
        { title: "C", checks: [createCheck({ status: "fail" })] },
      ];

      const report = createDoctorReport(
        { gymratVersion: "0.5.0", nodeVersion: "22.12.0", platform: "win32" },
        sections,
      );

      expect(report.okCount).toBe(1);
      expect(report.warnCount).toBe(1);
      expect(report.failCount).toBe(1);
      expect(report.hasFailures).toBe(true);
    });
  });

  describe("when sections are empty", () => {
    it("reports all zeros and no failures", () => {
      const report = createDoctorReport(
        { gymratVersion: "0.5.0", nodeVersion: "22.12.0", platform: "darwin" },
        [],
      );

      expect(report.okCount).toBe(0);
      expect(report.warnCount).toBe(0);
      expect(report.failCount).toBe(0);
      expect(report.hasFailures).toBe(false);
    });
  });

  it("preserves environment info and sections", () => {
    const env = { gymratVersion: "1.0.0", nodeVersion: "22.12.0", platform: "darwin" };
    const sections: CheckSection[] = [{ title: "Test", checks: [createCheck()] }];

    const report = createDoctorReport(env, sections);

    expect(report.environment).toStrictEqual(env);
    expect(report.sections).toStrictEqual(sections);
  });
});

// ---------------------------------------------------------------------------
// Environment section
// ---------------------------------------------------------------------------

describe("buildEnvironmentSection", () => {
  it("has title 'Environment'", () => {
    const section = buildEnvironmentSection({ gitAvailable: true, insideGitRepo: true });

    expect(section.title).toBe("Environment");
  });

  describe("when git is available on PATH", () => {
    it("produces an OK check", () => {
      const section = buildEnvironmentSection({ gitAvailable: true, insideGitRepo: true });

      const gitCheck = defined(section.checks.find((c) => c.name === "git"));
      expect(gitCheck.status).toBe("ok");
    });
  });

  describe("when git is not available on PATH", () => {
    it("produces a FAIL check with install hint", () => {
      const section = buildEnvironmentSection({ gitAvailable: false, insideGitRepo: false });

      const gitCheck = defined(section.checks.find((c) => c.name === "git"));
      expect(gitCheck.status).toBe("fail");
      expect(gitCheck.hint).toBeDefined();
    });
  });

  describe("when inside a git repository", () => {
    it("produces an OK check", () => {
      const section = buildEnvironmentSection({ gitAvailable: true, insideGitRepo: true });

      const repoCheck = defined(section.checks.find((c) => c.name === "git repository"));
      expect(repoCheck.status).toBe("ok");
    });
  });

  describe("when not inside a git repository", () => {
    it("produces a WARN check", () => {
      const section = buildEnvironmentSection({ gitAvailable: true, insideGitRepo: false });

      const repoCheck = defined(section.checks.find((c) => c.name === "git repository"));
      expect(repoCheck.status).toBe("warn");
    });
  });

  describe("when the repository root could not be resolved", () => {
    it("produces a WARN check surfacing the error message", () => {
      const section = buildEnvironmentSection({
        gitAvailable: true,
        insideGitRepo: true,
        gitError: "permission denied",
      });

      const rootCheck = defined(section.checks.find((c) => c.name === "git repository root"));
      expect(rootCheck.status).toBe("warn");
      expect(rootCheck.detail).toContain("permission denied");
    });
  });
});

// ---------------------------------------------------------------------------
// Config section
// ---------------------------------------------------------------------------

describe("buildConfigSection", () => {
  it("has title 'Configuration'", () => {
    const section = buildConfigSection(createInspection());

    expect(section.title).toBe("Configuration");
  });

  describe("when the config file is clean", () => {
    it("produces a single OK check naming the config path", () => {
      const inspection = createInspection({
        configPath: "/my/project/gymrat.json",
        configExists: true,
        problems: [],
      });

      const section = buildConfigSection(inspection);

      expect(section.checks).toHaveLength(1);
      const check = defined(section.checks[0]);
      expect(check.status).toBe("ok");
      expect(check.detail).toContain("/my/project/gymrat.json");
    });
  });

  describe("when no config file exists", () => {
    it("produces a single OK check indicating defaults-only", () => {
      const inspection = createInspection({
        configPath: undefined,
        configExists: false,
        problems: [],
        config: benchlessConfig(),
      });

      const section = buildConfigSection(inspection);

      expect(section.checks).toHaveLength(1);
      const check = defined(section.checks[0]);
      expect(check.status).toBe("ok");
      expect(check.detail).toMatch(/defaults/i);
    });
  });

  describe("when problems are present", () => {
    it("produces one FAIL check per problem with wording preserved", () => {
      const problems = [
        'Invalid value for "samples": expected a positive integer, got "abc"',
        'Invalid value for "adapter": expected a string, got 42',
      ];
      const inspection = createInspection({
        configPath: "/project/gymrat.json",
        configExists: true,
        problems,
      });

      const section = buildConfigSection(inspection);

      const failChecks = section.checks.filter((c) => c.status === "fail");
      expect(failChecks).toHaveLength(2);
      expect(defined(failChecks[0]).detail).toBe(problems[0]);
      expect(defined(failChecks[1]).detail).toBe(problems[1]);
    });
  });
});

// ---------------------------------------------------------------------------
// Workflow section
// ---------------------------------------------------------------------------

describe("buildWorkflowSection", () => {
  it("has title 'Workflow'", () => {
    const section = buildWorkflowSection({
      config: benchlessConfig(),

      skillFileExists: true,
    });

    expect(section.title).toBe("Workflow");
  });

  describe("when config has problems", () => {
    it("returns a single OK check that skips workflow evaluation", () => {
      const section = buildWorkflowSection({
        config: benchlessConfig(),
        problems: ["Invalid value for 'samples': expected a positive integer, got \"abc\""],
        skillFileExists: true,
      });

      expect(section.checks).toHaveLength(1);
      const check = defined(section.checks[0]);
      expect(check.status).toBe("ok");
      expect(check.detail).toMatch(/fix config/i);
    });

    it("does not produce skill file, checks, stop, or runbook warnings", () => {
      const section = buildWorkflowSection({
        config: benchlessConfig(),
        problems: ["bad value"],
        skillFileExists: false,
      });

      const names = section.checks.map((c) => c.name);
      expect(names).not.toContain("skill file");
      expect(names).not.toContain("checks");
      expect(names).not.toContain("stop");
      expect(names).not.toContain("runbook");
    });
  });

  describe("skill file check", () => {
    describe("when the skill file exists", () => {
      it("produces an OK check", () => {
        const section = buildWorkflowSection({
          config: benchlessConfig(),

          skillFileExists: true,
        });

        const skillCheck = defined(section.checks.find((c) => c.name === "skill file"));
        expect(skillCheck.status).toBe("ok");
      });
    });

    describe("when the skill file is missing", () => {
      it("produces a WARN check with hint naming install routes and the fresh-setup fix", () => {
        const section = buildWorkflowSection({
          config: benchlessConfig(),

          skillFileExists: false,
        });

        const skillCheck = defined(section.checks.find((c) => c.name === "skill file"));
        expect(skillCheck.status).toBe("warn");
        expect.soft(skillCheck.hint).toContain("npx skills add jeffzi/gymrat");
        expect(skillCheck.hint).toContain("gymrat init");
      });
    });
  });

  describe("checks setting", () => {
    describe("when checks is set", () => {
      it("produces an OK check echoing the value", () => {
        const config = { ...benchlessConfig(), checks: "npm test" };
        const section = buildWorkflowSection({
          config,

          skillFileExists: true,
        });

        const checksCheck = defined(section.checks.find((c) => c.name === "checks"));
        expect(checksCheck.status).toBe("ok");
        expect(checksCheck.detail).toContain("npm test");
      });
    });

    describe("when checks is unset", () => {
      it("produces a WARN check with hint about keep gating", () => {
        const section = buildWorkflowSection({
          config: benchlessConfig(),

          skillFileExists: true,
        });

        const checksCheck = defined(section.checks.find((c) => c.name === "checks"));
        expect(checksCheck.status).toBe("warn");
        expect(checksCheck.hint).toMatch(/keep/i);
      });
    });
  });

  describe("stop setting", () => {
    describe("when stop is set with one key", () => {
      it("produces an OK check whose detail carries the key value", () => {
        const config = {
          ...benchlessConfig(),
          stop: { maxIterations: 20 },
        };
        const section = buildWorkflowSection({
          config,

          skillFileExists: true,
        });

        const stopCheck = defined(section.checks.find((c) => c.name === "stop"));
        expect(stopCheck.status).toBe("ok");
        expect(stopCheck.detail).toContain("20");
      });
    });

    describe("when stop is set with two keys", () => {
      it("renders both parts in the detail", () => {
        const config = {
          ...benchlessConfig(),
          stop: { maxIterations: 20, targetValue: 1.5 },
        };
        const section = buildWorkflowSection({
          config,

          skillFileExists: true,
        });

        const stopCheck = defined(section.checks.find((c) => c.name === "stop"));
        expect(stopCheck.status).toBe("ok");
        expect(stopCheck.detail).toContain("20");
        expect(stopCheck.detail).toContain("1.5");
      });
    });

    describe("when stop is unset", () => {
      it("produces a WARN check with hint about session finish line", () => {
        const section = buildWorkflowSection({
          config: benchlessConfig(),

          skillFileExists: true,
        });

        const stopCheck = defined(section.checks.find((c) => c.name === "stop"));
        expect(stopCheck.status).toBe("warn");
        expect(stopCheck.hint).toBeDefined();
      });
    });

    describe("when stop is an empty object", () => {
      it("produces a WARN check same as absent stop", () => {
        const config = { ...benchlessConfig(), stop: {} };
        const section = buildWorkflowSection({
          config,

          skillFileExists: true,
        });

        const stopCheck = defined(section.checks.find((c) => c.name === "stop"));
        expect(stopCheck.status).toBe("warn");
        expect(stopCheck.hint).toBeDefined();
      });
    });
  });

  describe("runbook setting", () => {
    describe("when runbook is set", () => {
      it("produces an OK check echoing the value", () => {
        const config = { ...benchlessConfig(), runbook: "./RUNBOOK.md" };
        const section = buildWorkflowSection({
          config,

          skillFileExists: true,
        });

        const runbookCheck = defined(section.checks.find((c) => c.name === "runbook"));
        expect(runbookCheck.status).toBe("ok");
        expect(runbookCheck.detail).toContain("./RUNBOOK.md");
      });
    });

    describe("when runbook is unset", () => {
      it("produces a WARN check with hint about supervise and the fresh-setup fix", () => {
        const section = buildWorkflowSection({
          config: benchlessConfig(),

          skillFileExists: true,
        });

        const runbookCheck = defined(section.checks.find((c) => c.name === "runbook"));
        expect(runbookCheck.status).toBe("warn");
        expect.soft(runbookCheck.hint).toMatch(/supervise/i);
        expect(runbookCheck.hint).toContain("gymrat init");
      });
    });
  });
});
