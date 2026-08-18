import { stripVTControlCharacters as stripAnsi } from "node:util";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { CheckSection, DoctorReport, EnvironmentInfo } from "../../src/doctor/checks.js";
import { createDoctorReport } from "../../src/doctor/checks.js";
import { renderDoctorReport, renderDoctorJson } from "../../src/doctor/render.js";
import { reportLines } from "../fixtures/constants.js";
import { createCheck } from "../fixtures/doctor.js";

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------

function createEnv(overrides: Partial<EnvironmentInfo> = {}): EnvironmentInfo {
  return {
    gymratVersion: "0.5.0",
    nodeVersion: "22.12.0",
    platform: "darwin",
    ...overrides,
  };
}

function createReport(
  sections: CheckSection[] = [],
  env: EnvironmentInfo = createEnv(),
): DoctorReport {
  return createDoctorReport(env, sections);
}

// ---------------------------------------------------------------------------
// Text rendering: renderDoctorReport
// ---------------------------------------------------------------------------

describe("renderDoctorReport", () => {
  describe("environment header", () => {
    it("includes version, node, and platform in the first line", () => {
      const report = createReport(
        [],
        createEnv({
          gymratVersion: "1.2.3",
          nodeVersion: "22.12.0",
          platform: "linux",
        }),
      );

      const output = renderDoctorReport(report);

      const header = reportLines(output)[0] ?? "";
      expect(header).toBeDefined();
      expect.soft(header).toContain("1.2.3");
      expect.soft(header).toContain("22.12.0");
      expect(header).toContain("linux");
    });
  });

  describe("section rendering", () => {
    it("renders a bold section title", () => {
      const report = createReport([{ title: "Environment", checks: [createCheck()] }]);

      const output = renderDoctorReport(report);

      expect(stripAnsi(output)).toContain("Environment");
    });

    it("renders each check on its own line with a status glyph", () => {
      const report = createReport([
        {
          title: "Checks",
          checks: [
            createCheck({ name: "git", status: "ok", detail: "found" }),
            createCheck({ name: "repo", status: "warn", detail: "not inside" }),
            createCheck({ name: "config", status: "fail", detail: "missing" }),
          ],
        },
      ]);

      const output = reportLines(renderDoctorReport(report));

      const okLine = output.find((l) => l.includes("found"));
      const warnLine = output.find((l) => l.includes("not inside"));
      const failLine = output.find((l) => l.includes("missing"));
      expect.soft(okLine).toContain("✓");
      expect.soft(warnLine).toContain("⚠");
      expect(failLine).toContain("✗");
    });

    it("renders a four-space-indented hint line with backticks stripped", () => {
      const report = createReport([
        {
          title: "Env",
          checks: [
            createCheck({
              name: "git",
              status: "fail",
              detail: "not found",
              hint: "run `gymrat init` to set up",
            }),
          ],
        },
      ]);

      const output = reportLines(renderDoctorReport(report));

      const hintLine = output.find((l) => l.includes("gymrat init"));
      expect(hintLine).toBeDefined();
      if (hintLine === undefined) throw new Error("hint line not found");
      expect(stripAnsi(hintLine)).toMatch(/^ {4}/);
      expect(stripAnsi(hintLine)).not.toContain("`");
    });

    it("omits the hint line when a check has no hint", () => {
      const report = createReport([
        {
          title: "Env",
          checks: [createCheck({ name: "git", status: "ok", detail: "found" })],
        },
      ]);

      const output = reportLines(renderDoctorReport(report));

      const checkLines = output.filter((l) => l.includes("found"));
      expect(checkLines).toHaveLength(1);
    });
  });

  describe("caveat block", () => {
    it("includes a caveat about prepare not being run", () => {
      const report = createReport([{ title: "Env", checks: [createCheck()] }]);

      const output = stripAnsi(renderDoctorReport(report));

      expect(output).toMatch(/prepare.*not run|not.*run.*prepare/i);
    });

    it("includes a caveat about skill file location being the only check", () => {
      const report = createReport([{ title: "Env", checks: [createCheck()] }]);

      const output = stripAnsi(renderDoctorReport(report));

      expect(output).toMatch(/skill.*location|location.*skill|presence.*loaded|loaded.*presence/i);
    });
  });

  describe("summary line", () => {
    it("renders counts for all three statuses", () => {
      const report = createReport([
        {
          title: "Mixed",
          checks: [
            createCheck({ status: "ok" }),
            createCheck({ status: "ok" }),
            createCheck({ status: "warn" }),
            createCheck({ status: "fail" }),
          ],
        },
      ]);

      const output = stripAnsi(renderDoctorReport(report));

      expect(output).toMatch(/2 ok/);
      expect(output).toMatch(/1 warning/);
      expect(output).toMatch(/1 failure/);
    });

    it("never pluralizes ok to oks", () => {
      const report = createReport([
        {
          title: "Mixed",
          checks: [
            createCheck({ status: "ok" }),
            createCheck({ status: "ok" }),
            createCheck({ status: "warn" }),
            createCheck({ status: "fail" }),
          ],
        },
      ]);

      const output = stripAnsi(renderDoctorReport(report));

      expect(output).toContain("2 ok ·");
    });

    it("pluralizes correctly for singular and plural counts", () => {
      const report = createReport([
        {
          title: "All",
          checks: [
            createCheck({ status: "ok" }),
            createCheck({ status: "warn" }),
            createCheck({ status: "warn" }),
            createCheck({ status: "fail" }),
            createCheck({ status: "fail" }),
            createCheck({ status: "fail" }),
          ],
        },
      ]);

      const output = stripAnsi(renderDoctorReport(report));

      expect(output).toMatch(/1 ok/);
      expect(output).toMatch(/2 warnings/);
      expect(output).toMatch(/3 failures/);
    });
  });

  describe("color handling", () => {
    afterEach(() => {
      vi.unstubAllEnvs();
    });

    it("NO_COLOR suppresses ANSI escapes", () => {
      vi.stubEnv("NO_COLOR", "1");
      vi.stubEnv("FORCE_COLOR", undefined);
      const report = createReport([
        {
          title: "Env",
          checks: [createCheck({ status: "ok" }), createCheck({ status: "fail" })],
        },
      ]);

      const output = renderDoctorReport(report);

      expect(output).not.toMatch(/\x1b\[/);
    });

    it("FORCE_COLOR enables ANSI escapes", () => {
      vi.stubEnv("FORCE_COLOR", "1");
      vi.stubEnv("NO_COLOR", undefined);
      const report = createReport([
        {
          title: "Env",
          checks: [createCheck({ status: "ok" }), createCheck({ status: "fail" })],
        },
      ]);

      const output = renderDoctorReport(report);

      expect(output).toMatch(/\x1b\[/);
    });
  });
});

// ---------------------------------------------------------------------------
// JSON rendering: renderDoctorJson
// ---------------------------------------------------------------------------

describe("renderDoctorJson", () => {
  it("includes environment info, sections, checks, and counts", () => {
    const report = createReport(
      [
        {
          title: "Environment",
          checks: [
            createCheck({ name: "git", status: "ok", detail: "available" }),
            createCheck({
              name: "repo",
              status: "warn",
              detail: "not in repo",
              hint: "run inside repo",
            }),
          ],
        },
      ],
      createEnv({ gymratVersion: "1.0.0" }),
    );

    const parsed: unknown = JSON.parse(renderDoctorJson(report));

    expect(parsed).toHaveProperty("environment");
    expect(parsed).toHaveProperty("sections");
    expect(parsed).toHaveProperty("okCount", 1);
    expect(parsed).toHaveProperty("warnCount", 1);
    expect(parsed).toHaveProperty("failCount", 0);
  });

  it("includes check status and hint when present", () => {
    const report = createReport([
      {
        title: "Config",
        checks: [
          createCheck({
            name: "file",
            status: "fail",
            detail: "missing",
            hint: "create gymrat.json",
          }),
        ],
      },
    ]);

    const parsed: unknown = JSON.parse(renderDoctorJson(report));

    expect(parsed).toHaveProperty(["sections", 0, "checks", 0, "status"], "fail");
    expect(parsed).toHaveProperty(["sections", 0, "checks", 0, "hint"], "create gymrat.json");
  });
});
