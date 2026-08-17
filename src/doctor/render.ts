import { styleText } from "node:util";

import { highlightInlineCode, pluralize, type Style } from "../report/format.js";
import type { CheckStatus, DoctorReport } from "./checks.js";

const STATUS_GLYPHS: Record<CheckStatus, string> = {
  ok: "✓",
  warn: "⚠",
  fail: "✗",
};

const STATUS_STYLES: Record<CheckStatus, Style> = {
  ok: "green",
  warn: "yellow",
  fail: "red",
};

/**
 * Render a doctor report as styled text for the terminal.
 *
 * Color follows `styleText` auto-detection: `NO_COLOR` / `FORCE_COLOR` env
 * vars and the stream's TTY status govern whether ANSI escapes appear.
 */
export function renderDoctorReport(report: DoctorReport): string {
  const lines: string[] = [];

  const { gymratVersion, nodeVersion, platform } = report.environment;
  lines.push(
    styleText("bold", `gymrat v${gymratVersion}`) +
      styleText("dim", ` · node ${nodeVersion} · ${platform}`),
  );
  lines.push("");

  for (const section of report.sections) {
    lines.push(styleText("bold", section.title));
    for (const check of section.checks) {
      const glyph = styleText(STATUS_STYLES[check.status], STATUS_GLYPHS[check.status]);
      lines.push(`  ${glyph} ${check.detail}`);
      if (check.hint !== undefined) {
        lines.push(`    ${styleText("dim", highlightInlineCode(check.hint))}`);
      }
    }
    lines.push("");
  }

  lines.push(
    styleText(
      "dim",
      "Note: prepare scripts were not run; only the Claude skill file location was checked (presence ≠ loaded).",
    ),
  );
  lines.push("");

  const summary = [
    pluralize(report.okCount, "ok", "ok"),
    pluralize(report.warnCount, "warning"),
    pluralize(report.failCount, "failure"),
  ].join(" · ");
  lines.push(summary);

  return lines.join("\n");
}

/** Serialize the report as JSON for machine consumption. */
export function renderDoctorJson(report: DoctorReport): string {
  return JSON.stringify(report);
}
