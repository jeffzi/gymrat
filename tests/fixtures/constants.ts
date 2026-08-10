import { stripVTControlCharacters as stripAnsi } from "node:util";

/** User-facing hint attached to ref-target `CommandError` instances, duplicated here so tests assert against the same string production emits. */
export const REF_TARGET_HINT =
  "the worktree only contains files tracked at this ref; untracked, gitignored, or not-yet-committed files are absent";

/** The shape `toISOString` produces, for asserting a timestamp is one without pinning the instant. */
export const ISO_PATTERN = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;

/** Matches the start of any ANSI escape sequence, for asserting output is or is not styled. */
export const ANSI_RE = /\x1b\[/;

/** A session id of the shape `newSessionId` mints, fixed so records read back predictably. */
export const SESSION_ID = "20260808-141530-a3f2";

/**
 * A report's lines, stripped of color, for asserting on what it says.
 *
 * `trimLines` also drops each line's indentation, which the loop reports use to
 * nest a metric under its group.
 */
export function reportLines(report: string, options: { trimLines?: boolean } = {}): string[] {
  const lines = stripAnsi(report).split("\n");
  return options.trimLines === true ? lines.map((line) => line.trim()) : lines;
}
