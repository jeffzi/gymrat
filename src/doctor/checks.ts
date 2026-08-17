import type { BenchlessConfig, ConfigInspection } from "../config.js";

// ---------------------------------------------------------------------------
// Report model
// ---------------------------------------------------------------------------

/** The outcome severity of a single diagnostic check. */
export type CheckStatus = "ok" | "warn" | "fail";

/** One diagnostic check: a named probe with a status, a human detail line, and an optional fix hint. */
export interface Check {
  name: string;
  status: CheckStatus;
  detail: string;
  hint?: string;
}

/** A titled group of related checks (e.g. "Environment", "Configuration"). */
export interface CheckSection {
  title: string;
  checks: Check[];
}

/** Version and platform context printed at the top of the doctor report. */
export interface EnvironmentInfo {
  gymratVersion: string;
  nodeVersion: string;
  platform: string;
}

/**
 * The assembled doctor report: sections, environment, and derived counts.
 *
 * `hasFailures` is `failCount > 0` — the CLI uses it to choose exit code 0 vs 1.
 */
export interface DoctorReport {
  environment: EnvironmentInfo;
  sections: CheckSection[];
  okCount: number;
  warnCount: number;
  failCount: number;
  hasFailures: boolean;
}

/** Build a passing check. */
function okCheck(name: string, detail: string): Check {
  return { name, status: "ok", detail };
}

/** Build a warning or failing check, with an optional hint pointing at the fix. */
function issueCheck(name: string, status: CheckStatus, detail: string, hint?: string): Check {
  return hint === undefined ? { name, status, detail } : { name, status, detail, hint };
}

/** Assemble sections into a report, deriving ok/warn/fail counts from all checks. */
export function createDoctorReport(
  environment: EnvironmentInfo,
  sections: CheckSection[],
): DoctorReport {
  const allChecks = sections.flatMap((s) => s.checks);
  const counts: Record<CheckStatus, number> = { ok: 0, warn: 0, fail: 0 };
  for (const check of allChecks) {
    counts[check.status]++;
  }

  return {
    environment,
    sections,
    okCount: counts.ok,
    warnCount: counts.warn,
    failCount: counts.fail,
    hasFailures: counts.fail > 0,
  };
}

// ---------------------------------------------------------------------------
// Environment section
// ---------------------------------------------------------------------------

interface EnvironmentInput {
  gitAvailable: boolean;
  insideGitRepo: boolean;
  gitError?: string;
}

/**
 * FAILs when git is absent from PATH; WARNs when cwd is not inside a git
 * repository, or when the repository root could not be resolved.
 */
export function buildEnvironmentSection(input: EnvironmentInput): CheckSection {
  const checks: Check[] = [];

  checks.push(
    input.gitAvailable
      ? okCheck("git", "git is available on PATH")
      : issueCheck(
          "git",
          "fail",
          "git is not available on PATH",
          "Install git: https://git-scm.com/downloads",
        ),
  );

  checks.push(
    input.insideGitRepo
      ? okCheck("git repository", "current directory is inside a git repository")
      : issueCheck(
          "git repository",
          "warn",
          "current directory is not inside a git repository",
          "The compare command resolves refs against a git repository",
        ),
  );

  if (input.gitError !== undefined) {
    checks.push(
      issueCheck(
        "git repository root",
        "warn",
        `could not determine the repository root: ${input.gitError}`,
        "Falling back to the current directory; commands may operate on the wrong path",
      ),
    );
  }

  return { title: "Environment", checks };
}

// ---------------------------------------------------------------------------
// Config section
// ---------------------------------------------------------------------------

/** One FAIL per collected config problem; a single OK when clean or absent. */
export function buildConfigSection(inspection: ConfigInspection): CheckSection {
  const checks: Check[] = [];

  if (inspection.problems.length > 0) {
    for (const problem of inspection.problems) {
      checks.push({ name: "config", status: "fail", detail: problem });
    }
  } else if (inspection.configPath === undefined) {
    checks.push({
      name: "config",
      status: "ok",
      detail: "No config file found; operating with defaults only",
    });
  } else {
    checks.push({
      name: "config",
      status: "ok",
      detail: `Config file loaded: ${inspection.configPath}`,
    });
  }

  return { title: "Configuration", checks };
}

// ---------------------------------------------------------------------------
// Workflow section
// ---------------------------------------------------------------------------

interface WorkflowInput {
  config: BenchlessConfig;
  /** Config-level problems surfaced by {@link ConfigInspection}. When non-empty, workflow checks are skipped. */
  problems?: string[];
  skillFileExists: boolean;
}

/** WARNs for each missing workflow piece (skill file, checks, stop, runbook) with a fix hint. */
export function buildWorkflowSection(input: WorkflowInput): CheckSection {
  if (input.problems !== undefined && input.problems.length > 0) {
    return {
      title: "Workflow",
      checks: [okCheck("workflow", "Skipped — fix config errors first")],
    };
  }

  const checks: Check[] = [];
  const { config, skillFileExists } = input;

  checks.push(
    skillFileExists
      ? okCheck("skill file", "Skill file is installed")
      : issueCheck(
          "skill file",
          "warn",
          "No skill file — Claude Code agents won't have gymrat's workflow instructions",
          "Run `gymrat init` to scaffold the project, or install manually with `npx skills add jeffzi/gymrat`",
        ),
  );

  checks.push(
    config.checks !== undefined
      ? okCheck("checks", `checks: ${config.checks}`)
      : issueCheck(
          "checks",
          "warn",
          "checks is not configured",
          "Without checks, keep cannot gate commits",
        ),
  );

  if (
    config.stop !== undefined &&
    (config.stop.targetValue !== undefined || config.stop.maxIterations !== undefined)
  ) {
    const parts: string[] = [];
    if (config.stop.targetValue !== undefined) {
      parts.push(`targetValue: ${String(config.stop.targetValue)}`);
    }
    if (config.stop.maxIterations !== undefined) {
      parts.push(`maxIterations: ${String(config.stop.maxIterations)}`);
    }
    checks.push(okCheck("stop", `stop: ${parts.join(", ")}`));
  } else {
    checks.push(
      issueCheck(
        "stop",
        "warn",
        "stop is not configured",
        "Without stop, a session has no finish line",
      ),
    );
  }

  checks.push(
    config.runbook !== undefined
      ? okCheck("runbook", `runbook: ${config.runbook}`)
      : issueCheck(
          "runbook",
          "warn",
          "runbook is not configured",
          "Run `gymrat init` to create a runbook, or add `runbook` to gymrat.json. Without one, supervise has no instructions to follow.",
        ),
  );

  return { title: "Workflow", checks };
}
