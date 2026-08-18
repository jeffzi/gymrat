import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

import { readBundledSkill } from "../bundled-skill.js";
import {
  CONFIG_FILENAME,
  configFileValidator,
  GEOMEAN_PRIMARY,
  loopKeyProblems,
} from "../config.js";
import { GymratError } from "../errors.js";
import { DEFAULT_RUNBOOK_PATH, type WizardResult } from "./wizard.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** The outcome of writing one scaffold artifact: newly written, already present, or user-declined. */
export type ArtifactStatus = "created" | "exists" | "declined";

/** The return shape of {@link scaffold}: one entry per artifact it may write. */
export interface ScaffoldResult {
  config: { path: string; status: "created" };
  runbook: { path: string; status: ArtifactStatus };
  skill: { path: string; status: ArtifactStatus };
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Path, relative to the project root, where init writes and doctor checks for the skill file. */
export const SKILL_RELATIVE_PATH = ".claude/skills/gymrat/SKILL.md";

const RUNBOOK_STUB = `# Optimization Runbook

## Goal

<!-- Describe the optimization goal here. -->

## Gating metrics

<!-- List the metrics that must not regress. -->

## Constraints

<!-- List any constraints on the optimization. -->

## Approaches to try

<!-- List strategies for the agent to explore. -->

\`gymrat supervise\` injects this file into the agent's instructions.
`;

// ---------------------------------------------------------------------------
// Config construction
// ---------------------------------------------------------------------------

function buildConfig(wizardResult: WizardResult): Record<string, unknown> {
  const config: Record<string, unknown> = { bench: wizardResult.bench };

  if (wizardResult.adapter !== undefined) {
    config.adapter = wizardResult.adapter;
  }
  if (wizardResult.checks !== undefined) {
    config.checks = wizardResult.checks;
  }
  if (wizardResult.primary !== undefined) {
    config.primary = wizardResult.primary;
  }

  const stop: Record<string, unknown> = {};
  if (wizardResult.stopTarget !== undefined) {
    stop.targetValue = wizardResult.stopTarget;
  }
  if (wizardResult.stopMaxIterations !== undefined) {
    stop.maxIterations = wizardResult.stopMaxIterations;
  }
  if (Object.keys(stop).length > 0) {
    config.stop = stop;
  }

  if (wizardResult.runbook !== false) {
    config.runbook = wizardResult.runbook.path;
  }

  return config;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

function validateConfig(config: Record<string, unknown>): void {
  if (!configFileValidator.check(config)) {
    const issue = configFileValidator.firstIssue(config);
    if (issue === undefined) throw new GymratError("validation failed but no issue was reported");
    throw new GymratError(`Invalid config: ${issue.path} must be ${issue.expected}`);
  }

  const problems = loopKeyProblems({
    primary: config.primary ?? GEOMEAN_PRIMARY,
    ...(config.stop !== undefined && { stop: config.stop }),
  });
  const firstProblem = problems[0];
  if (firstProblem !== undefined) {
    throw new GymratError(firstProblem);
  }
}

// ---------------------------------------------------------------------------
// File writers
// ---------------------------------------------------------------------------

function writeRunbook(
  baseDir: string,
  wizardResult: WizardResult,
): { path: string; status: ArtifactStatus } {
  if (wizardResult.runbook === false) {
    return { path: DEFAULT_RUNBOOK_PATH, status: "declined" };
  }

  const runbookPath = wizardResult.runbook.path;
  const fullPath = resolve(baseDir, runbookPath);

  if (existsSync(fullPath)) {
    return { path: runbookPath, status: "exists" };
  }

  mkdirSync(dirname(fullPath), { recursive: true });
  writeFileSync(fullPath, RUNBOOK_STUB);
  return { path: runbookPath, status: "created" };
}

function writeSkill(
  baseDir: string,
  wizardResult: WizardResult,
  importMetaUrl: string,
): { path: string; status: ArtifactStatus } {
  if (!wizardResult.installSkill) {
    return { path: SKILL_RELATIVE_PATH, status: "declined" };
  }

  const fullPath = join(baseDir, SKILL_RELATIVE_PATH);

  if (existsSync(fullPath)) {
    return { path: SKILL_RELATIVE_PATH, status: "exists" };
  }

  const content = readBundledSkill(importMetaUrl);
  mkdirSync(dirname(fullPath), { recursive: true });
  writeFileSync(fullPath, content);
  return { path: SKILL_RELATIVE_PATH, status: "created" };
}

function writeConfig(
  baseDir: string,
  config: Record<string, unknown>,
): { path: string; status: "created" } {
  const fullPath = join(baseDir, CONFIG_FILENAME);
  writeFileSync(fullPath, JSON.stringify(config, null, 2) + "\n");
  return { path: CONFIG_FILENAME, status: "created" };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Write the gymrat config, runbook stub, and skill file to `baseDir`.
 *
 * `importMetaUrl` is the caller's `import.meta.url`, forwarded to
 * `readBundledSkill` so the bundled skill can be resolved from the
 * package's install location.
 *
 * Write order is runbook -> skill -> config so a failure partway through
 * never leaves a `gymrat.json` pointing at artifacts that were not created.
 */
export function scaffold(
  baseDir: string,
  wizardResult: WizardResult,
  importMetaUrl: string,
): ScaffoldResult {
  const config = buildConfig(wizardResult);
  validateConfig(config);

  const runbook = writeRunbook(baseDir, wizardResult);
  const skill = writeSkill(baseDir, wizardResult, importMetaUrl);
  const configResult = writeConfig(baseDir, config);

  return { config: configResult, runbook, skill };
}
