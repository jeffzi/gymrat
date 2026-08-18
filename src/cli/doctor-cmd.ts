import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import type { Command } from "commander";

import { inspectConfig } from "../config-inspect.js";
import { CONFIG_DEFAULTS, type CliFlags } from "../config.js";
import { buildBenchSection } from "../doctor/bench.js";
import {
  buildConfigSection,
  buildEnvironmentSection,
  buildWorkflowSection,
  createDoctorReport,
  type DoctorReport,
} from "../doctor/checks.js";
import { renderDoctorJson, renderDoctorReport } from "../doctor/render.js";
import { assertNever, GymratError } from "../errors.js";
import { SKILL_RELATIVE_PATH } from "../init/scaffold.js";
import {
  addSharedOptions,
  detectGitEnvironment,
  GATE_EXIT_CODE,
  runOrExit,
  suppressColor,
  writeAndFlush,
  type DoctorFlags,
} from "./shared.js";

/**
 * `import ... with { type: "json" }` would be tidier, but package.json sits
 * outside `rootDir`, so importing it would pull an extra directory level into
 * `dist/` and break the `dist/cli.js` bin path. Reading at runtime keeps the
 * emitted layout flat.
 *
 * The relative path resolves to the package root from both `src/cli/` (tests)
 * and `dist/cli/` (published).
 */
export function readPackageVersion(): string {
  const manifest: unknown = JSON.parse(
    readFileSync(new URL("../../package.json", import.meta.url), "utf8"),
  );

  if (
    typeof manifest !== "object" ||
    manifest === null ||
    !("version" in manifest) ||
    typeof manifest.version !== "string"
  ) {
    throw new GymratError("package.json has no string version field");
  }

  return manifest.version;
}

async function buildDoctorReport(options: DoctorFlags): Promise<DoctorReport> {
  const version = readPackageVersion();

  const cwd = process.cwd();
  const gitEnv = detectGitEnvironment(cwd);
  const baseDir = gitEnv.repoRootDir ?? cwd;

  const { bench: rawBench, ...rest } = options;
  const configFlags: CliFlags = {
    ...rest,
    ...(typeof rawBench === "string" && { bench: rawBench }),
  };
  const inspection = inspectConfig(configFlags, baseDir);

  const envSection = buildEnvironmentSection({
    gitAvailable: gitEnv.gitAvailable,
    insideGitRepo: gitEnv.insideGitRepo,
    ...(gitEnv.gitError !== undefined && { gitError: gitEnv.gitError }),
  });
  const configSection = buildConfigSection(inspection);
  const workflowSection = buildWorkflowSection({
    config: inspection.config ?? CONFIG_DEFAULTS,
    problems: inspection.problems,
    skillFileExists: existsSync(join(baseDir, SKILL_RELATIVE_PATH)),
  });

  const benchSection = await buildBenchSection({
    bench: inspection.bench,
    adapter: inspection.config?.adapter ?? CONFIG_DEFAULTS.adapter,
    timeoutSeconds: inspection.config?.timeoutSeconds ?? CONFIG_DEFAULTS.timeoutSeconds,
    primary: inspection.config?.primary ?? CONFIG_DEFAULTS.primary,
    ...(inspection.config?.metrics !== undefined && { metrics: inspection.config.metrics }),
    ...(inspection.config?.kinds !== undefined && { kinds: inspection.config.kinds }),
    repoRoot: baseDir,
    noBench: options.bench === false,
    configFailed: inspection.problems.length > 0,
  });

  return createDoctorReport(
    {
      gymratVersion: version,
      nodeVersion: process.versions.node,
      platform: process.platform,
    },
    [envSection, configSection, workflowSection, benchSection],
  );
}

/** Register the `doctor` subcommand and its `--no-bench` flag on `program`. */
export function registerDoctor(program: Command): void {
  addSharedOptions(
    program
      .command("doctor")
      .description("Check the project setup and report any problems")
      .option("--no-bench", "skip the smoke-run bench section"),
  ).action(async (options: DoctorFlags) => {
    if (!options.color) {
      suppressColor();
    }

    const report = await runOrExit(() => buildDoctorReport(options));

    let output: string;
    switch (options.format) {
      case "json":
        output = renderDoctorJson(report);
        break;
      case "text":
        output = renderDoctorReport(report);
        break;
      default:
        assertNever(options.format);
    }

    await writeAndFlush(process.stdout, output + "\n");

    if (report.hasFailures) {
      process.exit(GATE_EXIT_CODE);
    }
  });
}
