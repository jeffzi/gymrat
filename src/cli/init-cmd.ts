import { existsSync } from "node:fs";
import { join } from "node:path";

import { Option, type Command } from "commander";

import { CONFIG_FILENAME } from "../config.js";
import { assertNever, GymratError } from "../errors.js";
import { scaffold, type ArtifactStatus, type ScaffoldResult } from "../init/scaffold.js";
import { runWizard } from "../init/wizard.js";
import { highlightInlineCode } from "../report/format.js";
import {
  detectGitEnvironment,
  exitWithError,
  parsePositiveIntegerUpTo,
  parseStopTargetValue,
  runOrExit,
  writeAndFlush,
  type InitFlags,
} from "./shared.js";

function formatInitArtifact(
  label: string,
  artifact: { path: string; status: ArtifactStatus },
): string {
  switch (artifact.status) {
    case "created":
      return `  ${label} created at ${artifact.path}`;
    case "exists":
      return `  ${label} already exists at ${artifact.path}`;
    case "declined":
      return `  ${label} declined`;
    default:
      return assertNever(artifact.status);
  }
}

function formatInitSummary(result: ScaffoldResult): string {
  return [
    formatInitArtifact("Config:", result.config),
    formatInitArtifact("Runbook:", result.runbook),
    formatInitArtifact("Skill:", result.skill),
    "",
    highlightInlineCode("Run `gymrat doctor` to verify the setup."),
  ].join("\n");
}

/** Register the `init` subcommand and its scaffold flags on `program`. */
export function registerInit(program: Command): void {
  program
    .command("init")
    .description("Scaffold a gymrat.json, skill file, and runbook")
    .option("--bench <cmd>", "bench command")
    .option("--adapter <name>", "adapter type")
    .option("--checks <cmd>", "checks command")
    .option("--stop-target <number>", "stop target value", parseStopTargetValue)
    .option(
      "--stop-max-iterations <number>",
      "stop max iterations",
      parsePositiveIntegerUpTo(Number.MAX_SAFE_INTEGER),
    )
    .option("--primary <metric>", "primary metric name")
    .addOption(new Option("--runbook [path]", "create runbook").preset(true))
    .addOption(new Option("--no-runbook", "skip runbook"))
    .option("--skill", "install skill")
    .option("--no-skill", "skip skill")
    .option("-y, --yes", "non-interactive mode", false)
    .action(async (options: InitFlags) => {
      const cwd = process.cwd();
      const { repoRootDir } = detectGitEnvironment(cwd);
      const baseDir = repoRootDir ?? cwd;

      const configPath = join(baseDir, CONFIG_FILENAME);
      if (existsSync(configPath)) {
        await exitWithError(
          new GymratError(
            `${configPath} already exists.`,
            "Edit it directly, or run `gymrat doctor` to verify the setup.",
          ),
        );
      }

      const result = await runOrExit(async () => {
        const wizardResult = await runWizard({
          bench: options.bench,
          adapter: options.adapter,
          checks: options.checks,
          stopTarget: options.stopTarget,
          stopMaxIterations: options.stopMaxIterations,
          primary: options.primary,
          runbook: options.runbook,
          skill: options.skill,
          yes: options.yes,
          input: process.stdin,
          output: process.stderr,
        });
        return scaffold(baseDir, wizardResult, import.meta.url);
      });

      await writeAndFlush(process.stdout, `${formatInitSummary(result)}\n`);
    });
}
