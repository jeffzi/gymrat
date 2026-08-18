import { join } from "node:path";

import type { Command } from "commander";

import { resolveBenchlessConfig } from "../config.js";
import { GymratError } from "../errors.js";
import { runGit } from "../git.js";
import { pluralize } from "../report/format.js";
import { acquireLock } from "../session/lock.js";
import { repoRoot, superviseLockfilePath } from "../session/paths.js";
import { ensureGitExclude } from "../session/workspace.js";
import { createClaudeDriver } from "../supervisor/claude.js";
import type { LaunchEvent } from "../supervisor/events.js";
import { summarize } from "../supervisor/events.js";
import { composeKickoff } from "../supervisor/kickoff.js";
import { supervise } from "../supervisor/supervise.js";
import type { SupervisionResult } from "../supervisor/supervise.js";
import {
  exitWithError,
  GATE_EXIT_CODE,
  parseMaxMinutes,
  parsePositiveNumber,
  runOrExit,
  TOOL_FAILURE_EXIT_CODE,
  writeAndFlush,
  type SuperviseFlags,
} from "./shared.js";

function getDirtyFileCount(root: string): number {
  const output = runGit(["status", "--porcelain", "--untracked-files=all"], root).trim();
  if (output === "") return 0;
  return output.split("\n").length;
}

function resolveLogPath(root: string, explicitPath: string | undefined): string {
  if (explicitPath !== undefined) return explicitPath;
  ensureGitExclude(root);
  return join(root, ".gymrat", `supervisor-${Date.now()}.jsonl`);
}

async function validateWorkingTree(root: string, allowDirty: boolean): Promise<number> {
  const dirtyFileCount = getDirtyFileCount(root);
  if (dirtyFileCount > 0 && !allowDirty) {
    await exitWithError(
      new GymratError(
        `Working tree has ${pluralize(dirtyFileCount, "uncommitted file")}.`,
        "Commit or stash your changes, or pass --allow-dirty to proceed anyway.",
      ),
    );
  }
  if (dirtyFileCount > 0) {
    await writeAndFlush(
      process.stderr,
      `warning: working tree has ${pluralize(dirtyFileCount, "dirty file")} — proceeding because --allow-dirty was set\n`,
    );
  }
  return dirtyFileCount;
}

async function reportSupervisionResult(result: SupervisionResult, logPath: string): Promise<void> {
  const durationSec = Math.round(result.durationMs / 1000);
  const minutes = Math.floor(durationSec / 60);
  const seconds = durationSec % 60;
  const summary = [
    `outcome: ${result.outcome.reason}`,
    `ended by: ${result.endedBy}`,
    `duration: ${String(minutes)}m ${String(seconds)}s`,
    `cost: $${result.costUsd.toFixed(2)}`,
    `log: ${logPath}`,
  ].join("\n");

  await writeAndFlush(process.stdout, `${summary}\n`);

  if (result.outcome.reason === "error") {
    if (result.outcome.message) {
      await exitWithError(new GymratError(result.outcome.message));
    }
    process.exit(TOOL_FAILURE_EXIT_CODE);
  }

  if (result.endedBy !== "session") {
    process.exit(GATE_EXIT_CODE);
  }
}

/** Register the `supervise` subcommand and its wall-clock/spend cap options on `program`. */
export function registerSupervise(program: Command): void {
  program
    .command("supervise")
    .description("Run a supervised agent session with wall-clock and spend caps")
    .argument("[prompt]", "optimization prompt for the agent")
    .requiredOption("--max-minutes <number>", "wall-clock cap in minutes", parseMaxMinutes)
    .option("--max-usd <number>", "spend cap in USD", parsePositiveNumber)
    .option("--log <path>", "path for the JSONL event log")
    .option("--model <name>", "model to use for the agent session")
    .option("--allow-dirty", "allow launching with uncommitted changes", false)
    .action(async (prompt: string | undefined, options: SuperviseFlags) => {
      const root = await runOrExit(() => Promise.resolve(repoRoot()));
      const dirtyFileCount = await validateWorkingTree(root, options.allowDirty);

      const release = await runOrExit(() =>
        Promise.resolve(acquireLock(superviseLockfilePath(root), "supervise")),
      );

      process.once("exit", release);

      try {
        const logPath = resolveLogPath(root, options.log);

        const config = resolveBenchlessConfig({}, root);
        const kickoff = composeKickoff(config, import.meta.url, prompt);

        const headSha = runGit(["rev-parse", "HEAD"], root).trim();

        const launch: LaunchEvent = {
          type: "launch",
          timestamp: Date.now(),
          headSha,
          dirty: dirtyFileCount > 0 ? { fileCount: dirtyFileCount } : false,
          maxMinutes: options.maxMinutes,
          maxUsd: options.maxUsd,
          model: options.model,
          runbookPath: config.runbook ?? "",
          kickoffSummary: summarize(kickoff.kickoff),
        };

        const driver = createClaudeDriver();

        await writeAndFlush(process.stderr, `log: ${logPath}\n`);

        const result: SupervisionResult = await runOrExit(() =>
          supervise({
            driver,
            prompt: {
              kickoff: kickoff.kickoff,
              systemPromptAppend: kickoff.systemPromptAppend,
              cwd: root,
              ...(options.model !== undefined && { model: options.model }),
            },
            maxMinutes: options.maxMinutes,
            ...(options.maxUsd !== undefined && { maxUsd: options.maxUsd }),
            logPath,
            launch,
          }),
        );

        await reportSupervisionResult(result, logPath);
      } finally {
        process.removeListener("exit", release);
        release();
      }
    });
}
