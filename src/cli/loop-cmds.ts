import type { Command } from "commander";

import type { CliFlags } from "../config.js";
import { resolveBenchlessConfig, resolveConfig } from "../config.js";
import { confirmAction } from "../confirm.js";
import { finalizeSession } from "../loop/finalize.js";
import { iterateSession, LoopStopError } from "../loop/iterate.js";
import { discardSession, keepSession } from "../loop/settle.js";
import type { StartResult } from "../loop/start.js";
import { startSession } from "../loop/start.js";
import { statusSession } from "../loop/status.js";
import { pluralize } from "../report/format.js";
import { formatBaselineRef } from "../report/loop.js";
import { repoRoot } from "../session/paths.js";
import { requireOpenSession } from "../session/store.js";
import {
  addConfigOptions,
  GATE_EXIT_CODE,
  isTTY,
  runInterruptibly,
  runOrExit,
  suppressColor,
  TOOL_FAILURE_EXIT_CODE,
  withRepoLock,
  writeAndFlush,
  type FinalizeFlags,
  type KeepFlags,
  type StatusFlags,
} from "./shared.js";

function formatStartSummary(result: StartResult, runbook?: string): string {
  const { session, state } = result;
  const headline = result.resumed
    ? `Resumed session ${session.sessionId} — ${pluralize(state.iterationCount, "iteration")}, ${pluralize(state.keepCount, "keep")}`
    : `Started session ${session.sessionId}`;

  const rows: readonly (readonly [string, string])[] = [
    ["branch", session.branch],
    ["baseline", formatBaselineRef(session.baseline)],
    ["experiment worktree", session.worktrees.experiment],
    ["baseline worktree", session.worktrees.baseline],
  ];
  const labelWidth = Math.max(...rows.map(([label]) => label.length)) + 1;

  return [
    headline,
    ...rows.map(([label, value]) => `  ${`${label}:`.padEnd(labelWidth)} ${value}`),
    ...(runbook === undefined ? [] : [`  runbook: ${runbook} — read it before your first edit`]),
    ...(result.archivedPath === undefined
      ? []
      : [`  archived the finalized session ${result.archived} to ${result.archivedPath}`]),
  ].join("\n");
}

function registerSessionCommands(program: Command): void {
  addConfigOptions(
    program
      .command("start")
      .description("Create or resume this repository's optimization session")
      .argument("[ref]", "ref the baseline is pinned to; defaults to HEAD"),
  ).action(async (ref: string | undefined, options: CliFlags) => {
    const started = await withRepoLock("start", () => {
      const root = repoRoot();
      const config = resolveConfig(options, root);
      return Promise.resolve({
        result: startSession(root, ref, config),
        runbook: config.runbook,
      });
    });

    await writeAndFlush(process.stdout, `${formatStartSummary(started.result, started.runbook)}\n`);
  });

  addConfigOptions(
    program
      .command("iterate")
      .description("Measure the session's experiment worktree against its baseline"),
  ).action(async (options: CliFlags) => {
    const result = await withRepoLock(
      "iterate",
      async () => {
        const root = repoRoot();
        return runInterruptibly((signal) =>
          iterateSession(root, resolveConfig(options, root), { signal }),
        );
      },
      (error) => (error instanceof LoopStopError ? GATE_EXIT_CODE : TOOL_FAILURE_EXIT_CODE),
    );

    await writeAndFlush(process.stdout, `${result.report}\n`);
  });

  addConfigOptions(
    program
      .command("keep")
      .description("Commit the session's measured edit once its checks pass")
      .option("-m, --message <text>", "commit message for the kept edit"),
  ).action(async (options: KeepFlags) => {
    const result = await withRepoLock("keep", async () => {
      const root = repoRoot();
      return keepSession(root, resolveBenchlessConfig(options, root), {
        ...(options.message !== undefined && { message: options.message }),
      });
    });

    await writeAndFlush(process.stdout, `${result.report}\n`);

    if (result.record.status === "blocked") {
      process.exit(GATE_EXIT_CODE);
    }
  });
}

function registerSettleCommands(program: Command): void {
  program
    .command("discard")
    .description("Revert the session's experiment worktree to its last commit")
    .option("-f, --force", "skip the confirmation prompt")
    .action(async (options: { force?: boolean }) => {
      const root = repoRoot();

      if (isTTY(process.stdin) && options.force !== true) {
        const { session } = requireOpenSession(root, "discard");
        const confirmed = await confirmAction(
          `discard will revert uncommitted changes in ${session.worktrees.experiment}.\nProceed?`,
          process.stdin,
        );
        if (!confirmed) {
          await writeAndFlush(process.stderr, "discard cancelled\n");
          process.exit(GATE_EXIT_CODE);
        }
      }

      const result = await withRepoLock("discard", () => Promise.resolve(discardSession(root)));

      await writeAndFlush(process.stdout, `${result.report}\n`);
    });

  program
    .command("finalize")
    .description("Collapse the session's kept iterations into one commit and close it")
    .option("-m, --message <text>", "message for the squash commit")
    .option("--branch <name>", "branch to point at the squash commit (default: <branch>-final)")
    .action(async (options: FinalizeFlags) => {
      const result = await withRepoLock("finalize", () =>
        Promise.resolve(
          finalizeSession(repoRoot(), {
            ...(options.message !== undefined && { message: options.message }),
            ...(options.branch !== undefined && { branch: options.branch }),
          }),
        ),
      );

      await writeAndFlush(process.stdout, `${result.report}\n`);
    });

  addConfigOptions(
    program
      .command("status")
      .description("Show this repository's session history, read from its log")
      .option("--no-color", "print the report without ANSI styles"),
  ).action(async (options: StatusFlags) => {
    if (!options.color) {
      suppressColor();
    }

    const report = await runOrExit(() => {
      const root = repoRoot();
      return Promise.resolve(statusSession(root, resolveBenchlessConfig(options, root)));
    });

    await writeAndFlush(process.stdout, `${report}\n`);
  });
}

/** Register the `start`, `iterate`, `settle`, and `status` subcommands on `program`. */
export function registerLoopCommands(program: Command): void {
  registerSessionCommands(program);
  registerSettleCommands(program);
}
