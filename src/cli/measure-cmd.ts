import type { Command } from "commander";

import type { TargetSpec } from "../compare.js";
import { resolveConfig } from "../config.js";
import { measure } from "../measure.js";
import type { MeasureOptions } from "../measure.js";
import { renderMeasureJson } from "../report/json.js";
import { renderMeasureReport } from "../report/text.js";
import type { MeasurementResult } from "../report/types.js";
import { repoRoot } from "../session/paths.js";
import type { BaselineRecord } from "../session/records.js";
import { appendRecord, requireOpenSession } from "../session/store.js";
import {
  addSharedOptions,
  beginRun,
  colorOverrideOf,
  emitReport,
  parsePositional,
  runGuarded,
  runOptionsOf,
  withRepoLock,
  writeAndFlush,
  type MeasureFlags,
} from "./shared.js";

function baselineRecordOf(result: MeasurementResult): BaselineRecord {
  return {
    type: "baseline",
    at: new Date().toISOString(),
    label: result.label,
    samples: [...result.rounds],
  };
}

/** Register the `measure` subcommand and its target/format options on `program`. */
export function registerMeasure(program: Command): void {
  const currentDirectory: TargetSpec = { target: "." };

  addSharedOptions(
    program
      .command("measure")
      .description("Measure one revision or directory on its own, with nothing to compare it to")
      .argument(
        "[target]",
        "[label=]<ref|dir> to measure; defaults to the current directory",
        parsePositional,
        currentDirectory,
      ),
  )
    .option("-r, --record", "append the run to the session log as a baseline", false)
    .action(async (target: TargetSpec, options: MeasureFlags) => {
      const colorOverride = colorOverrideOf(options);

      const run = await withRepoLock("measure", async () => {
        const progress = beginRun(options, 1);

        const measured = await runGuarded(progress, async () => {
          const config = resolveConfig(options);

          const recording = options.record
            ? requireOpenSession(repoRoot(), "recording a measurement")
            : undefined;

          const measureOptions: MeasureOptions = {
            target,
            ...runOptionsOf(config, progress),
          };

          return { result: await measure(measureOptions), recording };
        });

        if (measured.recording !== undefined) {
          appendRecord(measured.recording.jsonlPath, baselineRecordOf(measured.result));
        }

        return measured;
      });

      await emitReport(
        run.result,
        options,
        { json: renderMeasureJson, text: renderMeasureReport },
        { color: colorOverride },
      );

      if (run.recording !== undefined) {
        await writeAndFlush(
          options.format === "json" ? process.stderr : process.stdout,
          `baseline "${run.result.label}" recorded to session ${run.recording.session.sessionId}\n`,
        );
      }
    });
}
