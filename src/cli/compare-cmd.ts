import type { Command } from "commander";

import { compare } from "../compare.js";
import type { CompareOptions, TargetSpec } from "../compare.js";
import { resolveConfig } from "../config.js";
import { renderJson } from "../report/json.js";
import { renderReport } from "../report/text.js";
import type { FailOnCondition } from "../report/types.js";
import { shouldFailGate, warnEmptyGeomeanGates } from "./gating.js";
import {
  addSharedOptions,
  beginRun,
  collectPositional,
  colorOverrideOf,
  emitReport,
  GATE_EXIT_CODE,
  parseFailOn,
  parsePositional,
  runGuarded,
  runOptionsOf,
  withRepoLock,
  type CompareFlags,
} from "./shared.js";

/** Register the `compare` subcommand and its target/fail-on options on `program`. */
export function registerCompare(program: Command): void {
  const noTargets: TargetSpec[] = [];
  const noFailOnConditions: FailOnCondition[] = [];

  addSharedOptions(
    program
      .command("compare")
      .description("Compare one baseline revision against one or more candidates")
      .argument("<baseline>", "[label=]<ref|dir> to measure against", parsePositional)
      .argument(
        "<candidates...>",
        "[label=]<ref|dir>, each judged against the baseline",
        collectPositional,
        noTargets,
      ),
  )
    .option("--verbose", "name the statistical method behind each verdict", false)
    .option(
      "--fail-on <condition>",
      'exit 1 when a condition trips (repeatable: "regressed", "geomean:<pct>")',
      parseFailOn,
      noFailOnConditions,
    )
    .action(async (baseline: TargetSpec, candidates: TargetSpec[], options: CompareFlags) => {
      const colorOverride = colorOverrideOf(options);

      const result = await withRepoLock("compare", async () => {
        const progress = beginRun(options, 1 + candidates.length);

        return runGuarded(progress, async () => {
          const config = resolveConfig(options);

          const compareOptions: CompareOptions = {
            baseline,
            candidates,
            ...runOptionsOf(config, progress),
            unstableNoisePct: config.unstableNoisePct,
          };

          return compare(compareOptions);
        });
      });

      await emitReport(
        result,
        options,
        { json: renderJson, text: renderReport },
        {
          verbose: options.verbose,
          color: colorOverride,
          failOn: options.failOn,
        },
      );

      await warnEmptyGeomeanGates(options.failOn, result);

      if (shouldFailGate(options.failOn, result)) {
        process.exit(GATE_EXIT_CODE);
      }
    });
}
