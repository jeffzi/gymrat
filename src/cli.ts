#!/usr/bin/env node

import { realpathSync } from "node:fs";
import { pathToFileURL } from "node:url";

import { createHelpConfig } from "@jeffzi/epaulettes";
import { Command, CommanderError } from "commander";

import { registerCompare } from "./cli/compare-cmd.js";
import { readPackageVersion, registerDoctor } from "./cli/doctor-cmd.js";
import { registerInit } from "./cli/init-cmd.js";
import { registerLoopCommands } from "./cli/loop-cmds.js";
import { registerMeasure } from "./cli/measure-cmd.js";
import { exitWithError, setDebugMode, TOOL_FAILURE_EXIT_CODE } from "./cli/shared.js";
import { registerSupervise } from "./cli/supervise-cmd.js";

export { formatCliError } from "./cli/shared.js";

const DOCS_URL = "https://github.com/jeffzi/gymrat#readme";
const BUGS_URL = "https://github.com/jeffzi/gymrat/issues";

const ROOT_EPILOGUE = `
Examples:

  • gymrat compare main my-branch --bench "npm run bench"

  • gymrat compare old=main new=perf/decode --bench "npm run bench" --fail-on regressed

  • gymrat measure --bench "npm run bench"

Docs: ${DOCS_URL}
Bugs: ${BUGS_URL}
`;

const COMPARE_EPILOGUE = `
Examples:

  • gymrat compare main perf/faster-decode --bench "npm run bench"

  • gymrat compare main perf/simd perf/lookup-table --bench "npm run bench"

  • gymrat compare old=main new=perf/faster-decode --bench "npm run bench"

  • gymrat compare main my-branch --bench "npm run bench" --fail-on geomean:2 --format json
`;

const MEASURE_EPILOGUE = `
Examples:

  • gymrat measure --bench "npm run bench"

  • gymrat measure release=v2.0.0 --bench "npm run bench" --adapter mitata

  • gymrat measure --bench "npm run bench" --record
`;

/**
 * Build the `gymrat` program: fully wired but inert until the caller invokes
 * `parseAsync` on it.
 *
 * `readPackageVersion()` runs during construction, so building a program reads
 * `package.json` off disk even before any command executes.
 */
export function createProgram(): Command {
  const program = new Command();

  program
    .name("gymrat")
    .description("Performance comparison tool for benchmarks")
    .version(readPackageVersion())
    .option("-d, --debug", "show stack traces on errors", false)
    .configureHelp(createHelpConfig())
    .exitOverride((err) => {
      throw new CommanderError(
        err.exitCode === 0 ? 0 : TOOL_FAILURE_EXIT_CODE,
        err.code,
        err.message,
      );
    })
    .hook("preSubcommand", (thisCommand) => {
      setDebugMode(thisCommand.opts<{ debug: boolean }>().debug);
    });

  registerCompare(program);
  registerMeasure(program);
  registerLoopCommands(program);
  registerInit(program);
  registerDoctor(program);
  registerSupervise(program);

  program.addHelpText("after", ROOT_EPILOGUE);
  const subcommandEpilogues: Record<string, string> = {
    compare: COMPARE_EPILOGUE,
    measure: MEASURE_EPILOGUE,
  };
  for (const sub of program.commands) {
    const epilogue = subcommandEpilogues[sub.name()];
    if (epilogue !== undefined) {
      sub.addHelpText("after", epilogue);
    }
  }

  return program;
}

/**
 * Resolve argv[1] to a canonical file URL, or `undefined` when it names nothing.
 *
 * Importing this module must never throw on the way in, and argv[1] is not
 * guaranteed to be a live path: `node -e` and the REPL leave it unset, and it
 * can point at a file that no longer exists. Neither case is this module being
 * the entry point, so both resolve to `undefined` rather than propagating.
 */
function resolveEntryUrl(entryPath: string | undefined): string | undefined {
  if (entryPath === undefined) {
    return undefined;
  }
  try {
    return pathToFileURL(realpathSync(entryPath)).href;
  } catch {
    return undefined;
  }
}

const isEntryPoint = resolveEntryUrl(process.argv[1]) === import.meta.url;

/* istanbul ignore next -- entry point. The "executes CLI when invoked through symlink"
   test in tests/cli.test.ts does run this block, but it spawns a child process, and
   in-process coverage cannot attribute execution that happens outside the worker. */
if (isEntryPoint) {
  try {
    const program = createProgram();
    await program.parseAsync(process.argv);
    process.exit(0);
  } catch (error) {
    /* Commander already wrote help, the version, or the usage error to its stream. */
    if (error instanceof CommanderError) {
      process.exit(error.exitCode);
    }
    await exitWithError(error);
  }
}
