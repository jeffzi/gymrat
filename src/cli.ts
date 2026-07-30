#!/usr/bin/env node

import { readFileSync, realpathSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { styleText } from "node:util";

import { createHelpConfig } from "@jeffzi/epaulettes";
import { Command, InvalidArgumentError } from "commander";

import { AdapterError } from "./adapters/index.js";
import { compare } from "./compare.js";
import type { CompareOptions } from "./compare.js";
import { resolveConfig, type CliFlags } from "./config.js";
import { GymratError } from "./errors.js";

/**
 * Parse label=ref syntax from a positional argument.
 * If no '=' is present, uses the whole string as the ref with undefined label.
 * Otherwise, splits on first '=' into label and ref.
 */
function parsePositional(positional: string): { label: string | undefined; ref: string } {
  const eqIndex = positional.indexOf("=");
  if (eqIndex === -1) {
    return { label: undefined, ref: positional };
  }
  return { label: positional.slice(0, eqIndex), ref: positional.slice(eqIndex + 1) };
}

/**
 * Coerce a numeric flag value to a positive integer, rejecting anything else.
 *
 * `parseInt` is too permissive for flag values: it returns `NaN` for
 * non-numeric input and silently truncates trailing garbage (`"10abc"` → `10`).
 * `NaN` is not nullish, so it survives the `??` default chain in
 * `resolveConfig` and only surfaces much later as a confusing benchmark error.
 *
 * Throwing `InvalidArgumentError` lets Commander render a usage error naming
 * the offending flag and value.
 */
function parsePositiveInteger(value: string): number {
  const parsed = Number(value);
  if (!/^\d+$/.test(value) || parsed <= 0) {
    throw new InvalidArgumentError("must be a positive integer.");
  }
  return parsed;
}

/**
 * `validateStream: false` tells `styleText` to skip its own TTY check — the
 * caller already decided whether color is appropriate via the `useColor` flag.
 */
function formatLabel(
  label: string,
  style: Parameters<typeof styleText>[0],
  useColor: boolean,
): string {
  return useColor ? styleText(style, label, { validateStream: false }) : label;
}

/**
 * Render an error for stderr, labelling adapter failures with their class name
 * and appending a styled hint line when the error carries one.
 *
 * An adapter message states what could not be parsed ("No valid METRIC lines
 * found") without saying which layer produced it, so the class name is what
 * tells the user the bench script's output — rather than gymrat's git or config
 * handling — is at fault. Errors raised elsewhere already name their own
 * subsystem, so prefixing them would only add noise.
 *
 * `useColor` defaults to auto-detection: true when stderr is a TTY and
 * `NO_COLOR` is not set.
 */
export function formatCliError(
  error: unknown,
  useColor: boolean = process.stderr.isTTY && process.env.NO_COLOR === undefined,
): string {
  if (error instanceof AdapterError) {
    return `${error.name}: ${error.message}`;
  }

  let output = error instanceof Error ? error.message : String(error);

  if (error instanceof GymratError && error.hint !== undefined) {
    const hintLabel = formatLabel("Hint:", ["yellow", "underline"], useColor);
    output += `\n${hintLabel} ${error.hint}`;
  }

  return output;
}

/**
 * Read the package version from the manifest next to the compiled entry point.
 *
 * `import ... with { type: "json" }` would be tidier, but package.json sits
 * outside `rootDir`, so importing it would pull an extra directory level into
 * `dist/` and break the `dist/cli.js` bin path. Reading at runtime keeps the
 * emitted layout flat and keeps the reported version in lockstep with the
 * manifest instead of a literal that has to be bumped by hand.
 *
 * The relative path resolves to the package root from both `src/cli.ts` (tests)
 * and `dist/cli.js` (published), and npm always ships package.json regardless
 * of the `files` allowlist.
 */
function readPackageVersion(): string {
  const manifest: unknown = JSON.parse(
    readFileSync(new URL("../package.json", import.meta.url), "utf8"),
  );

  if (
    typeof manifest !== "object" ||
    manifest === null ||
    !("version" in manifest) ||
    typeof manifest.version !== "string"
  ) {
    throw new Error("package.json has no string version field");
  }

  return manifest.version;
}

/** Returns a Commander Command instance that can be tested with exitOverride(). */
export function createProgram(): Command {
  const program = new Command();

  program
    .name("gymrat")
    .description("Performance comparison tool for benchmarks")
    .version(readPackageVersion());

  program
    .command("compare <old> <new>")
    .description("Compare performance between two revisions")
    .option("--bench <cmd>", "bench command")
    .option("--prepare <script>", "preparation script to run before each revision")
    .option("--adapter <type>", "adapter type for parsing benchmark output")
    .option("--samples <number>", "paired samples per target", parsePositiveInteger)
    .option("--timeout <number>", "timeout in seconds", parsePositiveInteger)
    .option("--config <file>", "configuration file path")
    .configureHelp(createHelpConfig())
    .action(async (oldRef: string, newRef: string, options: CliFlags) => {
      const oldParsed = parsePositional(oldRef);
      const newParsed = parsePositional(newRef);

      const config = resolveConfig({
        bench: options.bench,
        prepare: options.prepare,
        adapter: options.adapter,
        samples: options.samples,
        timeout: options.timeout,
        config: options.config,
      });

      const compareOptions: CompareOptions = {
        oldTarget: oldParsed.ref,
        newTarget: newParsed.ref,
        oldLabel: oldParsed.label,
        newLabel: newParsed.label,
        bench: config.bench,
        prepare: config.prepare,
        adapter: config.adapter,
        samples: config.samples,
        timeoutSeconds: config.timeoutSeconds,
        unstableNoisePct: config.unstableNoisePct,
        configMetrics: config.metrics,
      };

      const report = await compare(compareOptions);
      process.stdout.write(report + "\n");
    });

  return program;
}

/* v8 ignore next 16 -- entry point. The "executes CLI when invoked through symlink"
   test in tests/cli.test.ts does run this block, but it spawns a child process, and
   in-process v8 coverage cannot attribute execution that happens outside the worker. */
if (import.meta.url === pathToFileURL(realpathSync(process.argv[1]!)).href) {
  try {
    const program = createProgram();
    await program.parseAsync(process.argv);
    process.exit(0);
  } catch (error) {
    process.stderr.write(`${formatCliError(error)}\n`);
    process.exit(1);
  }
}
