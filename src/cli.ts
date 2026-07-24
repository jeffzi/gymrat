#!/usr/bin/env node

import { realpathSync } from "node:fs";
import { pathToFileURL } from "node:url";

import { createHelpConfig } from "@jeffzi/epaulettes";
import { Command } from "commander";

import { compare } from "./compare.js";
import type { CompareOptions } from "./compare.js";
import { resolveConfig } from "./config.js";

interface CompareCommandOptions {
  bench?: string;
  prepare?: string;
  adapter?: string;
  samples?: number;
  timeout?: number;
  config?: string;
}

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
 * Create and configure the CLI program.
 * Returns a Commander Command instance that can be tested with exitOverride().
 */
export function createProgram(): Command {
  const program = new Command();

  program.name("gymrat").description("Performance comparison tool for benchmarks").version("0.0.0");

  program
    .command("compare <old> <new>")
    .description("Compare performance between two revisions")
    .option("--bench <cmd>", "bench command")
    .option("--prepare <script>", "preparation script to run before each revision")
    .option("--adapter <type>", "adapter type for parsing benchmark output")
    .option("--samples <number>", "paired samples per target", (v) => parseInt(v, 10))
    .option("--timeout <number>", "timeout in seconds", (v) => parseInt(v, 10))
    .option("--config <file>", "configuration file path")
    .configureHelp(createHelpConfig())
    .action(async (oldRef: string, newRef: string, options: CompareCommandOptions) => {
      // Parse positional arguments to extract labels and refs
      const oldParsed = parsePositional(oldRef);
      const newParsed = parsePositional(newRef);

      // Resolve configuration from CLI flags
      const config = resolveConfig({
        bench: options.bench,
        prepare: options.prepare,
        adapter: options.adapter,
        samples: options.samples,
        timeout: options.timeout,
        config: options.config,
      });

      // Build compare options
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
      };

      // Run comparison
      const report = await compare(compareOptions);

      // Write report to stdout
      process.stdout.write(report + "\n");
    });

  return program;
}

/* v8 ignore next 16 -- entry point: only run if this is the main module (not imported for testing) */
if (import.meta.url === pathToFileURL(realpathSync(process.argv[1]!)).href) {
  try {
    const program = createProgram();
    await program.parseAsync(process.argv);
    process.exit(0);
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exit(1);
  }
}
