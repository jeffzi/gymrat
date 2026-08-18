import path from "node:path";

import { envStringResult, NUMBER_ENV_FIELDS, STRING_ENV_FIELDS } from "./config-env.js";
import {
  assignDefined,
  CONFIG_FILENAME,
  findImplicitBase,
  flagProblem,
  loadConfigFileCollecting,
  loopKeyProblems,
  mergeConfig,
  runbookProblem,
  type BenchlessConfig,
  type CliFlags,
  type ConfigFile,
} from "./config.js";

/** Result of loading and validating a config file — carries either a clean config or the problems found. */
export interface ConfigInspection {
  /** The resolved path to the config file, or undefined when no file was found. */
  configPath: string | undefined;
  /** Whether the resolved config path points to an existing file. */
  configExists: boolean;
  /** Human-worded problems found during inspection; empty when the config is clean. */
  problems: string[];
  /** The settled configuration, present only when no problems were found. */
  config?: BenchlessConfig;
  /** The resolved bench command, present only when clean and a bench value exists. */
  bench?: string;
}

function collectOneEnv<T>(
  envVar: string,
  reader: (name: string) => { value?: T; problem?: string },
  problems: string[],
): T | undefined {
  const r = reader(envVar);
  if (r.problem !== undefined) problems.push(r.problem);
  return r.value;
}

function collectEnvFlags(flags: CliFlags): { flags: CliFlags; problems: string[] } {
  const problems: string[] = [];
  const result: CliFlags = {};
  for (const { field, envVar, reader } of STRING_ENV_FIELDS) {
    if (flags[field] === undefined) {
      assignDefined(result, field, collectOneEnv(envVar, reader, problems));
    }
  }
  for (const { field, envVar, reader } of NUMBER_ENV_FIELDS) {
    if (flags[field] === undefined) {
      assignDefined(result, field, collectOneEnv(envVar, reader, problems));
    }
  }
  return { flags: result, problems };
}

/**
 * Resolve which config file to load, load and validate it, and report any problems.
 *
 * When the config source itself is broken (empty `--config` flag, empty
 * `GYMRAT_CONFIG`), file loading is skipped and an empty config file is
 * returned so the merge can still produce defaults.
 */
function resolveConfigSource(
  flags: CliFlags,
  baseDir: string | undefined,
): {
  configPath: string | undefined;
  configExists: boolean;
  configFile: ConfigFile | undefined;
  problems: string[];
} {
  const problems: string[] = [];

  let envConfigPath: string | undefined;
  let envConfigFailed = false;
  if (flags.config === undefined) {
    const r = envStringResult("GYMRAT_CONFIG");
    if (r.problem !== undefined) {
      problems.push(r.problem);
      envConfigFailed = true;
    }
    envConfigPath = r.value;
  }

  const explicitConfig = (flags.config !== "" ? flags.config : undefined) ?? envConfigPath;
  const skipLoading = flags.config === "" || envConfigFailed;

  if (skipLoading) {
    return { configPath: undefined, configExists: false, configFile: {}, problems };
  }

  const resolvedPath = explicitConfig ?? path.join(baseDir ?? findImplicitBase(), CONFIG_FILENAME);
  const required = explicitConfig !== undefined;
  const fileResult = loadConfigFileCollecting(resolvedPath, required);
  problems.push(...fileResult.problems);

  return {
    configPath: required || fileResult.exists ? resolvedPath : undefined,
    configExists: fileResult.exists,
    configFile: fileResult.configFile,
    problems,
  };
}

function collectFlagProblems(flags: CliFlags): string[] {
  const problems: string[] = [];
  for (const field of ["bench", "prepare", "adapter", "config"] as const) {
    const p = flagProblem(field, flags[field]);
    if (p !== undefined) problems.push(p);
  }
  return problems;
}

function buildEffectiveFlags(flags: CliFlags, envFlags: CliFlags): CliFlags {
  const effective: CliFlags = { ...envFlags };
  for (const field of ["bench", "prepare", "adapter"] as const) {
    if (flags[field] !== undefined && flags[field] !== "") effective[field] = flags[field];
  }
  if (flags.samples !== undefined) effective.samples = flags.samples;
  if (flags.timeout !== undefined) effective.timeout = flags.timeout;
  return effective;
}

function resolveRunbook(
  config: BenchlessConfig,
  configPath: string | undefined,
  problems: string[],
): void {
  if (config.runbook === undefined || configPath === undefined) return;
  const configDir = path.dirname(configPath);
  const rp = runbookProblem(config.runbook, configDir);
  if (rp !== undefined) {
    problems.push(rp);
  } else {
    config.runbook = path.resolve(configDir, config.runbook);
  }
}

export function inspectConfig(flags: CliFlags, baseDir?: string): ConfigInspection {
  const problems = collectFlagProblems(flags);

  const envResult = collectEnvFlags(flags);
  problems.push(...envResult.problems);

  const effective = buildEffectiveFlags(flags, envResult.flags);

  const source = resolveConfigSource(flags, baseDir);
  problems.push(...source.problems);
  const { configPath, configExists, configFile } = source;

  if (configFile === undefined) {
    return { configPath, configExists, problems };
  }

  const config = mergeConfig(effective, configFile);
  problems.push(...loopKeyProblems(config));
  resolveRunbook(config, configPath, problems);

  if (problems.length > 0) {
    return { configPath, configExists, problems };
  }

  const bench = effective.bench ?? configFile.bench;
  return {
    configPath,
    configExists,
    problems: [],
    config,
    ...(bench !== undefined ? { bench } : undefined),
  };
}
