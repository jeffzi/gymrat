import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { BenchlessConfig, CliFlags, ResolvedConfig } from "../src/config.js";
import { resolveBenchlessConfig, resolveConfig } from "../src/config.js";
import { GymratError } from "../src/errors.js";
import { freshRoot } from "./fixtures/scratch-repo.js";

function writeRawConfigFile(content: string): { dir: string; configPath: string } {
  const dir = freshRoot("gymrat-");
  const configPath = path.join(dir, "gymrat.json");
  fs.writeFileSync(configPath, content);
  return { dir, configPath };
}

function createConfigFile(content: Record<string, unknown>): { dir: string; configPath: string } {
  return writeRawConfigFile(JSON.stringify(content));
}

/** Registers an `afterEach` that restores the process cwd captured at call time. */
function restoreCwdAfterEach(): void {
  const originalCwd = process.cwd();
  afterEach(() => {
    process.chdir(originalCwd);
  });
}

describe("resolveConfig, GYMRAT_* environment variables", () => {
  let tmpdir: string;
  restoreCwdAfterEach();
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  describe("when a GYMRAT_* env var is set and its corresponding flag is absent", () => {
    it.each([
      {
        envVar: "GYMRAT_BENCH",
        envValue: "env-bench",
        flags: {},
        field: "bench",
        expected: "env-bench",
      },
      {
        envVar: "GYMRAT_PREPARE",
        envValue: "env-prepare",
        flags: { bench: "b" },
        field: "prepare",
        expected: "env-prepare",
      },
      {
        envVar: "GYMRAT_ADAPTER",
        envValue: "env-adapter",
        flags: { bench: "b" },
        field: "adapter",
        expected: "env-adapter",
      },
      {
        envVar: "GYMRAT_SAMPLES",
        envValue: "42",
        flags: { bench: "b" },
        field: "samples",
        expected: 42,
      },
      {
        envVar: "GYMRAT_TIMEOUT",
        envValue: "900",
        flags: { bench: "b" },
        field: "timeoutSeconds",
        expected: 900,
      },
    ])("uses $envVar as $field", ({ envVar, envValue, flags, field, expected }) => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      vi.stubEnv(envVar, envValue);

      const result = resolveConfig(flags);

      expect(result).toHaveProperty(field, expected);
    });
  });

  describe("when a GYMRAT_* env var is set and the config file provides the same field", () => {
    it.each([
      {
        envVar: "GYMRAT_BENCH",
        envValue: "env-bench",
        config: { bench: "config-bench" },
        field: "bench",
        expected: "env-bench",
      },
      {
        envVar: "GYMRAT_PREPARE",
        envValue: "env-prepare",
        config: { bench: "b", prepare: "config-prepare" },
        field: "prepare",
        expected: "env-prepare",
      },
      {
        envVar: "GYMRAT_ADAPTER",
        envValue: "env-adapter",
        config: { bench: "b", adapter: "config-adapter" },
        field: "adapter",
        expected: "env-adapter",
      },
      {
        envVar: "GYMRAT_SAMPLES",
        envValue: "42",
        config: { bench: "b", samples: 20 },
        field: "samples",
        expected: 42,
      },
      {
        envVar: "GYMRAT_TIMEOUT",
        envValue: "900",
        config: { bench: "b", timeoutSeconds: 3600 },
        field: "timeoutSeconds",
        expected: 900,
      },
    ])(
      "uses $envVar over the config file for $field",
      ({ envVar, envValue, config, field, expected }) => {
        tmpdir = createConfigFile(config).dir;
        process.chdir(tmpdir);
        vi.stubEnv(envVar, envValue);

        const result = resolveConfig({});

        expect(result).toHaveProperty(field, expected);
      },
    );
  });

  describe("when a string GYMRAT_* env var holds an empty string", () => {
    it.each([
      { envVar: "GYMRAT_BENCH", flags: {} },
      { envVar: "GYMRAT_PREPARE", flags: { bench: "b" } },
      { envVar: "GYMRAT_ADAPTER", flags: { bench: "b" } },
    ])("throws a GymratError naming $envVar", ({ envVar, flags }) => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      vi.stubEnv(envVar, "");
      const act = (): ResolvedConfig => resolveConfig(flags);

      expect.soft(act).toThrow(GymratError);
      expect(act).toThrow(new RegExp(`${envVar}.*non-empty`));
    });
  });

  describe("when GYMRAT_SAMPLES holds an invalid value", () => {
    it.each([
      { description: "a non-numeric string", value: "abc" },
      { description: "a non-integer", value: "1.5" },
      { description: "zero", value: "0" },
      { description: "a negative number", value: "-1" },
      { description: "an empty string", value: "" },
    ])("throws a GymratError naming GYMRAT_SAMPLES when it is $description", ({ value }) => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      vi.stubEnv("GYMRAT_SAMPLES", value);
      const act = (): ResolvedConfig => resolveConfig({ bench: "my-bench" });

      expect.soft(act).toThrow(GymratError);
      expect(act).toThrow(/GYMRAT_SAMPLES.*positive integer/);
    });
  });

  describe("when GYMRAT_TIMEOUT holds an invalid value", () => {
    it.each([
      { description: "a non-numeric string", value: "abc" },
      { description: "a non-integer", value: "1.5" },
      { description: "zero", value: "0" },
      { description: "a negative number", value: "-1" },
      { description: "an empty string", value: "" },
    ])("throws a GymratError naming GYMRAT_TIMEOUT when it is $description", ({ value }) => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      vi.stubEnv("GYMRAT_TIMEOUT", value);
      const act = (): ResolvedConfig => resolveConfig({ bench: "my-bench" });

      expect.soft(act).toThrow(GymratError);
      expect(act).toThrow(/GYMRAT_TIMEOUT.*positive integer/);
    });
  });

  describe("when GYMRAT_TIMEOUT exceeds the millisecond timer cap", () => {
    it("throws a GymratError naming GYMRAT_TIMEOUT and the cap", () => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      vi.stubEnv("GYMRAT_TIMEOUT", "2147484");
      const act = (): ResolvedConfig => resolveConfig({ bench: "my-bench" });

      expect.soft(act).toThrow(GymratError);
      expect.soft(act).toThrow(/GYMRAT_TIMEOUT/);
      expect(act).toThrow(/no greater than 2147483/);
    });
  });

  describe("when GYMRAT_CONFIG names an existing config file and --config is absent", () => {
    it("loads config from the env-specified path", () => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      const envConfigPath = path.join(tmpdir, "env-config.json");
      fs.writeFileSync(envConfigPath, JSON.stringify({ bench: "env-config-bench" }));
      vi.stubEnv("GYMRAT_CONFIG", envConfigPath);

      const result = resolveConfig({});

      expect(result.bench).toBe("env-config-bench");
    });
  });

  describe("when GYMRAT_CONFIG names a file that does not exist", () => {
    it("throws an error naming the missing path", () => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      const missingPath = path.join(tmpdir, "typo.json");
      vi.stubEnv("GYMRAT_CONFIG", missingPath);

      expect(() => resolveConfig({ bench: "my-bench" })).toThrow(missingPath);
    });
  });

  describe("when GYMRAT_CONFIG is set", () => {
    it("bypasses the implicit gymrat.json in the working directory", () => {
      tmpdir = createConfigFile({ bench: "implicit-bench", adapter: "implicit-adapter" }).dir;
      process.chdir(tmpdir);
      const envConfigPath = path.join(tmpdir, "alt-config.json");
      fs.writeFileSync(envConfigPath, JSON.stringify({ bench: "alt-bench" }));
      vi.stubEnv("GYMRAT_CONFIG", envConfigPath);

      const result = resolveConfig({});

      expect(result.bench).toBe("alt-bench");
      // adapter falls back to default, not the implicit gymrat.json value
      expect(result.adapter).toBe("metric-lines");
    });
  });

  describe("when GYMRAT_CONFIG holds an empty string", () => {
    it("throws a GymratError naming GYMRAT_CONFIG", () => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      vi.stubEnv("GYMRAT_CONFIG", "");
      const act = (): ResolvedConfig => resolveConfig({ bench: "my-bench" });

      expect.soft(act).toThrow(GymratError);
      expect(act).toThrow(/GYMRAT_CONFIG.*non-empty/);
    });
  });
});

describe("resolveBenchlessConfig", () => {
  let tmpdir: string;
  restoreCwdAfterEach();

  describe("when the config flag holds an empty string", () => {
    it("throws naming --config and the non-empty requirement", () => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      const act = (): BenchlessConfig => resolveBenchlessConfig({ config: "" });

      expect.soft(act).toThrow(GymratError);
      expect(act).toThrow(/--config.*non-empty/);
    });
  });
});

/**
 * Both resolvers settle the implicit `gymrat.json` through the same lookup, so a
 * base directory has to reach it through either one.
 */
const CONFIG_RESOLVERS = [
  {
    name: "resolveConfig",
    resolve: (flags: CliFlags, baseDir?: string): BenchlessConfig => resolveConfig(flags, baseDir),
  },
  {
    name: "resolveBenchlessConfig",
    resolve: (flags: CliFlags, baseDir?: string): BenchlessConfig =>
      resolveBenchlessConfig(flags, baseDir),
  },
];

describe.each(CONFIG_RESOLVERS)("$name, given a base directory", ({ resolve }) => {
  restoreCwdAfterEach();

  /**
   * A directory holding a `gymrat.json` of `baseConfig`, with a nested directory
   * holding one of `nestedConfig` — the two the lookup has to choose between.
   */
  function createNestedConfigDirs(
    baseConfig: Record<string, unknown>,
    nestedConfig: Record<string, unknown>,
  ): { baseDir: string; nestedDir: string } {
    const baseDir = freshRoot("gymrat-");
    const nestedDir = path.join(baseDir, "packages", "core");
    fs.mkdirSync(nestedDir, { recursive: true });
    fs.writeFileSync(path.join(baseDir, "gymrat.json"), JSON.stringify(baseConfig));
    fs.writeFileSync(path.join(nestedDir, "gymrat.json"), JSON.stringify(nestedConfig));
    return { baseDir, nestedDir };
  }

  describe("when the base directory and the working directory each hold a gymrat.json", () => {
    it("reads the base directory's", () => {
      const { baseDir, nestedDir } = createNestedConfigDirs(
        { bench: "a-bench", checks: "base-checks" },
        { bench: "a-bench", checks: "cwd-checks" },
      );
      process.chdir(nestedDir);

      const result = resolve({}, baseDir);

      expect(result.checks).toBe("base-checks");
    });
  });

  describe("when --config names a path relative to the working directory", () => {
    it("reads the named file, leaving the base directory's gymrat.json unread", () => {
      const { baseDir, nestedDir } = createNestedConfigDirs(
        { bench: "a-bench", checks: "base-checks" },
        { bench: "a-bench", checks: "cwd-checks" },
      );
      fs.writeFileSync(
        path.join(nestedDir, "custom.json"),
        JSON.stringify({ bench: "a-bench", checks: "named-checks" }),
      );
      process.chdir(nestedDir);

      const result = resolve({ config: "custom.json" }, baseDir);

      expect(result.checks).toBe("named-checks");
    });
  });
});

describe.each(CONFIG_RESOLVERS)("$name, runbook resolution", ({ resolve }) => {
  let tmpdir: string;
  restoreCwdAfterEach();

  describe("when the config file has no runbook key", () => {
    it("omits runbook from the resolved config", () => {
      tmpdir = createConfigFile({ bench: "a-bench" }).dir;
      process.chdir(tmpdir);

      const result = resolve({});

      expect(result).not.toHaveProperty("runbook");
    });
  });

  describe("when the config file names a runbook that exists as a file", () => {
    it("resolves the runbook path to an absolute path against the config directory", () => {
      tmpdir = createConfigFile({ bench: "a-bench", runbook: "RUNBOOK.md" }).dir;
      fs.writeFileSync(path.join(tmpdir, "RUNBOOK.md"), "# Steps\n");
      process.chdir(tmpdir);

      const result = resolve({});

      expect(result.runbook).toBe(path.join(tmpdir, "RUNBOOK.md"));
    });
  });

  describe("when the runbook path does not resolve to an existing file", () => {
    it("throws a GymratError naming the field and the path", () => {
      tmpdir = createConfigFile({ bench: "a-bench", runbook: "missing.md" }).dir;
      process.chdir(tmpdir);
      const act = (): BenchlessConfig => resolve({});

      expect.soft(act).toThrow(GymratError);
      expect.soft(act).toThrow(/runbook/);
      expect(act).toThrow(/missing\.md/);
    });
  });

  describe("when the runbook path points to a directory instead of a file", () => {
    it("throws a GymratError naming the field and the path", () => {
      tmpdir = createConfigFile({ bench: "a-bench", runbook: "docs" }).dir;
      fs.mkdirSync(path.join(tmpdir, "docs"));
      process.chdir(tmpdir);
      const act = (): BenchlessConfig => resolve({});

      expect.soft(act).toThrow(GymratError);
      expect.soft(act).toThrow(/runbook/);
      expect(act).toThrow(/docs/);
    });
  });
});

describe.each(CONFIG_RESOLVERS)("$name, implicit lookup in a git repository", ({ resolve }) => {
  restoreCwdAfterEach();

  describe("when no baseDir is passed and the cwd is inside a git repository", () => {
    it("finds gymrat.json at the repository root, not the cwd", () => {
      const repoRoot = freshRoot("gymrat-repo-");
      execFileSync("git", ["init"], { cwd: repoRoot, stdio: "ignore" });
      fs.writeFileSync(
        path.join(repoRoot, "gymrat.json"),
        JSON.stringify({ bench: "repo-bench", checks: "repo-checks" }),
      );
      const nestedDir = path.join(repoRoot, "packages", "core");
      fs.mkdirSync(nestedDir, { recursive: true });
      process.chdir(nestedDir);

      const result = resolve({ bench: "flag-bench" });

      expect(result.checks).toBe("repo-checks");
    });
  });
});
