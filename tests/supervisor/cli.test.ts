import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { BenchlessConfig } from "../../src/config.js";
import { GymratError } from "../../src/errors.js";
import type { Driver } from "../../src/supervisor/driver.js";
import type { SupervisionResult } from "../../src/supervisor/supervise.js";
import { MAX_TIMEOUT_SECONDS } from "../../src/timer-limits.js";
import {
  createRunnableProgram,
  exitCodeOf,
  mockProcessExit,
  stubWrite,
} from "../fixtures/cli-harness.js";
import { createScratchRepo, type ScratchRepo } from "../fixtures/scratch-repo.js";

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

vi.mock("../../src/config.js", () => ({
  resolveBenchlessConfig: vi.fn().mockReturnValue({
    adapter: "metric-lines",
    samples: 1,
    timeoutSeconds: 300,
    unstableNoisePct: 200,
    primary: "geomean",
  } satisfies BenchlessConfig),
}));

vi.mock("../../src/supervisor/kickoff.js", () => ({
  composeKickoff: vi.fn().mockReturnValue({
    systemPromptAppend: "system prompt",
    kickoff: "begin optimization",
  }),
}));

vi.mock("../../src/supervisor/claude.js", () => ({
  createClaudeDriver: vi.fn().mockReturnValue({
    start: vi.fn().mockReturnValue({
      interrupt: vi.fn().mockResolvedValue(undefined),
      outcome: Promise.resolve({ reason: "completed", costUsd: 0.05 }),
    }),
  } satisfies Driver),
}));

vi.mock("../../src/supervisor/supervise.js", () => ({
  supervise: vi.fn<() => Promise<SupervisionResult>>().mockResolvedValue({
    outcome: { reason: "completed", costUsd: 0.05 },
    endedBy: "session",
    durationMs: 60_000,
    costUsd: 0.05,
  }),
}));

vi.mock("../../src/session/workspace.js", async (importOriginal) => {
  const original = await importOriginal<typeof import("../../src/session/workspace.js")>();
  return { ...original, ensureGitExclude: vi.fn() };
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function superviseArgv(...args: string[]): string[] {
  return ["node", "cli.js", "supervise", ...args];
}

/** The text a stubbed stream spy was written, in call order. */
function textWrittenTo(spy: ReturnType<typeof vi.spyOn>): string {
  // oxlint-disable-next-line no-unsafe-member-access, no-unsafe-type-assertion -- vi.spyOn mock type is loosely typed
  const calls = spy.mock.calls as unknown[][];
  return calls.map((c) => String(c[0])).join("");
}

/**
 * Wires a fresh scratch repository as the CWD for every test in the enclosing
 * `describe` block, cleaning it up afterward. Returns a getter for the
 * current test's repo, since the repo itself is only created in `beforeEach`.
 */
function useScratchRepo(): () => ScratchRepo {
  let repo: ScratchRepo;
  beforeEach(() => {
    repo = createScratchRepo();
    process.chdir(repo.dir);
  });
  afterEach(() => {
    repo.cleanup();
  });
  return () => repo;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

afterEach(() => {
  vi.clearAllMocks();
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
});

describe("supervise command", () => {
  describe("flag parsing", () => {
    describe("when --max-minutes is missing", () => {
      it("rejects with a usage error", async () => {
        const program = createRunnableProgram({ exitOverride: "all", silent: true });
        mockProcessExit();

        const parsing = program.parseAsync(superviseArgv("my prompt"));

        await expect(parsing).rejects.toThrow(/--max-minutes/);
      });
    });

    describe("when --max-minutes receives an invalid value", () => {
      it.each([
        { value: "abc", why: "non-numeric" },
        { value: "0", why: "zero" },
        { value: "-5", why: "negative" },
        { value: "0x10", why: "hex notation" },
        { value: "1e-9", why: "scientific notation" },
      ])("rejects $value ($why) with a message naming the flag", async ({ value }) => {
        const program = createRunnableProgram({ exitOverride: "all", silent: true });
        mockProcessExit();

        const parsing = program.parseAsync(superviseArgv("my prompt", "--max-minutes", value));

        await expect(parsing).rejects.toThrow(/--max-minutes/);
      });
    });

    describe("when --max-minutes exceeds the timer ceiling", () => {
      it("rejects with a message naming the flag", async () => {
        const overLimit = String(Math.floor(MAX_TIMEOUT_SECONDS / 60) + 1);
        const program = createRunnableProgram({ exitOverride: "all", silent: true });
        mockProcessExit();

        const parsing = program.parseAsync(superviseArgv("my prompt", "--max-minutes", overLimit));

        await expect(parsing).rejects.toThrow(/--max-minutes/);
      });
    });

    describe("when --max-usd receives an invalid value", () => {
      it.each([
        { value: "abc", why: "non-numeric" },
        { value: "0", why: "zero" },
        { value: "-5", why: "negative" },
        { value: "0x10", why: "hex notation" },
        { value: "1e-9", why: "scientific notation" },
      ])("rejects $value ($why) with a message naming the flag", async ({ value }) => {
        const program = createRunnableProgram({ exitOverride: "all", silent: true });
        mockProcessExit();

        const parsing = program.parseAsync(
          superviseArgv("my prompt", "--max-minutes", "10", "--max-usd", value),
        );

        await expect(parsing).rejects.toThrow(/--max-usd/);
      });
    });

    describe("when valid flags are provided", () => {
      useScratchRepo();

      it("passes max-minutes through to supervise", async () => {
        const program = createRunnableProgram();
        stubWrite(process.stdout);
        stubWrite(process.stderr);
        const { supervise } = await import("../../src/supervisor/supervise.js");

        await program.parseAsync(superviseArgv("my prompt", "--max-minutes", "30"));

        expect(vi.mocked(supervise)).toHaveBeenCalledWith(
          expect.objectContaining({ maxMinutes: 30 }),
        );
      });

      it("passes max-usd through to supervise when provided", async () => {
        const program = createRunnableProgram();
        stubWrite(process.stdout);
        stubWrite(process.stderr);
        const { supervise } = await import("../../src/supervisor/supervise.js");

        await program.parseAsync(
          superviseArgv("my prompt", "--max-minutes", "30", "--max-usd", "5.50"),
        );

        expect(vi.mocked(supervise)).toHaveBeenCalledWith(expect.objectContaining({ maxUsd: 5.5 }));
      });
    });
  });

  describe("default log path", () => {
    useScratchRepo();

    it("defaults to .gymrat/supervisor-<timestamp>.jsonl under the repository root", async () => {
      const program = createRunnableProgram();
      const stderrSpy = stubWrite(process.stderr);
      stubWrite(process.stdout);

      await program.parseAsync(superviseArgv("optimize it", "--max-minutes", "10"));

      const output = textWrittenTo(stderrSpy);
      expect(output).toMatch(/\.gymrat\/supervisor-\d+\.jsonl/);
    });

    it("uses the path from --log when provided", async () => {
      const program = createRunnableProgram();
      const stderrSpy = stubWrite(process.stderr);
      stubWrite(process.stdout);

      await program.parseAsync(
        superviseArgv("optimize it", "--max-minutes", "10", "--log", "/tmp/custom.jsonl"),
      );

      const output = textWrittenTo(stderrSpy);
      expect(output).toContain("/tmp/custom.jsonl");
    });
  });

  describe("exit codes", () => {
    useScratchRepo();

    it("exits 0 when the session completed on its own", async () => {
      const program = createRunnableProgram();
      stubWrite(process.stdout);
      stubWrite(process.stderr);
      // oxlint-disable-next-line typescript/no-unsafe-assignment -- vi.spyOn return type is inherently `any`
      const exitSpy = mockProcessExit();

      await program.parseAsync(superviseArgv("optimize it", "--max-minutes", "10"));

      expect(exitSpy).not.toHaveBeenCalled();
    });

    it("exits 1 when a cap ended the session", async () => {
      const { supervise: superviseFn } = await import("../../src/supervisor/supervise.js");
      vi.mocked(superviseFn).mockResolvedValueOnce({
        outcome: { reason: "completed", costUsd: 1.0 },
        endedBy: "wall-clock",
        durationMs: 600_000,
        costUsd: 1.0,
      });
      const program = createRunnableProgram({ exitOverride: "all" });
      stubWrite(process.stdout);
      stubWrite(process.stderr);
      mockProcessExit();

      const error = await program
        .parseAsync(superviseArgv("optimize it", "--max-minutes", "10"))
        .then(
          () => undefined,
          (e: unknown) => e,
        );

      expect(exitCodeOf(error)).toBe(1);
      expect(vi.mocked(superviseFn)).toHaveBeenCalled();
    });

    it("exits 2 on operational errors", async () => {
      const { supervise } = await import("../../src/supervisor/supervise.js");
      vi.mocked(supervise).mockRejectedValueOnce(new GymratError("config broken"));
      const program = createRunnableProgram({ exitOverride: "all" });
      stubWrite(process.stdout);
      const stderrSpy = stubWrite(process.stderr);
      mockProcessExit();

      const parsing = program.parseAsync(superviseArgv("optimize it", "--max-minutes", "10"));

      await expect(parsing).rejects.toHaveProperty("exitCode", 2);
      expect(textWrittenTo(stderrSpy)).toContain("config broken");
    });

    it("exits 2 and surfaces the message when the session outcome is error", async () => {
      const { supervise: superviseFn } = await import("../../src/supervisor/supervise.js");
      vi.mocked(superviseFn).mockResolvedValueOnce({
        outcome: { reason: "error", costUsd: 0.03, message: "SDK connection lost" },
        endedBy: "session",
        durationMs: 5_000,
        costUsd: 0.03,
      });
      const program = createRunnableProgram({ exitOverride: "all" });
      stubWrite(process.stdout);
      const stderrSpy = stubWrite(process.stderr);
      mockProcessExit();

      const parsing = program.parseAsync(superviseArgv("optimize it", "--max-minutes", "10"));

      await expect(parsing).rejects.toHaveProperty("exitCode", 2);
      expect(textWrittenTo(stderrSpy)).toContain("SDK connection lost");
    });

    it.each([
      { label: "undefined (omitted)", outcome: { reason: "error" as const, costUsd: 0.01 } },
      { label: "empty string", outcome: { reason: "error" as const, costUsd: 0.01, message: "" } },
    ])(
      "exits 2 without writing an error to stderr when the error outcome message is $label",
      async ({ outcome }) => {
        const { supervise: superviseFn } = await import("../../src/supervisor/supervise.js");
        vi.mocked(superviseFn).mockResolvedValueOnce({
          outcome,
          endedBy: "session",
          durationMs: 5_000,
          costUsd: outcome.costUsd,
        });
        const program = createRunnableProgram({ exitOverride: "all" });
        stubWrite(process.stdout);
        const stderrSpy = stubWrite(process.stderr);
        mockProcessExit();

        const parsing = program.parseAsync(superviseArgv("optimize it", "--max-minutes", "10"));

        await expect(parsing).rejects.toHaveProperty("exitCode", 2);
        const stderr = textWrittenTo(stderrSpy);
        expect(stderr).not.toMatch(/\berror\b/i);
      },
    );
  });

  describe("closing summary", () => {
    useScratchRepo();

    it("reports how the run ended, duration, final cost estimate, and log path", async () => {
      const program = createRunnableProgram();
      const stderrSpy = stubWrite(process.stderr);
      const stdoutSpy = stubWrite(process.stdout);

      await program.parseAsync(superviseArgv("optimize it", "--max-minutes", "10"));

      const stdoutOutput = textWrittenTo(stdoutSpy);
      expect.soft(stdoutOutput).toMatch(/session/i);
      expect.soft(stdoutOutput).toMatch(/duration|time/i);
      expect.soft(stdoutOutput).toMatch(/\$?\d+\.\d+|cost/i);
      expect(textWrittenTo(stderrSpy)).toMatch(/\.jsonl/);
    });
  });

  describe("dirty-tree guard", () => {
    const repo = useScratchRepo();

    it("exits 2 when the tree is dirty and --allow-dirty is not set", async () => {
      writeFileSync(`${repo().dir}/uncommitted.txt`, "dirty");
      const program = createRunnableProgram({ exitOverride: "all" });
      stubWrite(process.stdout);
      const stderrSpy = stubWrite(process.stderr);
      mockProcessExit();

      const parsing = program.parseAsync(superviseArgv("optimize it", "--max-minutes", "10"));

      await expect(parsing).rejects.toHaveProperty("exitCode", 2);
      const output = textWrittenTo(stderrSpy);
      expect.soft(output).toMatch(/dirty|uncommitted|untracked/i);
      expect.soft(output).toMatch(/commit|stash/i);
      expect(output).toMatch(/--allow-dirty/);
    });

    it("proceeds with a warning when --allow-dirty is set and tree is dirty", async () => {
      writeFileSync(`${repo().dir}/uncommitted.txt`, "dirty");
      const program = createRunnableProgram();
      stubWrite(process.stdout);
      const stderrSpy = stubWrite(process.stderr);

      await program.parseAsync(
        superviseArgv("optimize it", "--max-minutes", "10", "--allow-dirty"),
      );

      const output = textWrittenTo(stderrSpy);
      expect(output).toMatch(/dirty|uncommitted|untracked/i);
    });

    it("launches without warning when the tree is clean", async () => {
      const program = createRunnableProgram();
      stubWrite(process.stdout);
      const stderrSpy = stubWrite(process.stderr);

      await program.parseAsync(superviseArgv("optimize it", "--max-minutes", "10"));

      const output = textWrittenTo(stderrSpy);
      expect(output).not.toMatch(/dirty|uncommitted|untracked/i);
    });
  });

  describe("driver wiring", () => {
    useScratchRepo();

    it("creates a Claude driver and passes it to supervise", async () => {
      const program = createRunnableProgram();
      stubWrite(process.stdout);
      stubWrite(process.stderr);
      const { createClaudeDriver } = await import("../../src/supervisor/claude.js");
      const { supervise } = await import("../../src/supervisor/supervise.js");

      await program.parseAsync(superviseArgv("optimize it", "--max-minutes", "10"));

      expect(vi.mocked(createClaudeDriver)).toHaveBeenCalled();
      expect(vi.mocked(supervise)).toHaveBeenCalledWith(
        // oxlint-disable-next-line no-unsafe-assignment -- expect.anything() returns any
        expect.objectContaining({ driver: expect.anything() }),
      );
    });

    it("composes kickoff from caller prompt", async () => {
      const program = createRunnableProgram();
      stubWrite(process.stdout);
      stubWrite(process.stderr);
      const { composeKickoff } = await import("../../src/supervisor/kickoff.js");

      await program.parseAsync(superviseArgv("optimize the decoder", "--max-minutes", "10"));

      expect(vi.mocked(composeKickoff)).toHaveBeenCalledWith(
        expect.anything(),
        expect.any(String),
        "optimize the decoder",
      );
    });

    it("calls composeKickoff without prompt when none is provided", async () => {
      const program = createRunnableProgram();
      stubWrite(process.stdout);
      stubWrite(process.stderr);
      const { composeKickoff } = await import("../../src/supervisor/kickoff.js");

      await program.parseAsync(superviseArgv("--max-minutes", "10"));

      expect(vi.mocked(composeKickoff)).toHaveBeenCalledWith(
        expect.anything(),
        expect.any(String),
        undefined,
      );
    });
  });

  describe("dirty-file count with untracked directories", () => {
    const repo = useScratchRepo();

    it("counts individual files inside an untracked directory, not the directory itself", async () => {
      const untrackedDir = path.join(repo().dir, "new-dir");
      mkdirSync(untrackedDir);
      writeFileSync(path.join(untrackedDir, "a.txt"), "a");
      writeFileSync(path.join(untrackedDir, "b.txt"), "b");
      writeFileSync(path.join(untrackedDir, "c.txt"), "c");
      const program = createRunnableProgram({ exitOverride: "all" });
      stubWrite(process.stdout);
      const stderrSpy = stubWrite(process.stderr);
      mockProcessExit();

      const parsing = program.parseAsync(superviseArgv("optimize it", "--max-minutes", "10"));

      await expect(parsing).rejects.toHaveProperty("exitCode", 2);
      const output = textWrittenTo(stderrSpy);
      expect(output).toMatch(/3/);
    });
  });

  describe("closing summary output stream", () => {
    useScratchRepo();

    it("prints the closing summary on stdout", async () => {
      const program = createRunnableProgram();
      const stdoutSpy = stubWrite(process.stdout);
      stubWrite(process.stderr);

      await program.parseAsync(superviseArgv("optimize it", "--max-minutes", "10"));

      const stdoutOutput = textWrittenTo(stdoutSpy);
      expect.soft(stdoutOutput).toMatch(/session/i);
      expect.soft(stdoutOutput).toMatch(/duration|time/i);
      expect(stdoutOutput).toMatch(/\.jsonl/);
    });
  });

  describe("git-exclude for default log path", () => {
    const repo = useScratchRepo();

    it("calls ensureGitExclude before writing the default .gymrat/ log", async () => {
      const program = createRunnableProgram();
      stubWrite(process.stdout);
      stubWrite(process.stderr);
      const { ensureGitExclude } = await import("../../src/session/workspace.js");

      await program.parseAsync(superviseArgv("optimize it", "--max-minutes", "10"));

      expect(vi.mocked(ensureGitExclude)).toHaveBeenCalledWith(repo().dir);
    });

    it("does not call ensureGitExclude when --log provides a custom path", async () => {
      const program = createRunnableProgram();
      stubWrite(process.stdout);
      stubWrite(process.stderr);
      const { ensureGitExclude } = await import("../../src/session/workspace.js");

      await program.parseAsync(
        superviseArgv("optimize it", "--max-minutes", "10", "--log", "/tmp/custom.jsonl"),
      );

      expect(vi.mocked(ensureGitExclude)).not.toHaveBeenCalled();
    });
  });

  describe("supervise lock", () => {
    const repo = useScratchRepo();

    it("exits 2 when another live process holds the supervise lock", async () => {
      const { superviseLockfilePath } = (await import("../../src/session/paths.js")) as {
        superviseLockfilePath?: (root: string) => string;
      };
      expect(superviseLockfilePath).toBeDefined();
      const lockPath = superviseLockfilePath!(repo().dir);
      mkdirSync(path.dirname(lockPath), { recursive: true });
      writeFileSync(
        lockPath,
        JSON.stringify({ pid: process.pid, command: "supervise", at: "2026-01-01T00:00:00.000Z" }),
      );

      const program = createRunnableProgram({ exitOverride: "all" });
      stubWrite(process.stdout);
      const stderrSpy = stubWrite(process.stderr);
      mockProcessExit();

      const parsing = program.parseAsync(superviseArgv("optimize it", "--max-minutes", "10"));

      await expect(parsing).rejects.toHaveProperty("exitCode", 2);
      const output = textWrittenTo(stderrSpy);
      expect(output).toMatch(/another gymrat/i);
    });
  });

  describe("session outcome in summary", () => {
    useScratchRepo();

    it("names 'interrupted' in the summary when the session was interrupted", async () => {
      const { supervise: superviseFn } = await import("../../src/supervisor/supervise.js");
      vi.mocked(superviseFn).mockResolvedValueOnce({
        outcome: { reason: "interrupted", costUsd: 0.02 },
        endedBy: "session",
        durationMs: 30_000,
        costUsd: 0.02,
      });
      const program = createRunnableProgram();
      const stdoutSpy = stubWrite(process.stdout);
      const stderrSpy = stubWrite(process.stderr);

      await program.parseAsync(superviseArgv("optimize it", "--max-minutes", "10"));

      const stdoutOutput = textWrittenTo(stdoutSpy);
      const stderrOutput = textWrittenTo(stderrSpy);
      const allOutput = stdoutOutput + stderrOutput;
      expect(allOutput).toMatch(/interrupted/i);
    });
  });
});
