import { writeFileSync } from "node:fs";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { BenchlessConfig } from "../../src/config.js";
import { GymratError } from "../../src/errors.js";
import type { Driver } from "../../src/supervisor/driver.js";
import type { SupervisionResult } from "../../src/supervisor/supervise.js";
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
      inject: vi.fn(),
      interrupt: vi.fn().mockResolvedValue(undefined),
      usage: vi.fn().mockReturnValue({ costUsd: 0.05 }),
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function superviseArgv(...args: string[]): string[] {
  return ["node", "cli.js", "supervise", ...args];
}

function stderrText(spy: ReturnType<typeof vi.spyOn>): string {
  // oxlint-disable-next-line no-unsafe-member-access, no-unsafe-type-assertion -- vi.spyOn mock type is loosely typed
  const calls = spy.mock.calls as unknown[][];
  return calls.map((c) => String(c[0])).join("");
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
      ])("rejects $value ($why) with a message naming the flag", async ({ value }) => {
        const program = createRunnableProgram({ exitOverride: "all", silent: true });
        mockProcessExit();

        const parsing = program.parseAsync(superviseArgv("my prompt", "--max-minutes", value));

        await expect(parsing).rejects.toThrow(/--max-minutes/);
      });
    });

    describe("when --max-usd receives an invalid value", () => {
      it.each([
        { value: "abc", why: "non-numeric" },
        { value: "0", why: "zero" },
        { value: "-5", why: "negative" },
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
      let repo: ScratchRepo;

      beforeEach(() => {
        repo = createScratchRepo();
        process.chdir(repo.dir);
      });

      afterEach(() => {
        repo.cleanup();
      });

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
    let repo: ScratchRepo;

    beforeEach(() => {
      repo = createScratchRepo();
      process.chdir(repo.dir);
    });

    afterEach(() => {
      repo.cleanup();
    });

    it("defaults to .gymrat/supervisor-<timestamp>.jsonl under the repository root", async () => {
      const program = createRunnableProgram();
      const stderrSpy = stubWrite(process.stderr);
      stubWrite(process.stdout);

      await program.parseAsync(superviseArgv("optimize it", "--max-minutes", "10"));

      const output = stderrText(stderrSpy);
      expect(output).toMatch(/\.gymrat\/supervisor-\d+\.jsonl/);
    });

    it("uses the path from --log when provided", async () => {
      const program = createRunnableProgram();
      const stderrSpy = stubWrite(process.stderr);
      stubWrite(process.stdout);

      await program.parseAsync(
        superviseArgv("optimize it", "--max-minutes", "10", "--log", "/tmp/custom.jsonl"),
      );

      const output = stderrText(stderrSpy);
      expect(output).toContain("/tmp/custom.jsonl");
    });
  });

  describe("exit codes", () => {
    let repo: ScratchRepo;

    beforeEach(() => {
      repo = createScratchRepo();
      process.chdir(repo.dir);
    });

    afterEach(() => {
      repo.cleanup();
    });

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
      expect(stderrText(stderrSpy)).toContain("config broken");
    });
  });

  describe("closing summary", () => {
    let repo: ScratchRepo;

    beforeEach(() => {
      repo = createScratchRepo();
      process.chdir(repo.dir);
    });

    afterEach(() => {
      repo.cleanup();
    });

    it("reports how the run ended, duration, final cost estimate, and log path", async () => {
      const program = createRunnableProgram();
      const stderrSpy = stubWrite(process.stderr);
      stubWrite(process.stdout);

      await program.parseAsync(superviseArgv("optimize it", "--max-minutes", "10"));

      const output = stderrText(stderrSpy);
      expect.soft(output).toMatch(/session/i);
      expect.soft(output).toMatch(/duration|time/i);
      expect.soft(output).toMatch(/\$?\d+\.\d+|cost/i);
      expect(output).toMatch(/\.jsonl/);
    });
  });

  describe("dirty-tree guard", () => {
    let repo: ScratchRepo;

    beforeEach(() => {
      repo = createScratchRepo();
      process.chdir(repo.dir);
    });

    afterEach(() => {
      repo.cleanup();
    });

    it("exits 2 when the tree is dirty and --allow-dirty is not set", async () => {
      writeFileSync(`${repo.dir}/uncommitted.txt`, "dirty");
      const program = createRunnableProgram({ exitOverride: "all" });
      stubWrite(process.stdout);
      const stderrSpy = stubWrite(process.stderr);
      mockProcessExit();

      const parsing = program.parseAsync(superviseArgv("optimize it", "--max-minutes", "10"));

      await expect(parsing).rejects.toHaveProperty("exitCode", 2);
      const output = stderrText(stderrSpy);
      expect.soft(output).toMatch(/dirty|uncommitted|untracked/i);
      expect.soft(output).toMatch(/commit|stash/i);
      expect(output).toMatch(/--allow-dirty/);
    });

    it("proceeds with a warning when --allow-dirty is set and tree is dirty", async () => {
      writeFileSync(`${repo.dir}/uncommitted.txt`, "dirty");
      const program = createRunnableProgram();
      stubWrite(process.stdout);
      const stderrSpy = stubWrite(process.stderr);

      await program.parseAsync(
        superviseArgv("optimize it", "--max-minutes", "10", "--allow-dirty"),
      );

      const output = stderrText(stderrSpy);
      expect(output).toMatch(/dirty|uncommitted|untracked/i);
    });

    it("launches without warning when the tree is clean", async () => {
      const program = createRunnableProgram();
      stubWrite(process.stdout);
      const stderrSpy = stubWrite(process.stderr);

      await program.parseAsync(superviseArgv("optimize it", "--max-minutes", "10"));

      const output = stderrText(stderrSpy);
      expect(output).not.toMatch(/dirty|uncommitted|untracked/i);
    });
  });

  describe("driver wiring", () => {
    let repo: ScratchRepo;

    beforeEach(() => {
      repo = createScratchRepo();
      process.chdir(repo.dir);
    });

    afterEach(() => {
      repo.cleanup();
    });

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
});
