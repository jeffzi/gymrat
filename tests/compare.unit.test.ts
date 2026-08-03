import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, it, expect, vi } from "vitest";

import { compare, CommandError } from "../src/compare.js";
import type {
  CommandErrorContext,
  CompareOptions,
  ExitFailure,
  TimeoutFailure,
} from "../src/compare.js";
import { GymratError } from "../src/errors.js";
import type { InPlaceTarget, RefTarget } from "../src/targets.js";
import { REF_TARGET_HINT } from "./fixtures/constants.js";

// The bundled adapters raise AdapterError when a bench emits nothing parseable,
// so compare()'s own empty-metrics guard is only reachable behind an adapter
// that parses successfully into no metrics at all.
vi.mock("../src/adapters/index.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/adapters/index.js")>();
  return {
    ...actual,
    getAdapter: (name: string) => ({ ...actual.getAdapter(name), parse: () => ({}) }),
  };
});

function createRefTarget(ref = "abc123", resolvedSha = "def456"): RefTarget {
  return { kind: "ref", ref, resolvedSha };
}

function createInPlaceTarget(dir = "/projects/my-app"): InPlaceTarget {
  return { kind: "in-place", dir };
}

function createContext(overrides: Partial<CommandErrorContext> = {}): CommandErrorContext {
  return {
    phase: "bench",
    position: "old",
    label: "baseline",
    command: "npm run bench",
    target: createRefTarget(),
    dir: "/tmp/worktree-abc",
    ...overrides,
  };
}

function createExitFailure(overrides: Partial<ExitFailure> = {}): ExitFailure {
  return {
    exitCode: 1,
    stderr: "Error: benchmark crashed",
    stdout: "",
    ...overrides,
  };
}

function createTimeoutFailure(overrides: Partial<TimeoutFailure> = {}): TimeoutFailure {
  return {
    timeoutMs: 30000,
    stderr: "partial output before timeout",
    stdout: "",
    ...overrides,
  };
}

describe("CommandError", () => {
  describe("when exit code is non-zero", () => {
    it("includes phase, position, label, ref, worktree dir, command, exit code, and stderr for a ref target", () => {
      const ctx = createContext({ target: createRefTarget("main", "abc123") });
      const failure = createExitFailure({
        exitCode: 2,
        stderr: "segfault in runner",
      });

      const error = new CommandError(ctx, failure);

      expect.soft(error.message).toContain("bench");
      expect.soft(error.message).toContain("old");
      expect.soft(error.message).toContain("baseline");
      expect.soft(error.message).toContain("main");
      expect.soft(error.message).toContain("/tmp/worktree-abc");
      expect.soft(error.message).toContain("npm run bench");
      expect.soft(error.message).toContain("exit code: 2");
      expect(error.message).toContain("segfault in runner");
    });

    it("includes phase, position, label, directory, command, exit code, and stderr for an in-place target", () => {
      const ctx = createContext({
        position: "new",
        label: "candidate",
        target: createInPlaceTarget("/projects/my-app"),
        dir: "/projects/my-app",
      });
      const failure = createExitFailure({
        exitCode: 3,
        stderr: "module not found",
      });

      const error = new CommandError(ctx, failure);

      expect.soft(error.message).toContain("bench");
      expect.soft(error.message).toContain("new");
      expect.soft(error.message).toContain("candidate");
      expect.soft(error.message).toContain("/projects/my-app");
      expect.soft(error.message).toContain("npm run bench");
      expect.soft(error.message).toContain("exit code: 3");
      expect(error.message).toContain("module not found");
    });
  });

  describe("when command times out", () => {
    it("includes phase, position, label, ref, worktree dir, command, timeout duration, and stderr for a ref target", () => {
      const ctx = createContext({
        phase: "prepare",
        command: "npm install",
        target: createRefTarget("feature-branch", "sha789"),
        dir: "/tmp/worktree-feature",
      });
      const failure = createTimeoutFailure({
        timeoutMs: 60000,
        stderr: "still installing...",
      });

      const error = new CommandError(ctx, failure);

      expect.soft(error.message).toContain("prepare");
      expect.soft(error.message).toContain("old");
      expect.soft(error.message).toContain("baseline");
      expect.soft(error.message).toContain("feature-branch");
      expect.soft(error.message).toContain("/tmp/worktree-feature");
      expect.soft(error.message).toContain("npm install");
      expect.soft(error.message).toContain("60000");
      expect(error.message).toContain("still installing...");
      expect(error.message).not.toContain("exit code");
    });

    it("includes phase, position, label, directory, command, timeout duration, and stderr for an in-place target", () => {
      const ctx = createContext({
        phase: "prepare",
        position: "new",
        label: "candidate",
        command: "npm install",
        target: createInPlaceTarget("/projects/other"),
        dir: "/projects/other",
      });
      const failure = createTimeoutFailure({
        timeoutMs: 45000,
        stderr: "hanging on postinstall",
      });

      const error = new CommandError(ctx, failure);

      expect.soft(error.message).toContain("prepare");
      expect.soft(error.message).toContain("new");
      expect.soft(error.message).toContain("candidate");
      expect.soft(error.message).toContain("/projects/other");
      expect.soft(error.message).toContain("npm install");
      expect.soft(error.message).toContain("45000");
      expect(error.message).toContain("hanging on postinstall");
    });
  });

  describe("when phase is bench", () => {
    it("includes the 1-indexed sample number in the message", () => {
      const ctx = createContext({ sample: 3 });
      const failure = createExitFailure();

      const error = new CommandError(ctx, failure);

      expect(error.message).toContain("sample 3");
    });
  });

  describe("when target is a ref", () => {
    it("does not include the hint text in the message", () => {
      const ctx = createContext();
      const failure = createExitFailure();

      const error = new CommandError(ctx, failure);

      expect(error.message).not.toContain("hint:");
      expect(error.message).not.toContain("the worktree only contains files tracked at this ref");
    });
  });

  describe("when accessed via property getters", () => {
    it("exposes context and exit failure fields for programmatic access", () => {
      const target = createRefTarget("v1.0", "sha-abc");
      const ctx = createContext({ target, dir: "/tmp/worktree", sample: 2 });
      const failure = createExitFailure({ exitCode: 5 });

      const error = new CommandError(ctx, failure);

      expect.soft(error.phase).toBe("bench");
      expect.soft(error.position).toBe("old");
      expect.soft(error.label).toBe("baseline");
      expect.soft(error.command).toBe("npm run bench");
      expect.soft(error.target).toStrictEqual(target);
      expect.soft(error.dir).toBe("/tmp/worktree");
      expect.soft(error.sample).toBe(2);
      expect.soft(error.exitCode).toBe(5);
      expect.soft(error.timeoutMs).toBeUndefined();
      expect(error.hint).toBe(REF_TARGET_HINT);
    });

    it("exposes context and timeout failure fields for programmatic access", () => {
      const ctx = createContext({ target: createInPlaceTarget() });
      const failure = createTimeoutFailure({ timeoutMs: 10000 });

      const error = new CommandError(ctx, failure);

      expect.soft(error.exitCode).toBeUndefined();
      expect.soft(error.timeoutMs).toBe(10000);
      expect.soft(error.sample).toBeUndefined();
      expect(error.hint).toBeUndefined();
    });
  });

  describe("when stderr and stdout are both non-empty", () => {
    it("shows both under labeled separators", () => {
      const ctx = createContext();
      const failure = createExitFailure({
        stderr: "error output here",
        stdout: "normal output here",
      });

      const error = new CommandError(ctx, failure);

      expect.soft(error.message).toContain("--- stderr ---");
      expect.soft(error.message).toContain("error output here");
      expect.soft(error.message).toContain("--- stdout ---");
      expect(error.message).toContain("normal output here");
    });
  });

  describe("when only stdout is non-empty", () => {
    it("shows stdout without separators", () => {
      const ctx = createContext();
      const failure = createExitFailure({ stderr: "", stdout: "bench output here" });

      const error = new CommandError(ctx, failure);

      expect.soft(error.message).toContain("bench output here");
      expect.soft(error.message).not.toContain("--- stderr ---");
      expect(error.message).not.toContain("--- stdout ---");
    });
  });

  describe("when stderr and stdout are both empty", () => {
    it("does not include any output separator or body", () => {
      const ctx = createContext();
      const failure = createExitFailure({ stderr: "", stdout: "" });

      const error = new CommandError(ctx, failure);

      expect(error.message).not.toContain("--- stderr ---");
      expect(error.message).not.toContain("--- stdout ---");
    });
  });
});

/**
 * Two sibling directories holding no files, usable as in-place targets.
 *
 * In-place targets need not live in a git repository, so a bare temp tree is
 * enough to drive a whole comparison run.
 */
function createInPlaceDirs(): { root: string; cleanup: () => void } {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-unit-"));
  fs.mkdirSync(path.join(root, "old"));
  fs.mkdirSync(path.join(root, "new"));
  return {
    root,
    cleanup: () => {
      fs.rmSync(root, { recursive: true, force: true });
    },
  };
}

describe("compare", () => {
  describe("when the run produces no metrics at all", () => {
    it("rejects with a GymratError so the CLI reports it as a gymrat failure", async () => {
      const dirs = createInPlaceDirs();
      const savedCwd = process.cwd();

      try {
        process.chdir(dirs.root);

        const options: CompareOptions = {
          baseline: { target: "old" },
          candidates: [{ target: "new" }],
          bench: "echo benched",
          adapter: "metric-lines",
          samples: 1,
          timeoutSeconds: 10,
        };

        const error = await compare(options).catch((cause: unknown) => cause);

        expect.soft(error).toBeInstanceOf(GymratError);
        expect(error).toHaveProperty("message", "No metrics found in benchmark output");
      } finally {
        process.chdir(savedCwd);
        dirs.cleanup();
      }
    });
  });
});
