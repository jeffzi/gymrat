import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { measure } from "../src/measure.js";
import type { MeasureOptions } from "../src/measure.js";
import type { MeasurementResult } from "../src/report/types.js";
import { CommandError } from "../src/sampling.js";
import { captureRejection } from "./fixtures/errors.js";
import {
  createInPlaceTargetDir,
  createScratchRepo,
  listWorktreeDirs,
  removeStrandedWorktrees,
  toShellPath,
} from "./fixtures/scratch-repo.js";
import type { ScratchRepo } from "./fixtures/scratch-repo.js";

/** Generous timeout for runs that create real worktrees and spawn real bench processes. */
const LONG_RUN_TIMEOUT_MS = 60_000;

/**
 * The MeasureOptions fields shared by every call site here, with only `target`
 * required — everything else follows the standard bench/adapter/samples/timeout
 * defaults unless overridden.
 */
function measureOptions(
  overrides: Partial<MeasureOptions> & Pick<MeasureOptions, "target">,
): MeasureOptions {
  return {
    bench: "sh bench.sh",
    adapter: "metric-lines",
    samples: 2,
    timeoutSeconds: 10,
    ...overrides,
  };
}

/** Write `files` into the repo and commit them, so a ref target's worktree carries them. */
function commitFiles(repo: ScratchRepo, files: Record<string, string>): void {
  for (const [name, content] of Object.entries(files)) {
    fs.writeFileSync(path.join(repo.dir, name), content);
  }
  execFileSync("git", ["add", ...Object.keys(files)], { cwd: repo.dir, stdio: "pipe" });
  execFileSync("git", ["commit", "-m", "bench scripts"], { cwd: repo.dir, stdio: "pipe" });
}

describe("measure – integration", () => {
  describe("when the target is a git ref", () => {
    let repo: ScratchRepo;
    let savedCwd: string;
    let result: MeasurementResult;
    let prepareCwd: string;
    let benchCwd: string;

    beforeAll(async () => {
      savedCwd = process.cwd();
      repo = createScratchRepo();
      process.chdir(repo.dir);

      // The scripts run inside a throwaway worktree, so they record where they
      // ran into files in the main repo dir that outlive the worktree's removal.
      const prepareCwdFile = path.join(repo.dir, "prepare-cwd.txt");
      const benchCwdFile = path.join(repo.dir, "bench-cwd.txt");

      commitFiles(repo, {
        "prepare.sh": `#!/bin/sh\npwd > "${toShellPath(prepareCwdFile)}"\n`,
        "bench.sh": `#!/bin/sh\npwd > "${toShellPath(benchCwdFile)}"\necho "METRIC latency=42"\n`,
      });

      result = await measure(
        measureOptions({ target: { target: "HEAD" }, prepare: "sh prepare.sh" }),
      );
      prepareCwd = fs.readFileSync(prepareCwdFile, "utf-8").trim();
      benchCwd = fs.readFileSync(benchCwdFile, "utf-8").trim();
    }, LONG_RUN_TIMEOUT_MS);

    afterAll(() => {
      process.chdir(savedCwd);
      removeStrandedWorktrees(repo);
      repo.cleanup();
    });

    it("parses the bench's own stdout through the real metric-lines adapter", () => {
      expect(result.metrics["latency"]?.median).toBe(42);
    });

    it("labels the result with the ref name when no label is given", () => {
      expect(result.label).toBe("HEAD");
    });

    it("carries one raw sample map per round, in the order they were collected", () => {
      // The aggregates alone cannot be recorded as history: a baseline record
      // keeps every round's own values.
      // Adapter sample maps are null-prototype (src/metric-record.ts invariant).
      expect(result.rounds).toEqual([{ latency: 42 }, { latency: 42 }]);
    });

    it("removes the throwaway worktree it created", () => {
      expect.soft(result.worktreesRemoved).toBe(1);
      expect.soft(result.worktreesLeftBehind).toStrictEqual([]);
      expect.soft(listWorktreeDirs(repo.dir, { includeMain: false })).toStrictEqual([]);
      // Only the directory itself says the worktree is gone: git clears its own
      // registry entry whether or not the files survived.
      expect(fs.existsSync(benchCwd)).toBe(false);
    });

    it("runs prepare in the throwaway worktree the bench runs in", () => {
      expect.soft(prepareCwd).toBe(benchCwd);
      expect(path.basename(prepareCwd)).toMatch(/^gymrat-wt-/);
    });
  });

  describe("when the target is a plain directory", () => {
    it(
      "benches it in place, creating no worktree",
      async () => {
        const savedCwd = process.cwd();
        const repo = createScratchRepo();

        try {
          process.chdir(repo.dir);
          createInPlaceTargetDir(repo, "in-place", '#!/bin/sh\necho "METRIC latency=7"\n');

          const result = await measure(measureOptions({ target: { target: "in-place" } }));

          // The metric proves the bench really ran, so the worktree assertions
          // below are about a run that happened rather than one that did not.
          expect.soft(result.metrics["latency"]?.median).toBe(7);
          expect.soft(result.worktreesRemoved).toBe(0);
          expect(listWorktreeDirs(repo.dir, { includeMain: false })).toStrictEqual([]);
        } finally {
          process.chdir(savedCwd);
          removeStrandedWorktrees(repo);
          repo.cleanup();
        }
      },
      LONG_RUN_TIMEOUT_MS,
    );
  });

  describe("when the bench fails on a ref target", () => {
    let repo: ScratchRepo;
    let savedCwd: string;
    let failure: Error;

    beforeAll(async () => {
      savedCwd = process.cwd();
      repo = createScratchRepo();
      process.chdir(repo.dir);

      commitFiles(repo, { "bench.sh": '#!/bin/sh\necho "bench boom" >&2\nexit 1\n' });

      failure = await captureRejection(measure(measureOptions({ target: { target: "HEAD" } })));
    }, LONG_RUN_TIMEOUT_MS);

    afterAll(() => {
      process.chdir(savedCwd);
      removeStrandedWorktrees(repo);
      repo.cleanup();
    });

    it("rejects with a CommandError carrying the bench's own failure output", () => {
      expect.soft(failure).toBeInstanceOf(CommandError);
      expect.soft(failure.message).toContain("bench boom");
      expect(failure.message).toContain("exit code: 1");
    });

    it("removes the worktree it created before rejecting", () => {
      expect.soft(failure.message).not.toContain("left behind");
      expect(listWorktreeDirs(repo.dir, { includeMain: false })).toStrictEqual([]);
    });
  });
});
