import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";

import { compare } from "../src/compare.js";
import type { CompareOptions } from "../src/compare.js";
import { createScratchRepo } from "./fixtures/scratch-repo.js";

interface BranchSetup {
  name: string;
  benchScript: string;
  prepareScript?: string;
}

/**
 * Create a git branch with bench and optional prepare scripts.
 */
function createBranch(
  repo: ReturnType<typeof createScratchRepo>,
  setup: BranchSetup,
  baseRef = "main",
) {
  execSync(`git checkout -b ${setup.name} ${baseRef}`, {
    cwd: repo.dir,
    stdio: "pipe",
  });

  fs.writeFileSync(path.join(repo.dir, "bench.sh"), setup.benchScript);

  if (setup.prepareScript) {
    fs.writeFileSync(path.join(repo.dir, "prepare.sh"), setup.prepareScript);
    execSync("chmod +x prepare.sh bench.sh && git add prepare.sh bench.sh", {
      cwd: repo.dir,
      stdio: "pipe",
    });
  } else {
    execSync("chmod +x bench.sh && git add bench.sh", {
      cwd: repo.dir,
      stdio: "pipe",
    });
  }

  execSync(`git commit -m '${setup.name}'`, {
    cwd: repo.dir,
    stdio: "pipe",
  });
}

/**
 * Verify all worktrees except the main repo dir have been cleaned up.
 */
function assertWorktreesCleanedUp(repo: ReturnType<typeof createScratchRepo>) {
  const worktreeList = execSync("git worktree list", {
    cwd: repo.dir,
    stdio: "pipe",
    encoding: "utf-8",
  });
  const worktreeLines = worktreeList
    .split("\n")
    .filter((line: string) => line.trim())
    .filter((line: string) => !line.includes(repo.dir));
  expect(worktreeLines.length).toBe(0);
}

describe("compare – integration", () => {
  let originalCwd: string;

  beforeEach(() => {
    originalCwd = process.cwd();
  });

  afterEach(() => {
    process.chdir(originalCwd);
    vi.restoreAllMocks();
  });

  describe("when prepare command is provided", () => {
    it("runs prepare before bench on both targets", async () => {
      const repo = createScratchRepo();

      try {
        process.chdir(repo.dir);

        createBranch(repo, {
          name: "old-prep",
          benchScript: '#!/bin/sh\necho "METRIC latency=100"',
          prepareScript: '#!/bin/sh\necho "prepared" > /tmp/prepared-old.txt',
        });

        createBranch(repo, {
          name: "new-prep",
          benchScript: '#!/bin/sh\necho "METRIC latency=90"',
          prepareScript: '#!/bin/sh\necho "prepared" > /tmp/prepared-new.txt',
        });

        const options: CompareOptions = {
          oldTarget: "old-prep",
          newTarget: "new-prep",
          bench: "./bench.sh",
          prepare: "./prepare.sh",
          adapter: "metric-lines",
          samples: 3,
          timeoutSeconds: 10,
        };

        const report = await compare(options);

        expect(report).toContain("latency");
      } finally {
        repo.cleanup();
      }
    });
  });

  describe("when comparing refs with metric-lines adapter and different metric sets", () => {
    it("renders union of metrics from both refs with one-sided rows", async () => {
      const repo = createScratchRepo();

      try {
        process.chdir(repo.dir);

        createBranch(repo, {
          name: "old-branch",
          benchScript: '#!/bin/sh\necho "METRIC latency=100"\necho "METRIC memory=200"',
        });

        createBranch(repo, {
          name: "new-branch",
          benchScript: '#!/bin/sh\necho "METRIC latency=80"\necho "METRIC throughput=500"',
        });

        const options: CompareOptions = {
          oldTarget: "old-branch",
          newTarget: "new-branch",
          bench: "./bench.sh",
          adapter: "metric-lines",
          samples: 3,
          timeoutSeconds: 10,
        };

        const report = await compare(options);

        expect(report).toContain("latency");
        expect(report).toContain("throughput");
        assertWorktreesCleanedUp(repo);
      } finally {
        repo.cleanup();
      }
    });
  });

  describe("when using mitata adapter with fixture replay", () => {
    it("parses mitata JSON fixture and generates report", async () => {
      const repo = createScratchRepo();
      const fixturePath = path.resolve(originalCwd, "tests/fixtures/mitata.json");

      try {
        process.chdir(repo.dir);

        const mitataBenchScript = `#!/bin/sh\ncat "${fixturePath}"`;
        createBranch(repo, {
          name: "mitata-branch",
          benchScript: mitataBenchScript,
        });

        createBranch(repo, {
          name: "mitata-branch-2",
          benchScript: mitataBenchScript,
        });

        const options: CompareOptions = {
          oldTarget: "mitata-branch",
          newTarget: "mitata-branch-2",
          bench: "./bench.sh",
          adapter: "mitata",
          samples: 3,
          timeoutSeconds: 10,
        };

        const report = await compare(options);

        expect(report).toContain("decode");
        expect(report).toContain("encode");
        expect(report).toContain("time");
      } finally {
        repo.cleanup();
      }
    });
  });

  describe("when using signed-rank method with sufficient samples", () => {
    it("applies signed-rank test for n=10 samples", async () => {
      const repo = createScratchRepo();

      try {
        process.chdir(repo.dir);

        createBranch(repo, {
          name: "old-sr",
          benchScript: '#!/bin/sh\necho "METRIC latency=100"',
        });

        createBranch(repo, {
          name: "new-sr",
          benchScript: '#!/bin/sh\necho "METRIC latency=80"',
        });

        const options: CompareOptions = {
          oldTarget: "old-sr",
          newTarget: "new-sr",
          bench: "./bench.sh",
          adapter: "metric-lines",
          samples: 10,
          timeoutSeconds: 10,
        };

        const report = await compare(options);

        expect(report).toBeTruthy();
        expect(report).toContain("latency");
      } finally {
        repo.cleanup();
      }
    });
  });

  describe("when using band method with few samples", () => {
    it("applies band method for n=3 samples", async () => {
      const repo = createScratchRepo();

      try {
        process.chdir(repo.dir);

        createBranch(repo, {
          name: "old-band",
          benchScript: '#!/bin/sh\necho "METRIC latency=100"',
        });

        createBranch(repo, {
          name: "new-band",
          benchScript: '#!/bin/sh\necho "METRIC latency=90"',
        });

        const options: CompareOptions = {
          oldTarget: "old-band",
          newTarget: "new-band",
          bench: "./bench.sh",
          adapter: "metric-lines",
          samples: 3,
          timeoutSeconds: 10,
        };

        const report = await compare(options);

        expect(report).toBeTruthy();
        expect(report).toContain("latency");
      } finally {
        repo.cleanup();
      }
    });
  });

  describe("when bench command fails", () => {
    it("throws error with captured stderr and cleans up worktrees", async () => {
      const repo = createScratchRepo();

      try {
        process.chdir(repo.dir);

        createBranch(repo, {
          name: "old-fail",
          benchScript: '#!/bin/sh\necho "stderr output" >&2\nexit 1',
        });

        createBranch(repo, {
          name: "new-fail",
          benchScript: '#!/bin/sh\necho "METRIC latency=80"',
        });

        const options: CompareOptions = {
          oldTarget: "old-fail",
          newTarget: "new-fail",
          bench: "./bench.sh",
          adapter: "metric-lines",
          samples: 3,
          timeoutSeconds: 10,
        };

        await expect(compare(options)).rejects.toThrow();
        assertWorktreesCleanedUp(repo);
      } finally {
        repo.cleanup();
      }
    });

    it("throws error when new target bench command fails", async () => {
      const repo = createScratchRepo();

      try {
        process.chdir(repo.dir);

        createBranch(repo, {
          name: "old-success",
          benchScript: '#!/bin/sh\necho "METRIC latency=100"',
        });

        createBranch(repo, {
          name: "new-fail-new",
          benchScript: '#!/bin/sh\necho "error" >&2\nexit 1',
        });

        const options: CompareOptions = {
          oldTarget: "old-success",
          newTarget: "new-fail-new",
          bench: "./bench.sh",
          adapter: "metric-lines",
          samples: 3,
          timeoutSeconds: 10,
        };

        await expect(compare(options)).rejects.toThrow();
        assertWorktreesCleanedUp(repo);
      } finally {
        repo.cleanup();
      }
    });
  });

  describe("when metric values are zero", () => {
    it("handles zero values gracefully in median and spread calculation", async () => {
      const repo = createScratchRepo();

      try {
        process.chdir(repo.dir);

        createBranch(repo, {
          name: "old-zero",
          benchScript: '#!/bin/sh\necho "METRIC latency=0"',
        });

        createBranch(repo, {
          name: "new-zero",
          benchScript: '#!/bin/sh\necho "METRIC latency=0"',
        });

        const options: CompareOptions = {
          oldTarget: "old-zero",
          newTarget: "new-zero",
          bench: "./bench.sh",
          adapter: "metric-lines",
          samples: 3,
          timeoutSeconds: 10,
        };

        const report = await compare(options);

        expect(report).toContain("latency");
      } finally {
        repo.cleanup();
      }
    });
  });

  describe("when bench script produces no metrics", () => {
    it("throws error indicating no metrics found", async () => {
      const repo = createScratchRepo();

      try {
        process.chdir(repo.dir);

        createBranch(repo, {
          name: "old-empty",
          benchScript: '#!/bin/sh\necho "no metrics here"',
        });

        createBranch(repo, {
          name: "new-empty",
          benchScript: '#!/bin/sh\necho "also no metrics"',
        });

        const options: CompareOptions = {
          oldTarget: "old-empty",
          newTarget: "new-empty",
          bench: "./bench.sh",
          adapter: "metric-lines",
          samples: 3,
          timeoutSeconds: 10,
        };

        await expect(compare(options)).rejects.toThrow(/[Nn]o valid METRIC|[Nn]o metrics/);
      } finally {
        repo.cleanup();
      }
    });
  });
});
