import { execFile } from "node:child_process";
import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join, resolve } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  createScratchRepo,
  freshRoot,
  toShellPath,
  type ScratchRepo,
} from "./fixtures/scratch-repo.js";

interface CliProcessResult {
  exitCode: number;
  stdout: string;
  stderr: string;
}

/**
 * Run a CLI entry file in a child process and collect its exit code and streams.
 *
 * The entry-point block only runs when the file is the process entry, so these
 * behaviors are unreachable in-process. The child also yields the real exit code
 * and the raw stderr text, both of which the assertions inspect, and it reads
 * the real `config` module — which this suite mocks for every in-process test.
 *
 * `cwd` names the repository the run resolves its configuration and session
 * against; it defaults to the suite's own working directory.
 */
function runCliProcess(
  entry: string,
  args: string[],
  options: { cwd?: string; env?: Record<string, string | undefined> } = {},
): Promise<CliProcessResult> {
  return new Promise<CliProcessResult>((settle) => {
    execFile(
      process.execPath,
      ["--import", import.meta.resolve("tsx"), entry, ...args],
      {
        timeout: 10000,
        cwd: options.cwd,
        env: { ...process.env, FORCE_COLOR: undefined, NO_COLOR: "1", ...options.env },
      },
      (error, stdout, stderr) => {
        settle({
          exitCode: typeof error?.code === "number" ? error.code : 0,
          stdout,
          stderr,
        });
      },
    );
  });
}

describe("a repository with no config file", () => {
  let repo: ScratchRepo;

  beforeEach(() => {
    repo = createScratchRepo();
  });

  afterEach(() => {
    repo.cleanup();
  });

  it.each([
    {
      desc: "keep reaches its own no-session hint",
      args: ["keep"],
      expected: "gymrat start",
    },
    {
      desc: "compare asks for the bench command it has to run",
      args: ["compare", "main", "branch"],
      expected: "bench is required. Provide it via --bench flag or in config file.",
    },
    {
      desc: "measure asks for the bench command it has to run",
      args: ["measure", "main"],
      expected: "bench is required. Provide it via --bench flag or in config file.",
    },
  ])(
    "$desc",
    async ({ args, expected }) => {
      const { exitCode, stderr } = await runCliProcess(resolve("src/cli.ts"), args, {
        cwd: repo.dir,
      });

      expect.soft(exitCode).toBe(2);
      expect(stderr).toContain(expected);
    },
    30_000,
  );
});

describe("the repository a benchmark run locks", () => {
  const DUBIOUS_OWNERSHIP = "fatal: detected dubious ownership in repository";

  function benchRecordingItRan(marker: string): string {
    return `#!/bin/sh\necho ran > "${toShellPath(marker)}"\necho "METRIC latency=100"\n`;
  }

  /**
   * A PATH whose `git` fails every invocation the way one refusing an untrusted
   * repository does.
   *
   * "Not a git repository" is the single git answer that places the run outside
   * a repository; every other failure leaves the question unanswered, and this
   * shim stands in for that whole class — dubious ownership, an unreadable
   * `.git`, a git that cannot run at all.
   */
  function pathWithFailingGit(dir: string): string {
    const shimDir = join(dir, "git-shim");
    mkdirSync(shimDir, { recursive: true });
    const shim = join(shimDir, "git");
    writeFileSync(shim, `#!/bin/sh\necho "${DUBIOUS_OWNERSHIP} at '$PWD'" >&2\nexit 128\n`);
    chmodSync(shim, 0o755);
    return `${shimDir}${delimiter}${process.env.PATH ?? ""}`;
  }

  describe("when git fails inside a repository", () => {
    let repo: ScratchRepo;

    beforeEach(() => {
      repo = createScratchRepo();
    });

    afterEach(() => {
      repo.cleanup();
    });

    it.skipIf(process.platform === "win32")(
      "exits 2 with the git error instead of benchmarking unlocked",
      async () => {
        const benchMarker = join(repo.dir, "bench-ran");
        writeFileSync(join(repo.dir, "bench.sh"), benchRecordingItRan(benchMarker));

        const { exitCode, stderr } = await runCliProcess(
          resolve("src/cli.ts"),
          ["measure", "--bench", "sh bench.sh", "--samples", "1"],
          { cwd: repo.dir, env: { PATH: pathWithFailingGit(repo.dir) } },
        );

        expect.soft(exitCode).toBe(2);
        expect.soft(stderr).toContain(DUBIOUS_OWNERSHIP);
        expect(existsSync(benchMarker)).toBe(false);
      },
      30_000,
    );
  });

  describe("when the working directory is outside every repository", () => {
    let nonRepoDir: string;

    beforeEach(() => {
      nonRepoDir = freshRoot("gymrat-not-a-repo-");
    });

    afterEach(() => {
      rmSync(nonRepoDir, { recursive: true, force: true, maxRetries: 3 });
    });

    it("benchmarks lock-free on git's explicit not-a-repository answer", async () => {
      writeFileSync(join(nonRepoDir, "bench.sh"), '#!/bin/sh\necho "METRIC latency=100"\n');

      const { exitCode, stdout } = await runCliProcess(
        resolve("src/cli.ts"),
        ["measure", "--bench", "sh bench.sh", "--samples", "1"],
        { cwd: nonRepoDir },
      );

      expect.soft(exitCode).toBe(0);
      expect(stdout).toContain("latency");
    }, 30_000);
  });
});

describe("entry point", () => {
  it("executes CLI when invoked through symlink", async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), "gymrat-cli-test-"));
    const cliPath = resolve("src/cli.ts");
    const symlinkPath = join(tmpDir, "cli-symlink.ts");

    try {
      symlinkSync(cliPath, symlinkPath);

      const { stdout } = await runCliProcess(symlinkPath, ["compare", "--help"]);

      expect(stdout).toContain("Usage: gymrat compare");
    } finally {
      rmSync(tmpDir, { recursive: true, force: true });
    }
  }, 20_000);

  describe("when Commander rejects the arguments", () => {
    it("reports the usage error on stderr exactly once", async () => {
      const { exitCode, stderr } = await runCliProcess(resolve("src/cli.ts"), [
        "compare",
        "main",
        "branch",
        "--bogus",
      ]);

      expect.soft(exitCode).toBe(2);
      expect(stderr.match(/unknown option '--bogus'/g) ?? []).toHaveLength(1);
    }, 20_000);
  });

  describe("when the process has no entry-path argument", () => {
    const originalArgv = process.argv;

    afterEach(() => {
      process.argv = originalArgv;
      vi.resetModules();
    });

    it("imports cleanly when process.argv has no entry path", async () => {
      vi.resetModules();
      process.argv = [process.execPath];

      const loading = import("../src/cli.js");

      await expect(loading).resolves.toHaveProperty("createProgram");
    });
  });

  describe("when the entry path does not exist", () => {
    const originalArgv = process.argv;

    afterEach(() => {
      process.argv = originalArgv;
      vi.resetModules();
    });

    it("imports cleanly when process.argv points at a missing entry path", async () => {
      vi.resetModules();
      process.argv = [process.execPath, join(tmpdir(), "gymrat-missing-entry-a1b2c3.ts")];

      const loading = import("../src/cli.js");

      await expect(loading).resolves.toHaveProperty("createProgram");
    });
  });
});
