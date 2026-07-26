import { execSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export interface ScratchRepo {
  dir: string;
  cleanup: () => void;
}

export function createScratchRepo(): ScratchRepo {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-test-"));

  try {
    // -b main pins the initial branch: without it the name comes from the
    // developer's init.defaultBranch, and tests that check out from "main" break.
    execSync("git init -b main", { cwd: dir, stdio: "pipe" });

    // Configure user and settings for commits
    const configs = [
      ["user.name", "Test User"],
      ["user.email", "test@example.com"],
      ["commit.gpgsign", "false"],
    ];
    for (const [key, value] of configs) {
      execSync(`git config ${key} '${value}'`, { cwd: dir, stdio: "pipe" });
    }

    // Create initial commit
    fs.writeFileSync(path.join(dir, "README.md"), "# Test Repo\n");
    execSync("git add README.md", { cwd: dir, stdio: "pipe" });
    execSync("git commit -m 'Initial commit'", { cwd: dir, stdio: "pipe" });
  } catch (error) {
    fs.rmSync(dir, { recursive: true, force: true });
    throw error;
  }

  return {
    dir,
    cleanup: () => {
      fs.rmSync(dir, { recursive: true, force: true });
    },
  };
}
