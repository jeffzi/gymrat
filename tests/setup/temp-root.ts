import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterAll, beforeAll } from "vitest";

/** Variables `os.tmpdir()` reads: `TMPDIR` on POSIX, `TEMP` then `TMP` on Windows. */
const TEMP_ENV_VARS = ["TMPDIR", "TEMP", "TMP"] as const;

const originalTempEnv = new Map<string, string | undefined>();

let tempRoot: string | undefined;

/**
 * Delete a temp root and everything under it, reporting failure instead of raising.
 *
 * Removal goes straight through the filesystem: a worktree whose `.git` link is
 * gone, or points at a repository that no longer exists, is one git refuses to
 * remove but `rm -rf` takes without complaint.
 *
 * A root that survives — a file another process still holds open on Windows, say
 * — is a leaked directory, not a broken test, so the failure is warned about and
 * the run continues.
 */
export function removeTempRoot(root: string): void {
  try {
    fs.rmSync(root, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
  } catch (error) {
    process.stderr.write(`could not remove temp root ${root}: ${String(error)}\n`);
  }
}

beforeAll(() => {
  // realpathSync.native resolves the macOS /var -> /private/var symlink and
  // Windows 8.3 short names, so paths built from os.tmpdir() compare equal to
  // the ones git prints in `worktree list`.
  const root = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-vitest-")));

  for (const name of TEMP_ENV_VARS) {
    originalTempEnv.set(name, process.env[name]);
    process.env[name] = root;
  }
  tempRoot = root;
});

afterAll(() => {
  if (tempRoot === undefined) {
    return;
  }
  const root = tempRoot;
  tempRoot = undefined;

  // Restore first: nothing that runs after teardown should see os.tmpdir()
  // pointing at a directory this file just deleted.
  for (const [name, value] of originalTempEnv) {
    if (value === undefined) {
      delete process.env[name];
    } else {
      process.env[name] = value;
    }
  }
  originalTempEnv.clear();

  removeTempRoot(root);
});
