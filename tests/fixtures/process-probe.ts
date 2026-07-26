import fs from "node:fs";

import { expect, vi } from "vitest";

/** True while a process with `pid` exists. */
export function isAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

/**
 * Reads a pid a shell wrote with `echo $$ > file` (its own) or `echo $! > file`
 * (the job it just started in the background).
 *
 * Returns NaN while the write is absent or incomplete: the trailing newline is
 * what marks the value as fully flushed, so a partially written file reads as no
 * pid rather than as a truncated one.
 */
export function readPid(pidPath: string): number {
  const raw = fs.readFileSync(pidPath, "utf8");
  return raw.endsWith("\n") ? Number.parseInt(raw, 10) : Number.NaN;
}

/**
 * Poll `pidPath` until it holds a complete pid, and return it.
 *
 * `timeoutMs` is the caller's, because how long the write can take depends on
 * what is being waited for: a bare shell writes almost immediately, while a pid
 * that appears only once a whole comparison run is under way needs far longer.
 */
export async function waitForPid(pidPath: string, timeoutMs: number): Promise<number> {
  let pid = Number.NaN;
  await vi.waitFor(
    () => {
      pid = readPid(pidPath);
      expect(pid).toBeGreaterThan(0);
    },
    { timeout: timeoutMs, interval: 25 },
  );
  return pid;
}
