import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

import { GymratError } from "../errors.js";
import type { SessionEvent, SessionObserver } from "./events.js";

/**
 * Creates a {@link SessionObserver} that appends each event as a JSON line
 * to the file at `logPath`.
 *
 * The parent directory is created (recursively) on the first write if it does
 * not already exist. A write failure surfaces as a {@link GymratError} naming
 * the log path so the user knows which file failed.
 */
export function createEventLogWriter(logPath: string): SessionObserver {
  let dirEnsured = false;

  return (event: SessionEvent): void => {
    try {
      if (!dirEnsured) {
        mkdirSync(dirname(logPath), { recursive: true });
        dirEnsured = true;
      }
      appendFileSync(logPath, JSON.stringify(event) + "\n");
    } catch (error) {
      throw new GymratError(`Failed to write event log: ${logPath}`, undefined, {
        cause: error,
      });
    }
  };
}
