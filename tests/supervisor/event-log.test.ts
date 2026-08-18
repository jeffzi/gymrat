import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { GymratError } from "../../src/errors.js";
import { createEventLogWriter } from "../../src/supervisor/event-log.js";
import type { SessionObserver, UsageUpdateEvent } from "../../src/supervisor/events.js";
import { combineObservers } from "../../src/supervisor/events.js";

function makeTempDir(): string {
  return mkdtempSync(join(tmpdir(), "event-log-"));
}

function makeTempLogPath(): { dir: string; logPath: string } {
  const dir = makeTempDir();
  return { dir, logPath: join(dir, "events.jsonl") };
}

function makeEvent(overrides: Partial<UsageUpdateEvent> = {}): UsageUpdateEvent {
  return {
    type: "usage_update",
    timestamp: Date.now(),
    costUsd: 0.01,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// createEventLogWriter
// ---------------------------------------------------------------------------

describe("createEventLogWriter", () => {
  describe("when observing events", () => {
    it("appends one JSON line per event to the log file", () => {
      const { logPath } = makeTempLogPath();
      const writer = createEventLogWriter(logPath);
      const event1 = makeEvent({ timestamp: 1000 });
      const event2 = makeEvent({ timestamp: 2000 });

      writer(event1);
      writer(event2);

      const lines = readFileSync(logPath, "utf-8").split("\n").filter(Boolean);
      expect(lines).toHaveLength(2);
      const line0 = lines[0] ?? "";
      const line1 = lines[1] ?? "";
      expect(JSON.parse(line0)).toStrictEqual(event1);
      expect(JSON.parse(line1)).toStrictEqual(event2);
    });

    it("terminates each line with a newline", () => {
      const { logPath } = makeTempLogPath();
      const writer = createEventLogWriter(logPath);

      writer(makeEvent());

      const raw = readFileSync(logPath, "utf-8");
      expect(raw.endsWith("\n")).toBe(true);
    });
  });

  describe("when the parent directory does not exist", () => {
    it("creates the directory tree on first write", () => {
      const dir = makeTempDir();
      const logPath = join(dir, "nested", "deep", "events.jsonl");
      const writer = createEventLogWriter(logPath);

      writer(makeEvent());

      const lines = readFileSync(logPath, "utf-8").split("\n").filter(Boolean);
      expect(lines).toHaveLength(1);
    });
  });

  describe("when a write fails", () => {
    it("throws a GymratError that names the log path", () => {
      // A directory path cannot be opened as a file for appending
      const dir = makeTempDir();
      const writer = createEventLogWriter(dir);

      expect(() => {
        writer(makeEvent());
      }).toThrow(GymratError);
      expect(() => {
        writer(makeEvent());
      }).toThrow(dir);
    });
  });

  describe("SessionObserver compatibility", () => {
    it("is assignable to SessionObserver and works with combineObservers", () => {
      const { logPath } = makeTempLogPath();
      const writer = createEventLogWriter(logPath);

      // Type-level: the writer is a SessionObserver
      const observer: SessionObserver = writer;
      const combined = combineObservers(observer);
      const event = makeEvent();

      combined(event);

      const lines = readFileSync(logPath, "utf-8").split("\n").filter(Boolean);
      expect(lines).toHaveLength(1);
      const line0 = lines[0] ?? "";
      expect(JSON.parse(line0)).toStrictEqual(event);
    });
  });
});
