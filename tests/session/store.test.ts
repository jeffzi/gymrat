import fs from "node:fs";
import path from "node:path";

import { describe, expect, expectTypeOf, it } from "vitest";

import { GymratError, messageOf } from "../../src/errors.js";
import { sessionJsonlPath } from "../../src/session/paths.js";
import type {
  BaselineRecord,
  FinalizeRecord,
  HookRecord,
  IterationRecord,
  KeepRecord,
  SessionLogRecord,
  SessionRecord,
} from "../../src/session/records.js";
import type { SessionState } from "../../src/session/store.js";
import {
  appendRecord,
  foldSession,
  readRecords,
  requireOpenSession,
  requireSession,
} from "../../src/session/store.js";
import { captureGymratError, captureThrown } from "../fixtures/errors.js";
import { freshRoot } from "../fixtures/scratch-repo.js";
import {
  AT,
  blockedKeep,
  committedKeep,
  discardRecord as discard,
  finalizeRecord,
  iterationRecord,
  sessionRecord,
} from "../fixtures/session-records.js";

const SESSION: SessionRecord = sessionRecord({
  worktrees: { experiment: "/repo/.gymrat/experiment", baseline: "/repo/.gymrat/baseline" },
});

const BASELINE: BaselineRecord = {
  type: "baseline",
  at: AT,
  label: "main",
  samples: [{ total_ms: 15200 }, { total_ms: 15184 }],
};

const HOOK: HookRecord = {
  type: "hook",
  stage: "before",
  seq: 1,
  exitCode: 0,
  durationMs: 120,
  stdoutBytes: 80,
  timedOut: false,
};

/** An iteration record numbered `seq`, reaching the target metric or not. */
function iteration(seq: number, targetReached: boolean): IterationRecord {
  return iterationRecord({
    seq,
    samples: {
      experiment: [{ total_ms: 14100 }, { total_ms: 14088 }],
      baseline: [{ total_ms: 15200 }, { total_ms: 15190 }],
    },
    targetReached,
  });
}

/** A blocked keep of the iteration numbered `seq` that never recorded why it was blocked. */
function blockedKeepWithoutReason(seq: number): KeepRecord {
  return blockedKeep(seq, { reason: undefined });
}

const ITERATION_1 = iteration(1, false);
const ITERATION_1_ON_TARGET = iteration(1, true);
const ITERATION_2 = iteration(2, false);
const ITERATION_2_ON_TARGET = iteration(2, true);

const FINALIZE: FinalizeRecord = finalizeRecord({ branch: `${SESSION.branch}-final` });

const EMPTY_STATE: SessionState = {
  session: undefined,
  iterationCount: 0,
  lastIteration: undefined,
  unsettled: false,
  keepCount: 0,
  discardCount: 0,
  targetReachedAndKept: false,
  lastSeq: 0,
  finalized: undefined,
};

/** Path to a session log inside a fresh temp repo root, with no `.gymrat` directory yet. */
function freshJsonlPath(): string {
  return sessionJsonlPath(freshRoot());
}

/** A fresh temp repo root whose session log holds `records`, appended in order. */
function rootHolding(records: SessionLogRecord[]): string {
  const root = freshRoot();
  const jsonlPath = sessionJsonlPath(root);
  for (const record of records) {
    appendRecord(jsonlPath, record);
  }
  return root;
}

/** Path to a session log holding exactly `lines`, each written verbatim. */
function jsonlHolding(lines: string[]): string {
  const jsonlPath = freshJsonlPath();
  fs.mkdirSync(path.dirname(jsonlPath), { recursive: true });
  fs.writeFileSync(jsonlPath, lines.map((line) => `${line}\n`).join(""));
  return jsonlPath;
}

describe("appendRecord", () => {
  describe("when the session directory does not exist yet", () => {
    it("creates it and writes the record as a single JSON line", () => {
      // Arrange
      const jsonlPath = freshJsonlPath();

      // Act
      appendRecord(jsonlPath, SESSION);

      // Assert
      expect(fs.readFileSync(jsonlPath, "utf-8")).toBe(`${JSON.stringify(SESSION)}\n`);
    });
  });

  describe("when the log already holds records", () => {
    it("adds one line, leaving the earlier lines untouched", () => {
      // Arrange
      const jsonlPath = freshJsonlPath();
      appendRecord(jsonlPath, SESSION);

      // Act
      appendRecord(jsonlPath, ITERATION_1);

      // Assert
      expect(fs.readFileSync(jsonlPath, "utf-8")).toBe(
        `${JSON.stringify(SESSION)}\n${JSON.stringify(ITERATION_1)}\n`,
      );
    });
  });
});

describe("readRecords", () => {
  describe("when the log does not exist", () => {
    it("reads as no session", () => {
      // Arrange
      const jsonlPath = freshJsonlPath();

      // Act
      const records = readRecords(jsonlPath);

      // Assert
      expect(records).toStrictEqual([]);
    });
  });

  describe("when the log holds appended records", () => {
    it("returns the same typed records in file order", () => {
      // Arrange
      const jsonlPath = freshJsonlPath();
      const written = [SESSION, BASELINE, HOOK, ITERATION_1, committedKeep(1), discard(2)];
      for (const record of written) {
        appendRecord(jsonlPath, record);
      }

      // Act
      const records = readRecords(jsonlPath);

      // Assert
      expect(records).toStrictEqual(written);
      expectTypeOf(records).toEqualTypeOf<SessionLogRecord[]>();
    });
  });

  describe("when a line is not valid JSON", () => {
    it("throws a GymratError naming the log and the 1-based line number", () => {
      // Arrange
      const jsonlPath = jsonlHolding([JSON.stringify(SESSION), "{not json", "{}"]);

      // Act
      const error = captureThrown(() => readRecords(jsonlPath));

      // Assert
      expect.soft(error).toBeInstanceOf(GymratError);
      expect.soft(messageOf(error)).toContain(`${jsonlPath}:2`);
    });
  });

  describe("when a line is JSON that matches no record schema", () => {
    it("throws a GymratError naming the log line and the offending field", () => {
      // Arrange
      const { metrics: _metrics, ...withoutMetrics } = ITERATION_1;
      const jsonlPath = jsonlHolding([JSON.stringify(SESSION), JSON.stringify(withoutMetrics)]);

      // Act
      const error = captureThrown(() => readRecords(jsonlPath));

      // Assert
      expect.soft(error).toBeInstanceOf(GymratError);
      expect.soft(messageOf(error)).toContain(`${jsonlPath}:2`);
      expect.soft(messageOf(error)).toMatch(/\bmetrics\b/);
    });
  });

  describe("when the first record is not a session header", () => {
    it("throws a GymratError naming the log and its first line", () => {
      // Arrange
      const jsonlPath = jsonlHolding([JSON.stringify(ITERATION_1), JSON.stringify(discard(1))]);

      // Act
      const error = captureThrown(() => readRecords(jsonlPath));

      // Assert
      expect.soft(error).toBeInstanceOf(GymratError);
      expect.soft(messageOf(error)).toContain(`${jsonlPath}:1`);
      expect.soft(messageOf(error)).toMatch(/session/i);
    });
  });
});

describe("foldSession", () => {
  it.each([
    {
      description: "an empty log",
      records: [],
      expected: EMPTY_STATE,
    },
    {
      description: "a session with nothing measured",
      records: [SESSION],
      expected: { ...EMPTY_STATE, session: SESSION },
    },
    {
      description: "a measured iteration nobody has settled",
      records: [SESSION, ITERATION_1],
      expected: {
        ...EMPTY_STATE,
        session: SESSION,
        iterationCount: 1,
        lastIteration: ITERATION_1,
        unsettled: true,
        lastSeq: 1,
      },
    },
    {
      description: "baseline and hook records around a measured iteration",
      records: [SESSION, BASELINE, HOOK, ITERATION_1],
      expected: {
        ...EMPTY_STATE,
        session: SESSION,
        iterationCount: 1,
        lastIteration: ITERATION_1,
        unsettled: true,
        lastSeq: 1,
      },
    },
    {
      description: "an iteration settled by a keep",
      records: [SESSION, ITERATION_1, committedKeep(1)],
      expected: {
        ...EMPTY_STATE,
        session: SESSION,
        iterationCount: 1,
        lastIteration: ITERATION_1,
        keepCount: 1,
        lastSeq: 1,
      },
    },
    {
      description: "an iteration settled by a discard",
      records: [SESSION, ITERATION_1, discard(1)],
      expected: {
        ...EMPTY_STATE,
        session: SESSION,
        iterationCount: 1,
        lastIteration: ITERATION_1,
        discardCount: 1,
        lastSeq: 1,
      },
    },
    {
      description: "a fresh iteration after a settled one",
      records: [SESSION, ITERATION_1, committedKeep(1), ITERATION_2],
      expected: {
        ...EMPTY_STATE,
        session: SESSION,
        iterationCount: 2,
        lastIteration: ITERATION_2,
        unsettled: true,
        keepCount: 1,
        lastSeq: 2,
      },
    },
    {
      description: "a target-reaching iteration that was kept",
      records: [SESSION, ITERATION_1_ON_TARGET, committedKeep(1)],
      expected: {
        ...EMPTY_STATE,
        session: SESSION,
        iterationCount: 1,
        lastIteration: ITERATION_1_ON_TARGET,
        keepCount: 1,
        targetReachedAndKept: true,
        lastSeq: 1,
      },
    },
    {
      description: "a target-reaching iteration that was discarded",
      records: [SESSION, ITERATION_1_ON_TARGET, discard(1)],
      expected: {
        ...EMPTY_STATE,
        session: SESSION,
        iterationCount: 1,
        lastIteration: ITERATION_1_ON_TARGET,
        discardCount: 1,
        lastSeq: 1,
      },
    },
    {
      description: "a target-reaching iteration nobody has kept yet",
      records: [SESSION, ITERATION_1, committedKeep(1), ITERATION_2_ON_TARGET],
      expected: {
        ...EMPTY_STATE,
        session: SESSION,
        iterationCount: 2,
        lastIteration: ITERATION_2_ON_TARGET,
        unsettled: true,
        keepCount: 1,
        lastSeq: 2,
      },
    },
    {
      description: "a kept target followed by a discarded iteration",
      records: [SESSION, ITERATION_1_ON_TARGET, committedKeep(1), ITERATION_2, discard(2)],
      expected: {
        ...EMPTY_STATE,
        session: SESSION,
        iterationCount: 2,
        lastIteration: ITERATION_2,
        keepCount: 1,
        discardCount: 1,
        targetReachedAndKept: true,
        lastSeq: 2,
      },
    },
    {
      description: "a session closed by a finalize",
      records: [SESSION, ITERATION_1, committedKeep(1), FINALIZE],
      expected: {
        ...EMPTY_STATE,
        session: SESSION,
        iterationCount: 1,
        lastIteration: ITERATION_1,
        keepCount: 1,
        lastSeq: 1,
        finalized: FINALIZE,
      },
    },
  ] satisfies { description: string; records: SessionLogRecord[]; expected: SessionState }[])(
    "summarizes $description",
    ({ records, expected }) => {
      // Act
      const state = foldSession(records);

      // Assert
      expect(state).toStrictEqual(expected);
    },
  );

  describe("when a keep was blocked instead of committed", () => {
    it("leaves it out of the keep count", () => {
      // Arrange
      const records = [SESSION, ITERATION_1, blockedKeep(1)];

      // Act
      const state = foldSession(records);

      // Assert
      expect(state.keepCount).toBe(0);
    });
  });

  describe("when a keep was blocked without recording a reason", () => {
    it("leaves the iteration unsettled", () => {
      // Arrange
      const records = [SESSION, ITERATION_1, blockedKeepWithoutReason(1)];

      // Act
      const state = foldSession(records);

      // Assert
      expect(state.unsettled).toBe(true);
    });
  });
});

describe("requireSession", () => {
  describe("when the log holds a session", () => {
    it("hands back the session, the folded state, the log path, and the records", () => {
      // Arrange
      const root = freshRoot();
      const jsonlPath = sessionJsonlPath(root);
      appendRecord(jsonlPath, SESSION);
      appendRecord(jsonlPath, ITERATION_1);

      // Act
      const required = requireSession(root, "measuring an edit");

      // Assert
      expect(required).toStrictEqual({
        session: SESSION,
        state: {
          ...EMPTY_STATE,
          session: SESSION,
          iterationCount: 1,
          lastIteration: ITERATION_1,
          unsettled: true,
          lastSeq: 1,
        },
        jsonlPath,
        records: [SESSION, ITERATION_1],
      });
    });
  });

  describe("when no session has been opened", () => {
    it.each(["measuring an edit", "asking for its status"])(
      "throws a GymratError naming the root and hinting at %s",
      (verb) => {
        // Arrange
        const root = freshRoot();

        // Act
        const error = captureGymratError(() => requireSession(root, verb));

        // Assert
        expect.soft(error.message).toContain(root);
        expect.soft(error.hint).toBe(`Run gymrat start to open one before ${verb}.`);
      },
    );
  });

  describe("when the session was finalized", () => {
    it("still hands the closed session back", () => {
      // Arrange
      const root = rootHolding([SESSION, ITERATION_1, committedKeep(1), FINALIZE]);

      // Act
      const required = requireSession(root, "asking for its status");

      // Assert
      expect(required.state.finalized).toStrictEqual(FINALIZE);
    });
  });
});

describe("requireOpenSession", () => {
  describe("when the log holds a session nobody has finalized", () => {
    it("hands back the session, the folded state, the log path, and the records", () => {
      // Arrange
      const root = rootHolding([SESSION, ITERATION_1]);

      // Act
      const required = requireOpenSession(root, "measuring an edit");

      // Assert
      expect(required).toStrictEqual({
        session: SESSION,
        state: {
          ...EMPTY_STATE,
          session: SESSION,
          iterationCount: 1,
          lastIteration: ITERATION_1,
          unsettled: true,
          lastSeq: 1,
        },
        jsonlPath: sessionJsonlPath(root),
        records: [SESSION, ITERATION_1],
      });
    });
  });

  describe("when no session has been opened", () => {
    it("throws the same GymratError requireSession does", () => {
      // Arrange
      const root = freshRoot();

      // Act
      const error = captureGymratError(() => requireOpenSession(root, "measuring an edit"));

      // Assert
      expect.soft(error.message).toContain(root);
      expect.soft(error.hint).toBe("Run gymrat start to open one before measuring an edit.");
    });
  });

  describe("when the session was finalized", () => {
    it("throws a GymratError naming the closed session and pointing at a fresh start", () => {
      // Arrange
      const root = rootHolding([SESSION, ITERATION_1, committedKeep(1), FINALIZE]);

      // Act
      const error = captureGymratError(() => requireOpenSession(root, "measuring an edit"));

      // Assert
      expect.soft(error.message).toContain(SESSION.sessionId);
      expect.soft(error.hint).toContain("gymrat start");
    });
  });
});
