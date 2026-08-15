import type { DriverSession, SessionOutcome, SessionPrompt, Driver } from "./driver.js";
import { createEventLogWriter } from "./event-log.js";
import type { LaunchEvent, SessionEvent, SessionObserver } from "./events.js";
import { combineObservers } from "./events.js";

/** Grace period after interrupt before firing the abort signal. */
const GRACE_MS = 30_000;

type CapType = "wall-clock" | "spend-cap";

type EndedBy = "session" | CapType;

/**
 * The result of a supervised agent session.
 *
 * - `outcome`: how the session ended (completed, interrupted, or error).
 * - `endedBy`: whether the session ended naturally or was stopped by a cap.
 * - `durationMs`: wall-clock duration from start to settlement.
 * - `costUsd`: final cost reported by the session.
 */
export interface SupervisionResult {
  readonly outcome: SessionOutcome;
  readonly endedBy: EndedBy;
  readonly durationMs: number;
  readonly costUsd: number;
}

interface SuperviseOptions {
  readonly driver: Driver;
  readonly prompt: SessionPrompt;
  readonly maxMinutes: number;
  readonly maxUsd?: number;
  readonly logPath: string;
  readonly launch: LaunchEvent;
  readonly observer?: SessionObserver;
}

/**
 * Run a supervised agent session with wall-clock and spend caps.
 *
 * Starts the driver session, tees every event to a JSONL log and an optional
 * observer, enforces time and cost limits, and returns the session outcome
 * with metadata about how the session ended.
 */
export async function supervise(options: SuperviseOptions): Promise<SupervisionResult> {
  const { driver, prompt, maxMinutes, maxUsd, logPath, launch, observer } = options;

  const logWriter = createEventLogWriter(logPath);
  const abortController = new AbortController();

  let endedBy: EndedBy = "session";
  let capFired = false;
  // oxlint-disable-next-line prefer-const -- reassigned by the `wallClockTimer = setTimeout(…)` call below; declared here so triggerCap's closure can see it
  let wallClockTimer: ReturnType<typeof setTimeout> | undefined;
  let graceTimer: ReturnType<typeof setTimeout> | undefined;
  // oxlint-disable-next-line prefer-const -- reassigned by `session = driver.start(…)` below; declared here so triggerCap's closure can see it
  let session: DriverSession | undefined;
  let pendingCap: CapType | undefined;

  function triggerCap(cap: CapType): void {
    if (capFired) return;
    if (!session) {
      pendingCap = cap;
      return;
    }
    capFired = true;
    endedBy = cap;
    clearTimeout(wallClockTimer);
    try {
      void session.interrupt();
    } catch (error) {
      // Synchronous throw must not prevent grace timer setup
      // oxlint-disable-next-line no-console -- last-resort warning; fallback recovery continues via grace timer
      console.warn("session.interrupt() failed:", error);
    }
    graceTimer = setTimeout(() => {
      abortController.abort();
    }, GRACE_MS);
  }

  const costObserver: SessionObserver = (event: SessionEvent): void => {
    if (maxUsd !== undefined && event.type === "usage_update" && event.costUsd >= maxUsd) {
      triggerCap("spend-cap");
    }
  };

  const observers: SessionObserver[] = [costObserver, logWriter];
  if (observer) observers.push(observer);
  const combined = combineObservers(...observers);

  combined(launch);

  const startTime = Date.now();
  session = driver.start(prompt, combined, abortController.signal);
  if (pendingCap) triggerCap(pendingCap);

  // oxlint-disable-next-line typescript/no-unnecessary-condition -- capFired is set by triggerCap when a pending cap fires above
  if (!capFired) {
    wallClockTimer = setTimeout(() => {
      triggerCap("wall-clock");
    }, maxMinutes * 60_000);
  }

  try {
    const outcome = await session.outcome;

    return {
      outcome,
      endedBy,
      durationMs: Date.now() - startTime,
      costUsd: outcome.costUsd,
    };
  } finally {
    // oxlint-disable-next-line typescript/no-unnecessary-condition -- capFired is mutated by triggerCap during await
    if (!capFired) clearTimeout(wallClockTimer);
    clearTimeout(graceTimer);
  }
}
