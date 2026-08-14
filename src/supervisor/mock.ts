import { messageOf } from "../errors.js";
import { createSession } from "./driver.js";
import type { SessionOutcome } from "./driver.js";
import type { SessionEvent, SessionObserver } from "./events.js";

interface EmitStep {
  readonly emit: SessionEvent;
  readonly delayMs?: number;
}

interface ActionStep {
  readonly action: () => Promise<void>;
  readonly delayMs?: number;
}

interface CostStep {
  readonly costUsd: number;
  readonly delayMs?: number;
}

/** A single step in a mock driver script: emits an event, runs an async action, or reports a cost. */
export type MockStep = EmitStep | ActionStep | CostStep;

function isEmitStep(step: MockStep): step is EmitStep {
  return "emit" in step;
}

function isActionStep(step: MockStep): step is ActionStep {
  return "action" in step;
}

/**
 * Resolves after `ms` milliseconds, or immediately when the signal fires,
 * whichever comes first. Cleans up the unused listener/timer on resolution.
 */
function interruptibleDelay(ms: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.resolve();
  return new Promise<void>((resolve) => {
    let settled = false;
    function settle(): void {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal.removeEventListener("abort", settle);
      // oxlint-disable-next-line promise/no-multiple-resolved -- guarded by `settled` flag; linter can't trace runtime state
      resolve();
    }
    const timer = setTimeout(settle, ms);
    signal.addEventListener("abort", settle, { once: true });
  });
}

/**
 * Creates a driver whose `start` method runs a caller-supplied script of steps
 * in order. Designed for testing supervisor orchestration without a real agent
 * backend.
 *
 * The returned session exposes an `injections` list for test inspection and
 * supports interrupt/abort to cancel remaining steps.
 */
export function createMockDriver(steps: readonly MockStep[]) {
  return {
    start(_prompt: unknown, observer: SessionObserver, externalSignal?: AbortSignal) {
      const controller = new AbortController();
      const { signal } = controller;
      const injections: string[] = [];
      let costUsd = 0;
      let interruptedOutcome: SessionOutcome | undefined;

      function interrupted(): SessionOutcome {
        return { reason: "interrupted", costUsd };
      }

      function doInterrupt(): void {
        controller.abort();
        interruptedOutcome = interrupted();
      }

      if (externalSignal) {
        if (externalSignal.aborted) {
          doInterrupt();
        } else {
          externalSignal.addEventListener("abort", doInterrupt, { once: true });
        }
      }

      async function executeStep(step: MockStep): Promise<void> {
        if (isEmitStep(step)) {
          await Promise.resolve();
          if (!signal.aborted) observer(step.emit);
        } else if (isActionStep(step)) {
          await step.action();
        } else {
          await Promise.resolve();
          if (signal.aborted) return;
          costUsd = step.costUsd;
          observer({
            type: "usage_update",
            timestamp: Date.now(),
            costUsd: step.costUsd,
          });
        }
      }

      async function runScript(): Promise<SessionOutcome> {
        for (const step of steps) {
          if (signal.aborted) return interrupted();

          if (step.delayMs !== undefined && step.delayMs > 0) {
            await interruptibleDelay(step.delayMs, signal);
          }

          // oxlint-disable-next-line typescript/no-unnecessary-condition -- signal.aborted is externally mutated after await
          if (signal.aborted) return interrupted();

          try {
            await executeStep(step);
          } catch (err: unknown) {
            return { reason: "error", costUsd, message: messageOf(err) };
          }

          // oxlint-disable-next-line typescript/no-unnecessary-condition -- signal.aborted is externally mutated after await
          if (signal.aborted) return interrupted();
        }

        return { reason: "completed", costUsd };
      }

      const session = createSession({
        onMessage: (msg) => injections.push(msg),
        doInterrupt,
        getCostUsd: () => costUsd,
        getInterruptedOutcome: () => interruptedOutcome,
        runPromise: runScript(),
        observer,
      });

      return Object.assign(session, { injections });
    },
  };
}
