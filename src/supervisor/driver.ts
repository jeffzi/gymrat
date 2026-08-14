import type { SessionObserver } from "./events.js";
import { summarize, SUMMARY_MAX_CHARS } from "./events.js";

/** What the driver is asked to run. */
export interface SessionPrompt {
  readonly kickoff: string;
  readonly systemPromptAppend?: string;
  readonly cwd: string;
  readonly model?: string;
}

/** Why a session ended. */
export type SessionEndReason = "completed" | "interrupted" | "error";

/** The result of a completed session; `message` is present only when reason is `"error"`. */
export interface SessionOutcome {
  readonly reason: SessionEndReason;
  readonly costUsd: number;
  readonly message?: string;
}

/** A live agent session returned by {@link Driver.start}. */
export interface DriverSession {
  inject(message: string): void;
  interrupt(): Promise<void>;
  usage(): { readonly costUsd: number };
  readonly outcome: Promise<SessionOutcome>;
}

/** Launches and controls an agent session. */
export interface Driver {
  start(prompt: SessionPrompt, observer: SessionObserver, signal?: AbortSignal): DriverSession;
}

/** Parts that vary between driver implementations. */
export interface SessionParts {
  readonly onMessage: (message: string) => void;
  readonly doInterrupt: () => void;
  readonly getCostUsd: () => number;
  readonly getInterruptedOutcome: () => SessionOutcome | undefined;
  readonly runPromise: Promise<SessionOutcome>;
  readonly observer: SessionObserver;
}

/**
 * Assembles the common {@link DriverSession} surface from implementation-specific
 * parts. Both the mock and Claude drivers delegate to this so the inject-event
 * shape, interrupt protocol, and outcome getter stay in one place.
 */
export function createSession(parts: SessionParts): DriverSession {
  const { onMessage, doInterrupt, getCostUsd, getInterruptedOutcome, runPromise, observer } = parts;

  return {
    inject(message: string): void {
      onMessage(message);
      observer({
        type: "inject",
        timestamp: Date.now(),
        message,
        messageSummary: summarize(message, SUMMARY_MAX_CHARS),
      });
    },

    interrupt(): Promise<void> {
      doInterrupt();
      return Promise.resolve();
    },

    usage(): { readonly costUsd: number } {
      return { costUsd: getCostUsd() };
    },

    get outcome(): Promise<SessionOutcome> {
      const interrupted = getInterruptedOutcome();
      if (interrupted) {
        return Promise.resolve(interrupted);
      }
      return runPromise;
    },
  };
}
