import type { SessionObserver } from "./events.js";

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
  interrupt(): Promise<void>;
  readonly outcome: Promise<SessionOutcome>;
}

/** Launches and controls an agent session. */
export interface Driver {
  start(prompt: SessionPrompt, observer: SessionObserver, signal?: AbortSignal): DriverSession;
}
