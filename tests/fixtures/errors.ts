import { GymratError } from "../../src/errors.js";

/** Run `act` and hand back the error it threw, failing the test if it threw none. */
export function captureThrown(act: () => unknown): unknown {
  try {
    act();
  } catch (error) {
    return error;
  }
  throw new Error("expected the call to throw");
}

/**
 * Run `act` and hand back the {@link GymratError} it threw.
 *
 * A different error class is rethrown rather than returned: a test asking for a
 * `GymratError` is asserting the failure was one gymrat worded, and a raw
 * `TypeError` reaching the caller is the failure the test wants to see.
 */
export function captureGymratError(act: () => unknown): GymratError {
  const error = captureThrown(act);
  if (error instanceof GymratError) {
    return error;
  }
  throw error;
}

/** Await a promise expected to reject, and hand back the Error it rejected with. */
export async function captureRejection(promise: Promise<unknown>): Promise<Error> {
  const outcome: unknown = await promise.then(
    () => undefined,
    (error: unknown) => error,
  );
  if (!(outcome instanceof Error)) {
    throw new Error(`expected a rejection with an Error, got: ${String(outcome)}`);
  }
  return outcome;
}

/**
 * Run `act` — sync or async — and hand back the {@link GymratError} it failed with.
 *
 * A synchronous throw is turned into a rejection first, so a caller does not
 * have to know whether the function under test returns a promise.
 */
export async function captureRejectedGymratError(act: () => unknown): Promise<GymratError> {
  const error = await captureRejection(Promise.resolve().then(act));
  if (error instanceof GymratError) {
    return error;
  }
  throw error;
}
