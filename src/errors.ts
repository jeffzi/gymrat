/**
 * Base error for all gymrat-specific failures.
 *
 * Sets `.name` via `new.target.name` so subclasses get their own class name
 * automatically — no manual `this.name = "MyError"` or `Object.setPrototypeOf`
 * boilerplate needed.
 */
export class GymratError extends Error {
  readonly hint: string | undefined;

  constructor(message: string, hint?: string, options?: ErrorOptions) {
    super(message, options);
    this.name = new.target.name;
    this.hint = hint;
  }
}

/**
 * Extract a human-readable message from an unknown thrown value.
 *
 * Returns `error.message` when the value is an `Error`, otherwise falls back
 * to `String(error)`.
 */
export function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Exhaustiveness guard for discriminated unions.
 *
 * Place in the `default` branch of a switch to get a compile-time error when a
 * new variant is added but not handled.
 */
export function assertNever(value: never): never {
  throw new Error(`Unexpected value: ${JSON.stringify(value)}`);
}
