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
 * Whether an unknown thrown value is an `Error` whose `.code` matches `code`.
 *
 * Node's filesystem and process errors carry a string `.code` that is not part
 * of the base `Error` type, so the probe narrows through `instanceof` and an
 * `in` check before reading the property.
 */
export function hasErrorCode(err: unknown, code: string): boolean {
  return err instanceof Error && "code" in err && err.code === code;
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
