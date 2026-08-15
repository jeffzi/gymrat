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

function hasStderr(error: unknown): error is Error & { stderr: string } {
  return error instanceof Error && "stderr" in error && typeof error.stderr === "string";
}

/**
 * The diagnostics a failed child process wrote to stderr.
 *
 * `execFileSync` attaches the child's piped stderr to the thrown error, separate
 * from `message`, which it prefixes with `Command failed: <command>` noise.
 *
 * Falls back to `messageOf` for thrown values that carry no stderr, or whose
 * stderr is empty — git sometimes explains a failure on stdout instead (e.g.
 * "nothing to commit"), leaving stderr an empty string rather than absent.
 */
export function stderrTextOf(error: unknown): string {
  if (hasStderr(error) && error.stderr.trim() !== "") {
    return error.stderr.trim();
  }

  return messageOf(error);
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

export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/**
 * `Omit<T, K>` without the type-safety hole: `Omit` accepts any string for `K`,
 * so a typo or a renamed field in `T` silently produces `T` unchanged instead of
 * a compile error. Constraining `K extends keyof T` forces the mistake to surface
 * at the call site.
 */
export type StrictOmit<T, K extends keyof T> = Omit<T, K>;
