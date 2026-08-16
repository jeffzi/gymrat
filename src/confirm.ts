import { createInterface } from "node:readline/promises";
import type { Readable } from "node:stream";

/**
 * Prompt the user with a y/N question on stderr and return their answer.
 *
 * Returns `true` only for `y` or `Y`; everything else — including EOF, empty
 * input, `n`, and arbitrary text — returns `false`. The prompt is written to
 * stderr so stdout stays data-only.
 *
 * `readline.question()` never resolves when the input stream is already closed
 * or ends before a line arrives. The `close` event race handles this: when the
 * interface closes (EOF), the question is abandoned and the call returns
 * `false`.
 */
export async function confirmAction(message: string, input: Readable): Promise<boolean> {
  const rl = createInterface({ input, output: process.stderr });
  try {
    const answer = await Promise.race([
      rl.question(`${message} [y/N] `),
      new Promise<undefined>((resolve) => {
        rl.once("close", () => {
          resolve(undefined);
        });
      }),
    ]);
    return answer === "y" || answer === "Y";
  } finally {
    rl.close();
  }
}
