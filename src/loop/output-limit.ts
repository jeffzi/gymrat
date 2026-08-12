const NEWLINE_BYTE = 0x0a;

/**
 * How much of a command's output reaches the agent driving the loop.
 *
 * A command that dumps a build log would otherwise bury the measurement it was
 * meant to annotate, and the whole of it is on disk anyway.
 */
const OUTPUT_LIMIT_BYTES = 8192;

/**
 * At most {@link OUTPUT_LIMIT_BYTES} of `text`, cut back to the last whole line
 * and, when a single line already overruns the limit, to the last whole
 * character.
 *
 * The limit is counted in bytes because that is what a consumer sizing their
 * command's output can measure; cutting mid-character would put a replacement
 * character in the agent's transcript instead.
 */
export function limitOutput(text: string): string {
  const encoded = Buffer.from(text, "utf-8");
  if (encoded.byteLength <= OUTPUT_LIMIT_BYTES) {
    return text;
  }

  const head = encoded.subarray(0, OUTPUT_LIMIT_BYTES);
  const lastNewline = head.lastIndexOf(NEWLINE_BYTE);
  if (lastNewline >= 0) {
    return head.subarray(0, lastNewline).toString("utf-8");
  }

  // A streaming decoder holds back the bytes of a character the cut split
  // instead of emitting U+FFFD for them.
  return new TextDecoder("utf-8").decode(head, { stream: true });
}
