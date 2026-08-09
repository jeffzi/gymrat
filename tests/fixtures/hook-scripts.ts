import fs from "node:fs";
import path from "node:path";

/** Builders for hook commands, scoped to the directory a suite scripts into. */
export interface HookScripts {
  hookCommand: (body: string) => string;
  printing: (...lines: string[]) => string;
}

/**
 * A `hookCommand`/`printing` pair that writes scripts into `dir`, numbered by
 * a counter private to this call so repeated invocations across a suite never
 * collide on a script name.
 *
 * `hookCommand` runs `body` as an ES module under the Node binary the suite
 * itself runs on: no `sh` on PATH, no executable bit, no shell builtins. Only
 * the two paths are quoted, which both `sh` and `cmd.exe` read the same way.
 */
export function hookScripts(dir: string): HookScripts {
  let scriptCount = 0;

  function hookCommand(body: string): string {
    scriptCount += 1;
    const scriptPath = path.join(dir, `hook-${scriptCount}.mjs`);
    fs.writeFileSync(scriptPath, body);
    return `"${process.execPath}" "${scriptPath}"`;
  }

  /** A command printing each of `lines` on its own line. */
  function printing(...lines: string[]): string {
    const writes = lines.map((line) => `process.stdout.write(${JSON.stringify(`${line}\n`)});`);
    return hookCommand(`${writes.join("\n")}\n`);
  }

  return { hookCommand, printing };
}
