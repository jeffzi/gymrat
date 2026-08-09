import { execFileSync, type StdioOptions } from "node:child_process";

import { GymratError } from "./errors.js";

const GIT_STDIO: StdioOptions = ["pipe", "pipe", "pipe"];

/**
 * Run git with suppressed output, returning stdout.
 *
 * Arguments are passed as an array rather than a shell string so that refs and
 * paths containing shell metacharacters are treated as literal git arguments.
 *
 * @throws whatever `execFileSync` throws — callers decide how to wrap it.
 */
export function runGit(args: readonly string[], cwd: string): string {
  return execFileSync("git", args, { cwd, encoding: "utf-8", stdio: GIT_STDIO });
}

/**
 * The error every entry point throws when a directory it needs turns out not
 * to be inside a git repository, so the message and hint read the same no
 * matter which one noticed.
 */
export function notAGitRepositoryError(dir: string, cause: unknown): GymratError {
  return new GymratError(
    `Not a git repository: ${dir}`,
    "Run gymrat from inside a git repository.",
    {
      cause,
    },
  );
}
