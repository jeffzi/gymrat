import { execFileSync, type StdioOptions } from "node:child_process";

import { GymratError, stderrTextOf } from "./errors.js";

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
 * A directory git placed outside every repository.
 *
 * Its own class because callers act on the distinction: standing outside a
 * repository is a supported way to run gymrat, while a git that declined to
 * answer says nothing about where the directory sits and must never be read as
 * "no repository here".
 */
export class NotAGitRepositoryError extends GymratError {}

/** Git's wording when it places a directory outside every repository. */
const NOT_A_REPOSITORY_RE = /not a git repository/i;

/**
 * The error a failed repository lookup throws, classified by what git said, so
 * the message and hint read the same no matter which entry point noticed.
 *
 * Git reporting that `dir` is outside every repository is an answer callers can
 * act on. Every other failure — dubious ownership, an unreadable `.git`, a git
 * that cannot run at all — is git declining to answer, and carries git's own
 * diagnostics so the reader sees the reason rather than a wrong one.
 */
export function repositoryLookupError(dir: string, cause: unknown): GymratError {
  const diagnostics = stderrTextOf(cause);

  if (NOT_A_REPOSITORY_RE.test(diagnostics)) {
    return new NotAGitRepositoryError(
      `Not a git repository: ${dir}`,
      "Run gymrat from inside a git repository.",
      { cause },
    );
  }

  return new GymratError(
    `Cannot determine the git repository at ${dir}: ${diagnostics}`,
    undefined,
    {
      cause,
    },
  );
}
