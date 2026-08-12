import fs from "node:fs";
import path from "node:path";

import { GymratError, stderrTextOf } from "../errors.js";
import { repositoryLookupError, runGit, tryGit } from "../git.js";
import { baselineWorktreeDir, experimentWorktreeDir, SESSION_DIR_NAME } from "./paths.js";

/** Prefix of the branch a session's experiment worktree sits on. */
const BRANCH_PREFIX = "gymrat/";

/** A git ref together with the commit it resolved to when the session started. */
export interface BaselineRef {
  ref: string;
  sha: string;
}

/** A session's git state: the branch it edits on, its worktrees, and its pinned baseline. */
export interface WorkspaceResult {
  branch: string;
  worktrees: { experiment: string; baseline: string };
  baseline: BaselineRef;
}

/**
 * Create the branch and both worktrees a session runs in.
 *
 * The experiment worktree is checked out *on* `gymrat/<sessionId>` so edits and
 * commits land on the session's own branch; the baseline worktree is detached at
 * `baseline.sha` so it keeps measuring the same commit no matter what the ref it
 * came from does afterwards.
 *
 * Either all of the branch and both worktrees exist afterwards, or nothing this
 * call made does: a worktree git refuses takes the branch and whatever this
 * attempt had already checked out down with it, so a retry starts from the state
 * it found instead of tripping over its own leftovers. A worktree directory that
 * was already standing survives — see {@link unwindWorkspace}.
 *
 * @throws GymratError when `root` is not a git repository, or when git refuses
 *   to prune, to create the branch, or to create either worktree.
 */
export function createWorkspace(
  root: string,
  sessionId: string,
  baseline: BaselineRef,
): WorkspaceResult {
  const branch = `${BRANCH_PREFIX}${sessionId}`;

  ensureGitExclude(root);
  pruneStaleWorktrees(root);

  runGitStep(
    ["branch", branch, baseline.sha],
    root,
    `Cannot create the session branch '${branch}'`,
    `A crashed session may have left it behind. Delete it with: git branch -D ${branch}`,
  );
  // Read before the first add, so the unwind can tell a directory this attempt
  // checked out from one that was already there.
  const standing = [experimentWorktreeDir(root), baselineWorktreeDir(root)].filter(isDirectory);

  try {
    addExperimentWorktree(root, branch);
    addBaselineWorktree(root, baseline.sha);
  } catch (error) {
    unwindWorkspace(root, branch, standing);
    throw error;
  }

  return {
    branch,
    worktrees: { experiment: experimentWorktreeDir(root), baseline: baselineWorktreeDir(root) },
    baseline,
  };
}

/**
 * Keep git from reporting the session directory as untracked.
 *
 * The line goes in `.git/info/exclude` rather than `.gitignore` because it is
 * this checkout's business, not the project's: nothing gymrat writes should show
 * up in a commit the agent under test prepares.
 *
 * @throws GymratError when `root` is not a git repository.
 */
export function ensureGitExclude(root: string): void {
  const excludeFile = path.join(gitCommonDir(root), "info", "exclude");
  const line = `${SESSION_DIR_NAME}/`;
  const existing = fs.existsSync(excludeFile) ? fs.readFileSync(excludeFile, "utf-8") : "";

  if (existing.split("\n").some((entry) => entry.trim() === line)) {
    return;
  }

  fs.mkdirSync(path.dirname(excludeFile), { recursive: true });
  const separator = existing === "" || existing.endsWith("\n") ? "" : "\n";
  fs.writeFileSync(excludeFile, `${existing}${separator}${line}\n`);
}

/** Whether both of the session's worktree directories are on disk. */
export function detectWorkspace(root: string): boolean {
  return isDirectory(experimentWorktreeDir(root)) && isDirectory(baselineWorktreeDir(root));
}

/**
 * Put back whichever of a session's worktrees is no longer on disk.
 *
 * Resuming has to survive a worktree the user deleted, so this is a no-op when
 * both are present — an experiment worktree carrying uncommitted work is never
 * re-checked-out.
 *
 * @throws GymratError when git refuses to prune or to add a worktree.
 */
export function recreateWorkspace(root: string, branch: string, baselineSha: string): void {
  const needsExperiment = !isDirectory(experimentWorktreeDir(root));
  const needsBaseline = !isDirectory(baselineWorktreeDir(root));

  if (!needsExperiment && !needsBaseline) {
    return;
  }

  pruneStaleWorktrees(root);

  if (needsExperiment) {
    addExperimentWorktree(root, branch);
  }
  if (needsBaseline) {
    addBaselineWorktree(root, baselineSha);
  }
}

/**
 * Commit everything standing in the experiment worktree, and report the commit.
 *
 * Staging is `add -A` so the commit holds what the agent produced whether it
 * tracked its new files or not — the worktree belongs to the session, and a keep
 * that left half an edit behind would put the next iteration's baseline out of
 * step with the code that earned it.
 *
 * @throws GymratError when git refuses to stage or to commit — a worktree with
 *   nothing to commit included.
 */
export function commitWorkspace(experimentDir: string, message: string): string {
  runGitStep(
    ["add", "-A"],
    experimentDir,
    `Cannot stage the experiment worktree at ${experimentDir}`,
    "Inspect what is standing there with: git status",
  );
  runGitStep(
    ["commit", "-m", message],
    experimentDir,
    `Cannot commit the experiment worktree at ${experimentDir}`,
    "Inspect what is standing there with: git status",
  );
  return runGitStep(
    ["rev-parse", "HEAD"],
    experimentDir,
    `Cannot read the commit just made in ${experimentDir}`,
    "Inspect the branch with: git log -1",
  ).trim();
}

/**
 * Throw away everything the experiment worktree has not committed.
 *
 * Destructive by contract, and safe because the directory is one gymrat owns:
 * the reset covers tracked edits — staged or not — and the clean covers the
 * files the agent added, which a reset alone would leave behind to be picked up
 * by the next keep.
 *
 * @throws GymratError when git refuses to reset or to clean.
 */
export function revertWorkspace(experimentDir: string): void {
  runGitStep(
    ["reset", "--hard", "HEAD"],
    experimentDir,
    `Cannot revert the experiment worktree at ${experimentDir}`,
    "Inspect what is standing there with: git status",
  );
  runGitStep(
    ["clean", "-fd"],
    experimentDir,
    `Cannot remove the untracked files in ${experimentDir}`,
    "Inspect what is standing there with: git status --ignored",
  );
}

/**
 * Take both of a session's worktrees off disk and out of git's bookkeeping.
 *
 * Each worktree is named by path rather than swept for, because git clears the
 * entry of a directory that vanished behind its back when it is asked for that
 * path — which leaves a worktree of the user's own that is only temporarily
 * absent registered. A refusal over a directory that is already gone stays quiet:
 * nothing is standing there for the user to clear by hand.
 *
 * Removal failures are returned instead of thrown because this runs after the
 * finalize record is written — the session is closed either way, and the caller's
 * job is to tell the user which directory to clear by hand.
 *
 * @returns One warning per worktree git left standing, empty when both went.
 */
export function removeWorktrees(
  root: string,
  worktrees: { experiment: string; baseline: string },
): string[] {
  const warnings: string[] = [];

  for (const dir of [worktrees.experiment, worktrees.baseline]) {
    const error = tryGit(["worktree", "remove", "--force", dir], root);
    if (error !== undefined && isDirectory(dir)) {
      warnings.push(
        `Could not remove the worktree at ${dir}: ${error}\n` +
          `  remove it by hand with: git worktree remove --force ${dir}`,
      );
    }
  }

  return warnings;
}

/**
 * Whether `dir` holds work git has not committed — untracked files included.
 *
 * A directory that is not there reads as clean: a worktree the user deleted
 * carries no uncommitted work anyone can still act on, and refusing to finalize
 * over a directory that cannot be inspected would strand the session.
 */
export function isWorktreeDirty(dir: string): boolean {
  if (!isDirectory(dir)) {
    return false;
  }
  return (
    runGitStep(
      ["status", "--porcelain"],
      dir,
      `Cannot read the status of the worktree at ${dir}`,
      "Inspect what is standing there with: git status",
    ).trim() !== ""
  );
}

/**
 * Move the baseline worktree onto `sha`, detached as it was created.
 *
 * Detached is not incidental: `sha` is a commit on the session branch, which the
 * experiment worktree has checked out, and git refuses to check the same branch
 * out twice.
 *
 * @throws GymratError when git refuses the checkout.
 */
export function advanceBaseline(baselineDir: string, sha: string): void {
  runGitStep(
    ["checkout", "--detach", sha],
    baselineDir,
    `Cannot move the baseline worktree at ${baselineDir} to ${sha}`,
    `Check that ${sha} is a commit this repository has: git cat-file -t ${sha}`,
  );
}

/**
 * Drop git's bookkeeping for worktrees whose directories are no longer there.
 *
 * git keeps the admin entry of a worktree whose directory vanished, and refuses
 * to add that path back — or to check its branch out again — until the entry
 * goes. Only entries with nothing on disk behind them go, so a worktree the user
 * is still working in is left alone.
 *
 * @throws GymratError when git refuses to prune.
 */
function pruneStaleWorktrees(root: string): void {
  runGitStep(
    ["worktree", "prune"],
    root,
    "Cannot clear the session's stale worktree entries",
    "Inspect them with: git worktree list",
  );
}

function addExperimentWorktree(root: string, branch: string): void {
  const dir = experimentWorktreeDir(root);
  runGitStep(
    ["worktree", "add", dir, branch],
    root,
    `Cannot create the experiment worktree at ${dir}`,
    `Check whether ${branch} is already checked out elsewhere: git worktree list`,
  );
}

/**
 * Take back the branch and worktrees a failed {@link createWorkspace} had made.
 *
 * A directory listed in `standing` was there before the attempt began — what a
 * session whose log was lost leaves behind — so it stays, uncommitted work and
 * all. The error the caller is about to throw names the path, which is the only
 * notice the user gets that something is in the way.
 *
 * The worktrees go before the branch: git refuses to delete a branch one of them
 * still has checked out. Every step is best-effort — the caller is about to see
 * the git step that broke the session, and a cleanup that cannot finish must not
 * speak in its place.
 */
function unwindWorkspace(root: string, branch: string, standing: readonly string[]): void {
  for (const dir of [experimentWorktreeDir(root), baselineWorktreeDir(root)]) {
    if (isDirectory(dir) && !standing.includes(dir)) {
      tryGit(["worktree", "remove", "--force", dir], root);
    }
  }
  tryGit(["branch", "-D", branch], root);
}

function addBaselineWorktree(root: string, sha: string): void {
  const dir = baselineWorktreeDir(root);
  runGitStep(
    ["worktree", "add", "--detach", dir, sha],
    root,
    `Cannot create the baseline worktree at ${dir}`,
    `Check that ${sha} is a commit this repository has: git cat-file -t ${sha}`,
  );
}

/**
 * Absolute path of the repository's shared git directory.
 *
 * `info/exclude` lives in the common directory, so a linked worktree — whose
 * `.git` is a file pointing elsewhere — excludes through the same file the main
 * checkout uses. git prints the path relative to the working directory when it
 * sits inside it, hence the resolve against `root`.
 */
function gitCommonDir(root: string): string {
  try {
    const printed = runGit(["rev-parse", "--git-common-dir"], root).trim();
    return path.resolve(root, printed);
  } catch (error) {
    throw repositoryLookupError(root, error);
  }
}

function isDirectory(dir: string): boolean {
  return fs.statSync(dir, { throwIfNoEntry: false })?.isDirectory() ?? false;
}

/**
 * Run git in `cwd`, turning a non-zero exit into a `GymratError` that carries
 * git's own diagnostics after `message`.
 */
export function runGitStep(
  args: readonly string[],
  cwd: string,
  message: string,
  hint: string,
): string {
  try {
    return runGit(args, cwd);
  } catch (error) {
    throw new GymratError(`${message}: ${stderrTextOf(error)}`, hint, { cause: error });
  }
}
