import { describe, it, expect, afterEach } from "vitest";

import { NotAGitRepositoryError, repositoryLookupError, runGit } from "../src/git.js";
import { createScratchRepo } from "./fixtures/scratch-repo.js";

describe("repositoryLookupError", () => {
  describe("when git's stderr starts with the fatal not-a-repository diagnostic", () => {
    it("returns a NotAGitRepositoryError", () => {
      const cause = Object.assign(new Error("git failed"), {
        stderr: "fatal: not a git repository (or any of the parent directories): .git\n",
      });

      const error = repositoryLookupError("/some/dir", cause);

      expect(error).toBeInstanceOf(NotAGitRepositoryError);
    });
  });

  describe("when the phrase appears only inside a path, not as a fatal diagnostic", () => {
    it("does not return a NotAGitRepositoryError", () => {
      /**
       * A git error whose stderr mentions the phrase inside a path component
       * rather than as git's own fatal diagnostic. An unanchored regex would
       * wrongly classify this as a repository-missing error.
       */
      const cause = Object.assign(new Error("git failed"), {
        stderr: "error: cannot open /tmp/not a git repository/config: No such file\n",
      });

      const error = repositoryLookupError("/some/dir", cause);

      expect(error).not.toBeInstanceOf(NotAGitRepositoryError);
    });
  });
});

describe("runGit", () => {
  let repo: ReturnType<typeof createScratchRepo> | undefined;

  afterEach(() => {
    repo?.cleanup();
  });

  describe("when the command passes --end-of-options", () => {
    it("returns the output without ENOBUFS for modest output volumes", () => {
      repo = createScratchRepo();

      const result = runGit(["rev-parse", "HEAD"], repo.dir);

      expect(result.trim()).toMatch(/^[0-9a-f]{40}$/);
    });
  });
});
