import { describe, expect, it } from "vitest";

import { readBundledSkill } from "../src/bundled-skill.js";
import { captureGymratError } from "./fixtures/errors.js";
import { createPackageLayout } from "./fixtures/package-layout.js";

// ---------------------------------------------------------------------------
// readBundledSkill
// ---------------------------------------------------------------------------

describe("readBundledSkill", () => {
  describe("when the bundled skill file exists at the expected path", () => {
    it("returns the skill file content", () => {
      const { importMetaUrl } = createPackageLayout("bundled-skill-");

      const result = readBundledSkill(importMetaUrl);

      expect(result).toContain("# Test Skill");
    });
  });

  describe("when the bundled skill file is missing", () => {
    it("throws a GymratError naming the path, carrying a reinstall hint, and chaining cause", () => {
      const { importMetaUrl } = createPackageLayout("bundled-skill-", { skipSkill: true });

      const thrown = captureGymratError(() => readBundledSkill(importMetaUrl));

      expect(thrown.message).toContain("SKILL.md");
      expect(thrown.hint).toMatch(/reinstall/i);
      expect(thrown.cause).toBeDefined();
    });
  });
});
