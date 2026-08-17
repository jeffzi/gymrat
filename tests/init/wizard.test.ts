import { PassThrough } from "node:stream";

import { describe, expect, it } from "vitest";

import { GymratError } from "../../src/errors.js";
import { runWizard, type WizardOptions } from "../../src/init/wizard.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Build PassThrough streams for injectable input/output.
 *
 * When `lines` is given, each string is written to the input stream as a
 * separate line (with trailing newline appended). The stream remains open
 * unless `end` is true.
 */
function makeStreams(
  lines?: string[],
  opts?: { end?: boolean; isTTY?: boolean },
): { input: PassThrough; output: PassThrough } {
  const input = new PassThrough();
  const output = new PassThrough();

  if (opts?.isTTY !== undefined) {
    (input as PassThrough & { isTTY?: boolean }).isTTY = opts.isTTY;
  }

  if (lines) {
    for (const line of lines) {
      input.write(`${line}\n`);
    }
    if (opts?.end) {
      input.end();
    }
  }

  return { input, output };
}

/** Collect all data written to a PassThrough stream as a string. */
function collectOutput(stream: PassThrough): () => string {
  const chunks: Buffer[] = [];
  stream.on("data", (chunk: Buffer) => chunks.push(chunk));
  return () => Buffer.concat(chunks).toString();
}

/** Build default WizardOptions for non-interactive mode (yes=true). */
function nonInteractiveOptions(overrides?: Partial<WizardOptions>): WizardOptions {
  const { input, output } = makeStreams();
  return {
    input,
    output,
    yes: true,
    ...overrides,
  };
}

/** Build default WizardOptions for interactive mode. */
function interactiveOptions(lines: string[], overrides?: Partial<WizardOptions>): WizardOptions {
  const { input, output } = makeStreams(lines, { end: true, isTTY: true });
  return {
    input,
    output,
    ...overrides,
  };
}

/**
 * Inject a value of the wrong runtime type into a typed field, simulating
 * Commander passing raw, unconverted CLI input (e.g. a string where a number
 * is expected).
 */
// oxlint-disable-next-line typescript/no-unsafe-type-assertion -- single documented cast, intentional invalid-input injection
function asRawInput(value: unknown): number {
  return value as number;
}

// ---------------------------------------------------------------------------
// runWizard — non-interactive mode
// ---------------------------------------------------------------------------

describe("runWizard", () => {
  describe("non-interactive mode", () => {
    describe("when bench is provided via flag", () => {
      it("returns settled defaults without prompting", async () => {
        const result = await runWizard(nonInteractiveOptions({ bench: "npm run bench" }));

        expect(result).toStrictEqual({
          bench: "npm run bench",
          installSkill: true,
          runbook: { path: "gymrat-runbook.md" },
        });
      });
    });

    describe("when bench is missing", () => {
      it("throws GymratError naming --bench flag", async () => {
        await expect(runWizard(nonInteractiveOptions())).rejects.toThrow("--bench");
      });
    });

    describe("when adapter flag is provided", () => {
      it("includes adapter in the result", async () => {
        const result = await runWizard(
          nonInteractiveOptions({
            bench: "npm run bench",
            adapter: "mitata",
          }),
        );

        expect(result.adapter).toBe("mitata");
      });
    });

    describe("when invalid adapter flag is provided", () => {
      it("throws GymratError naming the adapter and listing valid ones", async () => {
        const promise = runWizard(
          nonInteractiveOptions({
            bench: "npm run bench",
            adapter: "nope",
          }),
        );

        await expect(promise).rejects.toThrow(GymratError);
        await expect(promise).rejects.toThrow(/Unknown adapter.*"nope"/);
        await expect(promise).rejects.toSatisfy((error: GymratError) =>
          /valid adapters are:/.test(error.hint ?? ""),
        );
      });
    });

    describe("when stop-target flag is invalid", () => {
      it("throws GymratError for non-numeric value", async () => {
        await expect(
          runWizard(
            nonInteractiveOptions({
              bench: "npm run bench",
              stopTarget: asRawInput("abc"),
            }),
          ),
        ).rejects.toThrow();
      });
    });

    describe("when stop-target without primary", () => {
      it("throws GymratError naming both flags", async () => {
        await expect(
          runWizard(
            nonInteractiveOptions({
              bench: "npm run bench",
              stopTarget: 1.5,
            }),
          ),
        ).rejects.toThrow("--primary");
      });
    });

    describe("when stop-target with primary", () => {
      it("includes both in the result", async () => {
        const result = await runWizard(
          nonInteractiveOptions({
            bench: "npm run bench",
            stopTarget: 1.5,
            primary: "latency",
          }),
        );

        expect(result.stopTarget).toBe(1.5);
        expect(result.primary).toBe("latency");
      });
    });

    describe("when stop-max-iterations flag is invalid", () => {
      it("throws for non-integer", async () => {
        await expect(
          runWizard(
            nonInteractiveOptions({
              bench: "npm run bench",
              stopMaxIterations: 0,
            }),
          ),
        ).rejects.toThrow();
      });
    });

    describe("when --no-runbook is passed", () => {
      it("sets runbook to false", async () => {
        const result = await runWizard(
          nonInteractiveOptions({
            bench: "npm run bench",
            runbook: false,
          }),
        );

        expect(result.runbook).toBe(false);
      });
    });

    describe("when --runbook with a path is passed", () => {
      it("uses the provided path", async () => {
        const result = await runWizard(
          nonInteractiveOptions({
            bench: "npm run bench",
            runbook: "custom-runbook.md",
          }),
        );

        expect(result.runbook).toStrictEqual({ path: "custom-runbook.md" });
      });
    });

    describe("when --runbook is passed bare (boolean true)", () => {
      it("treats true as create at default path", async () => {
        const result = await runWizard(
          nonInteractiveOptions({
            bench: "npm run bench",
            runbook: true,
          }),
        );

        expect(result.runbook).toStrictEqual({ path: "gymrat-runbook.md" });
      });
    });

    describe("when --no-skill is passed", () => {
      it("sets installSkill to false", async () => {
        const result = await runWizard(
          nonInteractiveOptions({
            bench: "npm run bench",
            skill: false,
          }),
        );

        expect(result.installSkill).toBe(false);
      });
    });

    describe("when --skill is passed", () => {
      it("sets installSkill to true", async () => {
        const result = await runWizard(
          nonInteractiveOptions({
            bench: "npm run bench",
            skill: true,
          }),
        );

        expect(result.installSkill).toBe(true);
      });
    });

    it("asks nothing (does not read from input)", async () => {
      const { input, output } = makeStreams();
      let readCalled = false;
      const originalRead = input.read.bind(input);
      input.read = (...args: Parameters<typeof input.read>) => {
        readCalled = true;
        // oxlint-disable-next-line typescript/no-unsafe-return -- Readable.read() returns any in Node types
        return originalRead(...args);
      };

      await runWizard({ bench: "npm run bench", yes: true, input, output });

      expect(readCalled).toBe(false);
    });
  });

  // ---------------------------------------------------------------------------
  // runWizard — interactive mode
  // ---------------------------------------------------------------------------

  describe("interactive mode", () => {
    describe("when all answers are provided via prompts", () => {
      it("settles each answer from interactive input", async () => {
        // Answers in order: bench, gate (y), adapter, checks, stop target,
        // stop max iterations, runbook create (y), runbook path (default), skill (y)
        const result = await runWizard(
          interactiveOptions([
            "npm run bench",
            "y",
            "metric-lines",
            "npm run lint",
            "",
            "",
            "y",
            "",
            "y",
          ]),
        );

        expect(result.bench).toBe("npm run bench");
        expect(result.checks).toBe("npm run lint");
        expect(result.installSkill).toBe(true);
        expect(result.runbook).toStrictEqual({ path: "gymrat-runbook.md" });
      });
    });

    describe("when bench is empty interactively", () => {
      it("reprompts until non-empty", async () => {
        const result = await runWizard(
          interactiveOptions(["", "", "npm run bench", "y", "metric-lines", "", "", "", "n", "n"]),
        );

        expect(result.bench).toBe("npm run bench");
      });
    });

    describe("when adapter is invalid interactively", () => {
      it("reprompts with the error message and valid adapter names", async () => {
        const { input, output } = makeStreams(
          ["npm run bench", "y", "nope", "metric-lines", "", "", "", "n", "n"],
          { end: true, isTTY: true },
        );
        const getOutput = collectOutput(output);

        const result = await runWizard({ input, output });

        expect(result.adapter).toBeUndefined();
        const out = getOutput();
        expect(out).toContain('Unknown adapter: "nope".');
        expect(out).toContain("valid adapters are:");
      });
    });

    describe("when flag pre-answers a question", () => {
      it("skips the prompt for that question", async () => {
        // Provide bench via flag; interactive answers: gate (y), adapter, checks,
        // stop target, stop max iterations, runbook create, skill
        const result = await runWizard(
          interactiveOptions(["y", "metric-lines", "", "", "", "n", "y"], {
            bench: "npm run bench",
          }),
        );

        expect(result.bench).toBe("npm run bench");
      });
    });

    describe("when stop target is provided interactively", () => {
      it("prompts for primary metric", async () => {
        const result = await runWizard(
          interactiveOptions([
            "npm run bench",
            "y",
            "metric-lines",
            "",
            "1.5",
            "latency",
            "3",
            "n",
            "y",
          ]),
        );

        expect(result.stopTarget).toBe(1.5);
        expect(result.primary).toBe("latency");
        expect(result.stopMaxIterations).toBe(3);
      });
    });

    describe("when stop target is empty interactively", () => {
      it("does not prompt for primary", async () => {
        const result = await runWizard(
          interactiveOptions(["npm run bench", "y", "metric-lines", "", "", "", "n", "y"]),
        );

        expect(result.stopTarget).toBeUndefined();
        expect(result.primary).toBeUndefined();
      });
    });

    describe("when primary is 'geomean' interactively", () => {
      it("reprompts for a different primary", async () => {
        const result = await runWizard(
          interactiveOptions([
            "npm run bench",
            "y",
            "metric-lines",
            "",
            "1.5",
            "geomean",
            "latency",
            "",
            "n",
            "y",
          ]),
        );

        expect(result.primary).toBe("latency");
      });
    });

    describe("when runbook is declined interactively", () => {
      it("sets runbook to false", async () => {
        const result = await runWizard(
          interactiveOptions(["npm run bench", "y", "metric-lines", "", "", "", "n", "y"]),
        );

        expect(result.runbook).toBe(false);
      });
    });

    describe("when stop-target is invalid interactively", () => {
      it("reprompts until valid number", async () => {
        const result = await runWizard(
          interactiveOptions([
            "npm run bench",
            "y",
            "metric-lines",
            "",
            "abc",
            "2.0",
            "latency",
            "",
            "n",
            "y",
          ]),
        );

        expect(result.stopTarget).toBe(2.0);
      });
    });

    describe("when stop-max-iterations is invalid interactively", () => {
      it("reprompts until valid integer >= 1", async () => {
        const result = await runWizard(
          interactiveOptions([
            "npm run bench",
            "y",
            "metric-lines",
            "",
            "1.5",
            "latency",
            "0",
            "-1",
            "3",
            "n",
            "y",
          ]),
        );

        expect(result.stopMaxIterations).toBe(3);
      });
    });

    // -------------------------------------------------------------------------
    // Advanced settings gate
    // -------------------------------------------------------------------------

    describe("advanced settings gate", () => {
      describe("when gate is accepted", () => {
        it.each(["y", "Y"])("prompts for advanced settings on %j", async (gate) => {
          const result = await runWizard(
            interactiveOptions([
              "npm run bench",
              gate,
              "mitata",
              "npm run lint",
              "1.5",
              "latency",
              "3",
              "n",
              "y",
            ]),
          );

          expect(result.adapter).toBe("mitata");
          expect(result.checks).toBe("npm run lint");
          expect(result.stopTarget).toBe(1.5);
          expect(result.primary).toBe("latency");
          expect(result.stopMaxIterations).toBe(3);
        });
      });

      describe("when gate is declined", () => {
        it.each(["n", ""])("skips advanced prompts and uses defaults on %j", async (gate) => {
          const result = await runWizard(interactiveOptions(["npm run bench", gate, "n", "y"]));

          expect(result.adapter).toBeUndefined();
          expect(result.checks).toBeUndefined();
          expect(result.stopTarget).toBeUndefined();
          expect(result.primary).toBeUndefined();
          expect(result.stopMaxIterations).toBeUndefined();
        });

        it("still honors flag-supplied advanced settings", async () => {
          const result = await runWizard(
            interactiveOptions(["npm run bench", "n", "n", "y"], {
              adapter: "mitata",
              stopTarget: 1.5,
              primary: "latency",
            }),
          );

          expect(result.adapter).toBe("mitata");
          expect(result.stopTarget).toBe(1.5);
          expect(result.primary).toBe("latency");
        });
      });
    });
  });

  // ---------------------------------------------------------------------------
  // runWizard — EOF at required prompts
  // ---------------------------------------------------------------------------

  describe("EOF at required prompts", () => {
    describe("when input closes before bench is answered", () => {
      it("throws GymratError naming --bench", async () => {
        // Arrange — stream ends immediately with no lines
        const { input, output } = makeStreams([], { end: true, isTTY: true });

        // Act
        const promise = runWizard({ input, output });

        // Assert
        await expect(promise).rejects.toThrow(GymratError);
        await expect(promise).rejects.toThrow(/Missing --bench/);
      });
    });

    describe("when input closes before primary is answered", () => {
      it("throws GymratError naming --primary", async () => {
        // Arrange — bench is pre-answered via flag, stop-target provided via flag,
        // interactive answers: gate (y), adapter, checks — then EOF before primary
        const { input, output } = makeStreams(["y", "metric-lines", ""], {
          end: true,
          isTTY: true,
        });

        // Act
        const promise = runWizard({
          input,
          output,
          bench: "npm run bench",
          stopTarget: 1.5,
        });

        // Assert
        await expect(promise).rejects.toThrow(GymratError);
        await expect(promise).rejects.toThrow(/Missing --primary/);
      });
    });
  });

  // ---------------------------------------------------------------------------
  // runWizard — stop-target strict parsing (interactive)
  // ---------------------------------------------------------------------------

  describe("stop-target strict parsing (interactive)", () => {
    describe("when Infinity is entered for stop-target", () => {
      it("reprompts and accepts a finite decimal", async () => {
        // Arrange — answers: bench, gate (y), adapter, checks, stop-target "Infinity"
        // (invalid), then "1.5" (valid), primary, max-iterations, runbook, skill
        const { input, output } = makeStreams(
          ["npm run bench", "y", "metric-lines", "", "Infinity", "1.5", "latency", "", "n", "y"],
          { end: true, isTTY: true },
        );
        const getOutput = collectOutput(output);

        // Act
        const result = await runWizard({ input, output });

        // Assert — the Infinity input was rejected (error shown), finite input accepted
        const out = getOutput();
        expect(out).toMatch(/finite|number|invalid/i);
        expect(result.stopTarget).toBe(1.5);
      });
    });
  });

  // ---------------------------------------------------------------------------
  // runWizard — bare --primary without --stop-target
  // ---------------------------------------------------------------------------

  describe("bare --primary without --stop-target", () => {
    it("includes primary in the result", async () => {
      const result = await runWizard(
        nonInteractiveOptions({
          bench: "npm run bench",
          primary: "latency",
        }),
      );

      expect(result.primary).toBe("latency");
      expect(result.stopTarget).toBeUndefined();
    });
  });

  // ---------------------------------------------------------------------------
  // runWizard — non-TTY stdin without --yes
  // ---------------------------------------------------------------------------

  describe("non-TTY stdin without --yes", () => {
    it("behaves as non-interactive (uses defaults)", async () => {
      const { input, output } = makeStreams([], { isTTY: false });
      input.end();

      const result = await runWizard({
        bench: "npm run bench",
        input,
        output,
      });

      expect(result).toStrictEqual({
        bench: "npm run bench",
        installSkill: true,
        runbook: { path: "gymrat-runbook.md" },
      });
    });
  });
});
