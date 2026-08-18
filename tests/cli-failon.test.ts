/* eslint-disable typescript/no-unsafe-assignment -- vi.spyOn's generic return erases to any; spy results are inherently untyped */
/* eslint-disable typescript/no-unsafe-member-access -- see above */
/* eslint-disable typescript/no-unsafe-argument -- see above */
/* eslint-disable typescript/no-unsafe-return -- see above */
/* eslint-disable typescript/no-unsafe-call -- see above */
/* eslint-disable typescript/no-unsafe-type-assertion -- process.exit mock requires never cast */

import { Command } from "commander";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ResolvedConfig } from "../src/config.js";
import type { ComparisonResult } from "../src/report/types.js";
import type { KindAggregate } from "../src/verdict/aggregate.js";
import type { GeomeanResult } from "../src/verdict/verdict.js";
import {
  createRunnableProgram,
  mockProcessExit,
  stubDeferredWrite,
  stubWrite,
  writtenChunks,
} from "./fixtures/cli-harness.js";
import {
  createCandidate,
  createComparisonResult,
  geomeanOf,
  kindMetric,
  type Metrics,
  metricMeta,
} from "./fixtures/comparison-result.js";

// `...actual` is spread so CommandError passes through unmocked and tests can construct real instances.
vi.mock("../src/compare.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/compare.js")>();
  return {
    ...actual,
    compare: vi.fn(),
  };
});

// `...actual` is spread for the same reason as compare's mock: measure re-exports
// CommandError, and tests construct real instances of it.
vi.mock("../src/measure.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/measure.js")>();
  return {
    ...actual,
    measure: vi.fn(),
  };
});

vi.mock("../src/config.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/config.js")>();
  return {
    ...actual,
    resolveConfig: vi.fn(),
    resolveBenchlessConfig: vi.fn(),
  };
});

vi.mock("../src/config-inspect.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/config-inspect.js")>();
  return {
    ...actual,
    inspectConfig: vi.fn(),
  };
});

const { mockConfirmAction } = vi.hoisted(() => ({
  mockConfirmAction: vi.fn<(message: string, stream: NodeJS.ReadableStream) => Promise<boolean>>(),
}));

vi.mock("../src/confirm.js", () => ({
  confirmAction: mockConfirmAction,
}));

vi.mock("../src/report/json.js", () => ({
  renderJson: vi.fn().mockReturnValue('{"report": true}'),
  renderMeasureJson: vi.fn().mockReturnValue('{"measurement": true}'),
}));

vi.mock("../src/doctor/checks.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/doctor/checks.js")>();
  return {
    ...actual,
    buildEnvironmentSection: vi.fn(),
    buildConfigSection: vi.fn(),
    buildWorkflowSection: vi.fn(),
  };
});

vi.mock("../src/doctor/bench.js", () => ({
  buildBenchSection: vi.fn(),
}));

vi.mock("../src/doctor/render.js", () => ({
  renderDoctorReport: vi.fn().mockReturnValue("doctor text report"),
  renderDoctorJson: vi.fn().mockReturnValue('{"doctor": true}'),
}));

const { mockSpinnerInstance, mockYoctoSpinner, mockEtaRecord, mockFormatEta } = vi.hoisted(() => {
  const instance = {
    start: vi.fn(),
    stop: vi.fn(),
    clear: vi.fn(),
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
    text: "",
    color: "cyan" as const,
    isSpinning: false,
  };
  // Self-referencing returns for chaining
  instance.start.mockReturnValue(instance);
  instance.stop.mockReturnValue(instance);
  instance.clear.mockReturnValue(instance);

  return {
    mockSpinnerInstance: instance,
    mockYoctoSpinner: vi.fn().mockReturnValue(instance),
    mockEtaRecord: vi.fn(),
    mockFormatEta: vi.fn<(ms: number) => string>(),
  };
});

vi.mock("yocto-spinner", () => ({
  default: mockYoctoSpinner,
}));

vi.mock("../src/eta.js", () => {
  class MockEtaTracker {
    record = mockEtaRecord;
  }
  return {
    EtaTracker: MockEtaTracker,
    formatEta: mockFormatEta,
  };
});

const { mockRunWizard, mockScaffold } = vi.hoisted(() => ({
  mockRunWizard: vi.fn(),
  mockScaffold: vi.fn(),
}));

vi.mock("../src/init/wizard.js", () => ({
  runWizard: mockRunWizard,
}));

vi.mock("../src/init/scaffold.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/init/scaffold.js")>();
  return {
    ...actual,
    scaffold: mockScaffold,
  };
});

function resolvedConfigFixture(overrides: Partial<ResolvedConfig> = {}): ResolvedConfig {
  return {
    bench: "bench.sh",
    adapter: "metric-lines",
    samples: 1,
    timeoutSeconds: 300,
    unstableNoisePct: 200,
    primary: "geomean",
    ...overrides,
  };
}

/** A mock async function's settle-branch, shared by `mockResolvedValue`/`mockRejectedValue` stubs. */
interface ResolvableMock<T> {
  mockResolvedValue(value: T): unknown;
  mockRejectedValue(reason: unknown): unknown;
}

/**
 * Settle `mock` on `outcome`: rejects with it when it's an `Error`, otherwise
 * resolves with it (or `fallback` when omitted).
 *
 * The branch `setupMocks` and `setupMeasureMocks` share, differing only in
 * which mock and fallback result they settle.
 */
function stubOutcome<T>(
  mock: ResolvableMock<T>,
  outcome: T | Error | undefined,
  fallback: T,
): void {
  if (outcome instanceof Error) {
    mock.mockRejectedValue(outcome);
  } else {
    mock.mockResolvedValue(outcome ?? fallback);
  }
}

async function setupMocks(
  compareMockReturn?: ComparisonResult | Error,
  resolveConfigMockReturn: Partial<ResolvedConfig> = {},
) {
  const { compare: compareMock } = await import("../src/compare.js");
  const { resolveConfig: resolveConfigMock } = await import("../src/config.js");

  vi.mocked(resolveConfigMock).mockReturnValue(resolvedConfigFixture(resolveConfigMockReturn));
  stubOutcome(vi.mocked(compareMock), compareMockReturn, createComparisonResult());

  return { compareMock, resolveConfigMock };
}

/** `setupMocks` for the single-target command: stubs `resolveConfig` and `measure`. */
/** Prepends the `["node", "cli.js", "compare"]` prefix Commander expects. */
function compareArgv(...args: string[]): string[] {
  return ["node", "cli.js", "compare", ...args];
}

/** Prepends the `["node", "cli.js", "measure"]` prefix Commander expects. */
function stderrWrites(stderrSpy: ReturnType<typeof vi.spyOn>): unknown[] {
  return stderrSpy.mock.calls.map((c: unknown[]) => c[0]);
}

afterEach(() => {
  vi.clearAllMocks();
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  mockSpinnerInstance.text = "";
  mockSpinnerInstance.isSpinning = false;
});

describe("createProgram", () => {
  describe("compare command", () => {
    describe("on compare error", () => {
      it("exits 2 and writes the error to stderr", async () => {
        const program = createRunnableProgram({ exitOverride: "all" });
        await setupMocks(new Error("Compare failed"));
        stubWrite(process.stdout);
        const stderrSpy = stubWrite(process.stderr);
        mockProcessExit();

        await expect(program.parseAsync(compareArgv("main", "branch"))).rejects.toHaveProperty(
          "exitCode",
          2,
        );

        expect(stderrWrites(stderrSpy)).toContainEqual(expect.stringContaining("Compare failed"));
      });

      it("holds the exit until the error text has landed", async () => {
        // A chunk the stream accepted is not a chunk it has handed on,
        // so exiting on the accepting `write` alone drops the diagnostic.
        const program = createRunnableProgram({ exitOverride: "all" });
        await setupMocks(new Error("Compare failed"));
        stubWrite(process.stdout);
        const stderr = stubDeferredWrite(process.stderr);
        const exitSpy = mockProcessExit();

        const parsing = program.parseAsync(compareArgv("main", "branch")).catch((e: unknown) => e);
        await vi.waitFor(() => {
          expect(writtenChunks(stderr.spy)).toContainEqual(
            expect.stringContaining("Compare failed"),
          );
        });

        expect(exitSpy).not.toHaveBeenCalled();
        stderr.flush();
        await expect(parsing).resolves.toHaveProperty("exitCode", 2);
      });

      it("still exits 2 when stderr fails while printing the error", async () => {
        // A closed pipe makes the write throw outright; the exit code
        // contract reserves 1 for a gate trip, so a failed diagnostic must not
        // downgrade the error exit to an unhandled-rejection 1.
        const program = createRunnableProgram({ exitOverride: "all" });
        await setupMocks(new Error("Compare failed"));
        stubWrite(process.stdout);
        vi.spyOn(process.stderr, "write").mockImplementation(() => {
          throw new Error("EPIPE: broken pipe");
        });
        mockProcessExit();

        await expect(program.parseAsync(compareArgv("main", "branch"))).rejects.toHaveProperty(
          "exitCode",
          2,
        );
      });
    });

    describe("--fail-on", () => {
      /**
       * A single-candidate result whose sole metric carries the given verdict.
       *
       * Defaults to a gating, lower-is-better metric so gate tests exercise the
       * standard path. Override `gating` or `geomeanValue` to test edge cases.
       */
      function createGatingResult(
        verdict: "regressed" | "improved" | "no-signal" | "unstable",
        {
          gating = true,
          geomeanValue = -5.0,
          geomeanN = 10,
        }: { gating?: boolean; geomeanValue?: number; geomeanN?: number } = {},
      ): ComparisonResult {
        const deltaByVerdict: Record<typeof verdict, number> = {
          regressed: 8,
          improved: -10,
          "no-signal": 0.2,
          unstable: 0.2,
        };
        const delta = deltaByVerdict[verdict];
        const aggregate = geomeanOf(geomeanValue, geomeanN);
        return createComparisonResult({
          candidates: [
            createCandidate({
              kinds: [
                {
                  kind: "other",
                  geomean: aggregate,
                  groups: [],
                  ...(gating ? { gatedGeomean: aggregate } : {}),
                },
              ],
            }),
          ],
          metrics: {
            "decode/time": {
              baselineMedian: 100,
              baselineSpread: 1,
              candidates: [
                {
                  median: 100 + delta,
                  spread: 1,
                  verdict: {
                    verdict,
                    method: "signed-rank",
                    delta,
                    n: 10,
                    p: 0.01,
                    noisePct: verdict === "unstable" ? 300 : 2.5,
                    noiseAbs: verdict === "unstable" ? 300 : 2.5,
                  },
                },
              ],
              meta: metricMeta("decode/time", { gating }),
            },
          },
        });
      }

      /**
       * Set up a --fail-on test: a program with subcommand overrides, `compare`
       * mocked to resolve (or reject) with `compareMockReturn`, stdout/stderr
       * stubbed, and process.exit converted into a catchable rejection that
       * carries the intended exit code.
       */
      async function setupFailOnTest(compareMockReturn: ComparisonResult | Error): Promise<{
        program: Command;
        stdoutSpy: ReturnType<typeof vi.spyOn>;
        stderrSpy: ReturnType<typeof vi.spyOn>;
        exitSpy: ReturnType<typeof vi.spyOn>;
      }> {
        const program = createRunnableProgram({ exitOverride: "all" });
        await setupMocks(compareMockReturn);
        const stdoutSpy = stubWrite(process.stdout);
        const stderrSpy = stubWrite(process.stderr);
        const exitSpy = mockProcessExit();
        return { program, stdoutSpy, stderrSpy, exitSpy };
      }

      /** The stderr line warning that a geomean gate had no stable metrics to measure, if any. */
      function geomeanGateWarning(stderrSpy: ReturnType<typeof vi.spyOn>): string | undefined {
        return stderrWrites(stderrSpy)
          .map((write) => String(write))
          .find((write) => write.includes("geomean gate"));
      }

      /** Forces plain (uncolored) output regardless of the ambient environment. */
      function disableColor(): void {
        vi.stubEnv("FORCE_COLOR", undefined);
        vi.stubEnv("NO_COLOR", "1");
      }

      /** A kind whose metrics gate, aggregating to `gated` both overall and when gated. */
      function gatingKind(kind: string, gated: GeomeanResult): KindAggregate {
        return { kind, geomean: gated, groups: [], gatedGeomean: gated };
      }

      it("exits 0 when no gating metric regressed", async () => {
        const { program, stdoutSpy, exitSpy } = await setupFailOnTest(
          createGatingResult("improved"),
        );

        await program.parseAsync(compareArgv("main", "branch", "--fail-on", "regressed"));

        expect(stdoutSpy).toHaveBeenCalled();
        expect(exitSpy).not.toHaveBeenCalled();
      });

      it.each([
        {
          desc: "exceeds",
          geomeanValue: 5.0,
          expected: expect.objectContaining({ exitCode: 1 }),
        },
        {
          desc: "sits exactly on",
          geomeanValue: 2.0,
          expected: expect.objectContaining({ exitCode: 1 }),
        },
        { desc: "is within", geomeanValue: 1.5, expected: undefined },
      ])(
        "$desc the threshold when geomean delta is $geomeanValue%",
        async ({ geomeanValue, expected }) => {
          // A 2% gate; trips exit 1 "at or worse than" the threshold
          const { program, stdoutSpy } = await setupFailOnTest(
            createGatingResult("no-signal", { geomeanValue }),
          );

          // A gate trip rejects with exitCode 1, otherwise the parse resolves
          const error = await program
            .parseAsync(compareArgv("main", "branch", "--fail-on", "geomean:2"))
            .then(
              () => undefined,
              (e: unknown) => e,
            );

          // The report must be written before exit — distinguishes a gate
          // trip from a Commander parse error, which would not produce any report output.
          expect(error).toStrictEqual(expected);
          expect(stdoutSpy).toHaveBeenCalled();
        },
      );

      describe("when a candidate's geomean covers no stable gating metrics", () => {
        /** A candidate whose geomean aggregated nothing, gated on the most permissive threshold. */
        async function setupEmptyGeomeanGate(): Promise<
          Awaited<ReturnType<typeof setupFailOnTest>>
        > {
          disableColor();
          return setupFailOnTest(createGatingResult("unstable", { geomeanValue: 0, geomeanN: 0 }));
        }

        it("warns on stderr that the gate had nothing to measure", async () => {
          const { program, stderrSpy } = await setupEmptyGeomeanGate();
          const { label } = createCandidate();

          await program.parseAsync(compareArgv("main", "branch", "--fail-on", "geomean:0"));

          expect(geomeanGateWarning(stderrSpy)).toMatch(
            new RegExp(
              `warning: geomean gate for "${label}" had no stable gating metrics to measure`,
            ),
          );
        });

        it("does not trip the gate", async () => {
          // A geomean of 0 would otherwise sit exactly on the 0% threshold
          const { program, exitSpy } = await setupEmptyGeomeanGate();

          await program.parseAsync(compareArgv("main", "branch", "--fail-on", "geomean:0"));

          expect(exitSpy).not.toHaveBeenCalled();
        });

        it("holds the exit until the warning has landed", async () => {
          // A regressed gating metric trips the other condition, so the
          // vacuous geomean warning is written on the way to an exit. The stream
          // accepted the chunk without handing it on: exiting now would drop it.
          disableColor();
          const { program, exitSpy } = await setupFailOnTest(
            createGatingResult("regressed", { geomeanValue: 0, geomeanN: 0 }),
          );
          const stderr = stubDeferredWrite(process.stderr);

          const parsing = program
            .parseAsync(
              compareArgv("main", "branch", "--fail-on", "regressed", "--fail-on", "geomean:0"),
            )
            .catch((error: unknown) => error);
          await vi.waitFor(() => {
            expect(geomeanGateWarning(stderr.spy)).toBeDefined();
          });

          // The exit waits on the write, then carries the gate's code
          expect(exitSpy).not.toHaveBeenCalled();
          stderr.flush();
          await expect(parsing).resolves.toHaveProperty("exitCode", 1);
        });
      });

      describe("when a candidate spans several metric kinds", () => {
        /** An informational kind: it aggregates a geomean but nothing of it gates. */
        function informationalKind(kind: string, geomean: GeomeanResult): KindAggregate {
          return { kind, geomean, groups: [] };
        }

        /**
         * A one-candidate result with the given kind aggregates and one metric
         * per kind so the rendered report and the aggregates describe the same run.
         */
        function createKindGatingResult(kinds: readonly KindAggregate[]): ComparisonResult {
          const metrics: Metrics = Object.fromEntries(
            kinds.map((aggregate) => [
              `decode/${aggregate.kind}`,
              kindMetric({
                kind: aggregate.kind,
                shortName: "decode",
                verdict: "no-signal",
                delta: 0.2,
                gating: aggregate.gatedGeomean !== undefined,
              }),
            ]),
          );
          return createComparisonResult({
            candidates: [createCandidate({ kinds })],
            metrics,
          });
        }

        it.each([
          {
            shape: "a gating kind is at or worse than the threshold",
            kinds: [
              gatingKind("time", geomeanOf(5, 1)),
              informationalKind("memory", geomeanOf(-5, 1)),
            ],
          },
          {
            shape: "the gating kind with data when another gating kind has none",
            kinds: [gatingKind("time", geomeanOf(0, 0)), gatingKind("cpu", geomeanOf(5, 2))],
          },
        ])("exits 1 when $shape", async ({ kinds }) => {
          const { program } = await setupFailOnTest(createKindGatingResult(kinds));

          await expect(
            program.parseAsync(compareArgv("main", "branch", "--fail-on", "geomean:2")),
          ).rejects.toHaveProperty("exitCode", 1);
        });

        it("does not trip on a non-gating kind's geomean", async () => {
          const { program, exitSpy } = await setupFailOnTest(
            createKindGatingResult([
              gatingKind("time", geomeanOf(-1, 1)),
              informationalKind("memory", geomeanOf(11, 1)),
            ]),
          );

          await program.parseAsync(compareArgv("main", "branch", "--fail-on", "geomean:2"));

          expect(exitSpy).not.toHaveBeenCalled();
        });

        describe("when no gating kind measured a stable metric", () => {
          const vacuousKinds: readonly KindAggregate[] = [
            gatingKind("time", geomeanOf(0, 0)),
            informationalKind("memory", geomeanOf(-5, 1)),
          ];

          it.each([
            { shape: "every gating kind measured nothing", kinds: vacuousKinds },
            {
              shape: "no kind gates at all",
              kinds: [
                informationalKind("memory", geomeanOf(-5, 1)),
                informationalKind("other", geomeanOf(1, 1)),
              ],
            },
          ])("warns on stderr when $shape", async ({ kinds }) => {
            disableColor();
            const { program, stderrSpy } = await setupFailOnTest(createKindGatingResult(kinds));
            const { label } = createCandidate();

            await program.parseAsync(compareArgv("main", "branch", "--fail-on", "geomean:0"));

            expect(geomeanGateWarning(stderrSpy)).toContain(
              `warning: geomean gate for "${label}" had no stable gating metrics to measure`,
            );
          });
        });
      });

      it("hands the conditions to the renderer, so the report echoes the tripped gate", async () => {
        // A gating "time" kind at +5.0%, well past the 2% gate
        disableColor();
        const gated = geomeanOf(5, 1);
        const { program, stdoutSpy } = await setupFailOnTest(
          createComparisonResult({
            candidates: [createCandidate({ kinds: [gatingKind("time", gated)] })],
            metrics: {
              "decode/time": kindMetric({
                kind: "time",
                shortName: "decode",
                verdict: "regressed",
                delta: 5,
              }),
            },
          }),
        );

        await expect(
          program.parseAsync(compareArgv("main", "branch", "--fail-on", "geomean:2")),
        ).rejects.toHaveProperty("exitCode", 1);
        const report = stdoutSpy.mock.calls.map((call: unknown[]) => String(call[0])).join("");
        expect(report).toContain("⚑ time gated geomean +5.0% exceeded --fail-on geomean:2");
      });

      it("trips when any of multiple conditions matches", async () => {
        // No regression verdict, but geomean +5.0% exceeds the 2% gate
        const { program, stdoutSpy } = await setupFailOnTest(
          createGatingResult("improved", { geomeanValue: 5.0 }),
        );

        await expect(
          program.parseAsync(
            compareArgv("main", "branch", "--fail-on", "regressed", "--fail-on", "geomean:2"),
          ),
        ).rejects.toHaveProperty("exitCode", 1);
        // The report must be written — distinguishes a gate trip from a parse error
        expect(stdoutSpy).toHaveBeenCalled();
      });

      it("rejects a malformed condition with the allowed grammar in the error", async () => {
        const program = createRunnableProgram({ exitOverride: "all", silent: true });
        mockProcessExit();

        await expect(
          program.parseAsync(compareArgv("main", "branch", "--fail-on", "bogus")),
        ).rejects.toThrow(/regressed.*geomean|geomean.*regressed/);
      });

      it.each([
        { form: "an empty percentage", value: "geomean:" },
        { form: "a whitespace percentage", value: "geomean: " },
        { form: "a hexadecimal percentage", value: "geomean:0x10" },
      ])("rejects geomean with $form", async ({ value }) => {
        const program = createRunnableProgram({ exitOverride: "all", silent: true });
        await setupMocks();
        mockProcessExit();

        const parsing = program.parseAsync(compareArgv("main", "branch", "--fail-on", value));

        await expect(parsing).rejects.toThrow(/regressed.*geomean|geomean.*regressed/);
      });

      it("prints the report to stdout before exiting 1 on gate trip", async () => {
        const { program, stdoutSpy, exitSpy } = await setupFailOnTest(
          createGatingResult("regressed"),
        );

        const error = await program
          .parseAsync(compareArgv("main", "branch", "--fail-on", "regressed"))
          .catch((e: unknown) => e);

        // Assert - report was written before exit was called
        expect(error).toHaveProperty("exitCode", 1);
        expect(stdoutSpy).toHaveBeenCalled();
        const reportOrder = stdoutSpy.mock.invocationCallOrder[0];
        expect(reportOrder).toBeDefined();
        const exitOrder = exitSpy.mock.invocationCallOrder[0];
        expect(exitOrder).toBeDefined();
        expect(reportOrder).toBeLessThan(exitOrder);
      });

      it("does not trip when the regressed metric is non-gating", async () => {
        // The only regressed metric has gating: false
        const { program, stdoutSpy } = await setupFailOnTest(
          createGatingResult("regressed", { gating: false }),
        );

        await program.parseAsync(compareArgv("main", "branch", "--fail-on", "regressed"));

        expect(stdoutSpy).toHaveBeenCalled();
      });

      describe("when stdout has accepted the report without handing it on", () => {
        it("holds the exit until the report has landed", async () => {
          // A `write` returning true only says the chunk fit under the
          // high-water mark; the bytes are still queued, so exiting on that
          // return alone truncates the report.
          const { program, exitSpy } = await setupFailOnTest(createGatingResult("regressed"));
          const stdout = stubDeferredWrite(process.stdout);

          const parsing = program
            .parseAsync(compareArgv("main", "branch", "--fail-on", "regressed"))
            .catch((e: unknown) => e);
          await vi.waitFor(() => {
            expect(stdout.spy).toHaveBeenCalled();
          });

          // The exit is held until the write completes
          expect(exitSpy).not.toHaveBeenCalled();
          stdout.flush();
          await expect(parsing).resolves.toHaveProperty("exitCode", 1);
        });
      });

      it("exits 2 for Commander usage errors", async () => {
        // The production exitOverride (which sets exit code 2) must
        // survive here rather than being replaced by the test helper's plain one
        const program = createRunnableProgram({ exitOverride: "none", silent: true });
        mockProcessExit();

        await expect(
          program.parseAsync(compareArgv("main", "branch", "--bogus")),
        ).rejects.toHaveProperty("exitCode", 2);
      });
    });
  });
});
