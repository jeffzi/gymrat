import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    coverage: {
      // Without include, untested src files never enter the report and can't
      // drag the thresholds down — the gate would only measure touched files.
      include: ["src/**"],
      thresholds: {
        lines: 95,
        functions: 95,
        branches: 95,
        statements: 95,
      },
    },
  },
});
