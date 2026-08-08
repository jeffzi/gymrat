import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    update: "none",
    // Redirects os.tmpdir() per test file. `isolate` and the default `forks` pool
    // give every file its own process, so the redirect cannot bleed across files.
    setupFiles: ["tests/setup/temp-root.ts"],
    coverage: {
      include: ["src/**/*.ts"],
      thresholds: {
        lines: 95,
        functions: 95,
        branches: 95,
        statements: 95,
      },
    },
  },
});
