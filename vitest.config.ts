import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    update: "none",
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
