import { defineConfig, devices } from '@playwright/test';

/**
 * Engine coverage = browser coverage (docs/spec.md §8): chromium covers
 * Chrome + Yandex Browser, webkit covers Safari, firefox covers Zen.
 * CI runs chromium only (PLAYWRIGHT_PROJECT=chromium); run all three locally.
 */
export default defineConfig({
  testDir: './e2e',
  timeout: 90_000,
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  use: {
    baseURL: 'http://localhost:4173',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
  ],
  webServer: {
    command: 'npm run build && npm run preview -- --port 4173 --strictPort',
    port: 4173,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
