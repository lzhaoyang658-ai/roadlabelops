import { defineConfig, devices } from "@playwright/test";

const port = process.env.PLAYWRIGHT_PORT ?? "3100";
const baseURL = `http://localhost:${port}`;
const serverCommand = process.env.CI
  ? `PORT=${port} npm run start`
  : `npm run dev -- --port ${port}`;

export default defineConfig({
  testDir: "./e2e",
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  use: {
    baseURL,
    trace: "on-first-retry",
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
  webServer: {
    command: serverCommand,
    url: baseURL,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
