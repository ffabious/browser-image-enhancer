/**
 * Headless run of the demo's ?bench mode (chromium): processes every bundled
 * sample with both brains and prints the timing table for the README.
 *
 *   cd web && npm run build && node ../scripts/run-bench.cjs
 */
const { chromium } = require('@playwright/test');
const { spawn } = require('node:child_process');

(async () => {
  const srv = spawn('npx', ['vite', 'preview', '--port', '4317', '--strictPort'], {
    cwd: `${__dirname}/../web`,
    stdio: 'ignore',
  });
  try {
    for (let i = 0; i < 40; i++) {
      try {
        await fetch('http://localhost:4317/');
        break;
      } catch {
        await new Promise((r) => setTimeout(r, 500));
      }
    }
    const browser = await chromium.launch();
    const page = await browser.newPage();
    await page.goto('http://localhost:4317/?bench');
    await page.waitForSelector('#bench table', { timeout: 180_000 });
    const rows = await page.$$eval('#bench tr', (trs) =>
      trs.map((tr) => [...tr.querySelectorAll('th,td')].map((td) => td.textContent).join(' | ')),
    );
    for (const r of rows) console.log(r);
    const mem = await page.$eval('#bench p', (p) => p.textContent).catch(() => null);
    if (mem) console.log(mem);
    await browser.close();
  } finally {
    srv.kill();
  }
  process.exit(0);
})();
