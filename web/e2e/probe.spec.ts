import { test, expect } from '@playwright/test';

/**
 * Automated run of the compatibility probe page (docs/decisions.md):
 * every assumption the architecture relies on must hold in every engine.
 * Native HEIC decode is the only check allowed to differ (Safari: yes,
 * others: no — both paths are supported).
 */
test('compat probe: all required checks pass', async ({ page, browserName }) => {
  await page.goto('/probe.html');
  await page.waitForSelector('#results table', { timeout: 30_000 });

  const rows = await page.$$eval('#results table tr', (trs) =>
    trs.slice(1).map((tr) => {
      const tds = tr.querySelectorAll('td');
      return {
        name: tds[0]?.textContent ?? '',
        ok: tds[1]?.textContent === '✓',
        note: tds[2]?.textContent ?? '',
      };
    }),
  );
  console.log(`[probe:${browserName}]`);
  for (const r of rows) console.log(`  ${r.ok ? '✓' : '✗'} ${r.name} ${r.note}`);

  for (const r of rows) {
    if (r.name.includes('HEIC')) continue; // informational: native vs libheif
    expect(r.ok, `${browserName}: ${r.name} (${r.note})`).toBe(true);
  }
});
