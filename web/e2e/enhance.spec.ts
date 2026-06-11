import { test, expect, type Page } from '@playwright/test';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const fixture = (name: string): string =>
  fileURLToPath(new URL(`./fixtures/${name}`, import.meta.url));

interface StatusEvent {
  taskId: string;
  status: string;
  progress: number;
}

async function openAndTrack(page: Page): Promise<void> {
  await page.goto('/');
  await page.waitForFunction(() => '__enhancer' in window);
  await page.evaluate(() => {
    const w = window as unknown as { __events: StatusEvent[]; __enhancer: EventTarget };
    w.__events = [];
    w.__enhancer.addEventListener('taskstatuschange', (e) => {
      w.__events.push((e as CustomEvent).detail as StatusEvent);
    });
  });
}

function events(page: Page): Promise<StatusEvent[]> {
  return page.evaluate(() => (window as unknown as { __events: StatusEvent[] }).__events);
}

const FORMATS = ['small.jpg', 'small.png', 'small.bmp'].map((f) => `generated/${f}`);
FORMATS.push('sample.heic');

for (const name of FORMATS) {
  test(`enhances ${name}`, async ({ page }) => {
    test.skip(!existsSync(fixture(name)), `fixture ${name} missing — run scripts/make-fixtures.py`);
    await openAndTrack(page);
    await page.setInputFiles('#file-input', fixture(name));
    const card = page.locator('.task').first();
    await expect(card.locator('.task-status.done')).toBeVisible({ timeout: 60_000 });
    await expect(card.locator('.compare img.after')).toBeVisible();
    // Both comparison sides must actually render (HEIC/BMP "before" needs a
    // re-encoded preview in browsers that can't display the original file).
    for (const sel of ['img.after', '.before-wrap img']) {
      await expect
        .poll(() => card.locator(sel).evaluate((el) => (el as HTMLImageElement).naturalWidth))
        .toBeGreaterThan(0);
    }
    const beforeSrc = await card.locator('.before-wrap img').getAttribute('src');
    const afterSrc = await card.locator('img.after').getAttribute('src');
    expect(beforeSrc, 'before and after must show different images').not.toBe(afterSrc);
    // The (β, γ, σ) readout proves the full pipeline ran.
    await expect(card.locator('.task-meta')).toContainText('яркость');
  });
}

test('progress is monotonic and ends at done/100', async ({ page }) => {
  test.skip(!existsSync(fixture('generated/small.jpg')), 'fixtures missing');
  await openAndTrack(page);
  await page.setInputFiles('#file-input', fixture('generated/small.jpg'));
  await expect(page.locator('.task-status.done').first()).toBeVisible({ timeout: 60_000 });

  const evts = await events(page);
  expect(evts.length).toBeGreaterThan(2);
  const ids = new Set(evts.map((e) => e.taskId));
  expect(ids.size).toBe(1);
  for (let i = 1; i < evts.length; i++) {
    expect(evts[i].progress, `event ${i} progress`).toBeGreaterThanOrEqual(evts[i - 1].progress);
  }
  const last = evts[evts.length - 1];
  expect(last.status).toBe('done');
  expect(last.progress).toBe(100);
});

test('cancelling mid-processing yields cancelled status', async ({ page }) => {
  test.skip(!existsSync(fixture('generated/big.png')), 'fixtures missing');
  await openAndTrack(page);
  await page.setInputFiles('#file-input', fixture('generated/big.png'));
  const card = page.locator('.task').first();
  // Wait until the worker is actually processing, then cancel.
  await expect(card.locator('.task-status')).not.toContainText('в очереди', { timeout: 30_000 });
  await card.locator('button.cancel').click();
  await expect(card.locator('.task-status.cancelled')).toBeVisible({ timeout: 15_000 });

  const evts = await events(page);
  expect(evts[evts.length - 1].status).toBe('cancelled');
});

test('15 Mpx JPEG completes within the 30 s budget', async ({ page }) => {
  test.skip(!existsSync(fixture('generated/big.jpg')), 'fixtures missing');
  await openAndTrack(page);
  const t0 = Date.now();
  await page.setInputFiles('#file-input', fixture('generated/big.jpg'));
  const card = page.locator('.task').first();
  await expect(card.locator('.task-status.done')).toBeVisible({ timeout: 35_000 });
  const elapsed = Date.now() - t0;
  expect(elapsed, `15 Mpx took ${elapsed} ms`).toBeLessThan(30_000);
  await expect(card.locator('.task-meta')).toContainText('5000×3000');
});

test('queued task can be cancelled instantly', async ({ page }) => {
  test.skip(!existsSync(fixture('generated/big.jpg')), 'fixtures missing');
  await openAndTrack(page);
  // Two tasks: the second sits in the queue while the first processes.
  await page.setInputFiles('#file-input', [
    fixture('generated/big.jpg'),
    fixture('generated/small.jpg'),
  ]);
  const queued = page.locator('.task').first(); // newest card on top
  await queued.locator('button.cancel').click();
  await expect(queued.locator('.task-status.cancelled')).toBeVisible({ timeout: 5_000 });
});
