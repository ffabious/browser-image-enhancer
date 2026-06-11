/**
 * THE contract test: the TS enhancement math must match the Python
 * implementation (which generated these fixtures) within ±1 LSB per channel.
 * Regenerate fixtures with: cd ml && uv run python -m imageenh.export_golden colormath
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { enhanceRgba } from '../src/lib/apply/enhance';

interface GoldenCase {
  width: number;
  height: number;
  factors: [number, number, number];
  input: string;
  expected: string;
}

const fixture = JSON.parse(
  readFileSync(fileURLToPath(new URL('./golden/colormath.json', import.meta.url)), 'utf-8'),
) as { tolerance: number; cases: GoldenCase[] };

function b64ToBytes(b64: string): Uint8ClampedArray {
  return new Uint8ClampedArray(Buffer.from(b64, 'base64'));
}

describe('colormath golden fixtures (Python ↔ TS contract)', () => {
  fixture.cases.forEach((c, idx) => {
    const [beta, gamma, sigma] = c.factors;
    it(`case ${idx}: β=${beta.toFixed(3)} γ=${gamma.toFixed(3)} σ=${sigma.toFixed(3)}`, () => {
      const data = b64ToBytes(c.input);
      const expected = b64ToBytes(c.expected);
      enhanceRgba(data, { brightness: beta, contrast: gamma, saturation: sigma });

      expect(data.length).toBe(expected.length);
      let maxDiff = 0;
      let firstBad = -1;
      for (let i = 0; i < data.length; i++) {
        const d = Math.abs(data[i] - expected[i]);
        if (d > maxDiff) {
          maxDiff = d;
          firstBad = i;
        }
      }
      expect(
        maxDiff,
        `max |Δ|=${maxDiff} at byte ${firstBad} (got ${data[firstBad]}, want ${expected[firstBad]})`,
      ).toBeLessThanOrEqual(fixture.tolerance);
    });
  });
});
