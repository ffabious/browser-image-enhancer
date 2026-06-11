/**
 * Heuristic parameter predictor — the non-ML baseline brain (M3), kept as a
 * comparison mode in the demo. Percentile autocontrast + mean-luma exposure +
 * mean-chroma saturation, all damped toward identity so good images are left
 * mostly alone.
 */

import type { Factors } from '../apply/enhance';

const clamp = (x: number, lo: number, hi: number) => Math.min(Math.max(x, lo), hi);
/** Pull a multiplicative factor toward 1 in log space (power < 1 = damping). */
const damp = (factor: number, power: number) => Math.exp(Math.log(factor) * power);

export function heuristicFactors(thumb: ImageData): Factors {
  const d = thumb.data;
  const n = d.length / 4;

  const hist = new Float64Array(256);
  let lumaSum = 0;
  let chromaSum = 0;
  for (let p = 0; p < n; p++) {
    const i = p * 4;
    const r = d[i] / 255;
    const g = d[i + 1] / 255;
    const b = d[i + 2] / 255;
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    lumaSum += y;
    chromaSum += (Math.abs(r - y) + Math.abs(g - y) + Math.abs(b - y)) / 3;
    hist[Math.min(255, (y * 255 + 0.5) | 0)]++;
  }
  const meanLuma = lumaSum / n;
  const meanChroma = chromaSum / n;

  const percentile = (q: number): number => {
    const target = q * n;
    let acc = 0;
    for (let v = 0; v < 256; v++) {
      acc += hist[v];
      if (acc >= target) return v / 255;
    }
    return 1;
  };
  const p5 = percentile(0.05);
  const p95 = percentile(0.95);

  // Exposure: aim for mid-gray-ish mean luma.
  const brightness = clamp(damp(clamp(0.45 / Math.max(meanLuma, 0.02), 0.5, 2), 0.7), 0.5, 2);
  // Contrast: aim for a healthy 5–95 percentile spread after the gain.
  const spread = Math.max((p95 - p5) * brightness, 0.05);
  const contrast = clamp(damp(clamp(0.8 / spread, 0.5, 2), 0.7), 0.6, 1.8);
  // Saturation: aim for a typical mean chroma, gently.
  const saturation = clamp(damp(clamp(0.09 / Math.max(meanChroma, 0.005), 0.4, 2.5), 0.6), 0.6, 1.9);

  return { brightness, contrast, saturation };
}
