/**
 * Canonical color math — the TS twin of ml/src/imageenh/colormath.py.
 * Implements docs/color-math.md exactly; verified against Python golden
 * fixtures in tests/colormath.golden.test.ts (±1 LSB).
 */

const LN2 = Math.log(2);
const LN25 = Math.log(2.5);

const LUMA_R = 0.299;
const LUMA_G = 0.587;
const LUMA_B = 0.114;

export interface Factors {
  brightness: number;
  contrast: number;
  saturation: number;
}

/** Map model output o ∈ [-1, 1]^3 to multiplicative factors. */
export function factorsFromOutput(o: ArrayLike<number>): Factors {
  return {
    brightness: Math.exp(o[0] * LN2),
    contrast: Math.exp(o[1] * LN2),
    saturation: Math.exp(o[2] * LN25),
  };
}

/**
 * Build the fused brightness+contrast LUT: lut[v] = (v/255)·β·γ + 0.5·(1−γ).
 * Stored unclamped in Float32Array; the single clamp happens at quantization.
 */
export function buildLut(beta: number, gamma: number): Float32Array {
  const lut = new Float32Array(256);
  const scale = (beta * gamma) / 255;
  const offset = 0.5 * (1 - gamma);
  for (let v = 0; v < 256; v++) lut[v] = v * scale + offset;
  return lut;
}

/**
 * Enhance one tile of an RGBA buffer in place.
 * `start`/`end` are pixel indices (not byte offsets). Alpha is untouched.
 */
export function enhanceTile(
  data: Uint8ClampedArray,
  start: number,
  end: number,
  lut: Float32Array,
  sigma: number,
): void {
  for (let p = start; p < end; p++) {
    const i = p * 4;
    const r = lut[data[i]];
    const g = lut[data[i + 1]];
    const b = lut[data[i + 2]];
    const y = LUMA_R * r + LUMA_G * g + LUMA_B * b;
    // Final clamp + round-half-up quantization (Math.floor(x*255+0.5)).
    let v = y + (r - y) * sigma;
    data[i] = v <= 0 ? 0 : v >= 1 ? 255 : (v * 255 + 0.5) | 0;
    v = y + (g - y) * sigma;
    data[i + 1] = v <= 0 ? 0 : v >= 1 ? 255 : (v * 255 + 0.5) | 0;
    v = y + (b - y) * sigma;
    data[i + 2] = v <= 0 ? 0 : v >= 1 ? 255 : (v * 255 + 0.5) | 0;
  }
}

/** Convenience: enhance a full RGBA buffer synchronously (used in tests). */
export function enhanceRgba(data: Uint8ClampedArray, f: Factors): void {
  const lut = buildLut(f.brightness, f.contrast);
  enhanceTile(data, 0, data.length / 4, lut, f.saturation);
}
