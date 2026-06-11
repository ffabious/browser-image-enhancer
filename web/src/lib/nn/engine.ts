/**
 * Hand-rolled CNN inference engine (docs/decisions.md ADR-1).
 * NHWC layout, Float32Array math, five op types: conv2d (stride 2, TF 'same'
 * padding), relu, global average pool, dense, tanh. Validated layer-by-layer
 * against Keras golden vectors in tests/nn.golden.test.ts.
 */

export type Activation = 'relu' | 'tanh' | 'none';

export interface Tensor3 {
  data: Float32Array;
  h: number;
  w: number;
  c: number;
}

function activate(data: Float32Array, activation: Activation): void {
  if (activation === 'relu') {
    for (let i = 0; i < data.length; i++) if (data[i] < 0) data[i] = 0;
  } else if (activation === 'tanh') {
    for (let i = 0; i < data.length; i++) data[i] = Math.tanh(data[i]);
  }
}

/**
 * 2D convolution, NHWC input, kernel layout (kh, kw, inC, outC) — exactly
 * Keras' Conv2D kernel layout. TF 'same' padding: total padding
 * max((outDim-1)·stride + k − inDim, 0), split with the extra row/col at the
 * bottom/right.
 */
export function conv2d(
  input: Tensor3,
  kernel: Float32Array,
  bias: Float32Array,
  k: number,
  stride: number,
  outC: number,
  activation: Activation,
): Tensor3 {
  const { data: src, h, w, c } = input;
  const outH = Math.ceil(h / stride);
  const outW = Math.ceil(w / stride);
  const padH = Math.max((outH - 1) * stride + k - h, 0);
  const padW = Math.max((outW - 1) * stride + k - w, 0);
  const padTop = padH >> 1;
  const padLeft = padW >> 1;

  const out = new Float32Array(outH * outW * outC);
  for (let oy = 0; oy < outH; oy++) {
    const inY0 = oy * stride - padTop;
    for (let ox = 0; ox < outW; ox++) {
      const inX0 = ox * stride - padLeft;
      const outBase = (oy * outW + ox) * outC;
      for (let ky = 0; ky < k; ky++) {
        const iy = inY0 + ky;
        if (iy < 0 || iy >= h) continue;
        for (let kx = 0; kx < k; kx++) {
          const ix = inX0 + kx;
          if (ix < 0 || ix >= w) continue;
          const srcBase = (iy * w + ix) * c;
          const kerBase = (ky * k + kx) * c * outC;
          for (let ic = 0; ic < c; ic++) {
            const v = src[srcBase + ic];
            if (v === 0) continue;
            const kBase = kerBase + ic * outC;
            for (let oc = 0; oc < outC; oc++) {
              out[outBase + oc] += v * kernel[kBase + oc];
            }
          }
        }
      }
      for (let oc = 0; oc < outC; oc++) out[outBase + oc] += bias[oc];
    }
  }
  activate(out, activation);
  return { data: out, h: outH, w: outW, c: outC };
}

/** Global average pool over H and W → vector of length C. */
export function globalAveragePool(input: Tensor3): Float32Array {
  const { data, h, w, c } = input;
  const out = new Float32Array(c);
  const n = h * w;
  for (let p = 0; p < n; p++) {
    const base = p * c;
    for (let ch = 0; ch < c; ch++) out[ch] += data[base + ch];
  }
  for (let ch = 0; ch < c; ch++) out[ch] /= n;
  return out;
}

/** Dense layer; kernel layout (in, out) — exactly Keras' Dense kernel. */
export function dense(
  input: Float32Array,
  kernel: Float32Array,
  bias: Float32Array,
  units: number,
  activation: Activation,
): Float32Array {
  const out = new Float32Array(units);
  for (let u = 0; u < units; u++) out[u] = bias[u];
  for (let i = 0; i < input.length; i++) {
    const v = input[i];
    if (v === 0) continue;
    const base = i * units;
    for (let u = 0; u < units; u++) out[u] += v * kernel[base + u];
  }
  activate(out, activation);
  return out;
}
