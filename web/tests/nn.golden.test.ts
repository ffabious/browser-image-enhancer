/**
 * Layer-by-layer validation of the hand-rolled inference engine against
 * Keras golden vectors (generated with fp16-rounded weights, the same
 * rounding the deployed weights.bin carries).
 * Regenerate: cd ml && uv run --extra train python -m imageenh.export_golden nn
 */
import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { conv2d, dense, globalAveragePool, type Tensor3 } from '../src/lib/nn/engine';
import { fp16ToFp32, type ModelManifest } from '../src/lib/nn/weights';

const goldenPath = fileURLToPath(new URL('./golden/nn.json', import.meta.url));
const manifestPath = fileURLToPath(new URL('../public/model/manifest.json', import.meta.url));
const weightsPath = fileURLToPath(new URL('../public/model/weights.bin', import.meta.url));

const available = existsSync(goldenPath) && existsSync(manifestPath) && existsSync(weightsPath);

// Mixed fp16-weight/fp32-accumulator math: small order-of-summation drift is
// expected; anything beyond this tolerance is a real engine bug.
const ATOL = 2e-3;

describe.skipIf(!available)('NN engine vs Keras golden vectors', () => {
  if (!available) return;

  const golden = JSON.parse(readFileSync(goldenPath, 'utf-8')) as {
    input: string;
    layers: { name: string; shape: number[]; output: string }[];
  };
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf-8')) as ModelManifest;
  const weights = fp16ToFp32(new Uint16Array(readFileSync(weightsPath).buffer.slice(0)));

  const f32 = (b64: string): Float32Array => {
    const buf = Buffer.from(b64, 'base64');
    return new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  };

  it('matches every layer output', () => {
    const inputData = f32(golden.input);
    let t: Tensor3 = {
      data: inputData,
      h: manifest.inputSize,
      w: manifest.inputSize,
      c: 3,
    };
    let vec: Float32Array | null = null;
    let gi = 0;

    for (const layer of manifest.layers) {
      let actual: Float32Array;
      if (layer.type === 'conv') {
        const wLen = layer.kernel * layer.kernel * layer.inC * layer.outC;
        t = conv2d(
          t,
          weights.subarray(layer.wOffset, layer.wOffset + wLen),
          weights.subarray(layer.bOffset, layer.bOffset + layer.outC),
          layer.kernel,
          layer.stride,
          layer.outC,
          layer.activation,
        );
        actual = t.data;
      } else if (layer.type === 'gap') {
        vec = globalAveragePool(t);
        actual = vec;
      } else {
        vec = dense(
          vec!,
          weights.subarray(layer.wOffset, layer.wOffset + layer.inDim * layer.units),
          weights.subarray(layer.bOffset, layer.bOffset + layer.units),
          layer.units,
          layer.activation,
        );
        actual = vec;
      }

      const g = golden.layers[gi++];
      const expected = f32(g.output);
      expect(actual.length, `layer ${g.name} size`).toBe(expected.length);
      let maxDiff = 0;
      let at = -1;
      for (let i = 0; i < actual.length; i++) {
        const d = Math.abs(actual[i] - expected[i]);
        if (d > maxDiff) {
          maxDiff = d;
          at = i;
        }
      }
      expect(
        maxDiff,
        `layer ${g.name}: max |Δ|=${maxDiff} at ${at} (got ${actual[at]}, want ${expected[at]})`,
      ).toBeLessThanOrEqual(ATOL);
    }
    expect(gi).toBe(golden.layers.length);
  });
});
