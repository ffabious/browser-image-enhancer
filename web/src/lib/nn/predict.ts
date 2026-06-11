/**
 * ML brain: run the loaded CNN on the 224×224 thumbnail and return the raw
 * model output o ∈ [-1,1]^3 (mapped to factors by factorsFromOutput).
 */

import { conv2d, dense, globalAveragePool, type Tensor3 } from './engine';
import { loadModel, type LoadedModel } from './weights';

let modelBaseUrl: string | null = null;
let modelPromise: Promise<LoadedModel> | null = null;

/** Set by the main thread (init message) — absolute URL of the model dir. */
export function configureModel(baseUrl: string): void {
  modelBaseUrl = baseUrl;
}

export function preloadModel(): Promise<LoadedModel> | null {
  if (!modelBaseUrl) return null;
  modelPromise ??= loadModel(modelBaseUrl);
  return modelPromise;
}

export function runModel(model: LoadedModel, input: Tensor3): Float32Array {
  let t = input;
  let vec: Float32Array | null = null;
  for (const layer of model.manifest.layers) {
    if (layer.type === 'conv') {
      const wLen = layer.kernel * layer.kernel * layer.inC * layer.outC;
      t = conv2d(
        t,
        model.weights.subarray(layer.wOffset, layer.wOffset + wLen),
        model.weights.subarray(layer.bOffset, layer.bOffset + layer.outC),
        layer.kernel,
        layer.stride,
        layer.outC,
        layer.activation,
      );
    } else if (layer.type === 'gap') {
      vec = globalAveragePool(t);
    } else {
      if (!vec) throw new Error('dense layer before gap');
      vec = dense(
        vec,
        model.weights.subarray(layer.wOffset, layer.wOffset + layer.inDim * layer.units),
        model.weights.subarray(layer.bOffset, layer.bOffset + layer.units),
        layer.units,
        layer.activation,
      );
    }
  }
  if (!vec) throw new Error('model produced no vector output');
  return vec;
}

export async function predictFactors(thumb: ImageData): Promise<Float32Array> {
  const promise = preloadModel();
  if (!promise) throw new Error('model URL not configured');
  const model = await promise;

  const size = model.manifest.inputSize;
  if (thumb.width !== size || thumb.height !== size) {
    throw new Error(`expected ${size}×${size} thumb, got ${thumb.width}×${thumb.height}`);
  }
  const rgba = thumb.data;
  const n = size * size;
  const input = new Float32Array(n * 3);
  for (let p = 0; p < n; p++) {
    input[p * 3] = rgba[p * 4] / 255;
    input[p * 3 + 1] = rgba[p * 4 + 1] / 255;
    input[p * 3 + 2] = rgba[p * 4 + 2] / 255;
  }
  return runModel(model, { data: input, h: size, w: size, c: 3 });
}
