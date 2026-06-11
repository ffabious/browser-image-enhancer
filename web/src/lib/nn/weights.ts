/**
 * Model weight loading: manifest.json describes the architecture and element
 * offsets into weights.bin (little-endian float16, exported by
 * ml/src/imageenh/export_weights.py). fp16 → fp32 upcast is exact.
 */

import type { Activation } from './engine';

export interface ConvSpec {
  type: 'conv';
  kernel: number;
  stride: number;
  inC: number;
  outC: number;
  activation: Activation;
  /** Element offsets into the decoded fp32 array. */
  wOffset: number;
  bOffset: number;
}

export interface DenseSpec {
  type: 'dense';
  inDim: number;
  units: number;
  activation: Activation;
  wOffset: number;
  bOffset: number;
}

export interface GapSpec {
  type: 'gap';
}

export type LayerSpec = ConvSpec | DenseSpec | GapSpec;

export interface ModelManifest {
  version: number;
  inputSize: number;
  totalElements: number;
  layers: LayerSpec[];
}

export interface LoadedModel {
  manifest: ModelManifest;
  weights: Float32Array;
}

export function fp16ToFp32(bits: Uint16Array): Float32Array {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) {
    const b = bits[i];
    const sign = b & 0x8000 ? -1 : 1;
    const exp = (b & 0x7c00) >> 10;
    const frac = b & 0x03ff;
    if (exp === 0) out[i] = sign * 2 ** -14 * (frac / 1024);
    else if (exp === 31) out[i] = frac ? NaN : sign * Infinity;
    else out[i] = sign * 2 ** (exp - 15) * (1 + frac / 1024);
  }
  return out;
}

export async function loadModel(baseUrl: string): Promise<LoadedModel> {
  const [manifestRes, weightsRes] = await Promise.all([
    fetch(new URL('manifest.json', baseUrl)),
    fetch(new URL('weights.bin', baseUrl)),
  ]);
  if (!manifestRes.ok) throw new Error(`model manifest fetch failed: ${manifestRes.status}`);
  if (!weightsRes.ok) throw new Error(`model weights fetch failed: ${weightsRes.status}`);
  const manifest = (await manifestRes.json()) as ModelManifest;
  const raw = await weightsRes.arrayBuffer();
  const weights = fp16ToFp32(new Uint16Array(raw));
  if (weights.length !== manifest.totalElements) {
    throw new Error(
      `weights.bin has ${weights.length} elements, manifest expects ${manifest.totalElements}`,
    );
  }
  return { manifest, weights };
}
