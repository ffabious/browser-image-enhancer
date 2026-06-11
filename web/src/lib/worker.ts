/**
 * The processing worker: decode → analyze → enhance (tiled, in place) →
 * encode. One task at a time; cancellation flags are checked at every yield
 * point between tiles and stages.
 */

import { decodeImage } from './codec/decode';
import { encodeImage } from './codec/encode';
import { buildLut, enhanceTile, factorsFromOutput, type Factors } from './apply/enhance';
import { heuristicFactors } from './analyze/heuristic';
import { configureModel, predictFactors, preloadModel } from './nn/predict';
import type { BrainKind, TaskStatus, WorkerRequest, WorkerResponse } from './types';

const TILE_PIXELS = 1 << 20; // ~1 Mpx per tile → ~15 progress ticks on 15 Mpx

const ctx = self as unknown as {
  postMessage(msg: WorkerResponse, transfer?: Transferable[]): void;
  addEventListener(type: 'message', fn: (e: MessageEvent<WorkerRequest>) => void): void;
};

const cancelled = new Set<string>();

class CancelledError extends Error {}

function post(msg: WorkerResponse): void {
  ctx.postMessage(msg);
}

/** Macrotask yield: lets queued 'cancel' messages be handled mid-task. */
function tick(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

async function checkpoint(taskId: string, status: TaskStatus, progress: number): Promise<void> {
  post({ type: 'progress', taskId, status, progress });
  await tick();
  if (cancelled.has(taskId)) throw new CancelledError();
}

async function process(taskId: string, buffer: ArrayBuffer, brain: BrainKind): Promise<void> {
  const timings: Record<string, number> = {};
  let t = performance.now();
  const lap = (stage: string) => {
    const now = performance.now();
    timings[stage] = Math.round(now - t);
    t = now;
  };

  await checkpoint(taskId, 'decoding', 0);
  const decoded = await decodeImage(buffer);
  lap('decode');

  await checkpoint(taskId, 'analyzing', 20);
  let factors: Factors;
  let usedBrain = brain;
  if (brain === 'ml') {
    try {
      factors = factorsFromOutput(await predictFactors(decoded.thumb));
    } catch (err) {
      // Model unavailable (e.g. weights not deployed) — fall back gracefully.
      console.warn('ML brain failed, falling back to heuristic:', err);
      factors = heuristicFactors(decoded.thumb);
      usedBrain = 'heuristic';
    }
  } else {
    factors = heuristicFactors(decoded.thumb);
  }
  lap('analyze');

  const data = decoded.image.data;
  const totalPixels = data.length / 4;
  const lut = buildLut(factors.brightness, factors.contrast);
  for (let start = 0; start < totalPixels; start += TILE_PIXELS) {
    const progress = 35 + Math.round((start / totalPixels) * 55);
    await checkpoint(taskId, 'enhancing', progress);
    enhanceTile(data, start, Math.min(start + TILE_PIXELS, totalPixels), lut, factors.saturation);
  }
  lap('enhance');

  await checkpoint(taskId, 'encoding', 90);
  const blob = await encodeImage(decoded.image, decoded.format);
  lap('encode');

  await checkpoint(taskId, 'encoding', 99);
  post({
    type: 'done',
    taskId,
    blob,
    details: {
      factors,
      timings,
      width: decoded.width,
      height: decoded.height,
      brain: usedBrain,
    },
  });
}

ctx.addEventListener('message', (e) => {
  const msg = e.data;
  if (msg.type === 'init') {
    configureModel(msg.modelBaseUrl);
    // Warm the weights fetch before the first task arrives; swallow errors
    // here — they resurface (with heuristic fallback) on first use.
    preloadModel()?.catch(() => {});
    return;
  }
  if (msg.type === 'cancel') {
    cancelled.add(msg.taskId);
    return;
  }
  if (msg.type === 'process') {
    void process(msg.taskId, msg.buffer, msg.brain)
      .catch((err: unknown) => {
        if (err instanceof CancelledError) post({ type: 'cancelled', taskId: msg.taskId });
        else post({ type: 'error', taskId: msg.taskId, message: String(err) });
      })
      .finally(() => cancelled.delete(msg.taskId));
  }
});
