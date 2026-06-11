/**
 * Public API of the enhancement module (see docs/spec.md §4).
 *
 *   const enhancer = new ImageEnhancer();
 *   const id = enhancer.submitTask(file);
 *   enhancer.addEventListener('taskstatuschange', (e) => ...);
 *   const blob = enhancer.getResult(id);
 */

import type {
  BrainKind,
  CancelResult,
  TaskDetails,
  TaskStatus,
  TaskStatusChangeDetail,
  TaskStatusInfo,
  WorkerRequest,
  WorkerResponse,
} from './types';
import type { Factors } from './apply/enhance';

export type { BrainKind, CancelResult, TaskDetails, TaskStatus, TaskStatusInfo, Factors };
export type { TaskStatusChangeDetail };

const TERMINAL: ReadonlySet<TaskStatus> = new Set(['done', 'error', 'cancelled']);

export interface ImageEnhancerOptions {
  /** Parameter predictor: 'ml' (default) or 'heuristic' baseline. */
  brain?: BrainKind;
  /** Hard per-task timeout, ms (brief requirement: 30 s). */
  maxTaskMs?: number;
  /** Absolute or page-relative URL of the directory holding model files. */
  modelBaseUrl?: string;
}

interface Task {
  id: string;
  status: TaskStatus;
  progress: number;
  error?: string;
  buffer?: ArrayBuffer;
  brain: BrainKind;
  result?: Blob;
  details?: TaskDetails;
}

let nextId = 0;

export class ImageEnhancer extends EventTarget {
  private readonly tasks = new Map<string, Task>();
  private readonly queue: string[] = [];
  private readonly options: Required<ImageEnhancerOptions>;
  private worker: Worker | null = null;
  private activeTaskId: string | null = null;
  private watchdog: ReturnType<typeof setTimeout> | null = null;

  constructor(options: ImageEnhancerOptions = {}) {
    super();
    this.options = {
      brain: options.brain ?? 'ml',
      maxTaskMs: options.maxTaskMs ?? 30_000,
      modelBaseUrl:
        options.modelBaseUrl ??
        new URL('model/', typeof document !== 'undefined' ? document.baseURI : self.location.href)
          .href,
    };
  }

  /** Submit an image for enhancement; returns the task id immediately. */
  submitTask(image: File | Blob | ArrayBuffer, brain?: BrainKind): string {
    const id = `task-${++nextId}-${Math.random().toString(36).slice(2, 8)}`;
    const task: Task = {
      id,
      status: 'queued',
      progress: 0,
      brain: brain ?? this.options.brain,
    };
    this.tasks.set(id, task);

    const bufferPromise =
      image instanceof ArrayBuffer ? Promise.resolve(image) : image.arrayBuffer();
    bufferPromise
      .then((buffer) => {
        if (task.status !== 'queued') return; // cancelled while reading
        task.buffer = buffer;
        this.queue.push(id);
        this.pump();
      })
      .catch((err: unknown) => {
        this.transition(task, 'error', task.progress, `failed to read input: ${String(err)}`);
      });

    // Make the initial 'queued' state observable via the event API.
    queueMicrotask(() => {
      if (task.status === 'queued') this.emit(task);
    });
    return id;
  }

  /** Current status and progress (0–100) of a task. */
  getStatus(taskId: string): TaskStatusInfo {
    const task = this.mustGet(taskId);
    return { status: task.status, progress: task.progress, error: task.error };
  }

  /** Cancel a queued or running task. */
  cancelTask(taskId: string): CancelResult {
    const task = this.mustGet(taskId);
    if (TERMINAL.has(task.status)) {
      return { success: false, reason: `task is already ${task.status}` };
    }
    if (taskId === this.activeTaskId && this.worker) {
      // The worker confirms with a 'cancelled' message at its next checkpoint.
      this.post({ type: 'cancel', taskId });
    } else {
      const qi = this.queue.indexOf(taskId);
      if (qi >= 0) this.queue.splice(qi, 1);
      task.buffer = undefined;
      this.transition(task, 'cancelled', task.progress);
    }
    return { success: true };
  }

  /** The enhanced image; only valid once status is 'done'. */
  getResult(taskId: string): Blob {
    const task = this.mustGet(taskId);
    if (task.status !== 'done' || !task.result) {
      throw new Error(`task ${taskId} is ${task.status}, result is not available`);
    }
    return task.result;
  }

  /** Extra diagnostics (factors, timings, dimensions) — demo/bench helper. */
  getDetails(taskId: string): TaskDetails | undefined {
    return this.mustGet(taskId).details;
  }

  /** Terminate the worker and drop all tasks. */
  dispose(): void {
    this.worker?.terminate();
    this.worker = null;
    this.clearWatchdog();
    this.tasks.clear();
    this.queue.length = 0;
    this.activeTaskId = null;
  }

  private mustGet(taskId: string): Task {
    const task = this.tasks.get(taskId);
    if (!task) throw new Error(`unknown task id: ${taskId}`);
    return task;
  }

  private emit(task: Task): void {
    this.dispatchEvent(
      new CustomEvent<TaskStatusChangeDetail>('taskstatuschange', {
        detail: { taskId: task.id, status: task.status, progress: task.progress },
      }),
    );
  }

  private transition(task: Task, status: TaskStatus, progress: number, error?: string): void {
    if (TERMINAL.has(task.status)) return;
    const changed = task.status !== status || task.progress !== progress;
    task.status = status;
    task.progress = progress;
    if (error !== undefined) task.error = error;
    if (changed) this.emit(task);
  }

  private ensureWorker(): Worker {
    if (!this.worker) {
      this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      this.worker.addEventListener('message', (e: MessageEvent<WorkerResponse>) =>
        this.onWorkerMessage(e.data),
      );
      this.worker.addEventListener('error', (e) => {
        this.failActive(`worker error: ${e.message}`);
      });
      this.post({ type: 'init', modelBaseUrl: this.options.modelBaseUrl });
    }
    return this.worker;
  }

  private post(msg: WorkerRequest, transfer?: Transferable[]): void {
    this.ensureWorker().postMessage(msg, transfer ?? []);
  }

  private pump(): void {
    if (this.activeTaskId) return;
    let task: Task | undefined;
    while (this.queue.length > 0) {
      const id = this.queue.shift()!;
      const t = this.tasks.get(id);
      if (t && !TERMINAL.has(t.status) && t.buffer) {
        task = t;
        break;
      }
    }
    if (!task) return;

    this.activeTaskId = task.id;
    const buffer = task.buffer!;
    task.buffer = undefined;
    this.post({ type: 'process', taskId: task.id, buffer, brain: task.brain }, [buffer]);
    this.watchdog = setTimeout(() => this.onTimeout(), this.options.maxTaskMs);
  }

  private clearWatchdog(): void {
    if (this.watchdog) clearTimeout(this.watchdog);
    this.watchdog = null;
  }

  private finishActive(): void {
    this.clearWatchdog();
    this.activeTaskId = null;
    this.pump();
  }

  private failActive(message: string): void {
    const task = this.activeTaskId ? this.tasks.get(this.activeTaskId) : undefined;
    if (task) this.transition(task, 'error', task.progress, message);
    this.finishActive();
  }

  private onTimeout(): void {
    // Hard 30 s cap: kill the worker (also reclaims its memory) and respawn
    // lazily for the next task.
    this.worker?.terminate();
    this.worker = null;
    this.failActive(`task exceeded ${this.options.maxTaskMs} ms and was aborted`);
  }

  private onWorkerMessage(msg: WorkerResponse): void {
    const task = this.tasks.get(msg.taskId);
    if (!task) return;
    switch (msg.type) {
      case 'progress':
        this.transition(task, msg.status, msg.progress);
        break;
      case 'done':
        task.result = msg.blob;
        task.details = msg.details;
        this.transition(task, 'done', 100);
        this.finishActive();
        break;
      case 'error':
        this.transition(task, 'error', task.progress, msg.message);
        this.finishActive();
        break;
      case 'cancelled':
        this.transition(task, 'cancelled', task.progress);
        this.finishActive();
        break;
    }
  }
}
