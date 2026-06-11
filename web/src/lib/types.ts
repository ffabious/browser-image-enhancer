import type { Factors } from './apply/enhance';

export type TaskStatus =
  | 'queued'
  | 'decoding'
  | 'analyzing'
  | 'enhancing'
  | 'encoding'
  | 'done'
  | 'error'
  | 'cancelled';

/** Which parameter-prediction "brain" to use. */
export type BrainKind = 'ml' | 'heuristic';

export interface TaskStatusInfo {
  status: TaskStatus;
  progress: number;
  error?: string;
}

export interface TaskStatusChangeDetail {
  taskId: string;
  status: TaskStatus;
  progress: number;
}

export interface CancelResult {
  success: boolean;
  reason?: string;
}

/** Per-task diagnostics surfaced to the demo UI (not part of the core brief API). */
export interface TaskDetails {
  factors?: Factors;
  timings?: Record<string, number>;
  width?: number;
  height?: number;
  brain?: BrainKind;
  /** Re-encoded original, present only when the input file itself cannot be
      rendered by an <img> in this browser (HEIC/BMP fallback decode paths). */
  beforePreview?: Blob;
}

export type WorkerRequest =
  | { type: 'init'; modelBaseUrl: string }
  | { type: 'process'; taskId: string; buffer: ArrayBuffer; brain: BrainKind }
  | { type: 'cancel'; taskId: string };

export type WorkerResponse =
  | { type: 'progress'; taskId: string; status: TaskStatus; progress: number }
  | {
      type: 'done';
      taskId: string;
      blob: Blob;
      details: Required<Omit<TaskDetails, 'brain' | 'beforePreview'>> &
        Pick<TaskDetails, 'beforePreview'> & { brain: BrainKind };
    }
  | { type: 'error'; taskId: string; message: string }
  | { type: 'cancelled'; taskId: string };
