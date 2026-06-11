import { ImageEnhancer, type BrainKind, type TaskStatus } from '../lib';

const STATUS_RU: Record<TaskStatus, string> = {
  queued: 'в очереди',
  decoding: 'декодирование',
  analyzing: 'анализ',
  enhancing: 'улучшение',
  encoding: 'кодирование',
  done: 'готово',
  error: 'ошибка',
  cancelled: 'отменено',
};

const SAMPLES = ['dark.jpg', 'hazy.jpg', 'dull.jpg'];

const enhancer = new ImageEnhancer();

// Exposed for e2e tests (event-order assertions) and console experiments.
declare global {
  interface Window {
    __enhancer: ImageEnhancer;
  }
}
window.__enhancer = enhancer;

const tasksEl = document.getElementById('tasks')!;
const dropzone = document.getElementById('dropzone')!;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const brainSelect = document.getElementById('brain-select') as HTMLSelectElement;
const samplesEl = document.getElementById('samples')!;

interface TaskView {
  card: HTMLElement;
  statusEl: HTMLElement;
  barEl: HTMLElement;
  metaEl: HTMLElement;
  actionsEl: HTMLElement;
  originalUrl: string;
  name: string;
  startedAt: number;
}

const views = new Map<string, TaskView>();

function currentBrain(): BrainKind {
  return brainSelect.value === 'heuristic' ? 'heuristic' : 'ml';
}

function submit(file: File | Blob, name: string): string {
  const id = enhancer.submitTask(file, currentBrain());
  const card = document.createElement('article');
  card.className = 'task';
  card.innerHTML = `
    <div class="task-head">
      <span class="task-name"></span>
      <span class="task-status"></span>
    </div>
    <div class="progress"><div></div></div>
    <div class="task-meta"></div>
    <div class="task-actions">
      <button class="btn danger cancel">Отменить</button>
    </div>`;
  card.querySelector('.task-name')!.textContent = name;
  const view: TaskView = {
    card,
    statusEl: card.querySelector('.task-status')!,
    barEl: card.querySelector('.progress > div')!,
    metaEl: card.querySelector('.task-meta')!,
    actionsEl: card.querySelector('.task-actions')!,
    originalUrl: URL.createObjectURL(file),
    name,
    startedAt: performance.now(),
  };
  card.querySelector<HTMLButtonElement>('.cancel')!.addEventListener('click', () => {
    enhancer.cancelTask(id);
  });
  views.set(id, view);
  tasksEl.prepend(card);
  return id;
}

function renderDone(id: string, view: TaskView): void {
  const blob = enhancer.getResult(id);
  const details = enhancer.getDetails(id);
  const resultUrl = URL.createObjectURL(blob);
  const elapsed = ((performance.now() - view.startedAt) / 1000).toFixed(2);

  const compare = document.createElement('div');
  compare.className = 'compare';
  compare.innerHTML = `
    <img class="after" alt="" />
    <div class="before-wrap"><img alt="" /></div>
    <div class="divider"></div>
    <span class="tag before">до</span>
    <span class="tag after">после</span>
    <input type="range" min="0" max="100" value="50" aria-label="Сравнение до/после" />`;
  compare.querySelector<HTMLImageElement>('.after')!.src = resultUrl;
  // HEIC/BMP fallback inputs can't be shown by <img> directly — the worker
  // supplies a re-encoded preview of the original in that case.
  const beforeUrl = details?.beforePreview
    ? URL.createObjectURL(details.beforePreview)
    : view.originalUrl;
  compare.querySelector<HTMLImageElement>('.before-wrap img')!.src = beforeUrl;
  const beforeWrap = compare.querySelector<HTMLElement>('.before-wrap')!;
  const divider = compare.querySelector<HTMLElement>('.divider')!;
  const slider = compare.querySelector<HTMLInputElement>('input')!;
  const setSplit = (pct: number) => {
    beforeWrap.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    divider.style.left = `calc(${pct}% - 1px)`;
  };
  slider.addEventListener('input', () => setSplit(Number(slider.value)));
  setSplit(50);
  view.card.appendChild(compare);

  if (details?.factors) {
    const f = details.factors;
    const t = details.timings ?? {};
    const brainName = details.brain === 'ml' ? 'нейросеть' : 'эвристика';
    view.metaEl.innerHTML =
      `яркость ×${f.brightness.toFixed(3)} · контраст ×${f.contrast.toFixed(3)} · ` +
      `цветность ×${f.saturation.toFixed(3)} · ${details.width}×${details.height} · ${brainName}<br>` +
      `декодирование ${t.decode} мс · анализ ${t.analyze} мс · ` +
      `улучшение ${t.enhance} мс · кодирование ${t.encode} мс · всего ${elapsed} с`;
  }

  const download = document.createElement('a');
  download.href = resultUrl;
  download.download = view.name.replace(/\.\w+$/, '') + '-enhanced' + (blob.type === 'image/png' ? '.png' : '.jpg');
  download.textContent = 'Скачать результат';
  view.actionsEl.replaceChildren(download);
}

enhancer.addEventListener('taskstatuschange', (e) => {
  const { taskId, status, progress } = (e as CustomEvent).detail;
  const view = views.get(taskId);
  if (!view) return;
  view.statusEl.textContent = `${STATUS_RU[status as TaskStatus]} · ${progress}%`;
  view.statusEl.className = `task-status ${status}`;
  view.barEl.style.width = `${progress}%`;
  if (status === 'done') {
    view.barEl.style.background = 'var(--ok)';
    renderDone(taskId, view);
  } else if (status === 'error' || status === 'cancelled') {
    view.barEl.style.background = 'var(--err)';
    const { error } = enhancer.getStatus(taskId);
    if (error) view.metaEl.textContent = error;
    view.actionsEl.replaceChildren();
  }
});

function handleFiles(files: Iterable<File>): void {
  for (const file of files) submit(file, file.name);
}

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer) handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files) handleFiles(fileInput.files);
  fileInput.value = '';
});

for (const sample of SAMPLES) {
  const link = document.createElement('a');
  link.textContent = sample;
  link.addEventListener('click', async () => {
    const res = await fetch(`samples/${sample}`);
    submit(await res.blob(), sample);
  });
  samplesEl.appendChild(link);
}

// --- ?bench: run every sample through both brains, print a timing table ---
async function runBench(): Promise<void> {
  const benchEl = document.getElementById('bench')!;
  benchEl.hidden = false;
  benchEl.innerHTML = '<h2>Бенчмарк</h2><p>Выполняется…</p>';
  const rows: string[] = [];
  for (const brain of ['ml', 'heuristic'] as const) {
    for (const sample of SAMPLES) {
      const res = await fetch(`samples/${sample}`);
      const blob = await res.blob();
      const t0 = performance.now();
      const id = enhancer.submitTask(blob, brain); // bench tasks render no card
      await new Promise<void>((resolve) => {
        const onChange = (e: Event) => {
          const d = (e as CustomEvent).detail;
          if (d.taskId === id && ['done', 'error', 'cancelled'].includes(d.status)) {
            enhancer.removeEventListener('taskstatuschange', onChange);
            resolve();
          }
        };
        enhancer.addEventListener('taskstatuschange', onChange);
      });
      const elapsed = performance.now() - t0;
      const details = enhancer.getDetails(id);
      const t = details?.timings ?? {};
      rows.push(
        `<tr><td>${sample}</td><td>${brain}</td><td>${(elapsed / 1000).toFixed(2)} с</td>` +
          `<td>${t.decode ?? '—'}</td><td>${t.analyze ?? '—'}</td>` +
          `<td>${t.enhance ?? '—'}</td><td>${t.encode ?? '—'}</td></tr>`,
      );
    }
  }
  const mem = (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory;
  benchEl.innerHTML =
    '<h2>Бенчмарк</h2><table><tr><th>Файл</th><th>Алгоритм</th><th>Всего</th>' +
    '<th>Декод., мс</th><th>Анализ, мс</th><th>Улучш., мс</th><th>Кодир., мс</th></tr>' +
    rows.join('') +
    '</table>' +
    (mem ? `<p>Куча JS: ${(mem.usedJSHeapSize / 1048576).toFixed(1)} МБ</p>` : '');
}

if (new URLSearchParams(location.search).has('bench')) {
  void runBench();
}
