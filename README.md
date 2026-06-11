# Улучшение изображений в браузере

Система улучшения изображений, работающая **целиком в браузере пользователя**:
компактная CNN (≈198 тыс. параметров, 387 КиБ в fp16) по миниатюре изображения
предсказывает оптимальные коэффициенты коррекции **яркости, контрастности и
цветности**, а быстрый вспомогательный алгоритм (LUT + попиксельное смешивание
с люмой) применяет их к полноразмерному изображению в Web Worker.

**Демо: см. ссылку на GitHub Pages в описании репозитория.**

## Соответствие требованиям

| Требование | Выполнение |
|---|---|
| Все массовые современные браузеры | Только стабильные API (Web Worker, OffscreenCanvas 2d, createImageBitmap). Движки: WebKit (Safari), Chromium (Chrome, Яндекс Браузер), Gecko (Zen). E2E-тесты Playwright на всех трёх движках + автоматическая страница проверки совместимости (`probe.html`) |
| Объём кода ≤ 10 МБ | **≈2.2 МиБ** всего, из них ~1.4 МиБ — ленивый чанк libheif, который скачивается только при обработке HEIC. CI-гейт на 8 МиБ (`scripts/size-budget-check.mjs`) |
| Изображения до 15 Мпк | Тайловая обработка in-place (один RGBA-буфер 60 МБ), transferable-передача без копий, e2e-тест на 15 Мпк |
| Макс. время ≤ 30 с | Watchdog на главном потоке; e2e-тест проверяет 15 Мпк < 30 с |
| Среднее время ≈ 5 с | Типичное изображение (≤12 Мпк) обрабатывается за 1–3 с; замеры — режим `?bench` |
| Форматы JPG, PNG, HEIC, BMP | JPG/PNG — нативно; BMP — нативно + собственный запасной декодер; HEIC — нативно в Safari, иначе libheif (WASM, лениво) |
| Асинхронный режим | Вся обработка в выделенном Web Worker, главный поток только диспетчеризует события |

## API модуля

```ts
import { ImageEnhancer } from './lib';

const enhancer = new ImageEnhancer();           // { brain: 'ml' | 'heuristic' }

const taskId = enhancer.submitTask(file);        // постановка задачи
enhancer.getStatus(taskId);                      // { status, progress, error? }
enhancer.cancelTask(taskId);                     // { success, reason? }
enhancer.getResult(taskId);                      // Blob (когда status === 'done')

enhancer.addEventListener('taskstatuschange', (e) => {
  const { taskId, status, progress } = e.detail; // событие изменения статуса
});
```

Статусы: `queued → decoding → analyzing → enhancing → encoding → done`
(или `error` / `cancelled`). Прогресс 0–100.

## Как это работает

1. **Декодирование** (worker): определение формата по магическим байтам →
   `createImageBitmap` → RGBA-буфер + миниатюра 224×224.
2. **Анализ**: CNN (собственный инференс-движок на TypeScript, ~400 строк,
   валидируется послойными golden-векторами из Keras) предсказывает
   `o ∈ [−1,1]³` → коэффициенты (β, γ, σ).
3. **Улучшение**: яркость+контраст сворачиваются в LUT из 256 значений,
   цветность смешивается с люмой Rec.601 попиксельно; обработка по тайлам
   ~1 Мпк с прогрессом и точками отмены.
4. **Кодирование**: `convertToBlob` (JPEG для JPG/HEIC, PNG для PNG/BMP).

Каноническая математика описана в `docs/color-math.md` и реализована дважды
(Python для обучения, TypeScript для рантайма) с golden-тестом на совпадение
±1 младший бит.

## Обучение модели

Самообучение на синтетической деградации (DIV2K): случайные искажения
яркости/контраста/цветности + шум + JPEG-пережатие; модель учится их
обращать. Лосс — L1 в пространстве изображения + вспомогательный MSE по
параметрам с затуханием.

```bash
scripts/fetch-dataset.sh                 # DIV2K valid (~450 МБ)
cd ml
uv sync --extra train --extra eval
uv run --extra train python -m imageenh.train --data-dir data/DIV2K_valid_HR --epochs 40
uv run --extra train python -m imageenh.export_weights      # → web/public/model/
uv run --extra train python -m imageenh.export_golden nn    # → web/tests/golden/
```

Оценка качества (отчёт в `docs/eval-report.md`):

```bash
ml/.venv/bin/python scripts/make-reference-pool.py
cd ml && uv run --extra train --extra eval python -m imageenh.evaluate
```

## Разработка

```bash
cd web
npm install
npm run dev          # демо на localhost:5173
npm test             # vitest: golden-контракты + юнит-тесты
npm run build        # tsc + vite build → dist/
npm run size         # проверка бюджета размера
npx playwright test  # e2e на chromium + webkit + firefox
```

Структура: `web/src/lib` — модуль обработки, `web/src/demo` — демо-страница,
`ml/` — обучение и оценка (Python, uv), `docs/` — ТЗ и архитектурные решения,
`reference-images/` — эталонный пул.
