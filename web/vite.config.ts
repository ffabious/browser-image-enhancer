import { defineConfig } from 'vitest/config';
import { resolve } from 'node:path';

export default defineConfig({
  // GitHub Pages serves the site from /<repo>/, so use relative asset paths.
  base: './',
  build: {
    target: 'es2022',
    rollupOptions: {
      input: {
        index: resolve(__dirname, 'index.html'),
        probe: resolve(__dirname, 'probe.html'),
      },
    },
  },
  worker: {
    format: 'es',
  },
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
  },
});
