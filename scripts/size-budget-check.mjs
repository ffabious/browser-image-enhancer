// Size budget gate (brief: ≤ 10 MB total; CI fails above the 8 MB soft ceiling).
import { readdirSync, statSync } from 'node:fs';
import { join, relative, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', 'web', 'dist');
const LIMIT = 8 * 1024 * 1024;

function* walk(dir) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) yield* walk(p);
    else yield p;
  }
}

let total = 0;
const rows = [];
for (const file of walk(root)) {
  const size = statSync(file).size;
  total += size;
  rows.push([relative(root, file), size]);
}
rows.sort((a, b) => b[1] - a[1]);
for (const [name, size] of rows) {
  console.log(`${String((size / 1024).toFixed(1)).padStart(10)} KiB  ${name}`);
}
console.log('─'.repeat(40));
console.log(`Total: ${(total / 1024 / 1024).toFixed(2)} MiB (limit 8 MiB, brief allows 10 MiB)`);
if (total > LIMIT) {
  console.error('SIZE BUDGET EXCEEDED');
  process.exit(1);
}
