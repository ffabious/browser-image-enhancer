import { describe, expect, it } from 'vitest';
import { sniffFormat } from '../src/lib/codec/sniff';

function bytes(...parts: (number[] | string)[]): Uint8Array {
  const out: number[] = [];
  for (const p of parts) {
    if (typeof p === 'string') for (const ch of p) out.push(ch.charCodeAt(0));
    else out.push(...p);
  }
  while (out.length < 16) out.push(0);
  return new Uint8Array(out);
}

describe('sniffFormat', () => {
  it('detects JPEG', () => {
    expect(sniffFormat(bytes([0xff, 0xd8, 0xff, 0xe0]))).toBe('jpeg');
  });
  it('detects PNG', () => {
    expect(sniffFormat(bytes([0x89], 'PNG', [0x0d, 0x0a, 0x1a, 0x0a]))).toBe('png');
  });
  it('detects BMP', () => {
    expect(sniffFormat(bytes('BM'))).toBe('bmp');
  });
  it('detects HEIC brands regardless of MIME', () => {
    for (const brand of ['heic', 'heix', 'mif1']) {
      expect(sniffFormat(bytes([0, 0, 0, 24], 'ftyp', brand))).toBe('heic');
    }
  });
  it('rejects non-images and short buffers', () => {
    expect(sniffFormat(bytes('GIF89a'))).toBe('unknown');
    expect(sniffFormat(new Uint8Array([0xff, 0xd8]))).toBe('unknown');
    expect(sniffFormat(bytes([0, 0, 0, 24], 'ftyp', 'mp42'))).toBe('unknown');
  });
});
