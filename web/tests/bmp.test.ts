import { describe, expect, it } from 'vitest';
import { decodeBmp } from '../src/lib/codec/bmp';

/** Build a minimal BMP file for tests. */
function buildBmp(opts: {
  width: number;
  height: number; // positive = bottom-up
  bpp: 8 | 24 | 32;
  /** Rows top→bottom; each pixel is [r,g,b,(a)] or a palette index for bpp=8. */
  pixels: (number[] | number)[][];
  palette?: number[][]; // [r,g,b] entries for bpp=8
  bitfields?: boolean; // BI_BITFIELDS V4 header with alpha mask (bpp=32)
}): Uint8Array {
  const { width, bpp } = opts;
  const height = Math.abs(opts.height);
  const topDown = opts.height < 0;
  const headerSize = opts.bitfields ? 108 : 40;
  const paletteBytes = opts.palette ? opts.palette.length * 4 : 0;
  const rowSize = Math.floor((bpp * width + 31) / 32) * 4;
  const pixelOffset = 14 + headerSize + paletteBytes;
  const size = pixelOffset + rowSize * height;

  const buf = new Uint8Array(size);
  const view = new DataView(buf.buffer);
  buf[0] = 0x42;
  buf[1] = 0x4d;
  view.setUint32(2, size, true);
  view.setUint32(10, pixelOffset, true);
  view.setUint32(14, headerSize, true);
  view.setInt32(18, width, true);
  view.setInt32(22, opts.height, true);
  view.setUint16(26, 1, true);
  view.setUint16(28, bpp, true);
  view.setUint32(30, opts.bitfields ? 3 : 0, true);
  if (opts.palette) view.setUint32(46, opts.palette.length, true);
  if (opts.bitfields) {
    view.setUint32(54, 0x00ff0000, true);
    view.setUint32(58, 0x0000ff00, true);
    view.setUint32(62, 0x000000ff, true);
    view.setUint32(66, 0xff000000, true);
  }
  if (opts.palette) {
    opts.palette.forEach(([r, g, b], i) => {
      const o = 14 + headerSize + i * 4;
      buf[o] = b;
      buf[o + 1] = g;
      buf[o + 2] = r;
    });
  }
  for (let y = 0; y < height; y++) {
    const srcRow = opts.pixels[y];
    const fileY = topDown ? y : height - 1 - y;
    let o = pixelOffset + fileY * rowSize;
    for (let x = 0; x < width; x++) {
      const px = srcRow[x];
      if (typeof px === 'number') {
        buf[o++] = px;
      } else {
        const [r, g, b, a = 255] = px;
        buf[o++] = b;
        buf[o++] = g;
        buf[o++] = r;
        if (bpp === 32) buf[o++] = a;
      }
    }
  }
  return buf;
}

describe('decodeBmp', () => {
  it('decodes 24-bit bottom-up with row padding', () => {
    const bmp = buildBmp({
      width: 3,
      height: 2,
      bpp: 24,
      pixels: [
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
      ],
    });
    const { width, height, data } = decodeBmp(bmp);
    expect([width, height]).toEqual([3, 2]);
    expect([...data.slice(0, 4)]).toEqual([255, 0, 0, 255]); // top-left
    expect([...data.slice(12, 16)]).toEqual([10, 20, 30, 255]); // row 2 starts
  });

  it('decodes 32-bit top-down BI_RGB ignoring the 4th byte', () => {
    const bmp = buildBmp({
      width: 2,
      height: -1,
      bpp: 32,
      pixels: [[[1, 2, 3, 0], [4, 5, 6, 7]]],
    });
    const { data } = decodeBmp(bmp);
    expect([...data]).toEqual([1, 2, 3, 255, 4, 5, 6, 255]);
  });

  it('decodes 32-bit BITFIELDS with alpha (V4 header)', () => {
    const bmp = buildBmp({
      width: 1,
      height: 1,
      bpp: 32,
      bitfields: true,
      pixels: [[[9, 8, 7, 128]]],
    });
    expect([...decodeBmp(bmp).data]).toEqual([9, 8, 7, 128]);
  });

  it('decodes 8-bit palettized', () => {
    const bmp = buildBmp({
      width: 2,
      height: 1,
      bpp: 8,
      palette: [
        [11, 22, 33],
        [44, 55, 66],
      ],
      pixels: [[1, 0]],
    });
    expect([...decodeBmp(bmp).data]).toEqual([44, 55, 66, 255, 11, 22, 33, 255]);
  });

  it('rejects unsupported compression', () => {
    const bmp = buildBmp({ width: 1, height: 1, bpp: 24, pixels: [[[1, 2, 3]]] });
    new DataView(bmp.buffer).setUint32(30, 1, true); // BI_RLE8
    expect(() => decodeBmp(bmp)).toThrow(/compression/);
  });
});
