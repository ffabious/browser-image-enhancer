/**
 * Minimal pure-TS BMP decoder — fallback for engines whose createImageBitmap
 * rejects BMP blobs (see docs/decisions.md ADR-4).
 *
 * Supports the common cases: BITMAPINFOHEADER/V4/V5, uncompressed BI_RGB
 * 8/24/32-bit (8-bit palettized) and BI_BITFIELDS 32-bit with the standard
 * BGRA/BGRX masks, bottom-up and top-down row order.
 */

export interface DecodedBmp {
  width: number;
  height: number;
  /** RGBA, tightly packed. */
  data: Uint8ClampedArray<ArrayBuffer>;
}

export function decodeBmp(bytes: Uint8Array): DecodedBmp {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.length < 54 || bytes[0] !== 0x42 || bytes[1] !== 0x4d) {
    throw new Error('not a BMP file');
  }
  const pixelOffset = view.getUint32(10, true);
  const headerSize = view.getUint32(14, true);
  if (headerSize < 40) throw new Error(`unsupported BMP header size ${headerSize}`);

  const width = view.getInt32(18, true);
  const rawHeight = view.getInt32(22, true);
  const topDown = rawHeight < 0;
  const height = Math.abs(rawHeight);
  const bpp = view.getUint16(28, true);
  const compression = view.getUint32(30, true);

  if (width <= 0 || height === 0) throw new Error('invalid BMP dimensions');
  const isPlainRgb = compression === 0; // BI_RGB
  const isBitfields = compression === 3 && bpp === 32; // BI_BITFIELDS
  if (!isPlainRgb && !isBitfields) {
    throw new Error(`unsupported BMP compression ${compression}`);
  }
  if (bpp !== 8 && bpp !== 24 && bpp !== 32) {
    throw new Error(`unsupported BMP bit depth ${bpp}`);
  }

  // For BITFIELDS we only support the ubiquitous BGRA/BGRX layout.
  let hasAlpha = false;
  if (isBitfields) {
    const rMask = view.getUint32(54, true);
    const gMask = view.getUint32(58, true);
    const bMask = view.getUint32(62, true);
    if (rMask !== 0x00ff0000 || gMask !== 0x0000ff00 || bMask !== 0x000000ff) {
      throw new Error('unsupported BMP channel masks');
    }
    // V4/V5 headers carry an alpha mask; honor it only when it is the
    // standard high byte (BGRX files have alphaMask = 0 → opaque).
    hasAlpha = headerSize >= 56 && view.getUint32(66, true) === 0xff000000;
  }

  // 8-bit: read the palette (BGRX entries right after the header + masks).
  let palette: Uint8Array | null = null;
  if (bpp === 8) {
    let colorsUsed = view.getUint32(46, true);
    if (colorsUsed === 0) colorsUsed = 256;
    const paletteStart = 14 + headerSize;
    palette = bytes.subarray(paletteStart, paletteStart + colorsUsed * 4);
  }

  const rowSize = Math.floor((bpp * width + 31) / 32) * 4;
  const data = new Uint8ClampedArray(width * height * 4);

  for (let y = 0; y < height; y++) {
    const srcY = topDown ? y : height - 1 - y;
    let src = pixelOffset + srcY * rowSize;
    let dst = y * width * 4;
    if (bpp === 8) {
      for (let x = 0; x < width; x++, dst += 4) {
        const p = bytes[src + x] * 4;
        data[dst] = palette![p + 2];
        data[dst + 1] = palette![p + 1];
        data[dst + 2] = palette![p];
        data[dst + 3] = 255;
      }
    } else if (bpp === 24) {
      for (let x = 0; x < width; x++, src += 3, dst += 4) {
        data[dst] = bytes[src + 2];
        data[dst + 1] = bytes[src + 1];
        data[dst + 2] = bytes[src];
        data[dst + 3] = 255;
      }
    } else {
      // 32-bit BGRA/BGRX. BI_RGB 32-bit ignores the 4th byte by spec.
      for (let x = 0; x < width; x++, src += 4, dst += 4) {
        data[dst] = bytes[src + 2];
        data[dst + 1] = bytes[src + 1];
        data[dst + 2] = bytes[src];
        data[dst + 3] = hasAlpha ? bytes[src + 3] : 255;
      }
    }
  }

  return { width, height, data };
}
