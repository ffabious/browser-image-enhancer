import { sniffFormat, mimeFor, type ImageFormat } from './sniff';
import { decodeBmp } from './bmp';
import { decodeHeic, canDecodeHeicNatively } from './heic';

export interface DecodedImage {
  format: ImageFormat;
  width: number;
  height: number;
  /** Full-resolution RGBA pixels (mutated in place by the enhance stage). */
  image: ImageData;
  /** 224×224 aspect-ignoring squash used by the parameter-prediction brain. */
  thumb: ImageData;
  /**
   * False when decoding needed a fallback decoder (libheif, BMP): an <img>
   * pointing at the original file would not render in this browser, so UIs
   * need a re-encoded preview to show the "before" state.
   */
  displayable: boolean;
}

export const THUMB_SIZE = 224;

function get2d(canvas: OffscreenCanvas): OffscreenCanvasRenderingContext2D {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('OffscreenCanvas 2d context unavailable');
  return ctx as OffscreenCanvasRenderingContext2D;
}

function thumbFromBitmap(source: ImageBitmap | OffscreenCanvas): ImageData {
  const canvas = new OffscreenCanvas(THUMB_SIZE, THUMB_SIZE);
  const ctx = get2d(canvas);
  ctx.drawImage(source, 0, 0, THUMB_SIZE, THUMB_SIZE);
  return ctx.getImageData(0, 0, THUMB_SIZE, THUMB_SIZE);
}

function fromRgba(
  format: ImageFormat,
  width: number,
  height: number,
  data: Uint8ClampedArray<ArrayBuffer>,
): DecodedImage {
  const image = new ImageData(data, width, height);
  // Route through a canvas to get the browser's bilinear downscale, the same
  // path the native-decode branch uses.
  const canvas = new OffscreenCanvas(width, height);
  get2d(canvas).putImageData(image, 0, 0);
  return { format, width, height, image, thumb: thumbFromBitmap(canvas), displayable: false };
}

/** Decode an image file (any supported format) into full-res RGBA + thumb. */
export async function decodeImage(buffer: ArrayBuffer): Promise<DecodedImage> {
  const bytes = new Uint8Array(buffer);
  const format = sniffFormat(bytes);
  if (format === 'unknown') {
    throw new Error('unsupported image format (expected JPG, PNG, HEIC or BMP)');
  }

  if (format === 'heic' && !(await canDecodeHeicNatively())) {
    const { width, height, data } = await decodeHeic(buffer);
    return fromRgba(format, width, height, data);
  }

  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(new Blob([buffer], { type: mimeFor(format) }));
  } catch (err) {
    if (format === 'bmp') {
      const { width, height, data } = decodeBmp(bytes);
      return fromRgba(format, width, height, data);
    }
    throw new Error(`failed to decode ${format} image: ${String(err)}`);
  }

  try {
    const { width, height } = bitmap;
    const canvas = new OffscreenCanvas(width, height);
    const ctx = get2d(canvas);
    ctx.drawImage(bitmap, 0, 0);
    const image = ctx.getImageData(0, 0, width, height);
    const thumb = thumbFromBitmap(bitmap);
    return { format, width, height, image, thumb, displayable: true };
  } finally {
    bitmap.close();
  }
}
