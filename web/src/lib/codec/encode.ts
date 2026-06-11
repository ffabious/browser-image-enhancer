import type { ImageFormat } from './sniff';

/** JPEG for lossy inputs (JPG/HEIC), PNG for lossless inputs (PNG/BMP). */
export function outputTypeFor(format: ImageFormat): { type: string; quality?: number } {
  return format === 'png' || format === 'bmp'
    ? { type: 'image/png' }
    : { type: 'image/jpeg', quality: 0.92 };
}

export async function encodeImage(image: ImageData, format: ImageFormat): Promise<Blob> {
  const canvas = new OffscreenCanvas(image.width, image.height);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('OffscreenCanvas 2d context unavailable');
  ctx.putImageData(image, 0, 0);
  const { type, quality } = outputTypeFor(format);
  return canvas.convertToBlob({ type, quality });
}
