/** Image format detection by magic bytes — file MIME types are never trusted. */

export type ImageFormat = 'jpeg' | 'png' | 'bmp' | 'heic' | 'unknown';

const HEIC_BRANDS = new Set(['heic', 'heix', 'hevc', 'hevx', 'heif', 'mif1', 'msf1']);

export function sniffFormat(bytes: Uint8Array): ImageFormat {
  if (bytes.length < 12) return 'unknown';
  if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return 'jpeg';
  if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) {
    return 'png';
  }
  if (bytes[0] === 0x42 && bytes[1] === 0x4d) return 'bmp';
  // ISO-BMFF: [size:4]['ftyp':4][major brand:4]
  if (bytes[4] === 0x66 && bytes[5] === 0x74 && bytes[6] === 0x79 && bytes[7] === 0x70) {
    const brand = String.fromCharCode(bytes[8], bytes[9], bytes[10], bytes[11]).toLowerCase();
    if (HEIC_BRANDS.has(brand)) return 'heic';
  }
  return 'unknown';
}

export function mimeFor(format: ImageFormat): string {
  switch (format) {
    case 'jpeg':
      return 'image/jpeg';
    case 'png':
      return 'image/png';
    case 'bmp':
      return 'image/bmp';
    case 'heic':
      return 'image/heic';
    default:
      return 'application/octet-stream';
  }
}
