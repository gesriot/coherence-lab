export const WORLD = 100;

export function clamp(v: number, lo = 0, hi = 1): number {
  return Math.max(lo, Math.min(hi, v));
}

export function wrapHue(h: number): number {
  return ((h % 360) + 360) % 360;
}

export function hueDelta(a: number, b: number): number {
  const d = Math.abs(wrapHue(a) - wrapHue(b)) % 360;
  return d > 180 ? 360 - d : d;
}

export function lerpHue(a: number, b: number, t: number): number {
  const d = ((wrapHue(b) - wrapHue(a) + 540) % 360) - 180;
  return wrapHue(a + d * t);
}

export function hypot(x: number, y: number): number {
  return Math.hypot(x, y);
}

export function hashUnit(n: number): number {
  let x = Math.imul(n ^ 0x9e3779b9, 0x85ebca6b);
  x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35);
  return ((x ^ (x >>> 16)) >>> 0) / 4294967296;
}

/** Roughly −1…1, stable for a frame and cell. */
export function hueNoise(seed: number, t: number, i: number): number {
  const u = hashUnit(seed + t * 997 + i * 131 + 3);
  const v = hashUnit(seed * 3 + t * 1597 + i * 313 + 11);
  const w = hashUnit(seed * 7 + t * 53 + i * 17 + 29);
  return (u + v + w) / 1.5 - 1;
}
