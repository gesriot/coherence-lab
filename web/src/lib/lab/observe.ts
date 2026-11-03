import { wrapHue } from "./math.ts";
import type { Blob, Cell, Expectation, Observation } from "./types.ts";

export function classify(cell: Cell): Expectation {
  if (cell.lightness < 15) return "empty";
  return cell.saturation < 20 ? "occluder" : "surface";
}

export function extractBlobs(obs: Observation): Blob[] {
  const n = obs.grid;
  const cell = (obs.aperture * 2) / n;
  const left = obs.gazeX - obs.aperture;
  const top = obs.gazeY - obs.aperture;
  const seen = new Uint8Array(n * n);
  const blobs: Blob[] = [];

  const hueClose = (a: number, b: number) => {
    const d = Math.abs(a - b) % 360;
    return (d > 180 ? 360 - d : d) < 38;
  };

  for (let i = 0; i < seen.length; i++) {
    if (seen[i] || classify(obs.cells[i]!) !== "surface") continue;
    const stack = [i];
    seen[i] = 1;
    const members: number[] = [];
    const seedHue = obs.cells[i]!.hue;
    while (stack.length) {
      const cur = stack.pop()!;
      members.push(cur);
      const r = Math.floor(cur / n);
      const c = cur % n;
      const neighbors = [
        r > 0 ? cur - n : -1,
        r < n - 1 ? cur + n : -1,
        c > 0 ? cur - 1 : -1,
        c < n - 1 ? cur + 1 : -1,
      ];
      for (const nb of neighbors) {
        if (nb < 0 || seen[nb]) continue;
        const cellData = obs.cells[nb]!;
        if (classify(cellData) !== "surface" || !hueClose(seedHue, cellData.hue)) continue;
        seen[nb] = 1;
        stack.push(nb);
      }
    }
    if (members.length === 0) continue;
    let sx = 0;
    let sy = 0;
    let sinHue = 0;
    let cosHue = 0;
    let minC = n;
    let maxC = 0;
    let minR = n;
    let maxR = 0;
    for (const idx of members) {
      const r = Math.floor(idx / n);
      const c = idx % n;
      sx += left + (c + 0.5) * cell;
      sy += top + (r + 0.5) * cell;
      const angle = obs.cells[idx]!.hue * Math.PI / 180;
      sinHue += Math.sin(angle);
      cosHue += Math.cos(angle);
      if (c < minC) minC = c;
      if (c > maxC) maxC = c;
      if (r < minR) minR = r;
      if (r > maxR) maxR = r;
    }
    const area = members.length;
    const box = (maxC - minC + 1) * (maxR - minR + 1);
    blobs.push({
      index: blobs.length,
      cx: sx / area,
      cy: sy / area,
      hue: wrapHue(Math.atan2(sinHue, cosHue) * 180 / Math.PI),
      area,
      fill: area / box,
    });
  }
  return blobs;
}

export type Location =
  | { status: "outside" }
  | { status: "cell"; kind: Expectation; hue: number };

export function locate(obs: Observation, x: number, y: number): Location {
  const half = obs.aperture;
  if (x < obs.gazeX - half || x > obs.gazeX + half || y < obs.gazeY - half || y > obs.gazeY + half) {
    return { status: "outside" };
  }
  const n = obs.grid;
  const cell = (half * 2) / n;
  const col = Math.floor((x - (obs.gazeX - half)) / cell);
  const row = Math.floor((y - (obs.gazeY - half)) / cell);
  if (col < 0 || row < 0 || col >= n || row >= n) return { status: "outside" };
  const found = obs.cells[row * n + col]!;
  return { status: "cell", kind: classify(found), hue: found.hue };
}
