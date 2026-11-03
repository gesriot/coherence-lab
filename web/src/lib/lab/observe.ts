import { WORLD, hueNoise, wrapHue } from "./math";
import type { Blob, Cell, Observation, World } from "./types";
import { contains, hiddenByOccluder } from "./world";

const GRID = 12;

function sample(world: World, x: number, y: number, i: number): Cell {
  if (x < 0 || y < 0 || x > WORLD || y > WORLD) {
    return { kind: "empty", hue: 90 };
  }
  const occ = world.occluder;
  if (x >= occ.x && x <= occ.x + occ.w && y >= occ.y && y <= occ.y + occ.h) {
    return { kind: "occluder", hue: 42 };
  }
  for (let k = world.objects.length - 1; k >= 0; k--) {
    const o = world.objects[k]!;
    if (hiddenByOccluder(o, occ)) continue;
    if (!contains(o, x, y)) continue;
    const hue = wrapHue(o.hue + world.light + hueNoise(world.seed, world.t, i) * world.noise);
    return { kind: "surface", hue };
  }
  return { kind: "empty", hue: 110 };
}

export function observe(
  world: World,
  gazeX: number,
  gazeY: number,
  aperture: number,
): Observation {
  const cells: Cell[] = [];
  const cell = (aperture * 2) / GRID;
  const left = gazeX - aperture;
  const top = gazeY - aperture;
  for (let r = 0; r < GRID; r++) {
    for (let c = 0; c < GRID; c++) {
      const x = left + (c + 0.5) * cell;
      const y = top + (r + 0.5) * cell;
      cells.push(sample(world, x, y, r * GRID + c));
    }
  }
  return {
    gazeX,
    gazeY,
    aperture,
    grid: GRID,
    cells,
    capturedGazeX: gazeX,
    capturedGazeY: gazeY,
  };
}

/** Replay keeps the captured cells but lets the agent interpret them at a new gaze. */
export function reinterpret(source: Observation, gazeX: number, gazeY: number, aperture: number): Observation {
  return {
    ...source,
    cells: source.cells,
    grid: source.grid,
    gazeX,
    gazeY,
    aperture,
  };
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
    if (seen[i] || obs.cells[i]!.kind !== "surface") continue;
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
        if (cellData.kind !== "surface" || !hueClose(seedHue, cellData.hue)) continue;
        seen[nb] = 1;
        stack.push(nb);
      }
    }
    if (members.length === 0) continue;
    let sx = 0;
    let sy = 0;
    let sh = 0;
    let minC = n;
    let maxC = 0;
    let minR = n;
    let maxR = 0;
    for (const idx of members) {
      const r = Math.floor(idx / n);
      const c = idx % n;
      sx += left + (c + 0.5) * cell;
      sy += top + (r + 0.5) * cell;
      sh += obs.cells[idx]!.hue;
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
      hue: sh / area,
      area,
      fill: area / box,
    });
  }
  return blobs;
}

export type Location =
  | { status: "outside" }
  | { status: "cell"; kind: Cell["kind"]; hue: number };

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
  return { status: "cell", kind: found.kind, hue: found.hue };
}
