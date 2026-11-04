import type { Vector } from "./learning.ts";
import { sceneSignal } from "./sensorimotor-task.ts";
import { WORLD, hueNoise, wrapHue } from "./math.ts";
import type { Cell, Observation, World } from "./types.ts";
import { contains, hiddenByOccluder } from "./world.ts";

const GRID = 12;

function background(world: World, x: number, y: number): Cell {
  const t = sceneSignal.texture;
  const lightness = world.bodyGain
    ? t.base + t.amplitude * Math.sin(t.xFrequency * x + t.xPhase) * Math.sin(t.yFrequency * y + t.yPhase)
    : 6;
  return { hue: 0, saturation: 0, lightness };
}

export type Occlusion = "whole-object" | "per-cell";

export function sample(
  world: World, x: number, y: number, i: number, occlusion: Occlusion = "whole-object",
): Cell {
  if (x < 0 || y < 0 || x > WORLD || y > WORLD) {
    return background(world, x, y);
  }
  const occ = world.occluder;
  if (x >= occ.x && x <= occ.x + occ.w && y >= occ.y && y <= occ.y + occ.h) {
    return { hue: 0, saturation: 0, lightness: (Math.floor((x + y) / 4) % 2) ? 32 : 54 };
  }
  for (let k = world.objects.length - 1; k >= 0; k--) {
    const o = world.objects[k]!;
    if (o.vanishAt != null && world.t >= o.vanishAt) continue;
    if (occlusion === "whole-object" && hiddenByOccluder(o, occ)) continue;
    if (!contains(o, x, y)) continue;
    const hue = wrapHue(o.hue + world.light + hueNoise(world.seed, world.t, i) * world.noise);
    return { hue, saturation: 70, lightness: 50 };
  }
  return background(world, x, y);
}

export function observe(
  world: World,
  gazeX: number,
  gazeY: number,
  aperture: number,
  estimate?: Readonly<Vector>,
): Observation {
  const cells: Cell[] = [];
  const cell = (aperture * 2) / GRID;
  const left = ((world.bodyGain?.[0] ?? 1) / (estimate?.[0] ?? 1)) * gazeX - aperture;
  const top = ((world.bodyGain?.[1] ?? 1) / (estimate?.[1] ?? 1)) * gazeY - aperture;
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
  };
}
