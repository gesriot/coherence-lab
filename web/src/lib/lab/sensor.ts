import { WORLD, hueNoise, wrapHue } from "./math.ts";
import type { Cell, Observation, World } from "./types.ts";
import { contains, hiddenByOccluder } from "./world.ts";

const GRID = 12;

function sample(world: World, x: number, y: number, i: number): Cell {
  if (x < 0 || y < 0 || x > WORLD || y > WORLD) {
    return { hue: 0, saturation: 0, lightness: 6 };
  }
  const occ = world.occluder;
  if (x >= occ.x && x <= occ.x + occ.w && y >= occ.y && y <= occ.y + occ.h) {
    return { hue: 0, saturation: 0, lightness: (Math.floor((x + y) / 4) % 2) ? 32 : 54 };
  }
  for (let k = world.objects.length - 1; k >= 0; k--) {
    const o = world.objects[k]!;
    if (hiddenByOccluder(o, occ)) continue;
    if (!contains(o, x, y)) continue;
    const hue = wrapHue(o.hue + world.light + hueNoise(world.seed, world.t, i) * world.noise);
    return { hue, saturation: 70, lightness: 50 };
  }
  return { hue: 0, saturation: 0, lightness: 6 };
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
  };
}
