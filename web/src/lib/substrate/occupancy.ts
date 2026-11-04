import { advanceWorld, contains } from "../lab/world.ts";
import type { World } from "../lab/types.ts";
import { CELL_SIZE, GRID, substrateWorld } from "./input.ts";
import type { MapFactory } from "./training.ts";

export function occupancy(world: World) {
  return Float64Array.from({ length: GRID * GRID }, (_, i) => {
    const x = (i % GRID + 0.5) * CELL_SIZE, y = (Math.floor(i / GRID) + 0.5) * CELL_SIZE;
    return Number(world.objects.some((o) =>
      (o.vanishAt === null || world.t < o.vanishAt) && contains(o, x, y)));
  });
}

// Only the positive control constructs this independent, synchronized source of labels.
export const occupancyHook: MapFactory = (scenario, seed, settings) => {
  const world = substrateWorld(scenario, seed, settings);
  return (time) => {
    while (world.t < time) advanceWorld(world);
    if (world.t !== time) throw new Error("Нарушен порядок кадров подсказки");
    return occupancy(world);
  };
};
