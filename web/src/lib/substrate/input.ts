import { WORLD } from "../lab/math.ts";
import { sample } from "../lab/sensor.ts";
import { advanceWorld, makeWorld, scheduleOutdoorLosses } from "../lab/world.ts";
import type { ScenarioId, World } from "../lab/types.ts";

export const GRID = 16;
export const CELL_SIZE = WORLD / GRID;
export const CHANNELS = 3;

export function frame(world: World): Float64Array {
  const x = new Float64Array(GRID * GRID * CHANNELS);
  for (let i = 0; i < GRID * GRID; i++) {
    const col = i % GRID;
    const row = Math.floor(i / GRID);
    const { lightness, saturation, hue } = sample(
      world, (col + 0.5) * CELL_SIZE, (row + 0.5) * CELL_SIZE, i,
    );
    const h = hue * Math.PI / 180;
    x[3 * i] = lightness / 100;
    x[3 * i + 1] = saturation / 100 * Math.cos(h);
    x[3 * i + 2] = saturation / 100 * Math.sin(h);
  }
  return x;
}

// The training interface exposes sensory frames only, starting with t = 0.
export function* frames(scenario: ScenarioId, seed: number, steps: number) {
  const world = makeWorld(scenario, seed);
  if (scenario === "nursery") scheduleOutdoorLosses(world);
  yield frame(world);
  for (let t = 0; t < steps; t++) {
    advanceWorld(world);
    yield frame(world);
  }
}
