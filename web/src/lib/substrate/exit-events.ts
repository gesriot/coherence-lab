import { advanceWorld, hiddenByOccluder } from "../lab/world.ts";
import type { World } from "../lab/types.ts";
import { GRID, substrateWorld } from "./input.ts";
import type { MapFactory } from "./training.ts";
import { visibility } from "./visibility.ts";

export interface ReturnEvent {
  kind: "wall" | "gap";
  time: number;
  length: number;
  cells: number[];
}

// Observe every frame, including t = 0. Only contact during the invisible interval counts.
export function returnDetector() {
  let age = -1, touched = false;
  return (world: World) => {
    const mask = visibility(world, age);
    let event: ReturnEvent | undefined;
    if (mask.exit) {
      const cells: number[] = [];
      mask.visible.forEach((visible, i) => { if (visible) cells.push(i); });
      event = { kind: touched ? "wall" : "gap", time: world.t, length: age + 1, cells };
    }
    age = mask.age;
    if (age < 0) touched = false;
    else if (hiddenByOccluder(world.objects[0]!, world.occluder)) touched = true;
    return { event, visible: mask.visible };
  };
}

export const exitWeightHook: MapFactory = (scenario, seed, settings) => {
  const world = substrateWorld(scenario, seed, settings);
  const observe = returnDetector();
  observe(world);
  const ones = new Float64Array(GRID * GRID).fill(1);
  return (targetTime) => {
    if (targetTime !== world.t + 1) throw new Error("Нарушен порядок целевых кадров");
    advanceWorld(world);
    const { event } = observe(world);
    if (event?.kind !== "wall") return ones;
    const weights = ones.slice();
    for (const i of event.cells) weights[i] = 50;
    return weights;
  };
};
