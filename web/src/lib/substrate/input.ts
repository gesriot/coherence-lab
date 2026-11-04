import { WORLD, hashUnit } from "../lab/math.ts";
import { sample, type Occlusion } from "../lab/sensor.ts";
import { advanceWorld, makeWorld, scheduleOutdoorLosses } from "../lab/world.ts";
import type { ScenarioId, World } from "../lab/types.ts";

import { gazePath, WINDOW_CELL, type Command, type Gaze, type MotionSettings } from "./gaze.ts";

export type { Occlusion };
export const GRID = 16;
export const CELL_SIZE = WORLD / GRID;
export const CHANNELS = 3;

export type WallColumns = 1 | 2 | 4;
export interface WorldSettings {
  path?: "scenario" | "random";
  columns?: WallColumns;
}

export function substrateWorld(scenario: ScenarioId, seed: number, settings: WorldSettings = {}) {
  const world = makeWorld(scenario, seed);
  if (scenario === "nursery") scheduleOutdoorLosses(world);
  if (scenario !== "continuity") return world;
  if (settings.path === "random") {
    const u = (k: number) => hashUnit(7919 * seed + 31 * k + 5);
    const right = u(1) < 0.5, speed = 1.2 + 0.9 * u(2);
    Object.assign(world.objects[0]!, {
      y: 12 + 76 * u(3), x: (right ? 8 : 76) + 16 * u(4),
      vx: right ? speed : -speed, vy: 0.05 * (2 * u(5) - 1),
    });
  }
  if (settings.columns === 1 || settings.columns === 2) {
    world.occluder = { x: 44, w: 6 * settings.columns, y: 0, h: WORLD };
  }
  return world;
}

export function frame(world: World, occlusion: Occlusion = "whole-object", gaze?: Gaze): Float64Array {
  const x = new Float64Array(GRID * GRID * CHANNELS);
  for (let i = 0; i < GRID * GRID; i++) {
    const col = i % GRID;
    const row = Math.floor(i / GRID);
    const cx = gaze ? gaze.x + (col + 0.5 - GRID / 2) * WINDOW_CELL : (col + 0.5) * CELL_SIZE;
    const cy = gaze ? gaze.y + (row + 0.5 - GRID / 2) * WINDOW_CELL : (row + 0.5) * CELL_SIZE;
    const { lightness, saturation, hue } = sample(
      world, cx, cy, i, occlusion,
    );
    const h = hue * Math.PI / 180;
    x[3 * i] = lightness / 100;
    x[3 * i + 1] = saturation / 100 * Math.cos(h);
    x[3 * i + 2] = saturation / 100 * Math.sin(h);
  }
  return x;
}

// The training interface exposes sensory frames only, starting with t = 0.
export function* frames(
  scenario: ScenarioId, seed: number, steps: number, occlusion: Occlusion = "whole-object",
  settings?: WorldSettings,
) {
  const world = substrateWorld(scenario, seed, settings);
  yield frame(world, occlusion);
  for (let t = 0; t < steps; t++) {
    advanceWorld(world);
    yield frame(world, occlusion);
  }
}

export interface MotionInput { frame: Float64Array; copy: Command }

// The world is live only until the generator advances. The evaluator alone reads it.
export function* movingScene(scenario: ScenarioId, seed: number, steps: number, settings: MotionSettings) {
  const world = substrateWorld(scenario, seed);
  world.bodyGain = [1, 1]; // Enables the fixed background texture in the point sensor.
  const path = gazePath(seed, settings.gain);
  const alternate = settings.copy === "shuffled" ? gazePath(seed + 7777, settings.gain) : undefined;
  const zero: Command = [0, 0];
  for (let t = 0; t <= steps; t++) {
    const { gaze, command } = path();
    const copy = alternate ? alternate().command : settings.copy === "zero" ? zero : command;
    const input: MotionInput = { frame: frame(world, "per-cell", gaze), copy };
    yield { world, gaze, command, input };
    if (t < steps) advanceWorld(world);
  }
}

export function* movingFrames(scenario: ScenarioId, seed: number, steps: number, settings: MotionSettings) {
  for (const { input } of movingScene(scenario, seed, steps, settings)) yield input;
}
