import { contains, advanceWorld, hiddenByOccluder } from "../lab/world.ts";
import type { Occluder, SimObject, World } from "../lab/types.ts";
import { blankState, forward, stepBuffers, type Element } from "./element.ts";
import { CELL_SIZE, frame, GRID, substrateWorld, type Occlusion } from "./input.ts";

import { inWall, visibility } from "./visibility.ts";
import { occupancy } from "./occupancy.ts";

export const TEST_SEEDS = Array.from({ length: 30 }, (_, i) => 3001 + i);
export const PROBE_SEEDS = Array.from({ length: 30 }, (_, i) => 4001 + i);
export const OFFSETS = [-6, -5, -4, -3, -2, -1, 0, 1, 2] as const;
export const EVAL_STEPS = 72;

export interface Examples {
  velocity: { x: Float64Array[]; y: number[] };
  hidden: { x: Float64Array[]; y: number[] };
  late: { x: Float64Array[]; y: number[] };
}
export function examples(): Examples {
  return { velocity: { x: [], y: [] }, hidden: { x: [], y: [] }, late: { x: [], y: [] } };
}

// The probe sees a copy of the state. Labels and spatial selections never enter forward().
function probeSample(
  rows: Examples, a: SimObject, hidden: boolean, wall: number[],
  state: Float64Array, depth: number, late: boolean,
) {
  if (hidden) {
    const x = new Float64Array(wall.length * depth);
    for (let i = 0; i < wall.length; i++) {
      const start = wall[i]! * depth;
      x.set(state.subarray(start, start + depth), i * depth);
    }
    rows.hidden.x.push(x);
    rows.hidden.y.push(a.x);
    if (late) {
      rows.late.x.push(x);
      rows.late.y.push(a.x);
    }
  } else {
    const i = Math.floor(a.y / CELL_SIZE) * GRID + Math.floor(a.x / CELL_SIZE);
    rows.velocity.x.push(state.slice(i * depth, (i + 1) * depth));
    rows.velocity.y.push(Math.sign(a.vx));
  }
}

function wallCells(occ: Occluder) {
  const cells: number[] = [];
  for (let i = 0; i < GRID * GRID; i++) {
    const x = (i % GRID) * CELL_SIZE, y = Math.floor(i / GRID) * CELL_SIZE;
    if (x < occ.x + occ.w && x + CELL_SIZE > occ.x && y < occ.y + occ.h && y + CELL_SIZE > occ.y) {
      cells.push(i);
    }
  }
  return cells;
}

interface Errors { squared: number; copy: number; cells: number }
function errors(): Errors { return { squared: 0, copy: 0, cells: 0 }; }
function add(e: Errors, squared: number, copy: number) {
  e.squared += squared;
  e.copy += copy;
  e.cells++;
}
function sums() {
  return {
    all: errors(), changed: errors(), exit: errors(), wallExit: errors(),
    events: 0, wallEvents: 0,
    ghost: { absolute: 0, count: 0 },
    occupied: { sum: 0, n: 0 }, empty: { sum: 0, n: 0 },
    curve: OFFSETS.map(() => ({ value: 0, count: 0 })),
  };
}
type Sums = ReturnType<typeof sums>;

function scoreFrame(
  totals: Sums, target: Float64Array, previous: Float64Array, prediction: Float64Array,
  world: World, emergence: boolean, ghost: SimObject | null, exitCells: number[], visible?: Uint8Array, crossedWall = false,
) {
  const a = world.objects[0]!;
  for (let i = 0; i < GRID * GRID; i++) {
    const x = (i % GRID + 0.5) * CELL_SIZE, y = (Math.floor(i / GRID) + 0.5) * CELL_SIZE;
    let sq = 0, copy = 0, changed = false, absolute = 0;
    for (let d = 0; d < 3; d++) {
      const j = 3 * i + d;
      const e = prediction[j]! - target[j]!, delta = previous[j]! - target[j]!;
      sq += e * e;
      copy += delta * delta;
      absolute += Math.abs(e);
      if (Math.abs(delta) > 0.05) changed = true;
    }
    add(totals.all, sq, copy);
    if (changed) add(totals.changed, sq, copy);
    if (emergence && (visible ? visible[i] : contains(a, x, y))) {
      add(totals.exit, sq, copy);
      if (crossedWall) add(totals.wallExit, sq, copy);
      exitCells.push(i);
    }
    if (ghost && (x - ghost.x) ** 2 + (y - ghost.y) ** 2 <= (ghost.r + 4) ** 2) {
      totals.ghost.absolute += absolute;
      totals.ghost.count += 3;
    }
  }
}

function replay(
  model: Element | null, scenario: "continuity" | "vanish", seed: number,
  rows: Examples, totals?: Sums, occlusion: Occlusion = "whole-object",
) {
  const world = substrateWorld(scenario, seed);
  const wall = wallCells(world.occluder), a = world.objects[0]!;
  const tape = model ? [stepBuffers(model), stepBuffers(model)] : null;
  let state = model ? blankState(model) : new Float64Array(0);
  let previous: Float64Array | undefined, wasHidden = false, lastSeen: SimObject | null = null;
  // Index by the target time, not the time at which the prediction was made.
  let age = -1, crossedWall = false;
  const predicted: Float64Array[] = [];
  const exits: { time: number; cells: number[] }[] = [];
  for (let t = 0; t <= EVAL_STEPS; t++) {
    const x = frame(world, occlusion);
    const mask = occlusion === "per-cell" ? visibility(world, age) : undefined;
    if (mask) {
      age = mask.age;
      if (age >= 0 && hiddenByOccluder(a, world.occluder)) crossedWall = true;
    }
    const hidden = mask ? age >= 0 : hiddenByOccluder(a, world.occluder);
    if (totals && previous) {
      const emergence = mask ? mask.exit : scenario === "continuity" && wasHidden && !hidden;
      const since = a.vanishAt == null ? -1 : t - a.vanishAt;
      const exitCells: number[] = [];
      // Ten predictions made after absence was already observable, not the surprise at disappearance.
      const ghost = since >= 1 && since <= 10 ? lastSeen : null;
      scoreFrame(totals, x, previous, predicted[t]!, world, emergence, ghost, exitCells, mask?.visible, crossedWall);
      if (mask?.exit) {
        totals.events++;
        if (crossedWall) totals.wallEvents++;
      }
      if (exitCells.length) exits.push({ time: t, cells: exitCells });
    }
    if (model && tape) {
      const out = tape[t % 2]!;
      forward(model, state, x, out);
      state = out.state;
      if (totals && mask && hidden && out.occupancy) {
        const labels = occupancy(world);
        for (let i = 0; i < labels.length; i++) {
          const cx = (i % GRID + 0.5) * CELL_SIZE, cy = (Math.floor(i / GRID) + 0.5) * CELL_SIZE;
          const group = contains(a, cx, cy) ? totals.occupied :
            !labels[i] && inWall(world.occluder, cx, cy) ? totals.empty : null;
          if (group) {
            group.sum += out.occupancy[i]!;
            group.n++;
          }
        }
      }
      if (totals && t < EVAL_STEPS) predicted[t + 1] = out.prediction.slice();
    } else {
      state = x;
      if (totals && t < EVAL_STEPS) predicted[t + 1] = x;
    }
    if (scenario === "continuity") probeSample(rows, a, hidden, wall, state, model?.depth ?? 3, age >= 4);
    if (!hidden && (a.vanishAt === null || t < a.vanishAt)) lastSeen = { ...a };
    if (age < 0) crossedWall = false;
    wasHidden = hidden;
    previous = x;
    if (t < EVAL_STEPS) advanceWorld(world);
  }
  if (totals) {
    for (const { time, cells } of exits) {
      for (let j = 0; j < OFFSETS.length; j++) {
        const p = predicted[time + OFFSETS[j]!];
        if (!p) continue;
        for (const i of cells) {
          totals.curve[j]!.value += Math.hypot(p[3 * i + 1]!, p[3 * i + 2]!);
          totals.curve[j]!.count++;
        }
      }
    }
  }
}

export function probeTrainingData(model: Element | null, seeds: readonly number[], occlusion: Occlusion = "whole-object") {
  const rows = examples();
  for (const seed of seeds) replay(model, "continuity", seed, rows, undefined, occlusion);
  return rows;
}

export function evaluationData(model: Element | null, seeds: readonly number[], occlusion: Occlusion = "whole-object") {
  const totals = sums(), rows = examples();
  for (const seed of seeds) {
    replay(model, "continuity", seed, rows, totals, occlusion);
    replay(model, "vanish", seed, rows, totals, occlusion);
  }
  const ratio = (e: Errors) => {
    if (!e.copy) throw new Error("Нет опорной ошибки копии в выборке");
    return e.squared / e.copy;
  };
  return {
    rows,
    diagnostic: {
      events: totals.events, wallEvents: totals.wallEvents,
      wallExit: totals.wallExit.copy ? ratio(totals.wallExit) : null,
    },
    hint: totals.occupied.n && totals.empty.n
      ? { occupied: totals.occupied.sum / totals.occupied.n, empty: totals.empty.sum / totals.empty.n }
      : undefined,
    metrics: {
      all: ratio(totals.all), changed: ratio(totals.changed), exit: ratio(totals.exit),
      ghost: totals.ghost.absolute / totals.ghost.count,
      curve: totals.curve.map(({ value, count }) => value / count),
    },
  };
}
