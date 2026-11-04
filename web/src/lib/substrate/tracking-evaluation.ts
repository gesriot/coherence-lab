import { advanceWorld } from "../lab/world.ts";
import type { SimObject } from "../lab/types.ts";
import { blankState, forward, stepBuffers, type Element } from "./element.ts";
import { EVAL_STEPS, OFFSETS, TEST_SEEDS } from "./evaluation.ts";
import { returnDetector, type ReturnEvent } from "./exit-events.ts";
import { CELL_SIZE, frame, GRID, substrateWorld, type WorldSettings } from "./input.ts";

export interface TrackingEpisode {
  frames: Float64Array[];
  events: ReturnEvent[];
  ghostCells: number[][];
}

function aroundLastSeen(a: SimObject) {
  const cells: number[] = [];
  for (let i = 0; i < GRID * GRID; i++) {
    const x = (i % GRID + 0.5) * CELL_SIZE, y = (Math.floor(i / GRID) + 0.5) * CELL_SIZE;
    if ((x - a.x) ** 2 + (y - a.y) ** 2 <= (a.r + 4) ** 2) cells.push(i);
  }
  return cells;
}

export function trackingEpisodes(settings: WorldSettings, seeds: readonly number[] = TEST_SEEDS) {
  const episodes: TrackingEpisode[] = [];
  for (const seed of seeds) {
    for (const scenario of ["continuity", "vanish"] as const) {
      const world = substrateWorld(scenario, seed, settings), a = world.objects[0]!;
      const observe = returnDetector();
      const ep: TrackingEpisode = { frames: [], events: [], ghostCells: [] };
      let lastSeen: SimObject | undefined;
      for (let t = 0; t <= EVAL_STEPS; t++) {
        const { event, visible } = observe(world);
        if (event) ep.events.push(event);
        ep.frames.push(frame(world, "per-cell"));
        if (visible.includes(1)) lastSeen = { ...a };
        const since = a.vanishAt === null ? 0 : t - a.vanishAt;
        ep.ghostCells.push(lastSeen && since >= 1 && since <= 10 ? aroundLastSeen(lastSeen) : []);
        if (t < EVAL_STEPS) advanceWorld(world);
      }
      episodes.push(ep);
    }
  }
  return episodes;
}

function errorSum() { return { squared: 0, copy: 0 }; }
function sampleSum() { return { sum: 0, count: 0 }; }
function eventSum() { return { ...errorSum(), count: 0, min: Infinity, max: 0 }; }
export function trackingTotals() {
  return {
    all: errorSum(), changed: errorSum(), wall: eventSum(), gap: eventSum(),
    ghost: sampleSum(), other: sampleSum(), after: sampleSum(), curve: OFFSETS.map(sampleSum),
  };
}
type Totals = ReturnType<typeof trackingTotals>;

function sample(total: ReturnType<typeof sampleSum>, value: number) {
  total.sum += value;
  total.count++;
}

function saturation(frame: Float64Array, i: number) {
  return Math.hypot(frame[3 * i + 1]!, frame[3 * i + 2]!);
}

export function otherRowCells(exitCells: number[]) {
  const columns = new Set(exitCells.map((i) => i % GRID));
  const rows = exitCells.map((i) => Math.floor(i / GRID));
  const selected: number[] = [];
  for (let row = 0; row < GRID; row++) {
    if (rows.every((exitRow) => Math.abs(exitRow - row) >= 3)) {
      for (const col of columns) selected.push(row * GRID + col);
    }
  }
  return selected;
}

// Predictions are indexed by target frame. No prediction for t = 0 or outside the episode.
export function scoreTrackingEpisode(totals: Totals, ep: TrackingEpisode, predictions: Float64Array[]) {
  for (let t = 1; t < ep.frames.length; t++) {
    const target = ep.frames[t]!, previous = ep.frames[t - 1]!, pred = predictions[t]!;
    for (let i = 0; i < target.length; i += 3) {
      let squared = 0, copy = 0, changed = false;
      for (let j = i; j < i + 3; j++) {
        const delta = previous[j]! - target[j]!;
        squared += (pred[j]! - target[j]!) ** 2;
        copy += delta * delta;
        if (Math.abs(delta) > 0.05) changed = true;
      }
      totals.all.squared += squared;
      totals.all.copy += copy;
      if (changed) {
        totals.changed.squared += squared;
        totals.changed.copy += copy;
      }
    }
    for (const i of ep.ghostCells[t]!) {
      for (let j = 3 * i; j < 3 * i + 3; j++) sample(totals.ghost, Math.abs(pred[j]! - target[j]!));
    }
  }

  for (const { time, cells, kind, length } of ep.events) {
    const group = totals[kind];
    group.count++;
    group.min = Math.min(group.min, length);
    group.max = Math.max(group.max, length);
    for (const i of cells) {
      for (let j = 3 * i; j < 3 * i + 3; j++) {
        group.squared += (predictions[time]![j]! - ep.frames[time]![j]!) ** 2;
        group.copy += (ep.frames[time - 1]![j]! - ep.frames[time]![j]!) ** 2;
      }
    }
    if (kind !== "wall") continue;
    if (time >= 7) {
      for (const [j, offset] of OFFSETS.entries()) {
        const pred = predictions[time + offset];
        if (pred) for (const i of cells) sample(totals.curve[j]!, saturation(pred, i));
      }
    }
    const others = otherRowCells(cells);
    for (let t = Math.max(1, time - 6); t <= time; t++) {
      for (const i of others) sample(totals.other, saturation(predictions[t]!, i));
    }
    for (let t = time + 8; t <= time + 15 && t < ep.frames.length; t++) {
      for (const i of cells) {
        if (saturation(ep.frames[t]!, i) <= 0.05) sample(totals.after, saturation(predictions[t]!, i));
      }
    }
  }
}

export function evaluateTracking(model: Element, episodes: TrackingEpisode[]) {
  const totals = trackingTotals(), buffers = [stepBuffers(model), stepBuffers(model)];
  for (const ep of episodes) {
    let state = blankState(model);
    const predictions: Float64Array[] = [];
    for (let t = 0; t < ep.frames.length - 1; t++) {
      const out = buffers[t % 2]!;
      forward(model, state, ep.frames[t]!, out);
      predictions[t + 1] = out.prediction.slice();
      state = out.state;
    }
    scoreTrackingEpisode(totals, ep, predictions);
  }
  const mean = (s: ReturnType<typeof sampleSum>) => s.sum / s.count;
  const ratio = (s: ReturnType<typeof errorSum>) => s.squared / s.copy;
  const span = ({ count, min, max }: ReturnType<typeof eventSum>) => ({ count, min, max });
  return {
    all: ratio(totals.all), changed: ratio(totals.changed), ghost: mean(totals.ghost),
    wall: ratio(totals.wall), gap: ratio(totals.gap), other: mean(totals.other), after: mean(totals.after),
    curve: totals.curve.map(mean), wallEvents: span(totals.wall), gapEvents: span(totals.gap),
  };
}
export type TrackingScores = ReturnType<typeof evaluateTracking>;
