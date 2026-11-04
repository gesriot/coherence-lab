import type { World } from "../lab/types.ts";
import { contains } from "../lab/world.ts";
import { blankState, forward, stepBuffers, type Element } from "./element.ts";
import { EVAL_STEPS, PROBE_SEEDS, TEST_SEEDS } from "./evaluation.ts";
import { WINDOW_CELL, type Command, type Gaze, type MotionSettings } from "./gaze.ts";
import { GRID, movingScene, type MotionInput } from "./input.ts";
import { contextGroup, fitContext } from "./motion-context.ts";
import { ridge, rSquared, shuffled } from "./probes.ts";
import { inWall } from "./visibility.ts";

export const metrics = ["all", "moved", "still", "early", "late"] as const;
export const targets = ["own", "world", "apparent"] as const;
export type Metric = (typeof metrics)[number];
export type Target = (typeof targets)[number];
type Errors = Record<Metric, { sum: number; copy: number }>;

function errors(): Errors {
  const zero = () => ({ sum: 0, copy: 0 });
  return { all: zero(), moved: zero(), still: zero(), early: zero(), late: zero() };
}

export function scoreTransition(
  totals: Errors, t: number, executed: Command, prediction: Float64Array,
  current: Float64Array, next: Float64Array,
) {
  let sum = 0, copy = 0;
  for (let i = 0; i < current.length; i++) {
    sum += (prediction[i]! - next[i]!) ** 2;
    copy += (current[i]! - next[i]!) ** 2;
  }
  function add(key: Metric) {
    totals[key].sum += sum;
    totals[key].copy += copy;
  }
  add("all");
  if (executed[0] === 0 && executed[1] === 0) add("still");
  else {
    add("moved");
    if (t < 24) add("early");
    if (t >= 48) add("late");
  }
}

export function probeTarget(world: World, gaze: Gaze, previousAx: number, previousGx: number) {
  const a = world.objects[0]!;
  if (world.t < 1 || (a.vanishAt !== null && world.t >= a.vanishAt)) return;
  const col = Math.floor((a.x - gaze.x) / WINDOW_CELL + GRID / 2);
  const row = Math.floor((a.y - gaze.y) / WINDOW_CELL + GRID / 2);
  if (col < 0 || col >= GRID || row < 0 || row >= GRID) return;
  const x = gaze.x + (col + 0.5 - GRID / 2) * WINDOW_CELL;
  const y = gaze.y + (row + 0.5 - GRID / 2) * WINDOW_CELL;
  if (inWall(world.occluder, x, y) || !contains(a, x, y)) return;
  const own = gaze.x - previousGx, displacement = a.x - previousAx;
  return { cell: row * GRID + col, own, world: displacement, apparent: displacement - own };
}

// The same retinal cell in each frame; positions outside the window are zero padded.
export function rawPatch(cell: number, previous: MotionInput, current: MotionInput) {
  const patch = new Float64Array(56);
  let offset = 0;
  for (const { frame } of [previous, current]) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const r = Math.floor(cell / GRID) + dy, c = cell % GRID + dx;
        if (r >= 0 && r < GRID && c >= 0 && c < GRID) {
          const i = 3 * (r * GRID + c);
          patch.set(frame.subarray(i, i + 3), offset);
        }
        offset += 3;
      }
    }
  }
  patch.set(previous.copy, 54);
  return patch;
}

function predictor(m: Element) {
  let state = blankState(m), t = 0;
  const buffers = [stepBuffers(m), stepBuffers(m)];
  return (input: MotionInput) => {
    const out = buffers[t++ % 2]!;
    forward(m, state, input.frame, out, input.copy);
    state = out.state;
    return out;
  };
}

function collect(
  model: Element, initial: Element, settings: MotionSettings, seeds: readonly number[], totals?: Errors,
) {
  const rows = { model: [] as Float64Array[], initial: [] as Float64Array[], raw: [] as Float64Array[] };
  const groups: string[] = [];
  const labels: Record<Target, number[]> = { own: [], world: [], apparent: [] };
  for (const seed of seeds) {
    for (const scenario of ["continuity", "vanish"] as const) {
      const trainedStep = predictor(model), initialStep = predictor(initial);
      let prev: { input: MotionInput; executed: Command; prediction: Float64Array; ax: number; gx: number } | undefined;
      for (const { world, gaze, command, input } of movingScene(scenario, seed, EVAL_STEPS, settings)) {
        if (prev && totals) scoreTransition(totals, world.t - 1, prev.executed, prev.prediction, prev.input.frame, input.frame);
        const out = trainedStep(input), untrained = initialStep(input);
        if (prev) {
          const sample = probeTarget(world, gaze, prev.ax, prev.gx);
          if (sample) {
            const i = sample.cell * model.depth;
            rows.model.push(out.state.slice(i, i + model.depth));
            rows.initial.push(untrained.state.slice(i, i + model.depth));
            groups.push(contextGroup(scenario, world.t));
            rows.raw.push(rawPatch(sample.cell, prev.input, input));
            for (const key of targets) labels[key].push(sample[key]);
          }
        }
        prev = { input, executed: command, prediction: out.prediction, ax: world.objects[0]!.x, gx: gaze.x };
      }
    }
  }
  return { rows, labels, groups };
}

function evaluateSamples(
  model: Element, initial: Element, settings: MotionSettings,
  trainSeeds: readonly number[] = PROBE_SEEDS, testSeeds: readonly number[] = TEST_SEEDS,
) {
  if (trainSeeds.some((s) => testSeeds.includes(s))) throw new Error("Обучающие и проверочные зёрна зондов пересекаются");
  const train = collect(model, initial, settings, trainSeeds), totals = errors();
  const test = collect(model, initial, settings, testSeeds, totals);
  const ratios = {} as Record<Metric, number>;
  for (const key of metrics) ratios[key] = totals[key].sum / totals[key].copy;
  const fit = {
    model: ridge(train.rows.model), initial: ridge(train.rows.initial), raw: ridge(train.rows.raw),
  };
  const probes = {} as Record<Target, { learned: number; initial: number; raw: number; shuffled: number }>;
  for (const key of targets) {
    const y = train.labels[key];
    const score = (source: keyof typeof fit, labels: number[]) =>
      rSquared(test.labels[key], test.rows[source].map(fit[source](labels)));
    probes[key] = { learned: score("model", y), initial: score("initial", y),
      raw: score("raw", y), shuffled: score("model", shuffled(y)) };
  }
  const scores = { ratios, probes, trainCount: train.labels.own.length, testCount: test.labels.own.length };
  return { scores, train, test };
}

export function evaluateMotion(
  model: Element, initial: Element, settings: MotionSettings,
  trainSeeds?: readonly number[], testSeeds?: readonly number[],
) {
  return evaluateSamples(model, initial, settings, trainSeeds, testSeeds).scores;
}

export function evaluateMotionWithContext(
  model: Element, initial: Element, settings: MotionSettings,
  trainSeeds?: readonly number[], testSeeds?: readonly number[],
) {
  const { scores, train, test } = evaluateSamples(model, initial, settings, trainSeeds, testSeeds);
  const context = {} as Record<Target, number>;
  for (const key of targets) {
    const predict = fitContext(train.groups, train.labels[key]);
    context[key] = rSquared(test.labels[key], test.groups.map(predict));
  }
  return { ...scores, context };
}

export type MotionScores = ReturnType<typeof evaluateMotion>;
