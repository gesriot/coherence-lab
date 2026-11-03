import { clamp, hueDelta } from "./math.ts";
import type { MotionSample, Track } from "./types.ts";

export type Vector = [number, number];
export type Features = [number, number, number, number, number];
export type Context = "hidden" | "open";
export interface GapStep {
  context: Context;
  observation: "empty" | "none";
}
export const existenceKeys = ["hidden", "open", "miss"] as const;
type Existence = Record<typeof existenceKeys[number], number>;
export interface Models {
  weights: [Features, Features];
  logits: Existence;
}

export function existenceLogits(probabilities: Existence): Existence {
  const logit = (p: number) => Math.log(p / (1 - p));
  return { hidden: logit(probabilities.hidden), open: logit(probabilities.open), miss: logit(probabilities.miss) };
}

export function freshModels(): Models {
  return {
    weights: [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    logits: existenceLogits({ hidden: 0.28, open: 0.28, miss: 0.5 }),
  };
}

export function features(samples: MotionSample[]): Features {
  if (samples.length < 2) return [0, 0, 0, 0, 1];
  const first = samples[0]!;
  const previous = samples[samples.length - 2]!;
  const last = samples[samples.length - 1]!;
  const recentSteps = last.step - previous.step;
  const observedSteps = last.step - first.step;
  return [
    (last.x - previous.x) / recentSteps, (last.y - previous.y) / recentSteps,
    (last.x - first.x) / observedSteps, (last.y - first.y) / observedSteps, 1,
  ];
}

export function motion(models: Models, input: Features): Vector {
  const dot = (w: Features) => w.reduce((sum, value, i) => sum + value * input[i]!, 0);
  return [dot(models.weights[0]), dot(models.weights[1])];
}

export interface MotionExample {
  input: Features;
  target: Vector;
}

export function fitMotion(models: Models, example: MotionExample, rate: number): void {
  const { input, target } = example;
  const prediction = motion(models, input);
  const norm = 1 + input.reduce((sum, x) => sum + x * x, 0);
  for (const axis of [0, 1] as const) {
    const gain = rate * (target[axis] - prediction[axis]) / norm;
    for (let i = 0; i < input.length; i++) models.weights[axis][i]! += gain * input[i]!;
  }
}

export function probability(models: Models, key: keyof Existence): number {
  return 1 / (1 + Math.exp(-models.logits[key]));
}

export function filterExistence(models: Models, belief: number, step: GapStep): number {
  const predicted = belief * (1 - probability(models, step.context));
  if (step.observation === "none") return predicted;
  const missed = predicted * probability(models, "miss");
  return missed / (missed + (1 - predicted));
}

export interface GapExample {
  seen: number;
  steps: GapStep[];
  found: boolean;
}

export function gapLogLikelihood(models: Models, gap: GapExample): number {
  const hidden = probability(models, "hidden"), open = probability(models, "open");
  const miss = probability(models, "miss");
  const detected = gap.seen * (Math.log1p(-open) + Math.log1p(-miss));
  const decay = { hidden: Math.log1p(-hidden), open: Math.log1p(-open) };
  const logMiss = Math.log(miss);
  let logSurvival = 0, lost = 0;
  for (const { context, observation } of gap.steps) {
    // Loss on this step precedes its observation. Later empty views then have probability one.
    if (!gap.found) lost += Math.exp(logSurvival) * (context === "hidden" ? hidden : open);
    logSurvival += decay[context] + (observation === "empty" ? logMiss : 0);
  }
  return detected + (gap.found ? logSurvival : Math.log(lost));
}

export function fitExistence(models: Models, gap: GapExample, rate: number): void {
  const delta = 1e-4;
  const next = { ...models.logits };
  for (const key of existenceKeys) {
    const theta = models.logits[key];
    models.logits[key] = theta + delta;
    const plus = gapLogLikelihood(models, gap);
    models.logits[key] = theta - delta;
    const minus = gapLogLikelihood(models, gap);
    models.logits[key] = theta;
    next[key] = clamp(theta + rate * (plus - minus) / (2 * delta), -9, 9);
  }
  models.logits = next;
}

interface Gap {
  seen: number;
  steps: GapStep[];
  start: { x: number; y: number; step: number };
  trace: Track;
  deletedAt: number | null;
}

export interface ClosedGap extends GapExample {
  id: number;
  start: Gap["start"];
  forecast: { x: number; y: number; radius: number };
}

/** Owned by one episode. Unresolved records are censored when this book is discarded. */
export class GapBook {
  private pending = new Map<number, Gap>();

  step(
    step: number, before: Track[], after: Track[],
    advance: (track: Track) => Track, radius: (track: Track) => number,
  ): ClosedGap[] {
    const closed: ClosedGap[] = [];
    for (const prior of before) {
      if (prior.boundBlob == null || after.some((t) => t.id === prior.id && t.boundBlob != null)) continue;
      this.pending.set(prior.id, {
        seen: prior.seenRun ?? 0, steps: [],
        start: { x: prior.x + prior.vx, y: prior.y + prior.vy, step }, trace: prior, deletedAt: null,
      });
    }
    const births = after.filter((t) => t.age === 1);
    for (const [id, gap] of this.pending) {
      const live = after.find((t) => t.id === id);
      const predicted = live ?? advance(gap.trace);
      let found = live != null && live.boundBlob != null;
      if (!live) {
        gap.deletedAt ??= step;
        const index = births.findIndex((b) =>
          Math.hypot(b.x - predicted.x, b.y - predicted.y) <= radius(gap.trace) &&
          hueDelta(b.hue, predicted.hue) <= 40);
        if (index >= 0) {
          found = true;
          births.splice(index, 1);
        }
      }
      const previous = gap.trace;
      gap.trace = predicted;
      if (!found) gap.steps.push(predicted.gapStep!);
      if (found || (gap.deletedAt != null && step - gap.deletedAt >= 30)) {
        if (gap.steps.length) {
          const forecast = { x: previous.x + previous.vx, y: previous.y + previous.vy, radius: radius(previous) };
          closed.push({ id, start: gap.start, forecast, seen: gap.seen, steps: gap.steps, found });
        }
        this.pending.delete(id);
      }
    }
    return closed;
  }
}
