import { hueDelta } from "./math.ts";
import type { MotionSample, Track } from "./types.ts";

export type Vector = [number, number];
export type Features = [number, number, number, number, number];
export const causes = ["occlusion", "absence", "no-data"] as const;
export type Cause = typeof causes[number];
type Exposure = Record<Cause, number>;
export interface Models {
  weights: [Features, Features];
  logits: Exposure;
}

export function freshModels(): Models {
  const logit = Math.log(0.28 / 0.72);
  return {
    weights: [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    logits: { occlusion: logit, absence: logit, "no-data": logit },
  };
}

export function features(samples: MotionSample[]): Features {
  const displacement = (i: number): Vector => {
    if (i < 1) return [0, 0];
    const a = samples[i - 1]!, b = samples[i]!;
    return [(b.x - a.x) / (b.step - a.step), (b.y - a.y) / (b.step - a.step)];
  };
  return [...displacement(samples.length - 1), ...displacement(samples.length - 2), 1];
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

export function hazard(models: Models, cause: Cause): number {
  return 1 / (1 + Math.exp(-models.logits[cause]));
}

export interface GapExample {
  exposure: Exposure;
  last: Cause;
  found: boolean;
}

export function fitPermanence(models: Models, gap: GapExample, rate: number): void {
  for (const c of causes) {
    const event = Number(!gap.found && gap.last === c);
    // Exposure includes the terminal step. Its Bernoulli target is one only for a loss.
    models.logits[c] += rate * (event - hazard(models, c) * gap.exposure[c]);
  }
}

interface Gap {
  exposure: Exposure;
  last: Cause | null;
  start: { x: number; y: number; step: number };
  trace: Track;
  deletedAt: number | null;
}

export interface ClosedGap extends GapExample {
  id: number;
  start: Gap["start"];
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
        exposure: { occlusion: 0, absence: 0, "no-data": 0 }, last: null,
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
      gap.trace = predicted;
      const reason = predicted.reason;
      if (!found && (reason === "occlusion" || reason === "absence" || reason === "no-data")) {
        gap.exposure[reason]++;
        gap.last = reason;
      }
      if (found || (gap.deletedAt != null && step - gap.deletedAt >= 30)) {
        if (gap.last) closed.push({ id, start: gap.start, exposure: gap.exposure, last: gap.last, found });
        this.pending.delete(id);
      }
    }
    return closed;
  }
}
