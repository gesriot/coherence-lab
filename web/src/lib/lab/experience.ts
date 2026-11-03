import { hashUnit } from "./math.ts";
import {
  fitMotion, fitExistence, freshModels, gapLogLikelihood, motion,
  type GapExample, type Models, type MotionExample,
} from "./learning.ts";

export type Selection = "uniform" | "attention" | "error";
export interface Rates { motion: number; existence: number }

export function generator(seed: number): () => number {
  let draw = 0;
  return () => hashUnit(seed + Math.imul(++draw, 0x9e3779b9));
}

export class Replay<T> {
  readonly entries: { value: T; weight: number }[] = [];
  readonly capacity: number;

  constructor(capacity: number) { this.capacity = capacity; }

  add(value: T, weight: number): void {
    if (this.entries.length === this.capacity) this.entries.shift();
    this.entries.push({ value, weight });
  }

  sample(random: () => number, uniform: boolean): T | undefined {
    if (!this.entries.length) return undefined;
    const u = random();
    if (uniform) return this.entries[Math.floor(u * this.entries.length)]!.value;
    let threshold = u * this.entries.reduce((total, e) => total + e.weight, 0);
    for (const e of this.entries) {
      threshold -= e.weight;
      if (threshold < 0) return e.value;
    }
    return this.entries[this.entries.length - 1]!.value;
  }
}

export class Learner {
  readonly models = freshModels();
  readonly movements = new Replay<MotionExample>(256);
  readonly gaps = new Replay<GapExample>(32);
  readonly updates = { motion: 0, existence: 0 };
  readonly selection: Selection;
  readonly rates: Rates;
  private readonly random: () => number;

  constructor(rates: Rates, selection: Selection, seed: number) {
    this.rates = rates;
    this.selection = selection;
    this.random = generator(seed);
  }

  rememberMotion(example: MotionExample, attentionError: number | null): void {
    if (this.selection === "attention" && attentionError == null) return;
    let priority = attentionError ?? 0;
    if (this.selection === "error") {
      const [x, y] = motion(this.models, example.input);
      priority = Math.hypot(example.target[0] - x, example.target[1] - y);
    }
    this.movements.add(example, 0.05 + priority);
  }

  rememberGap(example: GapExample, attentionError: number | null): void {
    if (this.selection === "attention" && attentionError == null) return;
    const priority = this.selection === "error" ? -gapLogLikelihood(this.models, example) / (example.steps.length + example.seen) : attentionError ?? 0;
    this.gaps.add(example, 0.05 + priority);
  }

  update(): void {
    const uniform = this.selection === "uniform";
    const movement = this.movements.sample(this.random, uniform);
    const gap = this.gaps.sample(this.random, uniform);
    if (movement) {
      fitMotion(this.models, movement, this.rates.motion);
      this.updates.motion++;
    }
    if (gap) {
      fitExistence(this.models, gap, this.rates.existence);
      this.updates.existence++;
    }
  }
}
