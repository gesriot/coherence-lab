import { createEpisode, defaultConfig, snapshot, stepEpisode, type Episode } from "./agent.ts";
import { generator, Learner, type Rates } from "./experience.ts";
import { conditions, trainEpisode, type Condition, type Labels } from "./genesis-training.ts";
import { hazard, type Models } from "./learning.ts";
import type { Ablation, ScenarioId } from "./types.ts";
import { hiddenByOccluder } from "./world.ts";

export const checkpoints = [0, 4, 8, 16, 32] as const;
const curriculum: ScenarioId[] = ["continuity", "vanish", "binding", "nursery"];
const EVAL_SEEDS = 20;

function nearest<T extends { x: number; y: number }>(objects: T[], position: { x: number; y: number }): T {
  let best = objects[0]!, distance = Infinity;
  for (const obj of objects) {
    const d = (obj.x - position.x) ** 2 + (obj.y - position.y) ** 2;
    if (d < distance) { best = obj; distance = d; }
  }
  return best;
}

/** Privileged labels live in the experiment driver. Neither the agent nor replay receives a world. */
function oracleLabels(ep: Episode): Labels {
  const frame = () => ep.world.objects.map((o) => ({ id: o.id, x: o.x, y: o.y, vanishAt: o.vanishAt }));
  const frames = [frame()];
  return {
    observe() { frames.push(frame()); },
    displacement(previous) {
      const old = nearest(frames[ep.world.t - 1]!, previous);
      const now = frames[ep.world.t]!.find((o) => o.id === old.id)!;
      return [now.x - old.x, now.y - old.y];
    },
    found(gap) {
      const obj = nearest(frames[gap.start.step]!, gap.start);
      return obj.vanishAt == null || obj.vanishAt > ep.world.t;
    },
  };
}

export interface Point {
  accuracy: number;
  retention: number;
  ghostFree: number;
  untracked: number;
  motionError: number;
  occlusion: number;
  absence: number;
  noData: number;
}

function evaluate(models: Models, ablation: Ablation, firstSeed: number): Point {
  let held = 0, clean = 0, untracked = 0, error = 0, predictions = 0;
  const cfg = { ...defaultConfig(), ablation };
  for (let seed = firstSeed; seed < firstSeed + EVAL_SEEDS; seed++) {
    const ep = createEpisode("continuity", seed, ablation, models);
    let hidden = false, exit = 0;
    for (let step = 1; step <= 72; step++) {
      const forecasts = ep.agent.tracks.filter((t) => t.boundBlob != null).map((t) => ({
        object: nearest(ep.world.objects, t), x: t.x + t.vx, y: t.y + t.vy,
      }));
      stepEpisode(ep, cfg);
      // Each reference now holds the same object's actual position one step later.
      for (const f of forecasts) {
        error += Math.hypot(f.x - f.object.x, f.y - f.object.y);
        predictions++;
      }
      if (hiddenByOccluder(ep.world.objects[0]!, ep.world.occluder)) hidden = true;
      else if (hidden && !exit) exit = step;
      if (exit && step === exit + 1 && snapshot(ep, ablation).evaluator.retained) held++;
    }
    const vanish = createEpisode("vanish", seed, ablation, models);
    const at = vanish.world.objects[0]!.vanishAt!;
    let id: number | null = null;
    for (let step = 1; step <= 40; step++) {
      stepEpisode(vanish, cfg);
      if (step === at) {
        id = snapshot(vanish, ablation).evaluator.objects[0]!.nearestTrack;
        if (id == null) untracked++;
      }
      if (step === at + 8 && id != null && !vanish.agent.tracks.some((t) => t.id === id)) clean++;
    }
  }
  return {
    accuracy: (held + clean) / (2 * EVAL_SEEDS), retention: held / EVAL_SEEDS,
    ghostFree: clean / EVAL_SEEDS, untracked, motionError: error / predictions,
    occlusion: hazard(models, "occlusion"), absence: hazard(models, "absence"), noData: hazard(models, "no-data"),
  };
}

export function runTrial(condition: Condition, seed: number, firstEvalSeed: number, rates: Rates): Point[] {
  const { selection, gaze } = conditions[condition];
  const learner = new Learner(rates, selection, seed);
  const curve = [evaluate(learner.models, gaze, firstEvalSeed)];
  for (let episode = 1; episode <= 32; episode++) {
    const scenario = curriculum[(episode - 1) % curriculum.length]!;
    trainEpisode(learner, condition, scenario, 10000 + 100 * seed + episode, oracleLabels);
    if (checkpoints.some((n) => n === episode)) curve.push(evaluate(learner.models, gaze, firstEvalSeed));
  }
  return curve;
}

export function area(curve: Point[]): number {
  return mean(curve.slice(1).map((p) => p.accuracy));
}

export function mean(values: number[]): number {
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

export interface Estimate { mean: number; low: number; high: number }

export function bootstrap(values: number[]): Estimate {
  const random = generator(12345);
  const samples: number[] = [];
  for (let i = 0; i < 2000; i++) {
    let sum = 0;
    for (let j = 0; j < values.length; j++) sum += values[Math.floor(random() * values.length)]!;
    samples.push(sum / values.length);
  }
  samples.sort((a, b) => a - b);
  const quantile = (p: number) => {
    const index = p * (samples.length - 1);
    const lo = Math.floor(index);
    return samples[lo]! + (index - lo) * (samples[Math.ceil(index)]! - samples[lo]!);
  };
  return { mean: mean(values), low: quantile(0.025), high: quantile(0.975) };
}

/** Bootstrap integer success counts, then rescale. This preserves exact gate boundaries. */
export function scoreInterval(scores: number[], units: number): Estimate {
  const counts = bootstrap(scores.map((v) => Math.round(v * units)));
  return { mean: counts.mean / units, low: counts.low / units, high: counts.high / units };
}

export function effect(e: Estimate): "faster" | "slower" | "none" {
  if (e.mean > 0.10 && e.low > 0) return "faster";
  if (e.mean < -0.10 && e.high < 0) return "slower";
  return "none";
}

export function feasibility(oracle32: number, frozen: number[], full4: number, first4: number) {
  return {
    positive: oracle32 >= 0.9, lower: frozen.every((p) => p <= 0.75),
    saturated: full4 >= 0.9 && first4 >= 0.9,
  };
}
