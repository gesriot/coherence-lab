import { experimentSeeds, genesisTask } from "./genesis-task.ts";
import { createEpisode, defaultConfig, snapshot, stepEpisode, type Episode } from "./agent.ts";
import { generator, Learner, type Rates } from "./experience.ts";
import { conditions, trainEpisode, type Condition, type Labels } from "./genesis-training.ts";
import { probability, type Models } from "./learning.ts";
import type { Ablation, LabConfig } from "./types.ts";
import { hiddenByOccluder } from "./world.ts";

export const { checkpoints } = genesisTask;
const { curriculum, ghostDelay } = genesisTask;
const EVAL_SEEDS = experimentSeeds.evaluationCount;
export const tuningAreaUnits = (checkpoints.length - 1) * 2 * EVAL_SEEDS;

function nearest<T extends { x: number; y: number }>(objects: T[], position: { x: number; y: number }): T {
  let best = objects[0]!, distance = Infinity;
  for (const obj of objects) {
    const d = (obj.x - position.x) ** 2 + (obj.y - position.y) ** 2;
    if (d < distance) { best = obj; distance = d; }
  }
  return best;
}

export function confirmsForecast(
  object: { x: number; y: number; vanishAt: number | null }, step: number,
  forecast: { x: number; y: number; radius: number },
): boolean {
  return (object.vanishAt == null || object.vanishAt > step)
    && Math.hypot(object.x - forecast.x, object.y - forecast.y) <= forecast.radius;
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
      const now = frames[ep.world.t]!.find((o) => o.id === obj.id)!;
      return confirmsForecast(now, ep.world.t, gap.forecast);
    },
  };
}

export interface Point {
  accuracy: number;
  retention: number;
  ghostFree: number;
  untracked: number;
  motionError: number;
  hidden: number;
  open: number;
  miss: number;
}

export function evaluateDisappearance(models: Models, seed: number, cfg: LabConfig) {
  const ep = createEpisode("vanish", seed, cfg.ablation, models);
  const at = ep.world.objects[0]!.vanishAt!;
  // Identify A's track on the last visible frame, before the first empty observation.
  while (ep.world.t < at - 1) stepEpisode(ep, cfg);
  const id = snapshot(ep, cfg.ablation).evaluator.objects[0]!.nearestTrack;
  if (id == null) return "untracked";
  while (ep.world.t < at + ghostDelay) stepEpisode(ep, cfg);
  return ep.agent.tracks.some((t) => t.id === id) ? "ghost" : "cleared";
}

export function evaluate(models: Models, ablation: Ablation, firstSeed: number, seeds: number = EVAL_SEEDS): Point {
  let held = 0, clean = 0, untracked = 0, error = 0, predictions = 0;
  const cfg = { ...defaultConfig(), ablation };
  for (let seed = firstSeed; seed < firstSeed + seeds; seed++) {
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
    const outcome = evaluateDisappearance(models, seed, cfg);
    if (outcome === "untracked") untracked++;
    if (outcome === "cleared") clean++;
  }
  return {
    accuracy: (held + clean) / (2 * seeds), retention: held / seeds,
    ghostFree: clean / seeds, untracked, motionError: error / predictions,
    hidden: probability(models, "hidden"), open: probability(models, "open"), miss: probability(models, "miss"),
  };
}

export function trainingCurve<T>(
  condition: Condition, seed: number, rates: Rates, stages: readonly number[],
  measure: (models: Models, episode: number) => T,
): T[] {
  const learner = new Learner(rates, conditions[condition].selection, seed);
  const curve: T[] = [];
  if (stages.includes(0)) curve.push(measure(learner.models, 0));
  for (let episode = 1; episode <= stages[stages.length - 1]!; episode++) {
    const scenario = curriculum[(episode - 1) % curriculum.length]!;
    trainEpisode(learner, condition, scenario, 10000 + 100 * seed + episode, oracleLabels);
    if (stages.includes(episode)) curve.push(measure(learner.models, episode));
  }
  return curve;
}

export function runTrial(condition: Condition, seed: number, firstEvalSeed: number, rates: Rates): Point[] {
  return trainingCurve(condition, seed, rates, checkpoints, (models) =>
    evaluate(models, conditions[condition].gaze, firstEvalSeed));
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
