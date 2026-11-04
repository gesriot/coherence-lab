import { createEpisode } from "./agent.ts";
import { referenceModel } from "./feasibility.ts";
import { evaluate, mean } from "./genesis.ts";
import { genesisTask } from "./genesis-task.ts";
import { conditions, type Condition } from "./genesis-training.ts";
import type { Vector } from "./learning.ts";
import { bodyForRun, oracleGainLabel, sensorimotorTask } from "./sensorimotor-task.ts";
import { GainLearner, trainGainEpisode } from "./sensorimotor-training.ts";
import { scheduleOutdoorLosses } from "./world.ts";

export type GainCondition = Condition | "ideal";
export interface SeedRange { firstRun: number; runs: number; firstEvalSeed: number }

export function calibrationError(estimate: Vector, body: Vector): number {
  return (Math.abs(estimate[0] - body[0]) + Math.abs(estimate[1] - body[1])) / 2;
}

export function calibrationPoint(step: number, estimate: Vector, body: Vector, initialError: number) {
  const error = calibrationError(estimate, body);
  return { step, error, q: 1 - error / initialError, gainX: estimate[0], gainY: estimate[1] };
}

export function gainArea(curve: readonly { q: number }[]): number {
  return mean(curve.slice(1).map((p) => p.q));
}

export function gainTrial(id: GainCondition, seed: number, firstEvalSeed: number, rate: number) {
  const body = bodyForRun(seed);
  const models = referenceModel("ideal");
  const condition = id === "ideal" ? "frozen" : id;
  const policy = conditions[condition];
  const { checkpoints, evaluationCount } = sensorimotorTask;
  const initialError = calibrationError(models.gain, body);
  const learner = condition === "frozen" ? null : new GainLearner(models, rate, policy.selection, seed);
  if (id === "ideal") models.gain = [...body];

  let step = 0, saccades = 0;
  const curve = [calibrationPoint(step, models.gain, body, initialError)];
  const episodes = checkpoints.at(-1)! / genesisTask.trainingSteps;
  for (let episode = 1; episode <= episodes; episode++) {
    const scenario = genesisTask.curriculum[(episode - 1) % genesisTask.curriculum.length]!;
    const ep = createEpisode(scenario, 10000 + 100 * seed + episode, policy.gaze, models, body);
    if (scenario === "nursery") scheduleOutdoorLosses(ep.world);
    trainGainEpisode(learner, ep, condition, (shift, estimate) => oracleGainLabel(body, shift, estimate), (shift) => {
      step++;
      if (Math.hypot(...shift) > 4) saccades++;
      if (step === checkpoints[curve.length]) curve.push(calibrationPoint(step, models.gain, body, initialError));
    });
  }
  const secondary = evaluate(models, policy.gaze, firstEvalSeed, evaluationCount, body);
  return { curve, secondary, saccadesPerEpisode: saccades / episodes };
}

export function gainRuns(id: GainCondition, rate: number, seeds: SeedRange) {
  return Array.from({ length: seeds.runs }, (_, i) => gainTrial(id, seeds.firstRun + i, seeds.firstEvalSeed, rate));
}

export type GainRuns = ReturnType<typeof gainRuns>;

export function summarizeGainRuns(runs: GainRuns) {
  const units = 2 * sensorimotorTask.evaluationCount;
  const accuracy = mean(runs.map((r) => Math.round(r.secondary.accuracy * units))) / units;
  const curve = sensorimotorTask.checkpoints.map((step, i) => {
    const points = runs.map((r) => r.curve[i]!);
    return {
      step, q: mean(points.map((p) => p.q)), error: mean(points.map((p) => p.error)),
      gainX: mean(points.map((p) => p.gainX)), gainY: mean(points.map((p) => p.gainY)),
    };
  });
  return {
    curve, accuracy, retention: mean(runs.map((r) => r.secondary.retention)),
    ghostFree: mean(runs.map((r) => r.secondary.ghostFree)), saccadesPerEpisode: mean(runs.map((r) => r.saccadesPerEpisode)),
  };
}
