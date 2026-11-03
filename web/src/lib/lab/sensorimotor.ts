import { createEpisode } from "./agent.ts";
import { referenceModel } from "./feasibility.ts";
import { evaluate, mean } from "./genesis.ts";
import { genesisTask } from "./genesis-task.ts";
import { conditions, type Condition } from "./genesis-training.ts";
import { bodyForRun, oracleGainLabel, sensorimotorTask } from "./sensorimotor-task.ts";
import { GainLearner, trainGainEpisode } from "./sensorimotor-training.ts";
import { scheduleOutdoorLosses } from "./world.ts";

export type GainCondition = Condition | "ideal";
export interface SeedRange { firstRun: number; runs: number; firstEvalSeed: number }

export function gainTrial(id: GainCondition, seed: number, firstEvalSeed: number, rate: number) {
  const body = bodyForRun(seed);
  const models = referenceModel("ideal");
  const policy = conditions[id === "ideal" ? "frozen" : id];
  const { checkpoints, evaluationCount } = sensorimotorTask;
  const measure = () => ({
    ...evaluate(models, policy.gaze, firstEvalSeed, evaluationCount, body),
    gainX: models.gain[0], gainY: models.gain[1],
    calibrationError: (Math.abs(models.gain[0] - body[0]) + Math.abs(models.gain[1] - body[1])) / 2,
  });
  if (id === "ideal" || id === "frozen") {
    if (id === "ideal") models.gain = [...body];
    const point = measure();
    return checkpoints.map(() => point);
  }

  const learner = new GainLearner(models, rate, policy.selection, seed);
  const curve = [measure()];
  for (let episode = 1; episode <= checkpoints.at(-1)!; episode++) {
    const scenario = genesisTask.curriculum[(episode - 1) % genesisTask.curriculum.length]!;
    const ep = createEpisode(scenario, 10000 + 100 * seed + episode, policy.gaze, models, body);
    if (scenario === "nursery") scheduleOutdoorLosses(ep.world);
    trainGainEpisode(learner, ep, id, (shift, estimate) => oracleGainLabel(body, shift, estimate));
    if (checkpoints.some((n) => n === episode)) curve.push(measure());
  }
  return curve;
}

export function gainRuns(id: GainCondition, rate: number, seeds: SeedRange) {
  return Array.from({ length: seeds.runs }, (_, i) => gainTrial(id, seeds.firstRun + i, seeds.firstEvalSeed, rate));
}

export type GainRuns = ReturnType<typeof gainRuns>;

export function averageGainCurve(runs: GainRuns) {
  const units = 2 * sensorimotorTask.evaluationCount;
  return sensorimotorTask.checkpoints.map((step, i) => {
    const points = runs.map((r) => r[i]!);
    const avg = (key: keyof typeof points[number]) => mean(points.map((p) => p[key]));
    return {
      step, accuracy: points.reduce((sum, p) => sum + Math.round(p.accuracy * units), 0) / (units * runs.length),
      retention: avg("retention"), ghostFree: avg("ghostFree"), calibrationError: avg("calibrationError"),
      gainX: avg("gainX"), gainY: avg("gainY"),
    };
  });
}
