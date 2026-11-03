import { genesisTask } from "./genesis-task.ts";
import { checkpoints, evaluate, mean, trainingCurve } from "./genesis.ts";
import { causes, freshModels, type Models } from "./learning.ts";
import type { Rates } from "./experience.ts";

export const controlNames = {
  frozen: "Без обучения", "keep-all": "Хранить всё", ideal: "Идеальные параметры", oracle: "Оракул",
} as const;
export type Control = keyof typeof controlNames;
export const controls = Object.keys(controlNames) as Control[];

export function requireControl(id: string): Control {
  if (!Object.hasOwn(controlNames, id)) throw new Error(`Недопустимое условие проверки выполнимости: ${id}`);
  return id as Control;
}

export function referenceModel(id: "frozen" | "keep-all" | "ideal"): Models {
  const model = freshModels();
  if (id === "frozen") return model;
  model.weights[0][2] = 1;
  model.weights[1][3] = 1;
  for (const cause of causes) {
    const h = id === "keep-all" ? genesisTask.keepAllHazard : genesisTask.idealHazards[cause];
    model.logits[cause] = Math.log(h / (1 - h));
  }
  return model;
}

export function controlCurve(
  control: Control, seed: number, rates: Rates, stages: readonly number[] = checkpoints,
  firstEvalSeed = 3001, seeds = 20, splitFinal = false,
) {
  requireControl(control);
  const measure = (models: Models, step: number) => {
    const test = (m: Models) => evaluate(m, "first-order", firstEvalSeed, seeds);
    const ideal = referenceModel("ideal");
    const split = splitFinal && control === "oracle" && step === stages.at(-1)
      ? [
          test({ ...ideal, weights: models.weights }),
          test({ ...models, weights: ideal.weights }),
        ]
      : undefined;
    return {
      ...test(models), split, weights: models.weights.flat(), logits: causes.map((c) => models.logits[c]),
    };
  };
  if (control === "oracle") return trainingCurve("oracle", seed, rates, stages, measure);
  const point = measure(referenceModel(control), 0);
  return stages.map(() => point);
}

interface Design {
  firstRun: number;
  runs: number;
  firstEvalSeed: number;
  seeds: number;
  stages: readonly number[];
}

const design: Design = { firstRun: 701, runs: 10, firstEvalSeed: 3001, seeds: 20, stages: checkpoints };

export function designOracle(config: Design = design) {
  const final = [config.stages[config.stages.length - 1]!];
  let best = -1;
  let chosen: Rates = { motion: 0.03, permanence: 0.003 };
  for (const motion of [0.03, 0.1, 0.3, 1]) {
    for (const permanence of [0.003, 0.01, 0.03, 0.1]) {
      const rates = { motion, permanence };
      let successes = 0;
      for (let seed = config.firstRun; seed < config.firstRun + config.runs; seed++) {
        const [point] = controlCurve("oracle", seed, rates, final, config.firstEvalSeed, config.seeds);
        successes += Math.round(point!.accuracy * 2 * config.seeds);
      }
      if (successes > best) { best = successes; chosen = rates; }
    }
  }
  return chosen;
}

export function designCurve(control: Control, rates: Rates, config: Design = design) {
  const units = 2 * config.seeds;
  const runs = control === "oracle" ? config.runs : 1;
  const curves = Array.from({ length: runs }, (_, i) =>
    controlCurve(control, config.firstRun + i, rates, config.stages, config.firstEvalSeed, config.seeds, true));
  return config.stages.map((step, i) => {
    const points = curves.map((r) => r[i]!);
    const avg = (key: "accuracy" | "retention" | "ghostFree" | "occlusion" | "absence" | "noData") =>
      mean(points.map((p) => p[key]));
    return {
      step, accuracy: points.reduce((sum, p) => sum + Math.round(p.accuracy * units), 0) / (runs * units),
      retention: avg("retention"), ghostFree: avg("ghostFree"),
      split: points[0]!.split?.map((_, j) => ({
        accuracy: mean(points.map((p) => p.split![j]!.accuracy)),
        retention: mean(points.map((p) => p.split![j]!.retention)),
        ghostFree: mean(points.map((p) => p.split![j]!.ghostFree)),
      })),
      hazards: [avg("occlusion"), avg("absence"), avg("noData")],
      weights: points[0]!.weights.map((_, j) => mean(points.map((p) => p.weights[j]!))),
    };
  });
}

export function permitsComparison(curves: Record<Control, readonly number[]>): boolean {
  const belowCeiling = (values: readonly number[]) =>
    values.length > 0 && values.every((value) => value <= genesisTask.controlCeiling);
  return belowCeiling(curves.frozen) && belowCeiling(curves["keep-all"])
    && (curves.ideal.at(-1) ?? 0) >= genesisTask.controlFloor
    && (curves.oracle.at(-1) ?? 0) >= genesisTask.controlFloor;
}

export function runFeasibility(ids: Control[] = controls, config = design) {
  ids.forEach(requireControl);
  const rates = ids.includes("oracle") ? designOracle(config) : { motion: 0, permanence: 0 };
  const curves = new Map(ids.map((id) => [id, designCurve(id, rates, config)]));
  return { curves, rates };
}
