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
  model.weights[0][4] = 1.65;
  const ideal = { occlusion: 0.005, absence: 0.5, "no-data": 0.03 };
  for (const cause of causes) {
    const h = id === "keep-all" ? 0.001 : ideal[cause];
    model.logits[cause] = Math.log(h / (1 - h));
  }
  return model;
}

export function controlCurve(
  control: Control, seed: number, rates: Rates, stages: readonly number[] = checkpoints,
  firstEvalSeed = 3001, seeds = 20,
) {
  requireControl(control);
  const measure = (models: Models) => ({
    ...evaluate(models, "first-order", firstEvalSeed, seeds),
    weights: models.weights.flat(), logits: causes.map((c) => models.logits[c]),
  });
  if (control === "oracle") return trainingCurve("oracle", seed, rates, stages, measure);
  const point = measure(referenceModel(control));
  return stages.map(() => point);
}

export function designOracle() {
  const final = [checkpoints[checkpoints.length - 1]!];
  let best = -1;
  let chosen: Rates = { motion: 0.03, permanence: 0.003 };
  for (const motion of [0.03, 0.1, 0.3, 1]) {
    for (const permanence of [0.003, 0.01, 0.03, 0.1]) {
      const rates = { motion, permanence };
      let successes = 0;
      for (let seed = 701; seed <= 710; seed++) {
        const [point] = controlCurve("oracle", seed, rates, final);
        successes += Math.round(point!.accuracy * 40);
      }
      if (successes > best) { best = successes; chosen = rates; }
    }
  }
  return chosen;
}

export function designCurve(control: Control, rates: Rates) {
  const runs = control === "oracle" ? 10 : 1;
  const curves = Array.from({ length: runs }, (_, i) => controlCurve(control, 701 + i, rates));
  return checkpoints.map((step, i) => {
    const points = curves.map((r) => r[i]!);
    const avg = (key: "accuracy" | "retention" | "ghostFree" | "occlusion" | "absence" | "noData") =>
      mean(points.map((p) => p[key]));
    return {
      step, accuracy: points.reduce((sum, p) => sum + Math.round(p.accuracy * 40), 0) / (runs * 40),
      retention: avg("retention"), ghostFree: avg("ghostFree"),
      hazards: [avg("occlusion"), avg("absence"), avg("noData")],
      weights: points[0]!.weights.map((_, j) => mean(points.map((p) => p.weights[j]!))),
    };
  });
}
