import type { Element } from "./element.ts";
import { evaluationData, PROBE_SEEDS, probeTrainingData, TEST_SEEDS, type Examples } from "./evaluation.ts";
import { ridge, rSquared, shuffled } from "./probes.ts";

function probe(train: Examples, test: Examples, key: keyof Examples) {
  const fit = ridge(train[key].x), truth = test[key].y;
  function score(labels: number[]) {
    const predictions = test[key].x.map(fit(labels));
    if (key !== "velocity") return rSquared(truth, predictions);
    return predictions.filter((v, i) => (v >= 0 ? 1 : -1) === truth[i]).length / truth.length;
  }
  return { value: score(train[key].y), shuffled: score(shuffled(train[key].y)) };
}

export function evaluatePersistence(
  model: Element | null, trainSeeds: readonly number[] = PROBE_SEEDS,
  testSeeds: readonly number[] = TEST_SEEDS,
) {
  if (trainSeeds.some((seed) => testSeeds.includes(seed))) {
    throw new Error("Обучающие и проверочные зёрна зондов пересекаются");
  }
  const train = probeTrainingData(model, trainSeeds, "per-cell");
  const { metrics, hint, diagnostic, rows } = evaluationData(model, testSeeds, "per-cell");
  return {
    ...metrics, hint, diagnostic,
    velocity: probe(train, rows, "velocity"),
    hidden: probe(train, rows, "hidden"),
    late: probe(train, rows, "late"),
  };
}
export type PersistenceScores = ReturnType<typeof evaluatePersistence>;
