import { hashUnit } from "../lab/math.ts";
import type { Element } from "./element.ts";
import { evaluationData, PROBE_SEEDS, probeTrainingData, TEST_SEEDS } from "./evaluation.ts";

function mean(y: number[]) { return y.reduce((a, b) => a + b, 0) / y.length; }

function dot(x: Float64Array, y: Float64Array) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i]! * y[i]!;
  return s;
}

// Solve (X'X + lambda I)w = X'y. Centering leaves the intercept unpenalized.
// Use the equivalent dual system when there are fewer examples than features.
export function ridge(x: Float64Array[]) {
  const n = x.length, d = x[0]!.length;
  const center = new Float64Array(d);
  for (const row of x) for (let j = 0; j < d; j++) center[j] += row[j]!;
  for (let j = 0; j < d; j++) center[j] /= n;
  const centered = x.map((row) => row.map((v, j) => v - center[j]!));
  const dual = n < d;
  const basis = dual ? centered : Array.from({ length: d }, (_, j) =>
    Float64Array.from(centered, (row) => row[j]!));
  const size = basis.length;
  const l = new Float64Array(size * size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j <= i; j++) {
      let v = dot(basis[i]!, basis[j]!) + (i === j ? 0.01 : 0);
      for (let k = 0; k < j; k++) v -= l[i * size + k]! * l[j * size + k]!;
      l[i * size + j] = i === j ? Math.sqrt(v) : v / l[j * size + j]!;
    }
  }
  return function fit(labels: number[]) {
    const intercept = mean(labels);
    const y = Float64Array.from(labels, (v) => v - intercept);
    const solution = dual ? y : Float64Array.from(basis, (col) => dot(col, y));
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < i; j++) solution[i] -= l[i * size + j]! * solution[j]!;
      solution[i] /= l[i * size + i]!;
    }
    for (let i = size - 1; i >= 0; i--) {
      for (let j = i + 1; j < size; j++) solution[i] -= l[j * size + i]! * solution[j]!;
      solution[i] /= l[i * size + i]!;
    }
    const weights = dual ? new Float64Array(d) : solution;
    if (dual) {
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) weights[j] += solution[i]! * centered[i]![j]!;
      }
    }
    const bias = intercept - dot(weights, center);
    return (row: Float64Array) => dot(weights, row) + bias;
  };
}

function shuffled(labels: number[]) {
  const values = labels.slice();
  for (let i = values.length - 1; i > 0; i--) {
    const j = Math.floor(hashUnit(4001 + i) * (i + 1));
    [values[i], values[j]] = [values[j]!, values[i]!];
  }
  return values;
}

function rSquared(y: number[], prediction: number[]) {
  const avg = mean(y);
  let residual = 0, variation = 0;
  for (let i = 0; i < y.length; i++) {
    residual += (prediction[i]! - y[i]!) ** 2;
    variation += (y[i]! - avg) ** 2;
  }
  return 1 - residual / variation;
}

export function evaluate(
  model: Element | null, trainSeeds: readonly number[] = PROBE_SEEDS,
  testSeeds: readonly number[] = TEST_SEEDS,
) {
  if (trainSeeds.some((seed) => testSeeds.includes(seed))) {
    throw new Error("Обучающие и проверочные зёрна зондов пересекаются");
  }
  const train = probeTrainingData(model, trainSeeds);
  const { metrics, rows: test } = evaluationData(model, testSeeds);
  const sign = ridge(train.velocity.x)(train.velocity.y);
  const accuracy = mean(test.velocity.x.map((x, i) =>
    Number((sign(x) >= 0 ? 1 : -1) === test.velocity.y[i])));
  const fit = ridge(train.hidden.x);
  const score = (y: number[]) => rSquared(test.hidden.y, test.hidden.x.map(fit(y)));
  return { ...metrics, accuracy, hidden: score(train.hidden.y), shuffled: score(shuffled(train.hidden.y)) };
}
export type Scores = ReturnType<typeof evaluate>;
