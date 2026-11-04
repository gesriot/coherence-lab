import type { Element, Step } from "./element.ts";
import { product, transpose } from "./matrix.ts";

export function gradientBuffers(m: Element) {
  const { cells: n, depth: k, inputs: f } = m;
  return {
    grad: new Float64Array(m.weights.length), ds: new Float64Array(n * k),
    residual: new Float64Array(n * 3), df: new Float64Array(n * f),
    dz: m.gate ? new Float64Array(n * k) : undefined,
    retained: m.gate ? new Float64Array(n * k) : undefined,
    zt: m.gate ? new Float64Array(k * f) : undefined,
    ft: new Float64Array(n * f), wt: new Float64Array(k * f),
  };
}

// Scatter the transposed spatial filters. Edge means use the forward cell's count;
// contributions to the zero padding have no predecessor and are discarded.
function scatter(m: Element, df: Float64Array, ds: Float64Array) {
  const { cells, inputs: f, depth: k } = m;
  for (let i = 0; i < cells; i++) {
    for (let h = 0; h < k; h++) ds[i * k + h] = df[i * f + h]!;
  }
  for (let i = 0; i < cells; i++) {
    const base = i * f, mean = m.inverseCount[i]!;
    for (let j = 0; j < 8; j++) {
      const other = m.neighbors[8 * i + j]!;
      if (other === cells) continue;
      const index = j < 4 ? j : j + 1;
      const x = index % 3 - 1, y = Math.floor(index / 3) - 1;
      const sx = x * (y === 0 ? 2 : 1) / 8, sy = y * (x === 0 ? 2 : 1) / 8;
      for (let h = 0; h < k; h++) {
        let v = mean * df[base + k + h]!;
        if (m.directed) v += sx * df[base + 2 * k + h]! + sy * df[base + 3 * k + h]!;
        ds[other * k + h] += v;
      }
    }
  }
}

// Mean over the actual segment length, cells and channels. No adjoint crosses
// either end of the segment, even though its initial state may be nonzero.
export function backward(
  m: Element, tape: Step[], targets: Float64Array[], work: ReturnType<typeof gradientBuffers>,
  labels?: Float64Array[], weights?: Float64Array[],
): number {
  const { cells: n, depth: k, inputs: f } = m;
  const { grad, ds, residual, df, ft, wt } = work;
  const bias = m.w.length, readout = bias + k, end = readout + k * 3;
  const gateStart = end + 3, gateBias = gateStart + m.w.length;
  const hintStart = gateBias + k;
  if (m.hint && labels?.length !== targets.length) throw new Error("Нет меток подсказки");
  const norm = 1 / (targets.length * n * 3);
  let loss = 0;
  grad.fill(0);
  ds.fill(0);
  transpose(m.w, wt, f, k);
  if (m.gate) transpose(m.gate.w, work.zt!, f, k);
  for (let t = targets.length - 1; t >= 0; t--) {
    const { prediction, state, features } = tape[t]!;
    const cellWeights = weights?.[t];
    for (let i = 0; i < n; i++) {
      const weight = cellWeights ? cellWeights[i]! : 1;
      for (let d = 0; d < 3; d++) {
        const j = 3 * i + d;
        const error = prediction[j]! - targets[t]![j]!;
        loss += weight * (error * error);
        residual[j] = 2 * norm * error * weight;
        grad[end + d] += residual[j]!;
      }
      let hintError = 0;
      if (m.hint) {
        const error = tape[t]!.occupancy![i]! - labels![t]![i]!;
        loss += 3 * error * error;
        hintError = 6 * norm * error;
        grad[hintStart + k] += hintError;
      }
      for (let h = 0; h < k; h++) {
        const j = i * k + h, s = state[j]!;
        let v = ds[j]!;
        for (let d = 0; d < 3; d++) {
          const e = residual[3 * i + d]!;
          grad[readout + 3 * h + d] += s * e;
          v += m.u[3 * h + d]! * e;
        }
        if (m.hint) {
          grad[hintStart + h] += hintError * s;
          v += hintError * m.hint.v[h]!;
        }
        if (m.gate) {
          const z = tape[t]!.gate![j]!, candidate = tape[t]!.candidate![j]!;
          work.dz![j] = v * (candidate - features[i * f + h]!) * z * (1 - z);
          grad[gateBias + h] += work.dz![j]!;
          work.retained![j] = v * (1 - z);
          ds[j] = v * z * (1 - candidate * candidate);
        } else ds[j] = v * (1 - s * s);
        grad[bias + h] += ds[j]!;
      }
    }
    transpose(features, ft, n, f);
    product(ft, ds, grad, f, n, k, true);
    if (m.gate) product(ft, work.dz!, grad.subarray(gateStart, gateBias), f, n, k, true);
    if (t > 0) {
      product(ds, wt, df, n, k, f);
      if (m.gate) product(work.dz!, work.zt!, df, n, k, f, true);
      scatter(m, df, ds);
      if (m.gate) for (let j = 0; j < ds.length; j++) ds[j] += work.retained![j]!;
    }
  }
  return loss * norm;
}
