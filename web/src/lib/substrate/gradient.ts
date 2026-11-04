import type { Element, Step } from "./element.ts";
import { product, transpose } from "./matrix.ts";

export function gradientBuffers(m: Element) {
  const { cells: n, depth: k, inputs: f } = m;
  return {
    grad: new Float64Array(m.weights.length), ds: new Float64Array(n * k),
    residual: new Float64Array(n * 3), df: new Float64Array(n * f),
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
): number {
  const { cells: n, depth: k, inputs: f } = m;
  const { grad, ds, residual, df, ft, wt } = work;
  const bias = m.w.length, readout = bias + k, end = readout + k * 3;
  const norm = 1 / (targets.length * n * 3);
  let loss = 0;
  grad.fill(0);
  ds.fill(0);
  transpose(m.w, wt, f, k);
  for (let t = targets.length - 1; t >= 0; t--) {
    const { prediction, state, features } = tape[t]!;
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < 3; d++) {
        const j = 3 * i + d;
        const error = prediction[j]! - targets[t]![j]!;
        loss += error * error;
        residual[j] = 2 * norm * error;
        grad[end + d] += residual[j]!;
      }
      for (let h = 0; h < k; h++) {
        const j = i * k + h, s = state[j]!;
        let v = ds[j]!;
        for (let d = 0; d < 3; d++) {
          const e = residual[3 * i + d]!;
          grad[readout + 3 * h + d] += s * e;
          v += m.u[3 * h + d]! * e;
        }
        ds[j] = v * (1 - s * s);
        grad[bias + h] += ds[j]!;
      }
    }
    transpose(features, ft, n, f);
    product(ft, ds, grad, f, n, k, true);
    if (t > 0) {
      product(ds, wt, df, n, k, f);
      scatter(m, df, ds);
    }
  }
  return loss * norm;
}
