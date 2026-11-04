import { hashUnit } from "../lab/math.ts";
import { product } from "./matrix.ts";

export const CHANNEL_COUNT = 3;

export interface Element {
  grid: number;
  cells: number;
  depth: number;
  directed: boolean;
  inputs: number;
  // Features: own state, neighbor mean, optional Sobel x/y, then sensory channels.
  weights: Float64Array;
  w: Float64Array;
  b: Float64Array;
  u: Float64Array;
  c: Float64Array;
  neighbors: Int32Array;
  inverseCount: Float64Array;
}

export function createElement(seed: number, directed = true, grid = 16, depth = 16): Element {
  const cells = grid * grid;
  const inputs = (directed ? 4 : 2) * depth + CHANNEL_COUNT;
  const bias = inputs * depth;
  const readout = bias + depth;
  const end = readout + depth * CHANNEL_COUNT;
  const weights = new Float64Array(end + CHANNEL_COUNT);
  for (let i = 0; i < weights.length; i++) {
    // Omit the two Sobel blocks in the control; pair every shared initial parameter.
    const index = !directed && i >= 2 * depth * depth ? i + 2 * depth * depth : i;
    weights[i] = (2 * hashUnit(7919 * seed + index) - 1) * 0.3;
  }
  const neighbors = new Int32Array(cells * 8);
  const inverseCount = new Float64Array(cells);
  for (let i = 0; i < cells; i++) {
    let count = 0, n = 8 * i;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        const x = i % grid + dx, y = Math.floor(i / grid) + dy;
        const inside = x >= 0 && x < grid && y >= 0 && y < grid;
        neighbors[n++] = inside ? y * grid + x : cells;
        if (inside) count++;
      }
    }
    inverseCount[i] = count === 0 ? 0 : 1 / count;
  }
  return {
    grid, cells, depth, directed, inputs, weights, neighbors, inverseCount,
    w: weights.subarray(0, bias), b: weights.subarray(bias, readout),
    u: weights.subarray(readout, end), c: weights.subarray(end),
  };
}

// The last row is permanent zero padding for the spatial filters.
export function blankState(m: Element): Float64Array {
  return new Float64Array((m.cells + 1) * m.depth);
}

export function features(m: Element, prev: Float64Array, x: Float64Array, out: Float64Array) {
  const { cells, depth: k, inputs: f, neighbors: ns } = m;
  for (let i = 0; i < cells; i++) {
    const base = i * f, n = i * 8;
    const nw = ns[n]! * k, north = ns[n + 1]! * k, ne = ns[n + 2]! * k;
    const west = ns[n + 3]! * k, east = ns[n + 4]! * k;
    const sw = ns[n + 5]! * k, south = ns[n + 6]! * k, se = ns[n + 7]! * k;
    const scale = m.inverseCount[i]!;
    for (let h = 0; h < k; h++) {
      const a = prev[nw + h]!, b = prev[north + h]!, c = prev[ne + h]!;
      const d = prev[west + h]!, e = prev[east + h]!;
      const g = prev[sw + h]!, v = prev[south + h]!, z = prev[se + h]!;
      out[base + h] = prev[i * k + h]!;
      out[base + k + h] = (a + b + c + d + e + g + v + z) * scale;
      if (m.directed) {
        out[base + 2 * k + h] = (c + 2 * e + z - a - 2 * d - g) / 8;
        out[base + 3 * k + h] = (g + 2 * v + z - a - 2 * b - c) / 8;
      }
    }
    for (let d = 0; d < 3; d++) out[base + f - 3 + d] = x[3 * i + d]!;
  }
}

export function stepBuffers(m: Element) {
  return {
    features: new Float64Array(m.cells * m.inputs),
    state: blankState(m),
    prediction: new Float64Array(m.cells * CHANNEL_COUNT),
  };
}
export type Step = ReturnType<typeof stepBuffers>;

export function forward(m: Element, prev: Float64Array, x: Float64Array, out: Step): void {
  const { depth: k, cells } = m;
  features(m, prev, x, out.features);
  product(out.features, m.w, out.state, cells, m.inputs, k);
  for (let i = 0; i < cells; i++) {
    for (let h = 0; h < k; h++) {
      const j = i * k + h;
      out.state[j] = Math.tanh(out.state[j]! + m.b[h]!);
    }
  }
  product(out.state, m.u, out.prediction, cells, k, 3);
  for (let i = 0; i < cells; i++) {
    for (let d = 0; d < 3; d++) out.prediction[3 * i + d] += m.c[d]!;
  }
}
