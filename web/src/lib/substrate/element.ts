import { hashUnit } from "../lab/math.ts";
import type { Command } from "./gaze.ts";
import { product } from "./matrix.ts";

export const CHANNEL_COUNT = 3;

export interface Element {
  grid: number;
  cells: number;
  depth: number;
  directed: boolean;
  inputs: number;
  commandInput: boolean;
  // Features: own state, neighbor mean, optional Sobel x/y, optional command, then sensory channels.
  weights: Float64Array;
  w: Float64Array;
  b: Float64Array;
  u: Float64Array;
  c: Float64Array;
  gate?: { w: Float64Array; b: Float64Array };
  hint?: { v: Float64Array; bias: Float64Array };
  neighbors: Int32Array;
  inverseCount: Float64Array;
}

export function createElement(seed: number, directed = true, grid = 16, depth = 16,
  kind: "plain" | "memory" | "hinted" = "plain", commandInput = false,
): Element {
  const cells = grid * grid;
  const inputs = (directed ? 4 : 2) * depth + (commandInput ? 2 : 0) + CHANNEL_COUNT;
  const bias = inputs * depth;
  const readout = bias + depth;
  const end = readout + depth * CHANNEL_COUNT;
  const base = end + CHANNEL_COUNT;
  const gateEnd = base + (kind === "plain" ? 0 : bias + depth);
  const weights = new Float64Array(gateEnd + (kind === "hinted" ? depth + 1 : 0));
  for (let i = 0; i < weights.length; i++) {
    // Omit the two Sobel blocks in the control; pair every shared initial parameter.
    const index = !directed && i >= 2 * depth * depth ? i + 2 * depth * depth : i;
    weights[i] = (2 * hashUnit(7919 * seed + index) - 1) * 0.3;
  }
  const gate = kind === "plain" ? undefined : {
    w: weights.subarray(base, base + bias), b: weights.subarray(base + bias, gateEnd).fill(-1),
  };
  const hint = kind === "hinted" ? {
    v: weights.subarray(gateEnd, gateEnd + depth), bias: weights.subarray(gateEnd + depth),
  } : undefined;
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
    grid, cells, depth, directed, inputs, commandInput, weights, gate, hint, neighbors, inverseCount,
    w: weights.subarray(0, bias), b: weights.subarray(bias, readout),
    u: weights.subarray(readout, end), c: weights.subarray(end, base),
  };
}

// The last row is permanent zero padding for the spatial filters.
export function blankState(m: Element): Float64Array {
  return new Float64Array((m.cells + 1) * m.depth);
}

export function features(m: Element, prev: Float64Array, x: Float64Array, out: Float64Array, command?: Command) {
  if (m.commandInput && !command) throw new Error("Отсутствует копия команды");
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
    if (m.commandInput) {
      out[base + f - 5] = command![0];
      out[base + f - 4] = command![1];
    }
    for (let d = 0; d < 3; d++) out[base + f - 3 + d] = x[3 * i + d]!;
  }
}

export function stepBuffers(m: Element) {
  return {
    features: new Float64Array(m.cells * m.inputs),
    state: blankState(m),
    gate: m.gate ? new Float64Array(m.cells * m.depth) : undefined,
    candidate: m.gate ? new Float64Array(m.cells * m.depth) : undefined,
    occupancy: m.hint ? new Float64Array(m.cells) : undefined,
    prediction: new Float64Array(m.cells * CHANNEL_COUNT),
  };
}
export type Step = ReturnType<typeof stepBuffers>;

export function forward(m: Element, prev: Float64Array, x: Float64Array, out: Step, command?: Command): void {
  const { depth: k, cells } = m;
  features(m, prev, x, out.features, command);
  product(out.features, m.w, out.state, cells, m.inputs, k);
  if (m.gate) product(out.features, m.gate.w, out.gate!, cells, m.inputs, k);
  for (let i = 0; i < cells; i++) {
    for (let h = 0; h < k; h++) {
      const j = i * k + h;
      const candidate = Math.tanh(out.state[j]! + m.b[h]!);
      if (m.gate) {
        const z = 1 / (1 + Math.exp(-(out.gate![j]! + m.gate.b[h]!)));
        out.gate![j] = z;
        out.candidate![j] = candidate;
        out.state[j] = (1 - z) * prev[j]! + z * candidate;
      } else out.state[j] = candidate;
    }
  }
  if (m.hint) {
    product(out.state, m.hint.v, out.occupancy!, cells, k, 1);
    for (let i = 0; i < cells; i++) out.occupancy![i] += m.hint.bias[0]!;
  }
  product(out.state, m.u, out.prediction, cells, k, 3);
  for (let i = 0; i < cells; i++) {
    for (let d = 0; d < 3; d++) out.prediction[3 * i + d] += m.c[d]!;
  }
}
