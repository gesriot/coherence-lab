import { hashUnit } from "./math.ts";
import type { Vector } from "./learning.ts";

export const sensorimotorTask = {
  bodyMinOffset: 0.1,
  bodyMaxOffset: 0.2,
  minGazeShift: 0,
  minGain: 0.5,
  maxGain: 2,
  memory: 256,
  // Score mean relative error reduction at each post-initial checkpoint.
  metric: { baselineGain: [1, 1], firstScoredIndex: 1 },
  saccadeDistance: 4,
  checkpoints: [0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120],
  rates: [0.003, 0.01, 0.03, 0.1],
  evaluationCount: 20,
  design: { firstRun: 801, runs: 10, firstEvalSeed: 3101 },
  tuning: { firstRun: 531, runs: 10, firstEvalSeed: 2061 },
  comparison: { firstRun: 91, runs: 30, firstEvalSeed: 1061 },
  oracleTieTolerance: 0.005,
  // Smallest rate within the tolerance of the best final q on runs 801–810.
  oracleRate: 0.01,
} as const;

export function bodyForRun(seed: number): Vector {
  const { bodyMinOffset, bodyMaxOffset } = sensorimotorTask;
  const axis = (index: number) => {
    const sign = hashUnit(4 * seed + index) < 0.5 ? -1 : 1;
    const magnitude = bodyMinOffset + (bodyMaxOffset - bodyMinOffset) * hashUnit(4 * seed + index + 1);
    return 1 + sign * magnitude;
  };
  return [axis(0), axis(2)];
}

// Only the experiment driver supplies this label to oracle training.
export function oracleGainLabel(body: Readonly<Vector>, shift: Vector, estimate: Vector): Vector {
  return [(1 - body[0] / estimate[0]) * shift[0], (1 - body[1] / estimate[1]) * shift[1]];
}
