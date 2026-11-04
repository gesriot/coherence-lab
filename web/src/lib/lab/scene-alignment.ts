import type { Vector } from "./learning.ts";
import { classify } from "./observe.ts";
import { sceneSignal } from "./sensorimotor-task.ts";
import type { Cell } from "./types.ts";

export interface SceneFrame {
  grid: number;
  aperture: number;
  command: Vector;
  // NaN marks surfaces and their eight neighbors. All other cells keep their raw lightness.
  light: Float64Array;
}

export interface FramePair { previous: SceneFrame; current: SceneFrame }
export type AxisRatios = [number | null, number | null];

export function sceneFrame(cells: Cell[], grid: number, aperture: number, command: Vector): SceneFrame {
  const light = Float64Array.from(cells, (c) => c.lightness);
  for (let i = 0; i < cells.length; i++) {
    if (classify(cells[i]!) !== "surface") continue;
    const row = Math.floor(i / grid), col = i % grid;
    for (let y = Math.max(0, row - 1); y <= Math.min(grid - 1, row + 1); y++) {
      light.fill(NaN, y * grid + Math.max(0, col - 1), y * grid + Math.min(grid, col + 2));
    }
  }
  return { grid, aperture, command, light };
}

function alignmentScore({ previous, current }: FramePair, dx: number, dy: number): number | null {
  const n = previous.grid, size = 2 * previous.aperture / n;
  const scale = (2 * current.aperture / current.grid) / size;
  // Coordinates of the first current cell center in the previous cell-center grid.
  const x0 = (dx + previous.aperture - current.aperture) / size + (scale - 1) / 2;
  const y0 = (dy + previous.aperture - current.aperture) / size + (scale - 1) / 2;
  const old = previous.light;
  let sum = 0, count = 0;
  for (let row = 0; row < current.grid; row++) {
    const y = y0 + row * scale, iy = Math.floor(y), fy = y - iy;
    if (iy < 0 || iy + 1 >= n) continue;
    for (let col = 0; col < current.grid; col++) {
      const x = x0 + col * scale, ix = Math.floor(x), fx = x - ix;
      if (ix < 0 || ix + 1 >= n) continue;
      const k = iy * n + ix;
      const top = old[k]! + fx * (old[k + 1]! - old[k]!);
      const bottom = old[k + n]! + fx * (old[k + n + 1]! - old[k + n]!);
      const residual = current.light[row * current.grid + col]! - (top + fy * (bottom - top));
      if (Number.isNaN(residual)) continue;
      sum += residual * residual;
      count++;
    }
  }
  return count >= sceneSignal.minCells ? sum / count : null;
}

export function alignScene(pair: FramePair, estimate: Vector): AxisRatios {
  const shift = (axis: 0 | 1) =>
    estimate[axis] * (pair.current.command[axis] - pair.previous.command[axis]);
  const dx = shift(0), dy = shift(1);
  const ratios: AxisRatios = [null, null];
  for (const axis of [0, 1] as const) {
    if ((axis === 0 ? dx : dy) === 0) continue;
    let best = Infinity, worst = -Infinity, sum = 0, winner = 1, count = 0;
    for (const r of sceneSignal.ratios) {
      const score = alignmentScore(pair, axis === 0 ? dx * r : dx, axis === 1 ? dy * r : dy);
      if (score == null) break;
      if (score < best || (score === best && Math.abs(r - 1) < Math.abs(winner - 1))) {
        best = score;
        winner = r;
      }
      worst = Math.max(worst, score);
      sum += score;
      count++;
    }
    if (count === sceneSignal.ratios.length && best < worst && best <= sceneSignal.minToMean * (sum / count)) {
      ratios[axis] = winner;
    }
  }
  return ratios;
}
