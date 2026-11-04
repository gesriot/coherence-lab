import { hashUnit } from "../lab/math.ts";

export type Command = readonly [number, number];
export type GainMode = "fixed" | "episodic";
export type CopyMode = "executed" | "zero" | "shuffled";
export interface MotionSettings { gain: GainMode; copy: CopyMode }
export interface Gaze { x: number; y: number }
export const WINDOW_CELL = 3.125;

export function gazePath(seed: number, mode: GainMode) {
  const u = (k: number) => hashUnit(104729 * seed + 7 * k + 3);
  const gain = mode === "fixed" ? 1.2 : 0.7 + 0.7 * u(0);
  const distance = gain * 0.7 * WINDOW_CELL;
  let gaze: Gaze = { x: 35 + 30 * u(1), y: 35 + 30 * u(2) }, t = 0;
  function axis(position: number, k: number) {
    const command = Math.floor(3 * u(k)) - 1;
    const next = position + command * distance;
    return next < 25 || next > 75 ? -command : command;
  }
  return () => {
    const current = gaze;
    const command: Command = [axis(gaze.x, 11 + 3 * t), axis(gaze.y, 12 + 3 * t)];
    gaze = { x: gaze.x + command[0] * distance, y: gaze.y + command[1] * distance };
    t++;
    return { gaze: current, command };
  };
}
