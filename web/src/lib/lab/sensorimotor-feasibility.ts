import { stableOracle } from "./feasibility.ts";
import { genesisTask } from "./genesis-task.ts";
import { averageGainCurve, gainRuns } from "./sensorimotor.ts";
import { sensorimotorTask } from "./sensorimotor-task.ts";

export const gainControls = ["frozen", "ideal", "oracle"] as const;
type GainControl = typeof gainControls[number];

function requireGainControl(id: string): GainControl {
  const control = gainControls.find((c) => c === id);
  if (!control) throw new Error(`Недопустимое условие проверки выполнимости: ${id}`);
  return control;
}

export function gainControlsPass(frozen: number[], ideal: number[], oracles: number[][]): boolean {
  const count = sensorimotorTask.checkpoints.length;
  return frozen.length === count && frozen.every((a) => a <= genesisTask.controlCeiling)
    && ideal.length === count && ideal.every((a) => a >= genesisTask.controlFloor)
    && oracles.length > 0 && oracles.every((curve) => curve.length === count && stableOracle(curve));
}

export function designGainControls(requested: readonly string[] = gainControls) {
  const ids = new Set(requested.map(requireGainControl));
  return [...ids].flatMap((id) => (id === "oracle" ? [...sensorimotorTask.rates] : [0]).map((rate) => ({
    id, rate, curve: averageGainCurve(gainRuns(id, rate, sensorimotorTask.design)),
  })));
}
