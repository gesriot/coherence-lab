import { stableOracle } from "./feasibility.ts";
import { genesisTask } from "./genesis-task.ts";
import { summarizeGainRuns, gainRuns } from "./sensorimotor.ts";
import { sensorimotorTask } from "./sensorimotor-task.ts";

export const gainControls = ["frozen", "ideal", "oracle"] as const;
type GainControl = typeof gainControls[number];

function requireGainControl(id: string): GainControl {
  const control = gainControls.find((c) => c === id);
  if (!control) throw new Error(`Недопустимое условие проверки выполнимости: ${id}`);
  return control;
}

type ControlSummary = Pick<ReturnType<typeof summarizeGainRuns>, "accuracy" | "curve">;

export function gainControlsPass(frozen: ControlSummary, ideal: ControlSummary, oracles: ControlSummary[]): boolean {
  const count = sensorimotorTask.checkpoints.length;
  return frozen.curve.length === count && frozen.curve.every((p) => p.q === 0) && frozen.accuracy <= genesisTask.controlCeiling
    && ideal.curve.length === count && ideal.curve.every((p) => p.q === 1) && ideal.accuracy >= genesisTask.controlFloor
    && oracles.length > 0 && oracles.every((o) => o.curve.length === count
      && stableOracle(o.curve.map((p) => p.q)) && o.accuracy >= genesisTask.controlFloor);
}

export function designGainControls(requested: readonly string[] = gainControls) {
  const ids = new Set(requested.map(requireGainControl));
  return [...ids].flatMap((id) => (id === "oracle" ? [...sensorimotorTask.rates] : [0]).map((rate) => ({
    id, rate, ...summarizeGainRuns(gainRuns(id, rate, sensorimotorTask.design)),
  })));
}
