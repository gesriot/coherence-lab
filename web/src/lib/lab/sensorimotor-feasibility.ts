import { referenceModel, stableOracle } from "./feasibility.ts";
import { genesisTask } from "./genesis-task.ts";
import { summarizeGainRuns, gainRuns, runGainEpisodes, type SeedRange } from "./sensorimotor.ts";
import { bodyForRun, sceneSignal, sensorimotorTask } from "./sensorimotor-task.ts";

export const gainControls = ["frozen", "ideal", "oracle", "neutral"] as const;
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

export function designOracleRate(scores: { rate: number; finalQ: number }[]): number {
  const maximum = Math.max(...scores.map((s) => s.finalQ));
  const tied = scores.filter((s) => maximum - s.finalQ < sensorimotorTask.oracleTieTolerance);
  return Math.min(...tied.map((s) => s.rate));
}

export function designGainControls(requested: readonly string[] = gainControls) {
  const ids = new Set(requested.map(requireGainControl));
  return [...ids].flatMap((id) => (id === "neutral" || id === "oracle" ? [...sensorimotorTask.rates] : [0]).map((rate) => ({
    id, rate, ...summarizeGainRuns(gainRuns(id, rate, sensorimotorTask.design)),
  })));
}

export interface FixedPointScores { bias: number; biasLimit: number; direction: number }

export function fixedPointPass(p: FixedPointScores): boolean {
  return Math.abs(p.bias) <= p.biasLimit && p.direction >= sceneSignal.minCorrectDirection;
}

export function fixedPointCheck(seeds: SeedRange) {
  let biasSum = 0, informativeAxes = 0, correct = 0, directionalAxes = 0, offsetSum = 0;
  for (let i = 0; i < seeds.runs; i++) {
    const seed = seeds.firstRun + i, body = bodyForRun(seed);
    offsetSum += Math.abs(body[0] - 1) + Math.abs(body[1] - 1);
    for (const calibrated of [true, false]) {
      const models = referenceModel("ideal");
      if (calibrated) models.gain = [...body];
      runGainEpisodes("neutral", seed, body, models, null, (_, ratios) => {
        for (const axis of [0, 1] as const) {
          const r = ratios[axis];
          if (r == null) continue;
          if (calibrated) {
            biasSum += r - 1;
            informativeAxes++;
          } else if (r !== 1) {
            directionalAxes++;
            if (Math.sign(r - 1) === Math.sign(body[axis] - 1)) correct++;
          }
        }
      });
    }
  }
  const bias = biasSum / informativeAxes, direction = correct / directionalAxes;
  const biasLimit = sceneSignal.maxBiasFraction * offsetSum / (2 * seeds.runs);
  return { bias, biasLimit, direction, correct, directionalAxes, informativeAxes };
}
