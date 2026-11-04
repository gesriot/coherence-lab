import { defaultConfig, stepEpisode, type Episode } from "./agent.ts";
import { generator, Replay, type Selection } from "./experience.ts";
import { genesisTask } from "./genesis-task.ts";
import { conditions, type Condition } from "./genesis-training.ts";
import type { Models, Vector } from "./learning.ts";
import { clamp } from "./math.ts";
import { alignScene, sceneFrame, type AxisRatios, type FramePair } from "./scene-alignment.ts";
import { sensorimotorTask } from "./sensorimotor-task.ts";
import type { AttentionReason } from "./types.ts";

export const gainConditions = {
  ...conditions,
  neutral: { label: "Без модели", gaze: "no-schema", selection: "uniform" },
} as const;
export type GainLabel = (estimate: Vector) => Vector;
export type TrainingCondition = Exclude<Condition, "frozen">;

export function fitGain(estimate: Vector, ratios: AxisRatios, rate: number): void {
  for (const a of [0, 1] as const) {
    const r = ratios[a];
    if (r != null) estimate[a] = clamp(estimate[a] * (1 + rate * (r - 1)), sensorimotorTask.minGain, sensorimotorTask.maxGain);
  }
}

export class GainLearner {
  readonly memory = new Replay<FramePair>(sensorimotorTask.memory);
  readonly models: Models;
  private readonly rate: number;
  private readonly selection: Selection;
  private readonly random: () => number;

  constructor(models: Models, rate: number, selection: Selection, seed: number) {
    this.models = models;
    this.rate = rate;
    this.selection = selection;
    this.random = generator(seed);
  }

  observe(pair: FramePair, reason: AttentionReason, focusChanged: boolean, selfError: number, ratios: AxisRatios): void {
    if (this.selection === "attention" && reason === "pursue" && !focusChanged) return;
    const priority = this.selection === "error"
      ? Math.abs((ratios[0] ?? 1) - 1) + Math.abs((ratios[1] ?? 1) - 1) : selfError;
    this.memory.add(pair, 0.05 + priority);
  }

  update(label?: GainLabel): boolean {
    const pair = this.memory.sample(this.random, this.selection === "uniform");
    if (!pair) return false;
    const ratios = alignScene(pair, this.models.gain);
    if (ratios[0] == null && ratios[1] == null) return false;
    if (label) {
      const exact = label(this.models.gain);
      for (const a of [0, 1] as const) if (ratios[a] != null) ratios[a] = exact[a];
    }
    fitGain(this.models.gain, ratios, this.rate);
    return true;
  }
}

export type AfterGainStep = (shift: Vector, observed: AxisRatios, informativeDraw: boolean) => void;

export function trainGainEpisode(
  learner: GainLearner | null, ep: Episode, condition: Condition | "neutral", oracleLabel: GainLabel,
  afterStep?: AfterGainStep,
): void {
  const cfg = { ...defaultConfig(), ablation: gainConditions[condition].gaze };
  const agent = ep.agent;
  const capture = () => {
    const { x, y, aperture } = agent.obsGaze;
    const gain = agent.learning!.gain;
    return sceneFrame(agent.cells, agent.grid, aperture, [x / gain[0], y / gain[1]]);
  };
  let previous = capture(), previousFocus: number | null = null;
  for (let step = 1; step <= genesisTask.trainingSteps; step++) {
    const gaze = agent.obsGaze;
    const { executedFocusId, executedReason } = agent.attention;
    stepEpisode(ep, cfg);
    const current = capture();
    const pair = { previous, current };
    const ratios = alignScene(pair, agent.learning!.gain);
    learner?.observe(pair, executedReason, executedFocusId !== previousFocus, agent.selfError ?? 0, ratios);
    const informativeDraw = learner?.update(condition === "oracle" ? oracleLabel : undefined) ?? false;
    afterStep?.([agent.obsGaze.x - gaze.x, agent.obsGaze.y - gaze.y], ratios, informativeDraw);
    previous = current;
    previousFocus = executedFocusId;
  }
}
