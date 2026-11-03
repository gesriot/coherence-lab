import { defaultConfig, stepEpisode, type Episode } from "./agent.ts";
import { generator, Replay, type Selection } from "./experience.ts";
import { genesisTask } from "./genesis-task.ts";
import { conditions, type Condition } from "./genesis-training.ts";
import type { Models, Vector } from "./learning.ts";
import { clamp } from "./math.ts";
import { sensorimotorTask } from "./sensorimotor-task.ts";
import type { Track } from "./types.ts";

export interface GainExample {
  shift: Vector;
  error: Vector;
}

export type GainLabel = (shift: Vector, estimate: Vector) => Vector;
export type TrainingCondition = Exclude<Condition, "frozen">;

export function fitGain(estimate: Vector, shift: Vector, error: Vector, rate: number): void {
  for (const a of [0, 1] as const) {
    const delta = shift[a];
    estimate[a] = clamp(estimate[a] * (1 - rate * error[a] * delta / (1 + delta * delta)),
      sensorimotorTask.minGain, sensorimotorTask.maxGain);
  }
}

export class GainLearner {
  readonly memory = new Replay<GainExample>(sensorimotorTask.memory);
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

  observe(before: Track[], after: Track[], shift: Vector, focus: number | null, selfError: number): void {
    if (Math.hypot(...shift) < sensorimotorTask.minGazeShift) return;
    for (const track of after) {
      if (track.boundBlob == null || (this.selection === "attention" && track.id !== focus)) continue;
      const prior = before.find((t) => t.id === track.id);
      if (prior?.boundBlob == null) continue;
      const error: Vector = [track.x - prior.x - prior.vx, track.y - prior.y - prior.vy];
      const priority = this.selection === "error" ? Math.hypot(...error) : selfError;
      this.memory.add({ shift, error }, 0.05 + priority);
    }
  }

  update(label?: GainLabel): void {
    const sample = this.memory.sample(this.random, this.selection === "uniform");
    if (!sample) return;
    const error = label ? label(sample.shift, this.models.gain) : sample.error;
    fitGain(this.models.gain, sample.shift, error, this.rate);
  }
}

export function trainGainEpisode(learner: GainLearner, ep: Episode, condition: TrainingCondition, oracleLabel: GainLabel): void {
  const cfg = { ...defaultConfig(), ablation: conditions[condition].gaze };
  for (let step = 1; step <= genesisTask.trainingSteps; step++) {
    const agent = ep.agent;
    const before = agent.tracks, previous = agent.obsGaze;
    const focus = agent.attention.executedFocusId;
    stepEpisode(ep, cfg);
    learner.observe(before, agent.tracks, [agent.obsGaze.x - previous.x, agent.obsGaze.y - previous.y],
      focus, agent.selfError ?? 0);
    learner.update(condition === "oracle" ? oracleLabel : undefined);
  }
}
