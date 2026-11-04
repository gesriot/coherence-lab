import { blankState, forward, stepBuffers, type Element } from "./element.ts";
import { backward, gradientBuffers } from "./gradient.ts";
import type { Command, MotionSettings } from "./gaze.ts";
import { frames, movingFrames, type MotionInput, type Occlusion, type WorldSettings } from "./input.ts";

export const EPISODES = 2000;
export const STEPS = 80;
export const MEMORY = 30;
const scenarios = ["continuity", "vanish", "nursery", "binding"] as const;

export function adamState(m: Element) {
  return { step: 0, first: new Float64Array(m.weights.length), second: new Float64Array(m.weights.length) };
}

export function update(weights: Float64Array, grad: Float64Array, adam: ReturnType<typeof adamState>) {
  let squaredNorm = 0;
  for (const x of grad) squaredNorm += x * x;
  const clip = 1 / Math.max(1, Math.sqrt(squaredNorm));
  adam.step++;
  const mScale = 1 - 0.9 ** adam.step, vScale = 1 - 0.999 ** adam.step;
  for (let i = 0; i < weights.length; i++) {
    const g = grad[i]! * clip;
    adam.first[i] = 0.9 * adam.first[i]! + 0.1 * g;
    adam.second[i] = 0.999 * adam.second[i]! + 0.001 * g * g;
    weights[i] -= 0.003 * (adam.first[i]! / mScale) / (Math.sqrt(adam.second[i]! / vScale) + 1e-8);
  }
}

type FrameMap = (time: number) => Float64Array;
export type MapFactory = (
  scenario: (typeof scenarios)[number], seed: number, settings?: WorldSettings,
) => FrameMap;

interface TrainingSettings {
  world?: WorldSettings;
  motion?: MotionSettings;
  episodes?: number;
  weighting?: "uniform" | "wall-exits";
  weightHook?: MapFactory;
}

export function trainer(m: Element) {
  const tape = Array.from({ length: MEMORY }, () => stepBuffers(m));
  const work = gradientBuffers(m), adam = adamState(m), carry = blankState(m);
  return function trainEpisode(sequence: Iterable<Float64Array | MotionInput>, hint?: FrameMap, weightAt?: FrameMap) {
    carry.fill(0);
    if (m.hint && !hint) throw new Error("Для положительного контроля нужны метки");
    const weights = weightAt ? [] as Float64Array[] : undefined;
    const labels: Float64Array[] = [];
    let time = 0;
    let previous: Float64Array | undefined;
    let command: Command | undefined;
    const targets: Float64Array[] = [];
    function finish() {
      backward(m, tape, targets, work, labels, weights);
      carry.set(tape[targets.length - 1]!.state);
      update(m.weights, work.grad, adam);
      targets.length = 0;
      labels.length = 0;
      if (weights) weights.length = 0;
    }
    for (const observation of sequence) {
      const next = observation instanceof Float64Array ? observation : observation.frame;
      if (previous) {
        const t = targets.length;
        forward(m, t === 0 ? carry : tape[t - 1]!.state, previous, tape[t]!, command);
        if (m.hint) labels.push(hint!(time));
        time++;
        if (weightAt) weights!.push(weightAt(time));
        targets.push(next);
        if (targets.length === MEMORY) finish();
      }
      previous = next;
      command = observation instanceof Float64Array ? undefined : observation.copy;
    }
    if (targets.length) finish();
  };
}

export function train(
  m: Element, run: number, occlusion: Occlusion = "whole-object",
  hook?: MapFactory, onEpisode?: (completed: number) => void, settings: TrainingSettings = {},
) {
  const episode = trainer(m);
  const weighted = settings.weighting === "wall-exits";
  if (weighted && !settings.weightHook) throw new Error("Для веса выхода нужен хук оценщика");
  for (let i = 0; i < (settings.episodes ?? EPISODES); i++) {
    const scenario = scenarios[i % scenarios.length]!, seed = 20000 + 100 * run + i;
    episode(
      settings.motion ? movingFrames(scenario, seed, STEPS, settings.motion) :
        frames(scenario, seed, STEPS, occlusion, settings.world),
      m.hint ? hook?.(scenario, seed, settings.world) : undefined,
      weighted ? settings.weightHook!(scenario, seed, settings.world) : undefined,
    );
    onEpisode?.(i + 1);
  }
}
