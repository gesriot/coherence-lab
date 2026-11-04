import { blankState, forward, stepBuffers, type Element } from "./element.ts";
import { backward, gradientBuffers } from "./gradient.ts";
import { frames, type Occlusion } from "./input.ts";

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

type Hint = (time: number) => Float64Array;
export type HintFactory = (scenario: (typeof scenarios)[number], seed: number) => Hint;

export function trainer(m: Element) {
  const tape = Array.from({ length: MEMORY }, () => stepBuffers(m));
  const work = gradientBuffers(m), adam = adamState(m), carry = blankState(m);
  return function trainEpisode(sequence: Iterable<Float64Array>, hint?: Hint) {
    carry.fill(0);
    if (m.hint && !hint) throw new Error("Для положительного контроля нужны метки");
    const labels: Float64Array[] = [];
    let time = 0;
    let previous: Float64Array | undefined;
    const targets: Float64Array[] = [];
    function finish() {
      backward(m, tape, targets, work, labels);
      carry.set(tape[targets.length - 1]!.state);
      update(m.weights, work.grad, adam);
      targets.length = 0;
      labels.length = 0;
    }
    for (const next of sequence) {
      if (previous) {
        const t = targets.length;
        forward(m, t === 0 ? carry : tape[t - 1]!.state, previous, tape[t]!);
        if (m.hint) labels.push(hint!(time));
        time++;
        targets.push(next);
        if (targets.length === MEMORY) finish();
      }
      previous = next;
    }
    if (targets.length) finish();
  };
}

export function train(
  m: Element, run: number, occlusion: Occlusion = "whole-object",
  hook?: HintFactory, onEpisode?: (completed: number) => void,
) {
  const episode = trainer(m);
  for (let i = 0; i < EPISODES; i++) {
    const scenario = scenarios[i % scenarios.length]!, seed = 20000 + 100 * run + i;
    episode(frames(scenario, seed, STEPS, occlusion), m.hint ? hook?.(scenario, seed) : undefined);
    onEpisode?.(i + 1);
  }
}
