import { createEpisode, defaultConfig, stepEpisode, type Episode } from "./agent.ts";
import { Learner, type Selection } from "./experience.ts";
import { scheduleOutdoorLosses } from "./world.ts";
import { GapBook, type ClosedGap, type Vector } from "./learning.ts";
import { matchRadius, predictMiss } from "./perception.ts";
import type { Ablation, Observation, ScenarioId, Track } from "./types.ts";

export const conditions = {
  full: { label: "Полный контур", gaze: "full", selection: "attention" },
  "first-order": { label: "Первый порядок", gaze: "first-order", selection: "uniform" },
  "no-protocol": { label: "Без протокола", gaze: "full", selection: "uniform" },
  "strong-first-order": { label: "Сильный первый порядок", gaze: "novelty", selection: "error" },
  frozen: { label: "Без обучения", gaze: "first-order", selection: "uniform" },
  oracle: { label: "Оракул", gaze: "first-order", selection: "uniform" },
} as const satisfies Record<string, { label: string; gaze: Ablation; selection: Selection }>;
export type Condition = keyof typeof conditions;

export interface Labels {
  observe(): void;
  displacement(previous: Track): Vector;
  found(gap: ClosedGap): boolean;
}

export function trainEpisode(
  learner: Learner, condition: Condition, scenario: ScenarioId, seed: number,
  oracleHook: (ep: Episode) => Labels,
): void {
  const ablation = conditions[condition].gaze;
  const ep = createEpisode(scenario, seed, ablation, learner.models);
  if (scenario === "nursery") scheduleOutdoorLosses(ep.world);
  const labels = condition === "oracle" ? oracleHook(ep) : null;
  const cfg = { ...defaultConfig(), ablation };
  const book = new GapBook();
  const attentionErrors = new Map<number, number>();
  for (let step = 1; step <= 80; step++) {
    const agent = ep.agent;
    const before = agent.tracks;
    const focus = agent.attention.executedFocusId;
    stepEpisode(ep, cfg);
    if (condition === "frozen") continue;
    labels?.observe();
    const error = agent.selfError ?? 0;
    if (focus != null) attentionErrors.set(focus, Math.max(attentionErrors.get(focus) ?? 0, error));

    for (const t of agent.tracks) {
      if (t.boundBlob == null) continue;
      const samples = t.motionSamples;
      const oldest = samples.at(-3);
      const middle = samples.at(-2);
      if (!oldest || !middle || middle.step !== t.age - 1 || oldest.step !== t.age - 2) continue;
      const prior = before.find((p) => p.id === t.id)!;
      const target: Vector = labels?.displacement(prior) ?? [t.x - prior.x, t.y - prior.y];
      learner.rememberMotion({ input: prior.forecastInput!, target }, t.id === focus ? error : null);
    }

    const obs: Observation = {
      gazeX: agent.obsGaze.x, gazeY: agent.obsGaze.y, aperture: agent.obsGaze.aperture,
      cells: agent.cells, grid: agent.grid,
    };
    const closed = book.step(step, before, agent.tracks, (t) => predictMiss(t, obs, learner.models), matchRadius);
    for (const gap of closed) {
      learner.rememberGap({
        exposure: gap.exposure, last: gap.last, found: labels?.found(gap) ?? gap.found,
      }, attentionErrors.get(gap.id) ?? null);
    }
    learner.update();
  }
}
