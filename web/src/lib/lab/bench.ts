import { createEpisode, defaultConfig, snapshot, stepEpisode, type Episode } from "./agent";
import type { Ablation, LabConfig, ScenarioId } from "./types";

export interface BenchRow {
  id: string;
  label: string;
  detail: string;
  retention: number | null;
  pair: number | null;
  selfError: number | null;
  actionError: number | null;
}

export interface BenchReport {
  rows: BenchRow[];
  causal: "supported" | "unsupported" | "mixed";
  summary: string;
}

const SEEDS = [1, 2, 3, 4, 5, 6, 7];

function roll(scenario: ScenarioId, ablation: Ablation, seed: number, steps: number, mutate?: (ep: Episode, i: number, cfg: LabConfig) => void) {
  const ep = createEpisode(scenario, seed);
  const cfg = defaultConfig();
  cfg.ablation = ablation;
  const views = [];
  for (let i = 0; i < steps; i++) {
    mutate?.(ep, i, cfg);
    stepEpisode(ep, cfg);
    views.push(snapshot(ep, ablation));
  }
  return views;
}

function retentionOf(scenario: ScenarioId, ablation: Ablation, seed: number): boolean {
  const views = roll(scenario, ablation, seed, 72);
  let sawHidden = false;
  let exit = -1;
  for (let i = 0; i < views.length; i++) {
    const target = views[i]!.evaluator.objects[0];
    if (!target) return false;
    if (target.occluded) sawHidden = true;
    else if (sawHidden && exit < 0) exit = i;
  }
  if (exit < 0) return false;
  const look = views[Math.min(views.length - 1, exit + 1)]!;
  return look.evaluator.retained === true;
}

function pairOf(ablation: Ablation, seed: number): boolean {
  const views = roll("conflict", ablation, seed, 46);
  let streak = 0;
  let best = 0;
  for (const view of views) {
    const a = view.evaluator.objects.find((o) => o.id === "A");
    const b = view.evaluator.objects.find((o) => o.id === "B");
    const ok = a?.nearestTrack != null && b?.nearestTrack != null && a.nearestTrack !== b.nearestTrack;
    streak = ok ? streak + 1 : 0;
    if (streak > best) best = streak;
  }
  return best >= 6;
}

function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((s, v) => s + v, 0) / values.length;
}

function meanSelf(ablation: Ablation, seed: number): number | null {
  const views = roll("continuity", ablation, seed, 60);
  const vals = views.map((v) => v.energy.self).filter((v): v is number => v != null);
  return mean(vals);
}

function meanAction(ablation: Ablation, seed: number, replay: boolean): number | null {
  const views = roll("continuity", ablation, seed, 40, (_ep, i, cfg) => {
    if (replay && i >= 14) cfg.replay = true;
  });
  const slice = replay ? views.slice(16) : views.slice(4, 14);
  return mean(slice.map((v) => v.energy.action));
}

export function runBench(): BenchReport {
  const specs: { id: string; label: string; detail: string; ablation: Ablation }[] = [
    {
      id: "full",
      label: "Полный контур",
      detail: "Модель восприятия двигает взгляд",
      ablation: "full",
    },
    {
      id: "record",
      label: "Только запись",
      detail: "Модель есть, на действие не влияет",
      ablation: "record-only",
    },
    {
      id: "none",
      label: "Без модели",
      detail: "Взгляд за самой большой областью",
      ablation: "no-schema",
    },
    {
      id: "dummy",
      label: "Лишняя память",
      detail: "Буфер кадров без схемы внимания",
      ablation: "dummy-memory",
    },
  ];

  const rows: BenchRow[] = specs.map((spec) => {
    const retention =
      SEEDS.filter((seed) => retentionOf("continuity", spec.ablation, seed)).length / SEEDS.length;
    const pair = SEEDS.filter((seed) => pairOf(spec.ablation, seed)).length / SEEDS.length;
    const selves = SEEDS.map((seed) => meanSelf(spec.ablation, seed)).filter((v): v is number => v != null);
    const actions = SEEDS.map((seed) => meanAction(spec.ablation, seed, false)).filter((v): v is number => v != null);
    return {
      id: spec.id,
      label: spec.label,
      detail: spec.detail,
      retention,
      pair,
      selfError: mean(selves),
      actionError: mean(actions),
    };
  });

  const replayActions = SEEDS.map((seed) => meanAction("full", seed, true)).filter((v): v is number => v != null);
  rows.push({
    id: "replay",
    label: "Повтор записи",
    detail: "Те же глаза, но вход больше не следует действию",
    retention: null,
    pair: null,
    selfError: null,
    actionError: mean(replayActions),
  });

  const full = rows[0]!;
  const none = rows[2]!;
  const record = rows[1]!;
  const liveAction = full.actionError ?? 0;
  const replayAction = rows[4]!.actionError ?? 0;
  const holdsIdentity = (full.retention ?? 0) > (none.retention ?? 0) + 0.2;
  const recordIsNotEnough = (full.retention ?? 0) > (record.retention ?? 0) + 0.15;
  const couplingBreaks = replayAction > liveAction + 0.08;
  const causal = holdsIdentity && recordIsNotEnough && couplingBreaks ? "supported" : holdsIdentity ? "mixed" : "unsupported";

  const summary = holdsIdentity
    ? "На этих семенах модель внимания удерживает идентичность за окклюдером лучше, чем взгляд за яркостью. Это причинная роль для задачи, не свидетельство переживания."
    : "На этих семенах преимущество модели внимания не воспроизвелось. Стенд это показывает, а не прячет.";

  return {
    rows,
    causal: couplingBreaks && holdsIdentity ? causal : holdsIdentity ? "mixed" : "unsupported",
    summary: couplingBreaks
      ? summary
      : `${summary} Разрыв сенсомоторной связи ${replayAction > liveAction ? "повысил" : "не повысил"} ошибку действия.`,
  };
}
