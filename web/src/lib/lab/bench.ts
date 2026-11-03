import { createEpisode, defaultConfig, snapshot, stepEpisode } from "./agent.ts";
import type { Ablation, ScenarioId, ViewState } from "./types.ts";

const CONFIGURATIONS = 200;
const TASK_MARGIN = 0.2;
const AUC_THRESHOLD = 0.9;

export interface Proportion {
  k: number;
  n: number;
  rate: number;
  low: number;
  high: number;
}

export interface BenchRow {
  id: Ablation;
  label: string;
  detail: string;
  retention: Proportion;
  pair: Proportion;
  exploreSteps: number;
  selfError: number | null;
  actionError: number;
}

export interface ReplayResult {
  label: string;
  liveError: number;
  replayError: number;
  auc: number;
  medianDifference: number;
  higher: Proportion;
  detected: boolean;
}

export interface BenchReport {
  configurations: number;
  rows: BenchRow[];
  replay: ReplayResult[];
  causal: "supported" | "unsupported" | "mixed";
}

export const causalLabel = {
  supported: "подтверждена на этих конфигурациях",
  mixed: "смешанный результат",
  unsupported: "не подтверждена",
};

export const criterion = `На ${CONFIGURATIONS} конфигурациях полный контур должен опередить каждый из трёх контролей (первый порядок, только запись, лишняя память) более чем на ${TASK_MARGIN * 100} п. п. хотя бы по одной задаче. Отдельно оценщик различает живой эпизод и повтор по ошибке прогноза за шаги 17–40. Критерий обнаружения повтора: AUC ≥ ${decimal(AUC_THRESHOLD, 1)}.`;

export function decimal(value: number | null, digits = 3): string {
  return value == null ? "–" : value.toFixed(digits).replace(".", ",");
}

export function formatRate(p: Proportion): string {
  return `${p.k}/${p.n} · ${decimal(100 * p.rate, 1)}% [${decimal(100 * p.low, 1)}–${decimal(100 * p.high, 1)}%]`;
}

export function wilson(k: number, n: number): Proportion {
  const z = 1.959963984540054;
  const rate = k / n;
  const denominator = 1 + z ** 2 / n;
  const center = (rate + z ** 2 / (2 * n)) / denominator;
  const half = z / denominator * Math.sqrt(rate * (1 - rate) / n + z ** 2 / (4 * n ** 2));
  return { k, n, rate, low: k === 0 ? 0 : center - half, high: k === n ? 1 : center + half };
}

function mean(values: number[]): number {
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

export function replayStats(label: string, live: number[], replay: number[]): ReplayResult {
  if (!live.length || live.length !== replay.length) throw new Error("Expected paired episodes");
  let wins = 0;
  for (const positive of replay) {
    for (const negative of live) {
      if (positive > negative) wins++;
      else if (positive === negative) wins += 0.5;
    }
  }
  const auc = wins / (live.length * replay.length);
  const differences = replay.map((v, i) => v - live[i]!).sort((a, b) => a - b);
  const middle = Math.floor(differences.length / 2);
  const medianDifference = (differences[middle]! + differences[Math.floor((differences.length - 1) / 2)]!) / 2;
  return {
    label, liveError: mean(live), replayError: mean(replay), auc, medianDifference,
    higher: wilson(differences.filter((v) => v > 0).length, live.length),
    detected: auc >= AUC_THRESHOLD,
  };
}

export function assess(rows: BenchRow[]): BenchReport["causal"] {
  const row = (id: Ablation) => rows.find((item) => item.id === id)!;
  const full = row("full");
  const beats = (id: Ablation) => (["retention", "pair"] as const).some((metric) => {
    const control = row(id)[metric];
    // All policies use the same N. Compare counts to keep the strict boundary exact.
    return full[metric].k - control.k > TASK_MARGIN * control.n;
  });
  if (!beats("first-order")) return "unsupported";
  return beats("record-only") && beats("dummy-memory") ? "supported" : "mixed";
}

function roll(scenario: ScenarioId, ablation: Ablation, seed: number, steps: number, replay = false) {
  const ep = createEpisode(scenario, seed, ablation);
  const cfg = { ...defaultConfig(), ablation };
  const views: ViewState[] = [];
  for (let i = 0; i < steps; i++) {
    cfg.replay = replay && i >= 14;
    stepEpisode(ep, cfg);
    views.push(snapshot(ep, ablation));
  }
  return views;
}

function retained(views: ViewState[]): boolean {
  const hidden = views.findIndex((v) => v.evaluator.objects[0]!.occluded);
  if (hidden < 0) return false;
  const exit = views.findIndex((v, i) => i > hidden && !v.evaluator.objects[0]!.occluded);
  return exit >= 0 && views[Math.min(exit + 1, views.length - 1)]!.evaluator.retained === true;
}

function pairResolved(views: ViewState[]): boolean {
  let streak = 0;
  for (const view of views) {
    const [a, b] = view.evaluator.objects;
    const ok = a!.nearestTrack != null && b!.nearestTrack != null && a!.nearestTrack !== b!.nearestTrack;
    streak = ok ? streak + 1 : 0;
    if (streak >= 6) return true;
  }
  return false;
}

function actionScore(views: ViewState[]): number {
  // The grid is fixed. Sum integral mismatch counts so equally bad episodes tie exactly.
  const window = views.slice(16, 40);
  const cells = window[0]!.grid ** 2;
  const mismatches = window.reduce((sum, view) => sum + Math.round(view.energy.action * cells), 0);
  return mismatches / (window.length * cells);
}

const CONDITIONS: { id: Ablation; label: string; detail: string }[] = [
  { id: "full", label: "Полный контур", detail: "Модель восприятия двигает взгляд" },
  { id: "first-order", label: "Первый порядок", detail: "Сопровождение трека по прогнозу движения" },
  { id: "record-only", label: "Только запись", detail: "Модель есть, на действие не влияет" },
  { id: "no-schema", label: "Без модели", detail: "Взгляд направлен к самому крупному пятну" },
  { id: "dummy-memory", label: "Лишняя память", detail: "Буфер кадров без схемы внимания" },
];

export function runBench(): BenchReport {
  const replay: ReplayResult[] = [];
  const rows = CONDITIONS.map((spec) => {
    let holds = 0;
    let pairs = 0;
    let explores = 0;
    const selves: number[] = [];
    const live: number[] = [];
    const recordings: number[] = [];
    for (let seed = 1; seed <= CONFIGURATIONS; seed++) {
      const continuity = roll("continuity", spec.id, seed, 72);
      const conflict = roll("conflict", spec.id, seed, 46);
      if (retained(continuity)) holds++;
      if (pairResolved(conflict)) pairs++;
      for (const episode of [continuity, conflict]) {
        explores += episode.filter((v) => v.attention.executedReason === "explore").length;
      }
      const errors = continuity.slice(0, 60).map((v) => v.energy.self).filter((v) => v != null);
      if (errors.length) selves.push(mean(errors));
      live.push(actionScore(continuity));
      if (spec.id === "full" || spec.id === "first-order") {
        recordings.push(actionScore(roll("continuity", spec.id, seed, 40, true)));
      }
    }
    if (recordings.length) replay.push(replayStats(spec.label, live, recordings));
    return {
      ...spec, retention: wilson(holds, CONFIGURATIONS), pair: wilson(pairs, CONFIGURATIONS),
      exploreSteps: explores / (2 * CONFIGURATIONS),
      selfError: selves.length ? mean(selves) : null, actionError: mean(live),
    };
  });
  return { configurations: CONFIGURATIONS, rows, replay, causal: assess(rows) };
}

export function replayDescription(result: ReplayResult): string {
  return `${result.label}: ${result.detected ? "обнаружен" : "не обнаружен"} (AUC ${decimal(result.auc)}). Средняя ошибка в живом эпизоде / при повторе: ${decimal(result.liveError)} / ${decimal(result.replayError)}. Медиана парной разницы: ${decimal(result.medianDifference)}. Ошибка выше при повторе в ${formatRate(result.higher)}.`;
}

export const sharedDetection = "Оба условия различают повтор по ошибке прогноза. Этот сигнал доступен уже модели первого порядка.";
