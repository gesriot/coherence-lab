import { createEpisode, defaultConfig, snapshot, stepEpisode, type Episode } from "./agent.ts";
import type { Ablation, LabConfig, ScenarioId } from "./types.ts";

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
  configurations: number;
  rows: BenchRow[];
  causal: "supported" | "unsupported" | "mixed";
  summary: string;
}

const SEEDS = Array.from({ length: 20 }, (_, i) => i + 1);

function roll(scenario: ScenarioId, ablation: Ablation, seed: number, steps: number, mutate?: (ep: Episode, i: number, cfg: LabConfig) => void) {
  const ep = createEpisode(scenario, seed, ablation);
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
  const slice = views.slice(16);
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
      id: "first-order",
      label: "Первый порядок",
      detail: "Сопровождение трека по прогнозу движения",
      ablation: "first-order",
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
      detail: "Взгляд направлен к самому крупному пятну",
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
    detail: "Записанные кадры проигрываются независимо от взгляда",
    retention: null,
    pair: null,
    selfError: null,
    actionError: mean(replayActions),
  });

  const { causal, summary } = assess(rows, SEEDS.length);
  return { configurations: SEEDS.length, rows, causal, summary };
}

const TASK_MARGIN = 0.2;
const REPLAY_MARGIN = 0.08;
export const criterion = `Для подтверждения полный контур должен обойти каждый из трёх контролей (первый порядок, только запись, лишняя память) хотя бы по одной задаче более чем на ${TASK_MARGIN * 100} п. п. При повторе собственная ошибка прогноза должна вырасти более чем на ${REPLAY_MARGIN.toFixed(2).replace(".", ",")}.`;

export function assess(rows: BenchRow[], count: number): Pick<BenchReport, "causal" | "summary"> {
  const row = (id: string) => rows.find((item) => item.id === id)!;
  const full = row("full");
  const beats = (control: BenchRow) => (["retention", "pair"] as const).some((metric) =>
    full[metric] != null && control[metric] != null && full[metric] > control[metric] + TASK_MARGIN + 1e-12);
  const first = beats(row("first-order"));
  const controlsPassed = first && beats(row("record")) && beats(row("dummy"));
  const live = full.actionError;
  const replay = row("replay").actionError;
  const detected = live != null && replay != null && replay > live + REPLAY_MARGIN + 1e-12;
  const causal = controlsPassed && detected ? "supported" : first ? "mixed" : "unsupported";
  const comparison = first ? "превзошёл первый порядок с заданным запасом" : "не показал требуемого преимущества над первым порядком";
  const replayResult = detected ? "Повтор обнаружен по ошибке прогноза." : "Прирост ошибки прогноза слишком мал для обнаружения повтора.";
  return { causal, summary: `На ${count} конфигурациях полный контур ${comparison}. ${replayResult}` };
}
