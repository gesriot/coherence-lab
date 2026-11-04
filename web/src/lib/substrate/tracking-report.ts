import { OFFSETS } from "./evaluation.ts";
import type { WallColumns } from "./input.ts";
import type { TrackingScores } from "./tracking-evaluation.ts";

export interface Job { columns: WallColumns; weighted: boolean; run: number; episodes: number }
export interface Point { episode: number; wall: number; all: number }
export interface Result extends Job { scores: TrackingScores; formation: Point[] }

const mean = (numbers: number[]) => numbers.reduce((a, b) => a + b, 0) / numbers.length;
const fmt = (v: number) => v.toFixed(4).replace(".", ",");

export function trackingCriterion(base: number[], control: Pick<TrackingScores, "wall" | "other" | "after">[]) {
  const exit = mean(control.map((s) => s.wall)), worst = Math.max(...control.map((s) => s.wall));
  const other = mean(control.map((s) => s.other)), after = mean(control.map((s) => s.after));
  const baseline = mean(base);
  const passes = base.length === 5 && control.length === 5 && exit <= 0.5 && worst <= 0.65 &&
    other <= 0.03 && after <= 0.08 && baseline >= 0.9;
  return { exit, worst, other, after, baseline, passes };
}

export const condition = (weighted: boolean) => weighted ? "Вес ×50" : "База";

export function reportTracking(results: Result[]) {
  const ordered = results.slice().sort((a, b) =>
    a.columns - b.columns || Number(a.weighted) - Number(b.weighted) || a.run - b.run);
  type Group = { label: string; runs: Result[] };
  const groups: Group[] = [];
  for (const columns of [1, 2, 4]) {
    for (const weighted of [false, true]) {
      const runs = ordered.filter((r) => r.columns === columns && r.weighted === weighted);
      const prefix = `${columns} · ${condition(weighted)} · `;
      for (const r of runs) groups.push({ label: prefix + r.run, runs: [r] });
      groups.push({ label: prefix + "среднее", runs });
    }
  }
  const columns = (group: Group, values: (r: Result) => number[]) => {
    const rows = group.runs.map(values);
    return rows[0]!.map((_, i) => fmt(mean(rows.map((row) => row[i]!))));
  };
  const span = (s: TrackingScores["wallEvents"]) => `${s.count} (${s.min}–${s.max})`;
  console.log("Проверочные зёрна 3001–3030. Ошибки относительно копии, призрак – абсолютная ошибка.");
  console.log("Стена · условие · прогон\tВсе\tМеняющиеся\tПризрак\tВыход\tПропуск\tДругие строки\tПосле ухода\tВыходы (кадры)\tПропуски (кадры)");
  for (const g of groups) {
    const { wallEvents, gapEvents } = g.runs[0]!.scores;
    console.log([g.label, ...columns(g, ({ scores: s }) =>
      [s.all, s.changed, s.ghost, s.wall, s.gap, s.other, s.after]), span(wallEvents), span(gapEvents)].join("\t"));
  }
  console.log("\n" + ["Ожидание / шаг от выхода", ...OFFSETS].join("\t"));
  for (const g of groups) console.log([g.label, ...columns(g, (r) => r.scores.curve)].join("\t"));
  console.log("\nФормирование\tЭпизоды\tВыход\tВсе клетки");
  for (const g of groups) {
    g.runs[0]!.formation.forEach(({ episode }, i) => {
      console.log([g.label, episode, ...columns(g, (r) =>
        [r.formation[i]!.wall, r.formation[i]!.all])].join("\t"));
    });
  }
  for (const w of [1, 2]) {
    const at = ordered.filter((r) => r.columns === w);
    const gate = trackingCriterion(at.filter((r) => !r.weighted).map((r) => r.scores.wall),
      at.filter((r) => r.weighted).map((r) => r.scores));
    console.log(`Стена ${w} · выход ×50 ${fmt(gate.exit)} (≤ 0,50), худший ${fmt(gate.worst)} (≤ 0,65), другие строки ${fmt(gate.other)} (≤ 0,03), после ухода ${fmt(gate.after)} (≤ 0,08), база ${fmt(gate.baseline)} (≥ 0,90).`);
    console.log(`Стена ${w} · критерий «слежение выполнимо» ${gate.passes ? "выполнен" : "не выполнен"}.`);
  }
  console.log("Стена 4 · описательное условие без порога.");
}
