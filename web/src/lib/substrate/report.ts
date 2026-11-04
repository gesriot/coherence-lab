import { OFFSETS } from "./evaluation.ts";
import type { Scores } from "./probes.ts";

export function average(scores: Scores[]): Scores {
  const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;
  const field = (key: Exclude<keyof Scores, "curve">) => mean(scores.map((s) => s[key]));
  return {
    all: field("all"), changed: field("changed"), exit: field("exit"), ghost: field("ghost"),
    accuracy: field("accuracy"), hidden: field("hidden"), shuffled: field("shuffled"),
    curve: OFFSETS.map((_, i) => mean(scores.map((s) => s.curve[i]!))),
  };
}

export function viable(directed: Scores[], initial: Scores[]): boolean {
  const main = average(directed);
  return directed.length === 5 && initial.length === 5 &&
    main.changed <= 0.80 && directed.every((s) => s.changed <= 0.85) &&
    initial.every((s) => s.changed >= 1.0) && main.accuracy >= 0.75;
}

export interface RunResult {
  run: number;
  directed: boolean;
  seconds: number;
  learned: Scores;
  initial?: Scores;
}

const fmt = (n: number) => n.toFixed(4).replace(".", ",");

export function report(copy: Scores, runs: RunResult[]) {
  const directed = runs.filter((r) => r.directed), isotropic = runs.filter((r) => !r.directed);
  const initial = directed.map((r) => r.initial!);
  const lines: [string, Scores][] = [];
  function group(name: string, scores: Scores[]) {
    scores.forEach((s, i) => lines.push([`${name} ${i + 1}`, s]));
    lines.push([`${name} · среднее`, average(scores)]);
  }
  group("Собель", directed.map((r) => r.learned));
  lines.push(["Копия", copy]);
  group("Без обучения", initial);
  group("Без направлений", isotropic.map((r) => r.learned));

  console.log("Ошибки – доли ошибки копии; «Призрак» – средняя абсолютная ошибка по каналам.");
  console.log("У копии зонд читает три канала текущего кадра.");
  console.log("Условие\tВсе клетки\tМеняющиеся\tВыход\tПризрак\tЗнак скорости\tСкрытый x (R²)\tПеремешано (R²)");
  for (const [name, s] of lines) {
    console.log([name, ...[s.all, s.changed, s.exit, s.ghost, s.accuracy, s.hidden, s.shuffled].map(fmt)].join("\t"));
  }
  console.log("\nСредняя предсказанная насыщенность в клетках выхода");
  console.log(["Условие / шаг от выхода", ...OFFSETS].join("\t"));
  for (const [name, s] of lines) console.log([name, ...s.curve.map(fmt)].join("\t"));

  const scores = directed.map((r) => r.learned), main = average(scores);
  console.log("\nПроверка B0 по неокруглённым значениям:");
  console.log(`Меняющиеся: среднее ${fmt(main.changed)} (≤ 0,80); худший прогон ${fmt(Math.max(...scores.map((s) => s.changed)))} (≤ 0,85).`);
  console.log(`Без обучения: минимум ${fmt(Math.min(...initial.map((s) => s.changed)))} (≥ 1,0).`);
  console.log(`Зонд знака скорости: среднее ${fmt(main.accuracy)} (≥ 0,75).`);
  console.log(`Жизнеспособность B0: ${viable(scores, initial) ? "подтверждена" : "не подтверждена"}.`);
  console.log("Для метрик постоянства порог не задан.");
}
