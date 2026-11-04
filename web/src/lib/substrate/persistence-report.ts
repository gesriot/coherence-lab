import { OFFSETS } from "./evaluation.ts";
import type { PersistenceScores as Score } from "./persistence-probes.ts";

export const variants = {
  plain: { name: "Собель 16", kind: "plain", depth: 16, runs: 5 },
  memory: { name: "Память 16", kind: "memory", depth: 16, runs: 5 },
  wide: { name: "Память 28", kind: "memory", depth: 28, runs: 5 },
  hinted: { name: "Подсказка 16", kind: "hinted", depth: 16, runs: 3 },
} as const;
export type Variant = keyof typeof variants;
export interface Job { variant: Variant; run: number }
export interface Result extends Job { learned: Score; initial?: Score }

const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;

export function partialPersistence(main: Pick<Score, "exit" | "curve">[]) {
  const exit = mean(main.map((s) => s.exit)), worst = Math.max(...main.map((s) => s.exit));
  const at = mean(main.map((s) => s.curve[6]!)), before = mean(main.map((s) => mean(s.curve.slice(0, 3))));
  return { exit, worst, at, before, passes: main.length === 5 && exit <= 0.90 && worst <= 0.95 && at >= 0.03 && at > before };
}

const fmt = (v: number) => v.toFixed(4).replace(".", ",");
function line(label: string, values: number[]) {
  console.log([label, ...values.map(fmt)].join("\t"));
}

export function reportPersistence(raw: Score, runs: Result[]) {
  const families = Object.keys(variants) as Variant[];
  const group = (v: Variant) => runs.filter((r) => r.variant === v).sort((a, b) => a.run - b.run);
  function matched(r: Result): Score {
    const family = r.variant === "hinted" ? "memory" : r.variant;
    return runs.find((row) => row.variant === family && row.run === r.run)!.initial!;
  }
  function table(
    header: string, columns: (s: Score, initial: Score) => number[], includeInitial: boolean,
  ) {
    console.log("\n" + header);
    line("Копия / сырой вход", columns(raw, raw));
    function printGroup(name: string, rows: Result[], initial: boolean) {
      const matrix = rows.map((r) => {
        const values = columns(initial ? r.initial! : r.learned, matched(r));
        line(`${name} · ${r.run}`, values);
        return values;
      });
      line(`${name} · среднее`, matrix[0]!.map((_, i) => mean(matrix.map((v) => v[i]!))));
    }
    for (const family of families) printGroup(variants[family].name, group(family), false);
    if (includeInitial) for (const family of families.filter((f) => f !== "hinted")) {
      printGroup("Начальный " + variants[family].name, group(family), true);
    }
  }

  console.log("Оценка: 3001–3030; обучение зондов: 4001–4030. Ошибки относительно копии; призрак – абсолютная ошибка.");
  console.log(`Возврат видимости: ${raw.diagnostic.events} событий; в ${raw.diagnostic.wallEvents} из них A касался стены во время скрытия.`);
  table("Условие\tВсе клетки\tМеняющиеся\tВыход\tПризрак\tВыход после касания стены",
    (s) => [s.all, s.changed, s.exit, s.ghost, s.diagnostic.wallExit!], true);
  table(["Насыщенность / шаг от выхода", ...OFFSETS].join("\t"), (s) => s.curve, true);
  const probes = { velocity: "Знак скорости (доля верных)", hidden: "Скрытый x (R²)", late: "Скрытый x с шага 4 (R²)" } as const;
  for (const key of Object.keys(probes) as (keyof typeof probes)[]) {
    table(probes[key] + "\tЗонд\tНачальный\tСырой вход\tПеремешано\tПрирост",
      (s, initial) => [s[key].value, initial[key].value, raw[key].value, s[key].shuffled, s[key].value - initial[key].value], false);
  }

  console.log("\nПодсказка при полном скрытии\tКлетки A\tПустая стена\tВыход");
  const hinted = group("hinted").map(({ run, learned: s }) => {
    const values = [s.hint!.occupied, s.hint!.empty, s.exit];
    line("Прогон " + run, values);
    return values;
  });
  line("Среднее", hinted[0]!.map((_, i) => mean(hinted.map((v) => v[i]!))));
  const gate = partialPersistence(group("memory").map((r) => r.learned));
  console.log(`Память 16: выход ${fmt(gate.exit)} (≤ 0,90), худший ${fmt(gate.worst)} (≤ 0,95).`);
  console.log(`Ожидание в точке 0: ${fmt(gate.at)} (≥ 0,03), среднее в −6…−4: ${fmt(gate.before)} (требуется рост).`);
  console.log(`Критерий частичного постоянства: ${gate.passes ? "выполнен" : "не выполнен"}.`);
}
