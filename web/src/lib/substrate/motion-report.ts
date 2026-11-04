import type { MotionSettings } from "./gaze.ts";
import { metrics, targets, type MotionScores, type Target } from "./motion-evaluation.ts";

export const gains = ["fixed", "episodic"] as const;
export const copies = ["executed", "zero", "shuffled"] as const;
export const gainNames = { fixed: "Постоянный", episodic: "Свой в эпизоде" };
export const copyNames = { executed: "Копия", zero: "Без копии", shuffled: "Перемешанная" };
export interface Job extends MotionSettings { run: number }
export interface Result extends Job, MotionScores { context?: Record<Target, number> }

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
const fmt = (v: number) => v.toFixed(4).replace(".", ",");

export function commandCriterion(executed: number[], zero: number[], shuffled: number[]) {
  const withCopy = mean(executed), noCopy = mean(zero), mixed = mean(shuffled);
  const paired = executed.filter((v, i) => v < zero[i]! && v < shuffled[i]!).length;
  const passes = executed.length === 5 && zero.length === 5 && shuffled.length === 5 &&
    withCopy <= 0.85 * noCopy && withCopy <= 0.85 * mixed && paired === 5;
  return { vsZero: withCopy / noCopy, vsShuffled: withCopy / mixed, paired, passes };
}

type ContextProbe = { learned: number; initial: number; raw: number; context: number };

function gainOverBest(p: ContextProbe) {
  return p.learned - Math.max(p.initial, p.raw, p.context);
}

export function representationCriterion(probes: ContextProbe[]) {
  const gains = probes.map(gainOverBest);
  const average = mean(gains), wins = gains.filter((g) => g > 0).length;
  return { average, wins, passes: gains.length === 5 && average >= 0.10 && wins === 5 };
}

function withContext(r: Result, key: Target) {
  if (!r.context) throw new Error("Нет оценки контекстной базы");
  return { ...r.probes[key], context: r.context[key] };
}

export function reportMotion(results: Result[], randomPath = false) {
  const gainModes = randomPath ? ["fixed"] as const : gains;
  const group = (gain: Job["gain"], copy: Job["copy"]) =>
    results.filter((r) => r.gain === gain && r.copy === copy).sort((a, b) => a.run - b.run);
  function table(title: string, values: (r: Result) => number[]) {
    console.log("\n" + title);
    for (const gain of gainModes) {
      for (const copy of copies) {
        const label = gainNames[gain] + " · " + copyNames[copy];
        const matrix = group(gain, copy).map((r) => {
          const row = values(r);
          console.log([label, r.run, ...row.map(fmt)].join("\t"));
          return row;
        });
        console.log([label, "среднее", ...matrix[0]!.map((_, i) => fmt(mean(matrix.map((v) => v[i]!))))].join("\t"));
      }
    }
  }
  console.log("Проверка на зёрнах 3001–3030, обучение зондов на 4001–4030. Ошибки относительно копии кадра.");
  table("Коэффициент · условие\tПрогон\tВсе кадры\tПосле сдвига\tБез сдвига\tРанние\tПоздние",
    (r) => metrics.map((k) => r.ratios[k]));
  const names = { own: "Собственный сдвиг", world: "Смещение A в мире", apparent: "Смещение A в окне" };
  for (const key of targets) {
    if (randomPath) {
      table(names[key] + " (R²)\tПрогон\tОбученный\tНачальный\tСырой вход\tТолько контекст\tПеремешанные метки\tПрирост над лучшей базой",
        (r) => {
          const p = withContext(r, key);
          return [p.learned, p.initial, p.raw, p.context, p.shuffled, gainOverBest(p)];
        });
      continue;
    }
    table(names[key] + " (R²)\tПрогон\tОбученный\tНачальный\tСырой вход\tПеремешанные метки\tПрирост к начальному\tПрирост к сырому",
      (r) => {
        const p = r.probes[key];
        return [p.learned, p.initial, p.raw, p.shuffled, p.learned - p.initial, p.learned - p.raw];
      });
  }
  for (const gain of gainModes) {
    const shifted = (copy: Job["copy"]) => group(gain, copy).map((r) => r.ratios.moved);
    const c = commandCriterion(shifted("executed"), shifted("zero"), shifted("shuffled"));
    const { trainCount, testCount } = group(gain, "executed")[0]!;
    console.log(`\n${gainNames[gain]} · примеров зондов: обучение ${trainCount}, проверка ${testCount}.`);
    console.log(`Копия / без копии ${fmt(c.vsZero)}, копия / перемешанная ${fmt(c.vsShuffled)} (порог ≤ 0,85); ниже обоих контролей в ${c.paired} из 5 прогонов.`);
    console.log(`Критерий «копия команды используется» ${c.passes ? "выполнен" : "не выполнен"}.`);
  }
  if (randomPath) {
    console.log("\nДвижение объекта в мире\tСредний прирост\tПрогонов выше трёх баз\tКритерий");
    for (const copy of copies) {
      const criterion = representationCriterion(group("fixed", copy).map((r) => withContext(r, "world")));
      console.log([copyNames[copy], fmt(criterion.average), criterion.wins + "/5",
        criterion.passes ? "выполнен" : "не выполнен"].join("\t"));
    }
    const withCopy = group("fixed", "executed"), without = group("fixed", "zero");
    console.log("\nПрогон\tРазность R² смещения A в мире: копия минус без копии");
    const differences = withCopy.map((r, i) => {
      const difference = r.probes.world.learned - without[i]!.probes.world.learned;
      console.log([r.run, fmt(difference)].join("\t"));
      return difference;
    });
    console.log(["Среднее", fmt(mean(differences))].join("\t"));
  }
}
