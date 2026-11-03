import { decimal } from "./bench.ts";
import { controlCurve, controls, controlNames, permitsComparison, type Control } from "./feasibility.ts";
import { genesisTask } from "./genesis-task.ts";
import { conditions, type Condition } from "./genesis-training.ts";
import { area, checkpoints, effect, mean, runTrial, scoreInterval, type Estimate, type Point } from "./genesis.ts";

// Selected on the frozen 64-episode task using first-order runs 501–510 and evaluation seeds 2001–2020.
const selectedRates = { motion: 0.03, existence: 0.03 };

const runs = 30;
const evalSeed = 1001;
const comparisonIds = ["full", "first-order", "no-protocol", "strong-first-order"] as const;
type Id = Condition | Control;
const experiments = new Map<Id, Point[][]>();
const curve = (id: Id) => experiments.get(id)!;
const accuracy = (id: Id, i: number) => scoreInterval(curve(id).map((r) => r[i]!.accuracy), 40);
const interval = (e: Estimate) => `${decimal(e.mean)} [${decimal(e.low)}; ${decimal(e.high)}]`;
const label = (id: Id) => id === "ideal" || id === "keep-all" ? controlNames[id] : conditions[id].label;

for (const id of controls) {
  if (id === "oracle") {
    experiments.set(id, Array.from({ length: runs }, (_, i) =>
      controlCurve(id, i + 1, genesisTask.oracleRates, checkpoints, evalSeed)));
  } else {
    const fixed = controlCurve(id, 1, genesisTask.oracleRates, checkpoints, evalSeed);
    experiments.set(id, Array.from({ length: runs }, () => fixed));
  }
}
const scores = (id: Control) => checkpoints.map((_, i) => accuracy(id, i).mean);
const measured = {
  frozen: scores("frozen"), "keep-all": scores("keep-all"), ideal: scores("ideal"), oracle: scores("oracle"),
};
const ready = permitsComparison(measured);
console.log("Проверки выполнимости");
for (const id of controls) {
  const lower = id === "frozen" || id === "keep-all";
  const value = lower ? Math.max(...measured[id]) : measured[id].at(-1)!;
  console.log(`${label(id)} · ${decimal(value)} · порог ${lower ? "≤ 0,75 во всех точках" : "≥ 0,9 в последней точке"}`);
}

if (ready) {
  for (const id of comparisonIds) {
    experiments.set(id, Array.from({ length: runs }, (_, i) => runTrial(id, i + 1, evalSeed, selectedRates)));
  }
  const saturated = accuracy("full", 1).mean >= 0.9 && accuracy("first-order", 1).mean >= 0.9;
  console.log(saturated ? "Обучение слишком быстрое: задача не различает условия" : "Насыщения в точке 4 нет.");
} else {
  console.log("Сравнение не интерпретируется: проверка выполнимости не пройдена.");
}

console.log(`Прогонов: ${runs}. Оценочные зёрна 1001–1020. Интервалы 95% по прогонам.`);
console.log(`Шаги обучения оракула: ${decimal(genesisTask.oracleRates.motion)} / ${decimal(genesisTask.oracleRates.existence)}. Участники: ${decimal(selectedRates.motion)} / ${decimal(selectedRates.existence)}.`);
console.table(Array.from(experiments.keys()).flatMap((id) => checkpoints.map((n, i) => {
  const points = curve(id).map((r) => r[i]!);
  const avg = (key: keyof Point) => decimal(mean(points.map((p) => p[key])));
  return {
    Условие: label(id), Эпизоды: n, "Точность [95%]": interval(accuracy(id, i)),
    "Удержание A": avg("retention"), "Без призрака": avg("ghostFree"),
    "Без трека": points.reduce((sum, p) => sum + p.untracked, 0),
    "Ошибка движения": avg("motionError"), λ_hidden: avg("hidden"), λ_open: avg("open"), m: avg("miss"),
  };
})));

if (ready) {
  const comparisons = (["first-order", "no-protocol", "strong-first-order"] as const).map((id) => {
    const delta = curve("full").map((r, i) => area(r) - area(curve(id)[i]!));
    const estimate = scoreInterval(delta, (checkpoints.length - 1) * 40);
    console.log(`Разность площадей · полный контур минус «${conditions[id].label}» · ${interval(estimate)}`);
    return effect(estimate);
  });
  const labels = { faster: "подтверждено на этих конфигурациях", none: "не подтверждено", slower: "контур замедляет обучение" };
  console.log("Ускорение обучения контуром (Genesis):", labels[comparisons[0]!]);
  const protocol = { faster: "подтверждён на этих конфигурациях", none: "не подтверждён", slower: "протокол замедляет обучение" };
  console.log("Вклад протокола внимания:", protocol[comparisons[1]!]);
  console.log("Альтернативное объяснение правилами первого порядка:", comparisons[2] === "faster" ? "исключено" : "не исключено");
  if (comparisons[2] === "slower") console.log("Контур замедляет обучение относительно сильного первого порядка.");
}
console.log("Обучаются модели движения и существования объекта. Заданы классификация клеток, геометрия сенсора, эфферентная копия, оценка освещения, связывание и оттенки, правила внимания и веса E.");
