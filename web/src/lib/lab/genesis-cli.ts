import { decimal } from "./bench.ts";
import { conditions, type Condition } from "./genesis-training.ts";
import { area, checkpoints, effect, feasibility, mean, runTrial, scoreInterval, type Estimate } from "./genesis.ts";

// Selected by genesis:tune on first-order runs 501–510 before the comparison.
const selectedRates = { motion: 0.3, permanence: 0.003 };

const runs = 30;
const ids = Object.keys(conditions) as Condition[];
const experiments = new Map(ids.map((condition) => [
  condition, Array.from({ length: runs }, (_, i) => runTrial(condition, i + 1, 1001, selectedRates)),
]));
const curve = (id: Condition) => experiments.get(id)!;
const accuracy = (id: Condition, i: number) => scoreInterval(curve(id).map((r) => r[i]!.accuracy), 40);
const interval = (e: Estimate) => `${decimal(e.mean)} [${decimal(e.low)}; ${decimal(e.high)}]`;

const frozen = checkpoints.map((_, i) => accuracy("frozen", i).mean);
const oracle = accuracy("oracle", 4).mean;
const gates = feasibility(oracle, frozen, accuracy("full", 1).mean, accuracy("first-order", 1).mean);
console.log("Проверки выполнимости");
console.log(`Оракул после 32 эпизодов · ${decimal(oracle)} · порог ≥ 0,9 · ${gates.positive ? "пройдена" : "не пройдена"}`);
console.log(`Без обучения · максимум ${decimal(Math.max(...frozen))} · порог ≤ 0,75 · ${gates.lower ? "пройдена" : "не пройдена"}`);
console.log(gates.saturated ? "Обучение слишком быстрое: задача не различает условия" : "Насыщения в точке 4 нет.");
if (!gates.positive) console.log("Сравнение не интерпретируется: обучаемый блок не формирует модель даже на истинных метках");
if (!gates.lower) console.log("Сравнение не интерпретируется: задача решается без обучения.");

console.log(`Прогонов: ${runs}. Оценочные зёрна 1001–1020. Интервалы 95% по прогонам. η движения ${decimal(selectedRates.motion)}, η постоянства ${decimal(selectedRates.permanence)}.`);
console.table(ids.flatMap((id) => checkpoints.map((n, i) => {
  const points = curve(id).map((r) => r[i]!);
  const avg = (key: keyof typeof points[number]) => decimal(mean(points.map((p) => p[key])));
  return {
    Условие: conditions[id].label, Эпизоды: n, "Точность [95%]": interval(accuracy(id, i)),
    "Удержание A": avg("retention"), "Без призрака": avg("ghostFree"),
    "Без трека": points.reduce((sum, p) => sum + p.untracked, 0),
    "Ошибка движения": avg("motionError"), "h стены": avg("occlusion"),
    "h пустоты": avg("absence"), "h вне окна": avg("noData"),
  };
})));

const comparisons = (["first-order", "no-protocol", "strong-first-order"] as const).map((id) => {
  const delta = curve("full").map((r, i) => area(r) - area(curve(id)[i]!));
  const estimate = scoreInterval(delta, 160);
  console.log(`Разность площадей · полный контур минус «${conditions[id].label}» · ${interval(estimate)}`);
  return effect(estimate);
});

if (gates.positive && gates.lower) {
  const labels = { faster: "подтверждено на этих конфигурациях", none: "не подтверждено", slower: "контур замедляет обучение" };
  console.log("Ускорение обучения контуром (Genesis):", labels[comparisons[0]!]);
  const protocol = { faster: "подтверждён на этих конфигурациях", none: "не подтверждён", slower: "протокол замедляет обучение" };
  console.log("Вклад протокола внимания:", protocol[comparisons[1]!]);
  console.log("Альтернативное объяснение правилами первого порядка:", comparisons[2] === "faster" ? "исключено" : "не исключено");
  if (comparisons[2] === "slower") console.log("Контур замедляет обучение относительно сильного первого порядка.");
}
console.log("Обучаются движение и постоянство. Заданы классификация клеток, геометрия сенсора, эфферентная копия, оценка освещения, связывание и оттенки, правила внимания и веса E.");
