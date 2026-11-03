import { decimal } from "./bench.ts";
import { controlCurve, controls, controlNames, permitsComparison, stableOracle, type Control } from "./feasibility.ts";
import type { Rates } from "./experience.ts";
import { experimentSeeds, genesisTask } from "./genesis-task.ts";
import { conditions, type Condition } from "./genesis-training.ts";
import { area, checkpoints, effect, mean, runTrial, scoreInterval, type Estimate, type Point } from "./genesis.ts";

// Selected with the last-visible-frame metric on first-order runs 521–530 and evaluation seeds 2041–2060.
const selectedRates = { motion: 0.03, existence: 0.1 };

const { runs, firstRun, firstEvalSeed } = experimentSeeds.comparison;
const { evaluationCount } = experimentSeeds;
const comparisonIds = ["full", "first-order", "no-protocol", "strong-first-order"] as const;
type Id = Condition | Control | "participant-oracle";
const experiments = new Map<Id, Point[][]>();
const curve = (id: Id) => experiments.get(id)!;
const accuracy = (id: Id, i: number) => scoreInterval(curve(id).map((r) => r[i]!.accuracy), 40);
const interval = (e: Estimate) => `${decimal(e.mean)} [${decimal(e.low)}; ${decimal(e.high)}]`;
const label = (id: Id) => {
  if (id === "participant-oracle") return "Оракул (шаги участников)";
  return id === "ideal" || id === "keep-all" ? controlNames[id] : conditions[id].label;
};

const oracleRuns = (rates: Rates) => Array.from({ length: runs }, (_, i) =>
  controlCurve("oracle", firstRun + i, rates, checkpoints, firstEvalSeed, evaluationCount));
for (const id of controls) {
  if (id === "oracle") {
    experiments.set(id, oracleRuns(genesisTask.oracleRates));
  } else {
    const fixed = controlCurve(id, firstRun, genesisTask.oracleRates, checkpoints, firstEvalSeed, evaluationCount);
    experiments.set(id, Array.from({ length: runs }, () => fixed));
  }
}
const sameRates = genesisTask.oracleRates.motion === selectedRates.motion
  && genesisTask.oracleRates.existence === selectedRates.existence;
experiments.set("participant-oracle", sameRates ? curve("oracle") : oracleRuns(selectedRates));
const scores = (id: Id) => checkpoints.map((_, i) => accuracy(id, i).mean);
const measured = {
  frozen: scores("frozen"), "keep-all": scores("keep-all"), ideal: scores("ideal"), oracle: scores("oracle"),
};
const participantOracle = scores("participant-oracle");
const ready = permitsComparison(measured, [participantOracle]);
console.log("Проверки выполнимости");
for (const id of controls) {
  const lower = id === "frozen" || id === "keep-all";
  const value = lower ? Math.max(...measured[id]) : measured[id].at(-1)!;
  console.log(`${label(id)} · ${decimal(value)} · порог ${lower ? "≤ 0,75 во всех точках" : "≥ 0,9 в последней точке"}`);
}

for (const id of ["oracle", "participant-oracle"] as const) {
  const values = id === "oracle" ? measured.oracle : participantOracle;
  console.log(`${label(id)} · ${values.map((v) => decimal(v, 4)).join(" ")} · ${stableOracle(values) ? "устойчив" : "неустойчив"}`);
  const below = curve(id).filter((run) => run.at(-1)!.accuracy < genesisTask.controlFloor).length;
  console.log(`${label(id)} · прогонов с итоговой точностью ниже ${decimal(genesisTask.controlFloor, 1)}: ${below} из ${runs}`);
}

if (ready) {
  for (const id of comparisonIds) {
    experiments.set(id, Array.from({ length: runs }, (_, i) => runTrial(id, firstRun + i, firstEvalSeed, selectedRates)));
  }
  const saturated = accuracy("full", 1).mean >= 0.9 && accuracy("first-order", 1).mean >= 0.9;
  console.log(saturated ? "Обучение слишком быстрое: задача не различает условия" : "Насыщения в точке 4 нет.");
} else {
  console.log("Сравнение не интерпретируется: проверка выполнимости не пройдена.");
}

console.log(`Прогоны ${firstRun}–${firstRun + runs - 1}. Оценочные зёрна ${firstEvalSeed}–${firstEvalSeed + evaluationCount - 1}. Интервалы 95% по прогонам.`);
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
