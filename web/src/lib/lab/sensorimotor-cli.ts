import { decimal } from "./bench.ts";
import { controlNames, stableOracle } from "./feasibility.ts";
import { bootstrap, scoreInterval, type Estimate } from "./genesis.ts";
import { conditions } from "./genesis-training.ts";
import { compareGainConditions } from "./sensorimotor-comparison.ts";
import { sensorimotorTask } from "./sensorimotor-task.ts";
import { summarizeGainRuns, gainRuns, type GainRuns } from "./sensorimotor.ts";

const { comparison, evaluationCount, oracleRate, rates } = sensorimotorTask;
const args = process.argv.slice(2);
const selected = Number(args[0]?.replace(",", "."));
if (args.length !== 1 || !rates.some((rate) => rate === selected)) {
  throw new Error("Передайте шаг из результата подбора: npm run sensorimotor -- <шаг>");
}
const interval = (e: Estimate) => `${decimal(e.mean, 4)} [${decimal(e.low, 4)}; ${decimal(e.high, 4)}]`;
function printCurve(label: string, runs: GainRuns) {
  console.log(label);
  const summary = summarizeGainRuns(runs);
  console.table(summary.curve.map((p, i) => ({
    Шаг: p.step, "q [95%]": interval(bootstrap(runs.map((r) => r.curve[i]!.q))),
    "Ошибка калибровки": decimal(p.error, 5), "Оценка X": decimal(p.gainX, 5), "Оценка Y": decimal(p.gainY, 5),
  })));
  const accuracy = scoreInterval(runs.map((r) => r.secondary.accuracy), 2 * evaluationCount);
  console.log(`Итоговая точность ${interval(accuracy)} · удержание A ${decimal(summary.retention)} · без призрака ${decimal(summary.ghostFree)}`);
  console.log(`Саккад на эпизод: ${decimal(summary.saccadesPerEpisode)}`);
}

const { firstRun, runs, firstEvalSeed } = comparison;
console.log(`Прогоны ${firstRun}–${firstRun + runs - 1} · оценочные зёрна ${firstEvalSeed}–${firstEvalSeed + evaluationCount - 1}`);
const oracle = gainRuns("oracle", oracleRate, comparison);
const controls = {
  frozen: gainRuns("frozen", 0, comparison), ideal: gainRuns("ideal", 0, comparison), oracle,
  participantOracle: selected === oracleRate ? oracle : gainRuns("oracle", selected, comparison),
};
printCurve(controlNames.frozen, controls.frozen);
printCurve(controlNames.ideal, controls.ideal);
for (const [label, rate, curve] of [
  ["Оракул", oracleRate, oracle], ["Оракул (шаг участников)", selected, controls.participantOracle],
] as const) {
  printCurve(`${label} · η усиления ${decimal(rate)}`, curve);
  console.log(stableOracle(summarizeGainRuns(curve).curve.map((p) => p.q)) ? "Кривая устойчива." : "Кривая неустойчива.");
}
const result = compareGainConditions(controls, (id) => gainRuns(id, selected, comparison));
if (result) {
  console.log("Допуск по калибровке и итоговой точности пройден.");
  for (const [id, curve] of result.participants) printCurve(conditions[id].label, curve);
  console.log(result.saturated ? "Обучение слишком быстрое: задача не различает условия." : "Насыщения в точке 10 шагов нет.");
  for (const { id, estimate } of result.differences) {
    console.log(`Разность площадей · полный контур минус «${conditions[id].label}» · ${interval(estimate)}`);
  }
  for (const line of result.verdicts) console.log(line);
} else {
  console.log("Сравнение не интерпретируется: проверка выполнимости не пройдена.");
}
