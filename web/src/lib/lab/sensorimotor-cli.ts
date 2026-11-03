import { decimal } from "./bench.ts";
import { controlNames, stableOracle } from "./feasibility.ts";
import { scoreInterval, type Estimate } from "./genesis.ts";
import { conditions } from "./genesis-training.ts";
import { compareGainConditions } from "./sensorimotor-comparison.ts";
import { sensorimotorTask } from "./sensorimotor-task.ts";
import { averageGainCurve, gainRuns, type GainRuns } from "./sensorimotor.ts";

const { comparison, evaluationCount, oracleRate, rates } = sensorimotorTask;
const args = process.argv.slice(2);
const selected = Number(args[0]?.replace(",", "."));
if (args.length !== 1 || !rates.some((rate) => rate === selected)) {
  throw new Error("Передайте шаг из результата подбора: npm run sensorimotor -- <шаг>");
}
const interval = (e: Estimate) => `${decimal(e.mean)} [${decimal(e.low)}; ${decimal(e.high)}]`;
function printCurve(label: string, runs: GainRuns) {
  console.log(label);
  console.table(averageGainCurve(runs).map((p, i) => ({
    Эпизод: p.step, "Точность [95%]": interval(scoreInterval(runs.map((r) => r[i]!.accuracy), 2 * evaluationCount)),
    "Удержание A": decimal(p.retention), "Без призрака": decimal(p.ghostFree),
    "Ошибка калибровки": decimal(p.calibrationError, 5), "Оценка X": decimal(p.gainX, 5), "Оценка Y": decimal(p.gainY, 5),
  })));
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
  console.log(stableOracle(averageGainCurve(curve).map((p) => p.accuracy)) ? "Кривая устойчива." : "Кривая неустойчива.");
}
const result = compareGainConditions(controls, (id) => gainRuns(id, selected, comparison));
if (result) {
  for (const [id, curve] of result.participants) printCurve(conditions[id].label, curve);
  console.log(result.saturated ? "Обучение слишком быстрое: задача не различает условия." : "Насыщения в первой точке обучения нет.");
  for (const { id, estimate } of result.differences) {
    console.log(`Разность площадей · полный контур минус «${conditions[id].label}» · ${interval(estimate)}`);
  }
  for (const line of result.verdicts) console.log(line);
} else {
  console.log("Сравнение не интерпретируется: проверка выполнимости не пройдена.");
}
