import { decimal } from "./bench.ts";
import { controlNames, stableOracle } from "./feasibility.ts";
import { genesisTask } from "./genesis-task.ts";
import { designGainControls, designOracleRate, fixedPointCheck, fixedPointPass, gainControlsPass } from "./sensorimotor-feasibility.ts";
import { sensorimotorTask } from "./sensorimotor-task.ts";

import { printFixedPoint } from "./sensorimotor-report.ts";

const args = process.argv.slice(2);
const results = designGainControls(args.length ? args : undefined);
const { firstRun, runs, firstEvalSeed } = sensorimotorTask.design;
console.log(`Проверка сенсомоторной зависимости · прогоны ${firstRun}–${firstRun + runs - 1} · оценочные зёрна ${firstEvalSeed}–${firstEvalSeed + sensorimotorTask.evaluationCount - 1}`);
console.log("«Хранить всё» совпадает с «Без обучения»: окно считается расположенным там, куда скомандовал агент.");
for (const { id, rate, curve, accuracy, retention, ghostFree } of results) {
  const label = id === "neutral" ? "Самообучение при нейтральном взгляде" : controlNames[id];
  console.log(rate ? `${label} · η усиления ${decimal(rate)}` : label);
  console.table(curve.map((p) => ({
    Шаг: p.step, q: decimal(p.q, 5), "Ошибка калибровки": decimal(p.error, 5),
    "Оценка X": decimal(p.gainX, 5), "Оценка Y": decimal(p.gainY, 5),
  })));
  console.log(`Итоговая точность ${decimal(accuracy, 4)} · удержание A ${decimal(retention)} · без призрака ${decimal(ghostFree)}`);
  if (id === "oracle") console.log(stableOracle(curve.map((p) => p.q)) ? "Кривая устойчива." : "Кривая неустойчива.");
}
const frozen = results.find((r) => r.id === "frozen"), ideal = results.find((r) => r.id === "ideal");
const oracles = results.filter((r) => r.id === "oracle");
if (frozen && ideal && oracles.length === sensorimotorTask.rates.length) {
  const fixedPoint = fixedPointCheck(sensorimotorTask.design);
  printFixedPoint(fixedPoint);
  const ready = fixedPointPass(fixedPoint) && gainControlsPass(frozen, ideal, oracles);
  console.log(ready ? "Критерий готовности пройден." : "Критерий готовности не пройден.");
  if (ready) {
    const rate = designOracleRate(oracles.map((o) => ({ rate: o.rate, finalQ: o.curve.at(-1)!.q })));
    console.log(`Собственный шаг оракула по итоговому q: ${decimal(rate)}`);
  }
  const saturated = oracles.every((o) => o.curve[1]!.q >= genesisTask.controlFloor);
  console.log(saturated ? "Насыщение в первой точке обучения при всех шагах." : "Насыщения при всех шагах в первой точке обучения нет.");
}
