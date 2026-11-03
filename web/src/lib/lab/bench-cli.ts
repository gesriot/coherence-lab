import { causalLabel, criterion, decimal, explorationNote, formatRate, replayDescription, runBench, sharedDetection } from "./bench.ts";

const report = runBench();
console.log(criterion);
console.log("Доли успеха и 95%-е интервалы Уилсона");
console.table(report.rows.map((row) => ({
  Условие: row.label,
  "Удержание A": formatRate(row.retention),
  "Пара различена": formatRate(row.pair),
  "Доля шагов обхода": `${decimal(row.exploreRate * 100, 2)}%`,
  "Ошибка самомодели": decimal(row.selfError),
  "Ошибка действия": decimal(row.actionError),
})));
console.log("Причинная роль модели второго порядка:", causalLabel[report.causal]);
for (const message of report.saturation) console.log(message);
const note = explorationNote(report.rows);
if (note) console.log(note);
console.log("Обнаружение повтора по собственному прогнозу");
for (const result of report.replay) console.log(replayDescription(result));
if (report.replay.every((r) => r.detected)) console.log(sharedDetection);
