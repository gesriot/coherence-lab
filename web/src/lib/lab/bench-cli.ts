import { runBench } from "./bench.ts";

const report = runBench();
const rate = (value: number | null) => value == null ? "–" : `${(value * 100).toFixed(0)}%`;
const error = (value: number | null) => value?.toFixed(3) ?? "–";

console.log(`Конфигураций на условие: ${report.configurations}`);
console.table(report.rows.map((row) => ({
  Условие: row.label,
  "Удержание A": rate(row.retention),
  "Пара различена": rate(row.pair),
  "Ошибка самомодели": error(row.selfError),
  "Ошибка действия": error(row.actionError),
})));
console.log(report.causal, report.summary);
