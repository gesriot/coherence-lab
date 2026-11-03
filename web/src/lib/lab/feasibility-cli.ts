import { decimal } from "./bench.ts";
import { controls, controlNames, permitsComparison, requireControl, runFeasibility, stableOracle } from "./feasibility.ts";

const requested = process.argv.slice(2);
const ids = requested.length ? [...new Set(requested.map(requireControl))] : controls;
const { rates, curves, stability } = runFeasibility(ids);
console.log("Проверка выполнимости · прогоны 701–710 · оценочные зёрна 3001–3020");
if (ids.includes("oracle")) console.log(`Шаги обучения оракула: движение ${decimal(rates.motion)}, существование ${decimal(rates.existence)}.`);
for (const [id, curve] of curves) {
  console.log(controlNames[id]);
  console.table(curve.map((p) => ({
    Эпизод: p.step, Точность: decimal(p.accuracy, 4), Удержание: decimal(p.retention),
    "Без призрака": decimal(p.ghostFree), "λ_hidden / λ_open / m": p.existence.map((v) => decimal(v)).join(" / "),
    "W по строкам": p.weights.map((v) => decimal(v)).join(" "),
  })));
}
const split = curves.get("oracle")?.at(-1)?.split;
if (split) {
  const labels = ["Движение оракула + идеальные вероятности", "Идеальное движение + вероятности оракула"];
  for (const [i, p] of split.entries()) {
    console.log(`${labels[i]} · точность ${decimal(p.accuracy, 4)} · удержание ${decimal(p.retention)} · без призрака ${decimal(p.ghostFree)}`);
  }
}
const oracleCurves = Array.from(stability, ([existence, curve]) => {
  const accuracy = curve.map((p) => p.accuracy);
  console.log(`Оракул · η существования ${decimal(existence)} · ${accuracy.map((x) => decimal(x, 4)).join(" ")} · ${stableOracle(accuracy) ? "устойчив" : "неустойчив"}`);
  return accuracy;
});
if (controls.every((id) => curves.has(id))) {
  const accuracy = (id: keyof typeof controlNames) => curves.get(id)!.map((p) => p.accuracy);
  const passed = permitsComparison({
    frozen: accuracy("frozen"), "keep-all": accuracy("keep-all"), ideal: accuracy("ideal"), oracle: accuracy("oracle"),
  }, oracleCurves);
  console.log(passed ? "Постановка готова." : "Постановка не готова.");
  if (curves.get("oracle")!.find((p) => p.step === 4)!.accuracy >= 0.9) {
    console.log("Оракул достиг порога уже после четырёх эпизодов. Возможно раннее насыщение.");
  }
}
