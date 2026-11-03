import { decimal } from "./bench.ts";
import { controls, controlNames, designCurve, designOracle, requireControl } from "./feasibility.ts";

const requested = process.argv.slice(2);
const ids = requested.length ? [...new Set(requested.map(requireControl))] : controls;
const rates = ids.includes("oracle") ? designOracle() : { motion: 0, permanence: 0 };
console.log("Проверка выполнимости · прогоны 701–710 · оценочные зёрна 3001–3020");
if (ids.includes("oracle")) console.log(`Шаги обучения оракула: движение ${decimal(rates.motion)}, постоянство ${decimal(rates.permanence)}.`);
const curves = new Map(ids.map((id) => [id, designCurve(id, rates)]));
for (const [id, curve] of curves) {
  console.log(controlNames[id]);
  console.table(curve.map((p) => ({
    Эпизод: p.step, Точность: decimal(p.accuracy, 4), Удержание: decimal(p.retention),
    "Без призрака": decimal(p.ghostFree), "h стена / пусто / вне окна": p.hazards.map((v) => decimal(v)).join(" / "),
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
if (controls.every((id) => curves.has(id))) {
  const passed = curves.get("frozen")!.every((p) => p.accuracy <= 0.75)
    && curves.get("keep-all")!.every((p) => p.accuracy <= 0.75)
    && curves.get("ideal")!.at(-1)!.accuracy >= 0.9 && curves.get("oracle")!.at(-1)!.accuracy >= 0.9;
  console.log(passed ? "Постановка готова." : "Постановка не готова.");
  if (curves.get("oracle")!.find((p) => p.step === 4)!.accuracy >= 0.9) {
    console.log("Оракул достиг порога уже после четырёх эпизодов. Возможно раннее насыщение.");
  }
}
