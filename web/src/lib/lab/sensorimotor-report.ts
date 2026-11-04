import { decimal } from "./bench.ts";
import type { fixedPointCheck } from "./sensorimotor-feasibility.ts";

export function printFixedPoint(p: ReturnType<typeof fixedPointCheck>) {
  console.log(`Неподвижная точка · среднее r* − 1 при верной оценке: ${decimal(p.bias, 6)} · предел по модулю ${decimal(p.biasLimit, 6)} · информативных осей ${p.informativeAxes}`);
  console.log(`Направление при исходной оценке: ${p.correct} из ${p.directionalAxes} (${decimal(100 * p.direction, 2)}%) · порог 80%`);
  console.log(p.passed ? "Проверка неподвижной точки пройдена." : "Проверка неподвижной точки не пройдена.");
}
