import { decimal } from "./bench.ts";
import { sensorimotorTask } from "./sensorimotor-task.ts";
import { gainRuns } from "./sensorimotor.ts";

const { tuning, rates, evaluationCount, checkpoints } = sensorimotorTask;
const { firstRun, runs, firstEvalSeed } = tuning;
console.log(`Подбор на первом порядке · прогоны ${firstRun}–${firstRun + runs - 1} · оценочные зёрна ${firstEvalSeed}–${firstEvalSeed + evaluationCount - 1}`);
const units = 2 * evaluationCount;
let best = -1, chosen: number = rates[0];
for (const rate of rates) {
  let successes = 0;
  for (const curve of gainRuns("first-order", rate, tuning)) {
    for (const point of curve.slice(1)) successes += Math.round(point.accuracy * units);
  }
  const score = successes / (units * runs * (checkpoints.length - 1));
  console.log(`η усиления ${decimal(rate)} · площадь ${decimal(score, 6)}`);
  if (successes > best) { best = successes; chosen = rate; }
}
console.log(`Выбран шаг усиления ${decimal(chosen)}. Команда сравнения: npm run sensorimotor -- ${chosen}`);
