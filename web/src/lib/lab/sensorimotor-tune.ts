import { decimal } from "./bench.ts";
import { sensorimotorTask } from "./sensorimotor-task.ts";
import { mean } from "./genesis.ts";
import { gainArea, gainRuns } from "./sensorimotor.ts";

const { tuning, rates, evaluationCount } = sensorimotorTask;
const { firstRun, runs, firstEvalSeed } = tuning;
console.log(`Подбор на первом порядке · прогоны ${firstRun}–${firstRun + runs - 1} · оценочные зёрна ${firstEvalSeed}–${firstEvalSeed + evaluationCount - 1}`);
let best = -Infinity, chosen: number = rates[0];
for (const rate of rates) {
  const score = mean(gainRuns("first-order", rate, tuning).map((run) => gainArea(run.curve)));
  console.log(`η усиления ${decimal(rate)} · площадь q ${decimal(score, 6)}`);
  if (score > best) { best = score; chosen = rate; }
}
console.log(`Выбран шаг усиления ${decimal(chosen)}. Команда сравнения: npm run sensorimotor -- ${chosen}`);
