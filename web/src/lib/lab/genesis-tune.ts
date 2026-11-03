import { experimentSeeds, genesisTask } from "./genesis-task.ts";
import { decimal } from "./bench.ts";
import { area, runTrial, tuningAreaUnits } from "./genesis.ts";

const { runs, firstRun, firstEvalSeed } = experimentSeeds.tuning;
let best = -1;
let chosen = "";
console.log(`Подбор на первом порядке · прогоны ${firstRun}–${firstRun + runs - 1} · оценочные зёрна ${firstEvalSeed}–${firstEvalSeed + experimentSeeds.evaluationCount - 1}`);
for (const motion of [0.03, 0.1, 0.3, 1]) {
  for (const existence of genesisTask.existenceRates) {
    let successes = 0;
    for (let seed = firstRun; seed < firstRun + runs; seed++) {
      const curve = runTrial("first-order", seed, firstEvalSeed, { motion, existence });
      // Count successes across both tasks at every post-training checkpoint for an exact tie break.
      successes += Math.round(area(curve) * tuningAreaUnits);
    }
    const pair = `η движения ${decimal(motion)}, η существования ${decimal(existence)}`;
    console.log(`${pair} · площадь ${decimal(successes / (runs * tuningAreaUnits), 6)}`);
    if (successes > best) { best = successes; chosen = pair; }
  }
}
console.log("Выбрано:", chosen);
