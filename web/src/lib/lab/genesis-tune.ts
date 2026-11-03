import { genesisTask } from "./genesis-task.ts";
import { decimal } from "./bench.ts";
import { area, runTrial, tuningAreaUnits } from "./genesis.ts";

const runs = 10;
const firstRun = 511;
const evalSeed = 2021;
let best = -1;
let chosen = "";
console.log("Подбор на первом порядке · прогоны 511–520 · оценочные зёрна 2021–2040");
for (const motion of [0.03, 0.1, 0.3, 1]) {
  for (const existence of genesisTask.existenceRates) {
    let successes = 0;
    for (let seed = firstRun; seed < firstRun + runs; seed++) {
      const curve = runTrial("first-order", seed, evalSeed, { motion, existence });
      // Count successes across both tasks at every post-training checkpoint for an exact tie break.
      successes += Math.round(area(curve) * tuningAreaUnits);
    }
    const pair = `η движения ${decimal(motion)}, η существования ${decimal(existence)}`;
    console.log(`${pair} · площадь ${decimal(successes / (runs * tuningAreaUnits), 6)}`);
    if (successes > best) { best = successes; chosen = pair; }
  }
}
console.log("Выбрано:", chosen);
