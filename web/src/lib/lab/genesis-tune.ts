import { decimal } from "./bench.ts";
import { area, runTrial, tuningAreaUnits } from "./genesis.ts";

const runs = 10;
let best = -1;
let chosen = "";
console.log("Подбор на первом порядке · прогоны 501–510 · оценочные зёрна 2001–2020");
for (const motion of [0.03, 0.1, 0.3, 1]) {
  for (const existence of [0.003, 0.01, 0.03, 0.1]) {
    let successes = 0;
    for (let seed = 501; seed < 501 + runs; seed++) {
      const curve = runTrial("first-order", seed, 2001, { motion, existence });
      // Count successes across both tasks at every post-training checkpoint for an exact tie break.
      successes += Math.round(area(curve) * tuningAreaUnits);
    }
    const pair = `η движения ${decimal(motion)}, η существования ${decimal(existence)}`;
    console.log(`${pair} · площадь ${decimal(successes / (runs * tuningAreaUnits), 6)}`);
    if (successes > best) { best = successes; chosen = pair; }
  }
}
console.log("Выбрано:", chosen);
