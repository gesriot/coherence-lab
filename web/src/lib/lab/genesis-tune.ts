import { decimal } from "./bench.ts";
import { area, runTrial } from "./genesis.ts";

let best = -1;
let chosen = "";
console.log("Подбор на первом порядке · прогоны 501–510 · оценочные зёрна 2001–2020");
for (const motion of [0.03, 0.1, 0.3, 1]) {
  for (const permanence of [0.003, 0.01, 0.03, 0.1]) {
    let successes = 0;
    for (let seed = 501; seed <= 510; seed++) {
      const curve = runTrial("first-order", seed, 2001, { motion, permanence });
      // Four checkpoints, two tasks, twenty seeds. Use counts for an exact tie break.
      successes += Math.round(area(curve) * 160);
    }
    const pair = `η движения ${decimal(motion)}, η постоянства ${decimal(permanence)}`;
    console.log(`${pair} · площадь ${decimal(successes / 1600, 6)}`);
    if (successes > best) { best = successes; chosen = pair; }
  }
}
console.log("Выбрано:", chosen);
