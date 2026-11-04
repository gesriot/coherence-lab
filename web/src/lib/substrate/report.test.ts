import assert from "node:assert/strict";
import test from "node:test";
import { viable } from "./report.ts";
import type { Scores } from "./probes.ts";

test("the B0 gate checks unrounded means, each run and the untrained control independently", () => {
  const score = (changed: number, accuracy = 0.75): Scores =>
    ({ changed, accuracy, all: 0, exit: 2, ghost: 1, hidden: -2, shuffled: -2, curve: Array(9).fill(0) });
  const group = (changed: number) => Array.from({ length: 5 }, () => score(changed));
  assert.ok(viable(group(0.8), group(1)));
  assert.ok(!viable(group(0.80001), group(1)));
  assert.ok(!viable([...group(0.6).slice(1), score(0.85001)], group(1)));
  assert.ok(!viable(group(0.7), [...group(2).slice(1), score(0.99999)]));
  assert.ok(!viable([...group(0.7).slice(1), score(0.7, 0.74999)], group(1)));
});
