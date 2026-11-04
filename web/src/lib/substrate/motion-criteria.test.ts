import assert from "node:assert/strict";
import test from "node:test";
import { representationCriterion } from "./motion-report.ts";

test("world displacement must beat the strongest baseline of each paired run", () => {
  const passing = Array.from({ length: 5 }, () => ({ learned: 0.1, initial: 0, raw: -0.2, context: -0.1 }));
  assert.deepEqual(representationCriterion(passing), { average: 0.1, wins: 5, passes: true });
  assert.ok(!representationCriterion(passing.slice(1)).passes);
  for (const base of ["initial", "raw", "context"] as const) {
    const tied = passing.map((s) => ({ ...s, learned: 0.5 }));
    tied[2]![base] = 0.5;
    const c = representationCriterion(tied);
    assert.ok(c.average > 0.10);
    assert.equal(c.wins, 4);
    assert.equal(c.passes, false);
  }
});

test("the mean gain uses each run's own strongest baseline before averaging", () => {
  const rows = Array.from({ length: 5 }, (_, i) => ({
    learned: 0.5, initial: i === 0 ? 0.45 : 0, raw: i % 2 === 1 ? 0.45 : 0, context: i % 2 === 0 ? 0.45 : 0,
  }));
  const c = representationCriterion(rows);
  assert.ok(Math.abs(c.average - 0.05) < 1e-12);
  assert.equal(c.wins, 5);
  assert.equal(c.passes, false);
});
