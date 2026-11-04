import assert from "node:assert/strict";
import test from "node:test";
import { makeWorld } from "../lab/world.ts";
import { blankState, createElement, forward, stepBuffers } from "./element.ts";
import { frame } from "./input.ts";
import { PROBE_SEEDS, TEST_SEEDS } from "./evaluation.ts";
import { evaluate, ridge } from "./probes.ts";

test("copy scores unity on changing cells and emergence, and zero after observed disappearance", () => {
  const score = evaluate(null, [4111, 4112], [3111, 3112]);
  assert.deepEqual([score.all, score.changed, score.exit, score.ghost], [1, 1, 1, 0]);
  assert.deepEqual(score.curve.slice(0, 7), [0, 0, 0, 0, 0, 0, 0]);
  assert.ok(Math.abs(score.curve[7]! - 0.7) < 1e-12);
});

test("probe fitting uses disjoint seeds and leaves the frozen state trajectory unchanged", () => {
  assert.deepEqual(PROBE_SEEDS, Array.from({ length: 30 }, (_, i) => 4001 + i));
  assert.deepEqual(TEST_SEEDS, Array.from({ length: 30 }, (_, i) => 3001 + i));
  const m = createElement(17), before = m.weights.slice();
  const x = frame(makeWorld("continuity", 3111));
  const first = stepBuffers(m), next = stepBuffers(m);
  forward(m, blankState(m), x, first);
  forward(m, first.state, x, next);
  const saved = first.state.slice(), expected = next.state.slice();
  assert.throws(() => evaluate(m, [3111], [3111]), /пересекаются/);
  const score = evaluate(m, [4111, 4112], [3111, 3112]);
  assert.ok([score.accuracy, score.hidden, score.shuffled].every(Number.isFinite));
  assert.deepEqual(m.weights, before);
  assert.deepEqual(first.state, saved);
  forward(m, first.state, x, next);
  assert.deepEqual(next.state, expected);
});

test("primal and dual ridge give the analytic regularized slope and unpenalized intercept", () => {
  for (const d of [1, 5]) {
    const encode = (x: number) => Float64Array.from({ length: d }, (_, i) => i ? 0 : x);
    const predict = ridge([-1, 0, 1].map(encode))([5, 7, 9]);
    assert.ok(Math.abs(predict(encode(4)) - (7 + 4 * 4 / 2.01)) < 1e-12);
  }
});
