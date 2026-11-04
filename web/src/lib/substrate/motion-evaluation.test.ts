import assert from "node:assert/strict";
import test from "node:test";
import { blankState, createElement, forward, stepBuffers } from "./element.ts";
import { backward, gradientBuffers } from "./gradient.ts";
import { adamState, trainer, update } from "./training.ts";
import { movingFrames, substrateWorld } from "./input.ts";
import { evaluateMotion, probeTarget, rawPatch, scoreTransition } from "./motion-evaluation.ts";
import { commandCriterion } from "./motion-report.ts";

test("training pairs the outgoing copy with the current frame in the memory gate and candidate", () => {
  const make = () => createElement(19, true, 2, 2, "memory", true);
  const m = make(), reference = make();
  const first = { frame: Float64Array.from({ length: 12 }, (_, j) => j / 20), copy: [1, -1] as const };
  const next = { frame: first.frame.map((v) => v + 0.1), copy: [-1, 0] as const };
  trainer(m)([first, next]);
  const out = stepBuffers(reference), work = gradientBuffers(reference);
  forward(reference, blankState(reference), first.frame, out, first.copy);
  for (let i = 0; i < reference.cells; i++) {
    const start = (i + 1) * reference.inputs - 5;
    assert.deepEqual(Array.from(out.features.subarray(start, start + 5)),
      [...first.copy, ...first.frame.subarray(3 * i, 3 * i + 3)]);
  }
  backward(reference, [out], [next.frame], work);
  update(reference.weights, work.grad, adamState(reference));
  assert.deepEqual(m.weights, reference.weights);
});

test("prediction errors are pooled by executed transition and both time boundaries", () => {
  const init = () => ({ sum: 0, copy: 0 });
  const totals = { all: init(), moved: init(), still: init(), early: init(), late: init() };
  const before = Float64Array.of(0), next = Float64Array.of(1);
  for (const t of [0, 23, 24, 47, 48, 71]) {
    scoreTransition(totals, t, t === 0 || t === 71 ? [0, 0] : t === 24 ? [0, -1] : [1, 0],
      Float64Array.of(t + 1), before, next);
  }
  assert.deepEqual(totals.early, { sum: 23 ** 2, copy: 1 });
  assert.deepEqual(totals.late, { sum: 48 ** 2, copy: 1 });
  assert.deepEqual(totals.moved, { sum: 23 ** 2 + 24 ** 2 + 47 ** 2 + 48 ** 2, copy: 4 });
  assert.deepEqual(totals.still, { sum: 71 ** 2, copy: 2 });
  assert.deepEqual(totals.all, { sum: totals.moved.sum + 71 ** 2, copy: 6 });
});

test("the probe reads the visible center cell and labels the incoming displacements", () => {
  const w = substrateWorld("continuity", 17), a = w.objects[0]!;
  const gaze = { x: 39, y: 46 };
  Object.assign(a, { x: 34.3125, y: 47.5625, r: 4 });
  const target = () => probeTarget(w, gaze, 36, 36.375);
  assert.equal(target(), undefined);
  w.t = 1;
  assert.deepEqual(target(), { cell: 8 * 16 + 6, own: 2.625, world: -1.6875, apparent: -4.3125 });
  a.r = 0.1; a.y += 1;
  assert.equal(target(), undefined);
  a.r = 4; a.x = 42;
  assert.equal(target(), undefined);
  a.x = 13;
  assert.equal(target(), undefined);
  a.x = 34; a.vanishAt = w.t;
  assert.equal(target(), undefined);
});

test("the raw probe pairs two retinal patches with the previous command and pads missing neighbors", () => {
  const previous = { frame: Float64Array.from({ length: 768 }, (_, i) => i + 1), copy: [-1, 1] as const };
  const current = { frame: previous.frame.map((v) => v + 1000), copy: [1, -1] as const };
  const raw = rawPatch(0, previous, current);
  assert.equal(raw.length, 56);
  assert.deepEqual(raw.slice(0, 12), new Float64Array(12));
  assert.deepEqual(Array.from(raw.slice(12, 18)), [1, 2, 3, 4, 5, 6]);
  assert.deepEqual(Array.from(raw.slice(39, 42)), [1001, 1002, 1003]);
  assert.deepEqual(Array.from(raw.slice(-2)), [-1, 1]);
});

test("motion probes enforce held-out seeds and preserve frozen weights and trajectories", () => {
  const m = createElement(3, true, 16, 3, "memory", true), initial = createElement(3, true, 16, 3, "memory", true);
  m.b[0] += 0.2;
  const before = m.weights.slice(), beforeInitial = initial.weights.slice();
  const settings = { copy: "executed", gain: "fixed" } as const;
  function trajectory() {
    let state = blankState(m);
    for (const { frame, copy } of movingFrames("continuity", 71, 3, settings)) {
      const out = stepBuffers(m);
      forward(m, state, frame, out, copy);
      state = out.state;
    }
    return state;
  }
  const expected = trajectory();
  assert.throws(() => evaluateMotion(m, initial, settings, [71, 72], [72, 73]), /пересекаются/);
  const result = evaluateMotion(m, initial, settings, [8101,8102,8103,8104], [8201,8202,8203,8204]);
  assert.ok(result.trainCount > 0 && result.testCount > 0);
  assert.ok(result.probes.own.raw > 0.999);
  for (const probe of Object.values(result.probes)) assert.ok(Object.values(probe).every(Number.isFinite));
  assert.deepEqual(m.weights, before);
  assert.deepEqual(initial.weights, beforeInitial);
  assert.deepEqual(trajectory(), expected);
});

test("the command criterion requires both mean ratios and strict paired wins in all five runs", () => {
  const group = (n: number) => Array(5).fill(n) as number[];
  assert.ok(commandCriterion(group(0.85), group(1), group(1)).passes);
  assert.ok(!commandCriterion(group(0.85001), group(1), group(2)).passes);
  assert.ok(!commandCriterion(group(0.85001), group(2), group(1)).passes);
  assert.ok(!commandCriterion([0.5,0.5,0.5,0.5,1], group(1), group(2)).passes);
  assert.ok(!commandCriterion([0.5,0.5,0.5,0.5,1], group(2), group(1)).passes);
  assert.ok(!commandCriterion(group(0.5).slice(1), group(1), group(1)).passes);
});
