import assert from "node:assert/strict";
import test from "node:test";
import { makeWorld } from "../lab/world.ts";
import { blankState, createElement, forward, stepBuffers } from "./element.ts";
import { CELL_SIZE, frame, GRID } from "./input.ts";
import { evaluatePersistence } from "./persistence-probes.ts";
import { visibility } from "./visibility.ts";

test("the exit is the first partially visible frame and only its exposed cells are scored", () => {
  const w = makeWorld("continuity", 17), a = w.objects[0]!;
  a.r = 6;
  a.y = 8.5 * CELL_SIZE;
  let age = -1;
  const exits: [number, number[]][] = [];
  for (const [t, x] of [33, 44, 47, 50, 53, 56, 60, 65].entries()) {
    a.x = x;
    const mask = visibility(w, age);
    age = mask.age;
    assert.equal(age, t >= 1 && t <= 5 ? t - 1 : -1);
    if (mask.exit) exits.push([t, Array.from(mask.visible.keys()).filter((i) => mask.visible[i])]);
  }
  assert.deepEqual(exits, [[6, [8 * GRID + 10]]]);
  a.vanishAt = w.t;
  assert.equal(visibility(w, 5).exit, false);
});

test("a gap between grid centers counts under the cell definition, without a wall intersection", () => {
  const w = makeWorld("continuity", 19), a = w.objects[0]!;
  a.x = a.y = 2 * CELL_SIZE; a.r = 4;
  const gap = visibility(w, -1);
  assert.equal(gap.age, 0);
  a.x += CELL_SIZE / 2;
  assert.equal(visibility(w, gap.age).exit, true);
});

test("both hidden windows and the shuffled probes use disjoint seeds without changing frozen dynamics", () => {
  const m = createElement(4, true, GRID, 3, "hinted"), before = m.weights.slice();
  const x = frame(makeWorld("continuity", 8101), "per-cell");
  const trajectory = () => {
    const a = stepBuffers(m), b = stepBuffers(m);
    forward(m, blankState(m), x, a);
    forward(m, a.state, x, b);
    return b;
  };
  const expected = trajectory();
  assert.throws(() => evaluatePersistence(m, [8101], [8101]), /пересекаются/);
  const s = evaluatePersistence(m, [8201, 8202], [8101, 8102]);
  for (const p of [s.velocity, s.hidden, s.late]) assert.ok([p.value, p.shuffled].every(Number.isFinite));
  assert.ok(s.hint && Number.isFinite(s.hint.occupied) && Number.isFinite(s.hint.empty));
  assert.deepEqual(m.weights, before);
  assert.deepEqual(trajectory(), expected);
});
