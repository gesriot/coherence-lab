import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";
import { makeWorld } from "../lab/world.ts";
import { blankState, createElement, features, forward, stepBuffers } from "./element.ts";
import { backward, gradientBuffers } from "./gradient.ts";
import { CELL_SIZE, frame, frames } from "./input.ts";
import { trainer } from "./training.ts";

test("the fixed sensor encodes background, patterned wall and noisy object in circular channels", () => {
  const world = makeWorld("continuity", 23);
  const object = world.objects[0]!;
  object.x = 2.5 * CELL_SIZE;
  object.y = 8.5 * CELL_SIZE;
  const x = frame(world);
  const cell = (i: number) => Array.from(x.subarray(3 * i, 3 * i + 3));
  assert.deepEqual(cell(0), [0.06, 0, 0]);
  assert.deepEqual(cell(7), [0.54, 0, 0]);
  const [l, a, b] = cell(8 * 16 + 2);
  assert.equal(l, 0.5);
  assert.ok(Math.abs(Math.hypot(a, b) - 0.7) < 1e-14);
  assert.ok(Math.abs(Math.atan2(b, a) * 180 / Math.PI - object.hue) <= world.noise);
  world.objects[0]!.hue += 90;
  const y = frame(world), j = 3 * (8 * 16 + 2);
  assert.ok(Math.abs(y[j + 1]! + b) < 1e-14 && Math.abs(y[j + 2]! - a) < 1e-14);
});

test("Sobel impulse responses are antisymmetric and the edge mean excludes padding and self", () => {
  const m = createElement(1, true, 3, 1), prev = blankState(m);
  const out = stepBuffers(m).features, x = new Float64Array(27);
  const sx = [], sy = [];
  for (let i = 0; i < 9; i++) {
    prev.fill(0);
    prev[i] = 1;
    features(m, prev, x, out);
    sx.push(out[4 * m.inputs + 2]! * 8);
    sy.push(out[4 * m.inputs + 3]! * 8);
    assert.equal(out[1], [1, 3, 4].includes(i) ? 1 / 3 : 0);
  }
  assert.deepEqual(sx, [-1, 0, 1, -2, 0, 2, -1, 0, 1]);
  assert.deepEqual(sy, [-1, -2, -1, 0, 0, 0, 1, 2, 1]);
});

test("every shared parameter agrees with a central difference through three recurrent steps", () => {
  for (const directed of [false, true]) {
    const m = createElement(7, directed, 3, 3);
    const sequence = Array.from({ length: 4 }, (_, t) =>
      Float64Array.from({ length: 27 }, (_, i) => Math.sin(0.71 * i + t) / 2));
    const tape = sequence.slice(1).map(() => stepBuffers(m)), initial = blankState(m);
    function loss() {
      let total = 0;
      for (let t = 0; t < tape.length; t++) {
        forward(m, t ? tape[t - 1]!.state : initial, sequence[t]!, tape[t]!);
        for (let j = 0; j < 27; j++) total += (tape[t]!.prediction[j]! - sequence[t + 1]![j]!) ** 2;
      }
      return total / 81;
    }
    const expected = loss(), work = gradientBuffers(m);
    assert.ok(Math.abs(backward(m, tape, sequence.slice(1), work) - expected) < 1e-14);
    for (let i = 0; i < m.weights.length; i++) {
      const saved = m.weights[i]!;
      m.weights[i] = saved + 1e-5;
      const above = loss();
      m.weights[i] = saved - 1e-5;
      const numeric = (above - loss()) / 2e-5;
      m.weights[i] = saved;
      const analytic = work.grad[i]!;
      const relative = Math.abs(analytic - numeric) / Math.max(Math.abs(analytic), Math.abs(numeric));
      assert.ok(relative <= 1e-6, `parameter ${i}: ${relative}`);
    }
  }
});

test("identical seeds reproduce parameters across episodes and a partial final segment", () => {
  function run(seed: number) {
    const m = createElement(seed, true, 16, 2);
    const episode = trainer(m);
    for (let i = 0; i < 3; i++) episode(frames("continuity", 41 + i, 32));
    return m.weights;
  }
  const result = run(1);
  assert.deepEqual(result, run(1));
  assert.notDeepEqual(result, createElement(1, true, 16, 2).weights);
  assert.notDeepEqual(result, run(2));
});

test("only the sensory boundary can reach the world from the training dependency graph", () => {
  const seen = new Set<string>();
  function visit(file: string) {
    if (seen.has(file)) return;
    seen.add(file);
    const src = ts.createSourceFile(file, readFileSync(new URL(file, import.meta.url), "utf8"),
      ts.ScriptTarget.Latest, true);
    for (const dep of ts.preProcessFile(src.text).importedFiles) {
      const name = dep.fileName;
      if (name === "../lab/math.ts") continue;
      if (file === "./input.ts" && ["../lab/world.ts", "../lab/types.ts", "../lab/sensor.ts"].includes(name)) continue;
      assert.match(name, /^\.\/[a-z-]+\.ts$/);
      assert.ok(!name.includes("eval") && !name.includes("probe"));
      visit(name);
    }
  }
  visit("./training.ts");
  assert.ok(seen.has("./input.ts") && seen.has("./gradient.ts") && seen.has("./element.ts"));
});
