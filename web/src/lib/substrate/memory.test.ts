import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { makeWorld } from "../lab/world.ts";
import { blankState, createElement, forward, stepBuffers } from "./element.ts";
import { backward, gradientBuffers } from "./gradient.ts";
import { CELL_SIZE, frame, frames, GRID } from "./input.ts";
import { occupancy, occupancyHook } from "./occupancy.ts";
import { trainer } from "./training.ts";

test("default sensor reproduces the full reference episode byte for byte", () => {
  const hash = createHash("sha256");
  for (const x of frames("continuity", 41, 72)) hash.update(x);
  assert.equal(hash.digest("hex"), "99c5264c857356a642d44d1e89f394e8fb3dcb16ab55a7decee8a85bb49f42e4");
});

test("per-cell occlusion exposes only the part outside the wall; occupancy includes its hidden part", () => {
  const w = makeWorld("continuity", 41), a = w.objects[0]!;
  a.x = 40; a.y = 8.5 * CELL_SIZE; a.r = 6;
  const legacy = frame(w), local = frame(w, "per-cell"), labels = occupancy(w);
  const outside = 8 * GRID + 5, inside = outside + 1;
  assert.equal(legacy[3 * outside], 0.06);
  assert.equal(local[3 * outside], 0.5);
  assert.deepEqual(local.slice(3 * inside, 3 * inside + 3), legacy.slice(3 * inside, 3 * inside + 3));
  assert.deepEqual([labels[outside], labels[inside]], [1, 1]);
  a.vanishAt = 0;
  assert.equal(occupancy(w)[inside], 0);
});

test("gated gradients match central differences for frame loss and joint occupancy loss", () => {
  for (const kind of ["memory", "hinted"] as const) {
    const m = createElement(13, true, 2, 2, kind);
    assert.deepEqual(Array.from(m.gate!.b), [-1, -1]);
    const x = Array.from({ length: 4 }, (_, t) =>
      Float64Array.from({ length: 12 }, (_, i) => 0.3 + Math.sin(0.3 * i + t) / 4));
    const labels = x.slice(1).map((_, t) => Float64Array.from([0, 1, t % 2, 1]));
    const tape = labels.map(() => stepBuffers(m)), initial = blankState(m);
    for (let j = 0; j < 8; j++) initial[j] = 0.2 + Math.sin(j) / 3;
    function loss() {
      let sum = 0;
      for (let t = 0; t < tape.length; t++) {
        const out = tape[t]!;
        forward(m, t ? tape[t - 1]!.state : initial, x[t]!, out);
        for (let j = 0; j < 12; j++) sum += (out.prediction[j]! - x[t + 1]![j]!) ** 2;
        if (m.hint) for (let i = 0; i < 4; i++) sum += 3 * (out.occupancy![i]! - labels[t]![i]!) ** 2;
      }
      return sum / 36;
    }
    const expected = loss(), work = gradientBuffers(m);
    assert.ok(Math.abs(backward(m, tape, x.slice(1), work, m.hint ? labels : undefined) - expected) < 1e-14);
    for (let i = 0; i < m.weights.length; i++) {
      const saved = m.weights[i]!;
      m.weights[i] = saved + 1e-5;
      const plus = loss();
      m.weights[i] = saved - 1e-5;
      const numeric = (plus - loss()) / 2e-5;
      m.weights[i] = saved;
      const analytic = work.grad[i]!;
      const rel = Math.abs(analytic - numeric) / Math.max(Math.abs(analytic), Math.abs(numeric));
      assert.ok(rel <= 1e-6, `${kind} parameter ${i}: ${rel}`);
    }
  }
});

test("only the hinted model reads current-frame labels, including across a segment boundary", () => {
  for (const kind of ["plain", "memory", "hinted"] as const) {
    function trial() {
      const m = createElement(3, true, GRID, 2, kind), seen: number[] = [];
      const labels = occupancyHook("continuity", 15);
      trainer(m)(frames("continuity", 15, 32, "per-cell"), (t) => {
        seen.push(t);
        return labels(t);
      });
      assert.deepEqual(seen, kind === "hinted" ? Array.from({ length: 32 }, (_, i) => i) : []);
      return m.weights;
    }
    assert.deepEqual(trial(), trial());
  }
});
