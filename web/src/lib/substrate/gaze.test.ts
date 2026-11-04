import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { hashUnit } from "../lab/math.ts";
import { sample } from "../lab/sensor.ts";
import { blankState, createElement, forward, stepBuffers } from "./element.ts";
import { gazePath, WINDOW_CELL } from "./gaze.ts";
import { frame, frames, GRID, movingFrames, substrateWorld } from "./input.ts";

test("the original element reproduces the reference feature bytes", () => {
  const m = createElement(29, true, GRID, 4, "memory");
  const digest = createHash("sha256");
  let state = blankState(m);
  for (const x of frames("continuity", 31, 5)) {
    const out = stepBuffers(m);
    forward(m, state, x, out);
    digest.update(out.features);
    state = out.state;
  }
  assert.equal(digest.digest("hex"), "5dc38b15278b17301acbc858bd0b04f948a3cb867429473a6b7de464bbded3f1");
});

test("executed commands reflect at both bounds and move the gaze by the episode gain", () => {
  let reflections = 0;
  for (const mode of ["fixed", "episodic"] as const) {
    for (let seed = 1; seed <= 20; seed++) {
      const path = gazePath(seed, mode), repeat = gazePath(seed, mode);
      const u = (k: number) => hashUnit(104729 * seed + 7 * k + 3);
      const gain = mode === "fixed" ? 1.2 : 0.7 + 0.7 * u(0);
      let x = 35 + 30 * u(1), y = 35 + 30 * u(2);
      for (let t = 0; t < 140; t++) {
        const p = path();
        assert.deepEqual(p, repeat());
        assert.deepEqual(p.gaze, { x, y });
        assert.ok(x >= 25 && x <= 75 && y >= 25 && y <= 75);
        for (const axis of [0, 1] as const) {
          const raw = Math.floor(3 * u(11 + 3 * t + axis)) - 1;
          const next = (axis === 0 ? x : y) + raw * (gain * 0.7 * WINDOW_CELL);
          const reflected = next < 25 || next > 75;
          if (reflected) reflections++;
          assert.equal(p.command[axis], reflected ? -raw : raw);
        }
        x += p.command[0] * (gain * 0.7 * WINDOW_CELL);
        y += p.command[1] * (gain * 0.7 * WINDOW_CELL);
      }
    }
  }
  assert.ok(reflections > 0);
});

test("the moving window samples cell centers and translates stationary texture by a whole cell", () => {
  const w = substrateWorld("continuity", 31);
  w.objects = [];
  w.bodyGain = [1, 1];
  const gaze = { x: 44, y: 41 };
  const x = frame(w, "per-cell", gaze), y = frame(w, "per-cell", { x: gaze.x + WINDOW_CELL, y: gaze.y });
  for (let i = 0; i < GRID * GRID; i++) {
    const c = i % GRID, r = Math.floor(i / GRID);
    const cell = sample(w, 44 + (c + 0.5 - 8) * 3.125, 41 + (r + 0.5 - 8) * 3.125, i, "per-cell");
    assert.equal(x[i * 3], cell.lightness / 100);
    if (c < GRID - 1) assert.deepEqual(y.slice(3 * i, 3 * i + 3), x.slice(3 * i + 3, 3 * i + 6));
  }
});

test("all copy conditions share the sensory movie and supply only the frame and outgoing command copy", () => {
  const seed = 23, steps = 80;
  const settings = { gain: "episodic", copy: "executed" } as const;
  const actual = [...movingFrames("continuity", seed, steps, settings)];
  const otherSeed = [...movingFrames("continuity", seed + 7777, steps, settings)];
  const zero = [...movingFrames("continuity", seed, steps, { ...settings, copy: "zero" })];
  const shuffled = [...movingFrames("continuity", seed, steps, { ...settings, copy: "shuffled" })];
  const path = gazePath(seed, settings.gain);
  for (let t = 0; t <= steps; t++) {
    assert.deepEqual(Object.keys(actual[t]!).sort(), ["copy", "frame"]);
    assert.deepEqual(actual[t]!.frame, zero[t]!.frame);
    assert.deepEqual(actual[t]!.frame, shuffled[t]!.frame);
    assert.deepEqual(actual[t]!.copy, path().command);
    assert.deepEqual(zero[t]!.copy, [0, 0]);
    assert.deepEqual(shuffled[t]!.copy, otherSeed[t]!.copy);
  }
});
