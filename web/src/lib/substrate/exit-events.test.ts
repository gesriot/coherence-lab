import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { hashUnit } from "../lab/math.ts";
import { advanceWorld } from "../lab/world.ts";
import { createElement } from "./element.ts";
import { exitWeightHook, returnDetector } from "./exit-events.ts";
import { CELL_SIZE, frames, GRID, substrateWorld } from "./input.ts";
import { train, trainer } from "./training.ts";
import { inWall } from "./visibility.ts";

test("random paths reproduce the specified draws and wall widths affect only continuity", () => {
  for (const columns of [1, 2, 4] as const) {
    for (let seed = 1; seed <= 40; seed++) {
      const settings = { path: "random", columns } as const;
      const w = substrateWorld("continuity", seed, settings), a = w.objects[0]!;
      assert.deepEqual(w, substrateWorld("continuity", seed, settings));
      const u = (k: number) => hashUnit(7919 * seed + 31 * k + 5);
      const sign = u(1) < 0.5 ? 1 : -1;
      assert.equal(a.vx, sign * (1.2 + 0.9 * u(2)));
      assert.equal(a.x, (sign > 0 ? 8 : 76) + 16 * u(4));
      assert.equal(a.y, 12 + 76 * u(3));
      assert.equal(a.vy, 0.05 * (2 * u(5) - 1));
      const covered = Array.from({ length: GRID }, (_, c) => inWall(w.occluder, (c + 0.5) * CELL_SIZE, 50));
      assert.equal(covered.filter(Boolean).length, columns);
      for (const sc of ["vanish", "nursery", "binding", "conflict"] as const) {
        assert.deepEqual(substrateWorld(sc, seed, settings), substrateWorld(sc, seed));
      }
    }
  }
});

test("a return records contact during hiding, duration and visible cells; disappearance cancels it", () => {
  const w = substrateWorld("continuity", 1, { columns: 1 }), a = w.objects[0]!;
  const observe = returnDetector();
  function at(x: number, y: number) {
    a.x = x; a.y = y;
    return observe(w).event;
  }
  at(20, 50);
  w.t = 1;
  assert.equal(at(25, 50), undefined);
  w.t = 2;
  assert.deepEqual(at(28.125, 50), { time: 2, length: 1, kind: "gap", cells: [116, 132] });
  w.t = 3;
  at(46.875, 46.875);
  w.t = 4;
  at(47.5, 46.875);
  w.t = 5;
  assert.deepEqual(at(53.125, 46.875), { time: 5, length: 2, kind: "wall", cells: [120] });
  w.t = 6;
  at(46.875, 46.875);
  a.vanishAt = 7;
  w.t = 7;
  assert.equal(at(53.125, 46.875), undefined);
  w.t = 8;
  assert.equal(at(53.125, 46.875), undefined);
});

test("target-frame weights highlight only the exposed cells on wall returns", () => {
  const seed = 72, settings = { columns: 2, path: "random" } as const;
  const hook = exitWeightHook("continuity", seed, settings);
  const w = substrateWorld("continuity", seed, settings);
  const observe = returnDetector();
  observe(w);
  let hits = 0;
  for (let t = 1; t <= 72; t++) {
    advanceWorld(w);
    const { event } = observe(w), weights = hook(t);
    for (let i = 0; i < weights.length; i++) {
      assert.equal(weights[i], event?.kind === "wall" && event.cells.includes(i) ? 50 : 1);
      if (weights[i] === 50) hits++;
    }
  }
  assert.ok(hits > 0);
});

test("unweighted training reproduces the previous parameter bytes across all four scenarios", () => {
  const m = createElement(23, true, GRID, 4, "memory"), episode = trainer(m);
  for (const [i, sc] of (["continuity", "vanish", "nursery", "binding"] as const).entries()) {
    episode(frames(sc, 29000 + i, 32, "per-cell"));
  }
  assert.equal(createHash("sha256").update(m.weights).digest("hex"),
    "6921fe85eeb9ffbc79dd208e5c04752a501741f002dfadc5e66e63df69d68bca");
});

test("only exit-weighted training requests maps, once per target frame across segment boundaries", () => {
  for (const weighting of ["uniform", "wall-exits"] as const) {
    const seen: number[] = [];
    let factories = 0;
    train(createElement(1, true, GRID, 2, "memory"), 1, "per-cell", undefined, undefined, {
      weighting, episodes: 1, weightHook: () => {
        factories++;
        return (t) => { seen.push(t); return new Float64Array(GRID * GRID).fill(1); };
      },
    });
    assert.equal(factories, weighting === "uniform" ? 0 : 1);
    assert.deepEqual(seen, weighting === "uniform" ? [] : Array.from({ length: 80 }, (_, i) => i + 1));
  }
});
