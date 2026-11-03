import assert from "node:assert/strict";
import test from "node:test";
import { createEpisode, defaultConfig, snapshot, stepEpisode } from "./agent.ts";

test("the episode entry point advances one world step", () => {
  const ep = createEpisode("continuity", 1);
  stepEpisode(ep, defaultConfig());
  assert.equal(snapshot(ep, "full").step, 1);
});

import { extractBlobs } from "./observe.ts";
import { hueDelta } from "./math.ts";
import { advanceWorld, hiddenByOccluder, makeWorld } from "./world.ts";

test("connected reds across zero have a red circular mean", () => {
  const blobs = extractBlobs({
    grid: 2, aperture: 1, gazeX: 0, gazeY: 0, capturedGazeX: 0, capturedGazeY: 0,
    cells: [355, 5, 355, 5].map((hue) => ({ hue, kind: "surface" })),
  });
  assert.equal(blobs.length, 1);
  assert.ok(hueDelta(blobs[0]!.hue, 0) < 1e-9);
});

test("revisiting a frame does not add evidence", () => {
  const run = (budget: number) => {
    const ep = createEpisode("continuity", 4);
    ep.iterationBudget = budget;
    stepEpisode(ep, defaultConfig());
    return ep.tracks[0]!;
  };
  const once = run(1);
  const repeated = run(4);
  assert.equal(once.boundBlob, 0);
  for (const key of ["evidence", "prediction", "uncertainty", "fill", "hue"] as const) {
    assert.equal(once[key], repeated[key]);
  }
});

test("seeds vary geometry and preserve the events of each scenario", () => {
  assert.notDeepEqual(makeWorld("binding", 1).objects, makeWorld("binding", 2).objects);
  for (let seed = 1; seed <= 20; seed++) {
    const world = makeWorld("continuity", seed);
    let hidden = false;
    let emerged = false;
    for (let i = 0; i < 60; i++) {
      advanceWorld(world);
      if (hiddenByOccluder(world.objects[0]!, world.occluder)) hidden = true;
      else if (hidden) emerged = true;
    }
    assert.ok(emerged);
    const binding = makeWorld("binding", seed);
    const original = binding.objects[0]!.hue;
    for (let i = 1; i <= 16; i++) {
      advanceWorld(binding);
      assert.equal(binding.objects[0]!.hue === original, i < 16);
    }
    const [a, b] = makeWorld("conflict", seed).objects;
    assert.ok(Math.hypot(a!.x - b!.x, a!.y - b!.y) < 15);
  }
});

test("equal seeds reproduce the entire episode", () => {
  const run = () => {
    const ep = createEpisode("continuity", 9);
    for (let i = 0; i < 40; i++) stepEpisode(ep, defaultConfig());
    return ep;
  };
  assert.deepEqual(run(), run());
});
