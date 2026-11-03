import assert from "node:assert/strict";
import test from "node:test";
import { createEpisode, defaultConfig, snapshot, stepEpisode } from "./agent.ts";
import { classify, extractBlobs } from "./observe.ts";
import { hueDelta } from "./math.ts";
import { advanceWorld, hiddenByOccluder, makeWorld } from "./world.ts";
import { createAgent, stepAgent } from "./perception.ts";
import { assess, type BenchRow } from "./bench.ts";
import type { Ablation, Observation } from "./types.ts";

test("the episode entry point advances one world step", () => {
  const ep = createEpisode("continuity", 1);
  stepEpisode(ep, defaultConfig());
  assert.equal(snapshot(ep, "full").step, 1);
});

test("connected reds across zero have a red circular mean", () => {
  const blobs = extractBlobs({
    grid: 2, aperture: 1, gazeX: 0, gazeY: 0,
    cells: [355, 5, 355, 5].map((hue) => ({ hue, saturation: 70, lightness: 50 })),
  });
  assert.equal(blobs.length, 1);
  assert.ok(hueDelta(blobs[0]!.hue, 0) < 1e-9);
});

test("revisiting a frame does not add evidence", () => {
  const run = (budget: number) => {
    const ep = createEpisode("continuity", 4);
    ep.agent.iterationBudget = budget;
    stepEpisode(ep, defaultConfig());
    return ep.agent.tracks[0]!;
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

test("a replay frame contains only raw pixels and is graded against the previous command", () => {
  const ep = createEpisode("continuity", 3);
  for (let i = 0; i < 8; i++) stepEpisode(ep, defaultConfig());
  const guesses = [...ep.agent.guesses];
  const frame = ep.recording[0]!;
  assert.deepEqual(Object.keys(frame[0]!).sort(), ["hue", "lightness", "saturation"]);
  const expected = frame.filter((cell, i) => classify(cell) !== guesses[i]).length / frame.length;
  stepEpisode(ep, { ...defaultConfig(), replay: true });
  assert.deepEqual(ep.agent.cells, frame);
  assert.equal(guesses.length, frame.length);
  assert.equal(ep.agent.energy.action, expected);
});

test("the next retinal forecast projects an observed wall into the commanded window", () => {
  const wall = { hue: 0, saturation: 0, lightness: 40 };
  const dark = { hue: 0, saturation: 0, lightness: 6 };
  const obs = { gazeX: 50, gazeY: 50, aperture: 4, grid: 2, cells: [dark, wall, dark, wall] };
  const agent = createAgent(obs);
  stepAgent(agent, obs, { ...defaultConfig(), pinGaze: true, pinX: 54, pinY: 50 });
  assert.deepEqual(agent.guesses, ["occluder", "empty", "occluder", "empty"]);
  assert.equal(agent.energy.action, 0);
});

function patches(columns: number[], gazeX = 50): Observation {
  const grid = 8;
  return {
    gazeX, gazeY: 50, grid, aperture: 24,
    cells: Array.from({ length: grid ** 2 }, (_, i) => {
      const surface = Math.floor(i / grid) === 4 && columns.includes(i % grid);
      return { hue: 20, saturation: surface ? 70 : 0, lightness: surface ? 50 : 6 };
    }),
  };
}

test("first order keeps its focus and commands the motion forecast at fixed aperture", () => {
  const agent = createAgent(patches([1, 6]));
  agent.attention.focusId = agent.tracks[1]!.id;
  stepAgent(agent, patches([1, 6], 52), { ...defaultConfig(), ablation: "first-order" });
  const focus = agent.tracks[1]!;
  assert.equal(agent.attention.focusId, focus.id);
  assert.equal(agent.gazeX, focus.x + focus.vx);
  assert.equal(agent.aperture, 18);
  assert.equal(agent.attention.active, false);
});

test("only the salience policy with memory returns to a patch after a blank frame", () => {
  const run = (ablation: Ablation) => {
    const agent = createAgent(patches([1]));
    const cfg = { ...defaultConfig(), ablation };
    stepAgent(agent, patches([1]), cfg);
    const lastX = agent.gazeX;
    stepAgent(agent, patches([]), cfg);
    return agent.gazeX === lastX;
  };
  assert.equal(run("dummy-memory"), true);
  assert.equal(run("no-schema"), false);
});

test("record-only grades the attended patch even when the schema intends another focus", () => {
  const agent = createAgent(patches([1, 5, 6]));
  agent.attention.focusId = agent.tracks[0]!.id;
  agent.focusHold = 2;
  stepAgent(agent, patches([1, 5, 6]), { ...defaultConfig(), ablation: "record-only" });
  assert.notEqual(agent.attention.intendedX, agent.attention.executedX);
  assert.equal(agent.focusPrevId, agent.tracks[1]!.id);
  assert.equal(agent.predictedObserved, true);
});

test("workspace competition and injected confidence change the schema's own decision", () => {
  const run = (capacity: number, injectConfidence: boolean) => {
    const ep = createAgent(patches([1, 6]));
    ep.workspaceCapacity = capacity;
    for (let i = 0; i < 3; i++) stepAgent(ep, patches([1, 6]), defaultConfig());
    stepAgent(ep, patches([1]), { ...defaultConfig(), injectConfidence });
    return ep;
  };
  const full = run(3, false);
  const narrow = run(1, false);
  assert.equal(narrow.tracks.filter((t) => t.inWorkspace).length, 1);
  assert.notEqual(full.gazeX, narrow.gazeX);
  const injected = run(3, true);
  assert.ok(injected.tracks.every((t) =>
    t.uncertainty < 0.05 && t.confidence > 0.95 && !t.conflict && t.reason === "none"));
  assert.equal(injected.attention.influencesAction, true);
  assert.equal(injected.attention.intendedX, injected.attention.executedX);
  assert.notEqual(injected.gazeX, full.gazeX);
});

test("the verdict requires an advantage over the main controls and independent replay detection", () => {
  const rows: BenchRow[] = ["full", "first-order", "record", "dummy", "replay"].map((id) => ({
    id, label: id, detail: "", retention: 0, pair: 0, selfError: null, actionError: 0.05,
  }));
  rows[0]!.retention = 1;
  rows[1]!.retention = 1;
  assert.equal(assess(rows, 20).causal, "unsupported");
  rows[0]!.pair = 1;
  assert.equal(assess(rows, 20).causal, "mixed");
  rows[4]!.actionError = 0.14;
  assert.equal(assess(rows, 20).causal, "supported");
  rows[3]!.retention = rows[3]!.pair = 1;
  assert.equal(assess(rows, 20).causal, "mixed");
});

test("binding and motion weights resolve the same ambiguous crossing differently", () => {
  const frame = (swapped: boolean) => {
    const obs = patches([3, 4]);
    obs.cells[35]!.hue = swapped ? 200 : 20;
    obs.cells[36]!.hue = swapped ? 20 : 200;
    return obs;
  };
  const assignment = (binding: number, temporal: number) => {
    const agent = createAgent(frame(false));
    const ids = agent.tracks.map((t) => t.id);
    agent.weights = { ...agent.weights, binding, temporal };
    stepAgent(agent, frame(true), defaultConfig());
    return ids.map((id) => agent.tracks.find((t) => t.id === id)?.boundBlob);
  };
  assert.deepEqual(assignment(0.1, 2), [0, 1]);
  assert.deepEqual(assignment(2, 0.1), [1, 0]);
});

test("accepted energies descend to the displayed total and weak hypotheses can expire", () => {
  const agent = createAgent(patches([1, 6]));
  assert.equal(agent.tracks.length, 2);
  assert.ok(agent.energyTrace[0]! > agent.energy.total);
  for (let i = 0; i < 20; i++) {
    stepAgent(agent, patches([]), defaultConfig());
    const trace = agent.energyTrace;
    assert.equal(trace.length, agent.iterationsUsed + 1);
    assert.ok(agent.iterationsUsed <= agent.iterationBudget);
    assert.ok(trace.every((e, j) => j === 0 || e <= trace[j - 1]!));
    assert.equal(trace.at(-1), agent.energy.total);
  }
  assert.equal(agent.tracks.length, 0);
});

test("changing confidence alters later samples and bindings in a live episode", () => {
  const control = createEpisode("binding", 1);
  const injected = createEpisode("binding", 1);
  let samplesChanged = false;
  let bindingsChanged = false;
  for (let i = 0; i < 16; i++) {
    stepEpisode(control, defaultConfig());
    stepEpisode(injected, { ...defaultConfig(), injectConfidence: true });
    const a = control.agent;
    const b = injected.agent;
    if (i === 0) assert.deepEqual(a.cells, b.cells);
    if (a.cells.some((cell, j) => classify(cell) !== classify(b.cells[j]!))) samplesChanged = true;
    if (samplesChanged && a.tracks.some((t) =>
      !b.tracks.some((other) => other.id === t.id && other.boundBlob === t.boundBlob))) {
      bindingsChanged = true;
    }
  }
  assert.ok(samplesChanged && bindingsChanged);
});
