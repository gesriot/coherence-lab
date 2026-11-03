import assert from "node:assert/strict";
import test from "node:test";
import { createEpisode, defaultConfig, snapshot, stepEpisode } from "./agent.ts";
import { classify, extractBlobs } from "./observe.ts";
import { hueDelta } from "./math.ts";
import { advanceWorld, hiddenByOccluder, makeWorld } from "./world.ts";
import { createAgent, firstOrderDecision, stepAgent } from "./perception.ts";
import { assess, replayStats, wilson, type BenchRow } from "./bench.ts";
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

test("Wilson intervals match known binomial examples", () => {
  for (const [k, low, high] of [[0, 0, 0.161125], [10, 0.299298, 0.700702], [20, 0.838875, 1]]) {
    const p = wilson(k!, 20);
    assert.ok(Math.abs(p.low - low!) < 1e-6 && Math.abs(p.high - high!) < 1e-6);
  }
});

test("replay AUC compares all pairs, splits ties and includes the threshold", () => {
  const mixed = replayStats("", [0, 1], [1, 2]);
  assert.equal(mixed.auc, 0.875);
  assert.equal(mixed.detected, false);
  assert.equal(mixed.medianDifference, 1);
  assert.equal(mixed.higher.k, 2);
  assert.equal(replayStats("", [1], [1]).auc, 0.5);
  assert.equal(replayStats("", [2], [1]).auc, 0);
  const boundary = replayStats("", [0, 1, 2, 3, 4], [4, 4, 4, 4, 4]);
  assert.equal(boundary.auc, 0.9);
  assert.equal(boundary.detected, true);
});

test("causal verdict requires a strict task advantage independent of replay", () => {
  const ids: Ablation[] = ["full", "first-order", "record-only", "dummy-memory"];
  const rows: BenchRow[] = ids.map((id) => ({
    id, label: id, detail: "", retention: wilson(0, 200), pair: wilson(0, 200),
    exploreSteps: 0, selfError: null, actionError: 0,
  }));
  rows[0]!.retention = wilson(40, 200);
  assert.equal(assess(rows), "unsupported");
  rows[0]!.retention = wilson(41, 200);
  rows[3]!.retention = wilson(41, 200);
  assert.equal(assess(rows), "mixed");
  rows[0]!.pair = wilson(41, 200);
  assert.equal(assess(rows), "supported");
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

test("exploration uses the history of observed windows and breaks ties in row order", () => {
  const frame = (x: number, y: number, aperture = 12.5) => ({
    ...patches([]), gazeX: x, gazeY: y, aperture,
  });
  const run = (initialX: number) => {
    const agent = createAgent(frame(initialX, 12.5));
    stepAgent(agent, frame(87.5, 87.5), defaultConfig());
    assert.equal(agent.attention.executedReason, "explore");
    return [agent.gazeX, agent.gazeY];
  };
  assert.deepEqual(run(12.5), [37.5, 12.5]);
  assert.deepEqual(run(87.5), [12.5, 12.5]);
  assert.deepEqual(run(12.5), run(12.5));

  const agent = createAgent(frame(87.5, 87.5));
  stepAgent(agent, frame(25, 25), defaultConfig());
  assert.deepEqual(agent.lastViewed.slice(0, 6), [1, 1, -1, -1, 1, 1]);
  assert.deepEqual([agent.gazeX, agent.gazeY], [62.5, 12.5]);
  assert.equal(agent.aperture, 22);

  const seen = createAgent(frame(50, 50, 50));
  stepAgent(seen, frame(25, 25), defaultConfig());
  assert.deepEqual([seen.gazeX, seen.gazeY], [62.5, 12.5]);
});

test("first-order decisions are independent of second-order fields", () => {
  const agent = createAgent(patches([1, 6]));
  for (const cycle of [0, 8, 9]) {
    agent.cycle = cycle;
    const before = firstOrderDecision(agent);
    for (const t of agent.tracks) {
      t.source = t.source === "observed" ? "remembered" : "observed";
      t.reason = t.reason === "none" ? "occlusion" : "none";
      t.uncertainty = 1 - t.uncertainty;
      t.conflict = !t.conflict;
    }
    assert.deepEqual(firstOrderDecision(agent), before);
  }
});

test("scheduled first-order exploration requires the sticky focus to have a current binding", () => {
  const agent = createAgent(patches([1, 6]));
  agent.attention.focusId = agent.tracks[1]!.id;
  for (const cycle of [0, 7, 8, 9, 16]) {
    agent.cycle = cycle;
    const choice = firstOrderDecision(agent);
    const due = cycle === 8 || cycle === 16;
    assert.equal(choice.reason, due ? "explore" : "pursue");
    assert.equal(choice.aperture, due ? 22 : 18);
    assert.equal(choice.focusId, agent.attention.focusId);
  }
  agent.tracks[1]!.boundBlob = null;
  assert.equal(firstOrderDecision(agent).reason, "pursue");
});

test("one quantized centroid jump survives eighteen missed observations and favors rebinding", () => {
  function frame(y: number, hidden = false): Observation {
    const grid = 12;
    return {
      grid, gazeX: 50 - 1.5, gazeY: y - 1.5, aperture: 18,
      cells: Array.from({ length: grid ** 2 }, (_, i) => {
        if (hidden) return { hue: 0, saturation: 0, lightness: 50 };
        const spot = i === 6 * grid + 6;
        return { hue: 20, saturation: spot ? 70 : 0, lightness: spot ? 50 : 6 };
      }),
    };
  }
  const cfg = defaultConfig();
  const ep = createAgent(frame(50));
  for (let i = 0; i < 7; i++) stepAgent(ep, frame(50), cfg);
  stepAgent(ep, frame(53), cfg);
  const id = ep.tracks[0]!.id;
  assert.equal(ep.tracks[0]!.vy, 3 / 8);
  for (let i = 0; i < 18; i++) stepAgent(ep, frame(50, true), cfg);
  const prior = ep.tracks[0]!;
  assert.equal(prior.unseen, 18);
  assert.ok(Math.abs(prior.y - 50) < 16);

  // With a fixed denominator the innovation would cost more than the 0.5 birth term.
  const innovation = Math.abs(prior.y + prior.vy - 50);
  assert.ok(innovation / 16 > 0.5);
  ep.iterationBudget = 1;
  stepAgent(ep, frame(50), cfg);
  assert.equal(ep.tracks.length, 1);
  assert.equal(ep.tracks[0]!.id, id);
  assert.equal(ep.tracks[0]!.boundBlob, 0);
  assert.ok(ep.energy.temporal < 0.5);
});
