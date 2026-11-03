import { advanceWorld, hiddenByOccluder, makeWorld, scheduleOutdoorLosses } from "./world.ts";
import { hashUnit } from "./math.ts";
import { genesisTask } from "./genesis-task.ts";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { createEpisode, defaultConfig, snapshot, stepEpisode } from "./agent.ts";

test("version A reproduces the reference gaze, tracks and energy on every frame", () => {
  const reference = [
    ["continuity", 1, "32a12ead2e7b036fbd8cea6b4560640c3b8fe9805ccde0ac4a8f06c8e187be72"],
    ["continuity", 2, "abd24d99e3d86605f33f6ed274125a4daabdfdd7b37e1499de11419c739ac2b2"],
    ["conflict", 1, "829e213dc168d7d96f869a350891ca215cb7e6d6e15012dd6b8d8c65bce85a72"],
    ["conflict", 2, "c3e637300adaacffd0b477183be127fb5e2c2fe60cc3633f309199028f1ee98e"],
  ] as const;
  for (const [scenario, seed, expected] of reference) {
    const ep = createEpisode(scenario, seed);
    const digest = createHash("sha256");
    for (let i = 0; i <= (scenario === "continuity" ? 72 : 200); i++) {
      if (i) stepEpisode(ep, defaultConfig());
      const v = snapshot(ep, "full");
      digest.update(JSON.stringify([v.obsGaze, v.nextGaze, v.tracks.map((t) => [t.id, t.x, t.y]), v.energy.total]));
    }
    assert.equal(digest.digest("hex"), expected);
  }
});

test("the disk disappears from the sensor at the scheduled step and remains in the evaluator", () => {
  for (let seed = 1; seed <= 20; seed++) {
    const ep = createEpisode("vanish", seed);
    const at = ep.world.objects[0]!.vanishAt!;
    assert.ok(at >= 20 && at <= 23);
    for (let i = 1; i <= 40; i++) {
      stepEpisode(ep, { ...defaultConfig(), pinGaze: true, pinX: 26, pinY: 34 });
      const v = snapshot(ep, "full");
      assert.equal(v.evaluator.objects[0]!.vanished, i >= at);
      assert.equal(v.cells.some((c) => c.saturation > 0), i < at);
    }
  }
});

import {
  existenceLogits, features, filterExistence, fitExistence, fitMotion, freshModels, GapBook, gapLogLikelihood,
  motion, probability, type Features, type GapExample, type GapStep,
} from "./learning.ts";
import type { Track } from "./types.ts";
import { matchRadius, matchTrack, predictMiss } from "./perception.ts";

test("normalized motion learning converges on a constant displacement", () => {
  const models = freshModels();
  const input: Features = [1.5, -0.5, 1.5, -0.5, 1];
  assert.deepEqual(motion(models, input), [0, 0]);
  for (let i = 0; i < 500; i++) fitMotion(models, { input, target: [1.5, -0.5] }, 0.1);
  assert.ok(Math.hypot(motion(models, input)[0] - 1.5, motion(models, input)[1] + 0.5) < 1e-8);
});

test("a missing observation propagates survival and an empty view updates its posterior", () => {
  const models = freshModels();
  models.logits = existenceLogits({ hidden: 0.25, open: 0.5, miss: 0.2 });
  assert.equal(filterExistence(models, 0.8, { context: "open", observation: "none" }), 0.4);
  const posterior = filterExistence(models, 0.8, { context: "hidden", observation: "empty" });
  assert.ok(Math.abs(posterior - 0.12 / (0.12 + 0.4)) < 1e-12);
});

test("gap likelihood includes detections and marginalizes the unknown time of loss", () => {
  const models = freshModels();
  models.logits = existenceLogits({ hidden: 0.25, open: 0.5, miss: 0.2 });
  const steps: GapStep[] = [
    { context: "open", observation: "empty" }, { context: "hidden", observation: "none" },
  ];
  const detected = (0.5 * 0.8) ** 2;
  const found = gapLogLikelihood(models, { seen: 2, steps, found: true });
  const lost = gapLogLikelihood(models, { seen: 2, steps, found: false });
  assert.ok(Math.abs(found - Math.log(detected * 0.5 * 0.2 * 0.75)) < 1e-12);
  assert.ok(Math.abs(lost - Math.log(detected * (0.5 + 0.5 * 0.2 * 0.25))) < 1e-12);
});

test("synthetic gaps recover both disappearance rates and the missed detection probability", () => {
  const models = freshModels();
  const records: GapExample[] = [];
  // 25/100 hidden losses, 30/300 open losses (including 200 prior detections),
  // and 50 empty views of survivors alongside the 200 detections.
  for (let i = 0; i < 100; i++) {
    records.push({ seen: 1, found: i % 4 !== 0, steps: [{ context: "hidden", observation: "none" }] });
    records.push({ seen: 1, found: i % 10 >= 3, steps: [
      { context: "open", observation: i % 10 < 8 ? "empty" : "none" },
    ] });
  }
  for (let epoch = 1; epoch <= 150; epoch++) {
    for (const record of records) fitExistence(models, record, 0.04 / Math.sqrt(epoch));
  }
  for (const [key, expected] of [["hidden", 0.25], ["open", 0.1], ["miss", 0.2]] as const) {
    assert.ok(Math.abs(probability(models, key) - expected) < 0.002);
  }
});

test("gap outcomes use live bindings, retrospective births, timeout and episode censoring", () => {
  const track = { ...createEpisode("continuity", 1).agent.tracks[0]!, seenRun: 6 };
  const advance = (t: Track): Track => ({ ...t, boundBlob: null, unseen: t.unseen + 1, reason: "conflict",
    gapStep: { context: "open", observation: "none" },
  });
  const miss = advance(track);
  for (const outcome of ["live", "birth", "timeout", "censored"]) {
    const book = new GapBook();
    assert.deepEqual(book.step(1, [track], [miss], advance, matchRadius), []);
    if (outcome === "live") {
      assert.equal(book.step(2, [miss], [track], advance, matchRadius)[0]!.found, true);
      continue;
    }
    assert.deepEqual(book.step(2, [miss], [], advance, matchRadius), []);
    for (let i = 3; i <= 31; i++) assert.deepEqual(book.step(i, [], [], advance, matchRadius), []);
    if (outcome === "censored") continue;
    const births = outcome === "birth" ? [{ ...track, id: 2, age: 1 }] : [];
    const [record] = book.step(32, [], births, advance, matchRadius);
    assert.equal(record!.found, outcome === "birth");
    assert.equal(record!.seen, 6);
    assert.deepEqual(record!.steps, Array(outcome === "birth" ? 31 : 32).fill(miss.gapStep));
  }
});

test("zero motion and the initial existence model lose the target behind the wall", () => {
  let losses = 0;
  for (let seed = 1; seed <= 8; seed++) {
    const ep = createEpisode("continuity", seed, "first-order", freshModels());
    let hidden = false;
    let exit = 0;
    for (let i = 1; i <= 72; i++) {
      stepEpisode(ep, { ...defaultConfig(), ablation: "first-order" });
      const v = snapshot(ep, "first-order");
      if (v.evaluator.objects[0]!.occluded) hidden = true;
      else if (hidden && !exit) exit = i;
      if (exit && i === exit + 1 && !v.evaluator.retained) losses++;
    }
  }
  assert.ok(losses >= 3);
});

import { generator, Learner, Replay, type Selection } from "./experience.ts";
import { conditions, trainEpisode, type Condition } from "./genesis-training.ts";

test("bounded experience is FIFO and attention admits only executed-focus experience", () => {
  const learner = new Learner({ motion: 0.1, existence: 0.01 }, "attention", 1);
  const example = { input: [1, 0, 1, 0, 1] as Features, target: [1, 0] as [number, number] };
  const gap: GapExample = { found: true, steps: [{ context: "open", observation: "empty" }], seen: 1 };
  learner.rememberMotion(example, null);
  learner.rememberGap(gap, null);
  assert.equal(learner.movements.entries.length + learner.gaps.entries.length, 0);
  for (let i = 0; i < 300; i++) {
    learner.rememberMotion({ ...example, target: [i, 0] }, 0);
    learner.rememberGap({ ...gap, seen: i + 1 }, 0.3);
  }
  assert.equal(learner.movements.entries.length, 256);
  assert.equal(learner.movements.entries[0]!.value.target[0], 44);
  assert.equal(learner.gaps.entries.length, 32);
  assert.equal(learner.gaps.entries[0]!.value.seen, 269);
});

test("replay draws are reproducible and proportional to insertion priorities", () => {
  const replay = new Replay<number>(2);
  replay.add(0, 1);
  replay.add(1, 3);
  const sample = () => {
    const random = generator(3);
    return Array.from({ length: 10000 }, () => replay.sample(random, false));
  };
  const draws = sample();
  assert.deepEqual(draws, sample());
  assert.ok(Math.abs(draws.filter((x) => x === 1).length / draws.length - 0.75) < 0.02);
});

test("every nonempty memory receives exactly one update per training step", () => {
  for (const selection of ["uniform", "attention", "error"] as Selection[]) {
    const learner = new Learner({ motion: 0.1, existence: 0.01 }, selection, 3);
    learner.update();
    assert.deepEqual(learner.updates, { motion: 0, existence: 0 });
    learner.rememberMotion({ input: [0, 0, 0, 0, 1], target: [0, 1] }, 0);
    learner.update();
    learner.rememberGap({ seen: 1, steps: [{ context: "open", observation: "empty" }], found: false }, 0);
    learner.update();
    assert.deepEqual(learner.updates, { motion: 2, existence: 1 });
  }
});

test("only the oracle training condition requests privileged labels", () => {
  for (const condition of Object.keys(conditions) as Condition[]) {
    const learner = new Learner({ motion: 0.1, existence: 0.01 }, conditions[condition].selection, 1);
    const run = () => trainEpisode(learner, condition, "vanish", 1, () => { throw Error("label hook"); });
    if (condition === "oracle") assert.throws(run, /label hook/);
    else run();
  }
});

import { area, bootstrap, checkpoints, effect, evaluate, trainingCurve, tuningAreaUnits } from "./genesis.ts";

test("tuning recovers the exact number of successes across post-training checkpoints", () => {
  const curve = checkpoints.map((_, i) => ({
    accuracy: i === 0 ? 1 : i === 1 ? 3 / (2 * 20) : 0,
    retention: i === 0 ? 1 : i === 1 ? 3 / 20 : 0, ghostFree: i === 0 ? 1 : 0,
    untracked: 0, motionError: 0, hidden: 0.28, open: 0.28, miss: 0.5,
  }));
  assert.equal(tuningAreaUnits, (checkpoints.length - 1) * 2 * 20);
  assert.equal(Math.round(area(curve) * tuningAreaUnits), 3);
});

test("bootstrap and strict paired thresholds preserve both directions and zero boundaries", () => {
  const sample = [-0.1, 0.2, 0.4, 0.25];
  assert.deepEqual(bootstrap(sample), bootstrap(sample));
  assert.equal(effect({ mean: 0.10, low: 0.01, high: 0.25 }), "none");
  assert.equal(effect({ mean: 0.11, low: 0, high: 0.25 }), "none");
  assert.equal(effect({ mean: 0.11, low: 0.01, high: 0.25 }), "faster");
  assert.equal(effect({ mean: -0.10, low: -0.25, high: -0.01 }), "none");
  assert.equal(effect({ mean: -0.11, low: -0.25, high: 0 }), "none");
  assert.equal(effect({ mean: -0.11, low: -0.25, high: -0.01 }), "slower");
});

test("a seeded first-order training run reproduces its held-out learning curve", () => {
  const run = () => trainingCurve("first-order", 701, { motion: 0.03, existence: 0.003 }, [0, 1],
    (models) => evaluate(models, "first-order", 3001, 1));
  assert.deepEqual(run(), run());
});

import { confirmsForecast } from "./genesis.ts";
import { controlCurve, permitsComparison, referenceModel, requireControl, runFeasibility } from "./feasibility.ts";

test("design command rejects every comparison condition", () => {
  for (const id of ["full", "first-order", "no-protocol", "strong-first-order", "constructor"]) {
    assert.throws(() => requireControl(id), /Недопустимое условие/);
  }
});

test("oracle requires the original surviving object within the closing prediction radius", () => {
  const forecast = { x: 5, y: 7, radius: 5 };
  assert.equal(confirmsForecast({ x: 8, y: 11, vanishAt: null }, 10, forecast), true);
  assert.equal(confirmsForecast({ x: 8, y: 11, vanishAt: 10 }, 10, forecast), false);
  assert.equal(confirmsForecast({ x: 8, y: 12, vanishAt: null }, 10, forecast), false);
});

test("hand-set controls are evaluated without learning or parameter drift", () => {
  for (const id of ["ideal", "keep-all"] as const) {
    const expected = referenceModel(id);
    const curve = controlCurve(id, 701, { motion: 1, existence: 0.1 }, [0, 4], 3001, 2);
    assert.deepEqual(curve[0], curve[1]);
    assert.deepEqual(curve[1]!.weights, expected.weights.flat());
    assert.deepEqual(curve[1]!.logits, Object.values(expected.logits));
  }
});

test("motion features use elapsed time for both the latest observation and the long baseline", () => {
  const samples = [
    { step: 1, x: 3, y: 6, cellWidth: 3 },
    { step: 5, x: 9, y: 6, cellWidth: 3 },
    { step: 7, x: 15, y: 0, cellWidth: 3 },
  ];
  assert.deepEqual(features(samples.slice(0, 1)), [0, 0, 0, 0, 1]);
  assert.deepEqual(features(samples), [3, -3, 2, -1, 1]);
});

test("missing observations feed back displacement but preserve the last observed baseline", () => {
  const models = freshModels();
  const ep = createEpisode("continuity", 3001, "first-order", models);
  const obs = { gazeX: 90, gazeY: 90, aperture: 2, grid: 1, cells: [{ hue: 0, saturation: 0, lightness: 0 }] };
  const track = ep.agent.tracks[0]!;
  track.forecastInput = [3, -3, 2, -1, 1];
  track.vx = 1;
  track.vy = -2;
  const first = predictMiss(track, obs, models);
  assert.deepEqual(first.forecastInput, [1, -2, 2, -1, 1]);
  assert.deepEqual(predictMiss(first, obs, models).forecastInput, [0, 0, 2, -1, 1]);
});

test("reference controls predict the measured baseline without a scenario-specific offset", () => {
  for (const id of ["keep-all", "ideal"] as const) {
    const models = referenceModel(id);
    assert.deepEqual(models.weights, [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]);
    assert.deepEqual(motion(models, [1, 2, -3, 4, 1]), [-3, 4]);
  }
});

test("empty evidence uses visible cell centers inside the prediction disk on every miss", () => {
  const models = freshModels();
  const ep = createEpisode("continuity", 3001, "first-order", models);
  const prior = ep.agent.tracks[0]!;
  const obs = { gazeX: prior.x, gazeY: prior.y, aperture: 12, grid: 12,
    cells: Array.from({ length: 144 }, () => ({ hue: 0, saturation: 0, lightness: 0 })),
  };
  const first = predictMiss(prior, obs, models);
  assert.equal(first.reason, "no-data");
  assert.deepEqual(first.gapStep, { context: "open", observation: "empty" });
  assert.equal(predictMiss(prior, { ...obs, gazeX: prior.x - 5 }, models).gapStep!.observation, "none");
  // The cell at (7,5) intersects the radius-8 disk but its center is outside.
  obs.cells[8 * 12 + 9]!.lightness = 50;
  assert.equal(predictMiss(prior, obs, models).gapStep!.observation, "empty");
  obs.cells[6 * 12 + 6]!.lightness = 50;
  assert.deepEqual(predictMiss(prior, obs, models).gapStep, { context: "hidden", observation: "none" });
  obs.cells[6 * 12 + 6]!.saturation = 50;
  const conflict = predictMiss(prior, obs, models);
  assert.equal(conflict.reason, "conflict");
  assert.deepEqual(conflict.gapStep, { context: "open", observation: "none" });
  assert.equal(conflict.evidence, prior.evidence);
});

test("only consecutive matches with the whole forecast in view count towards the capped seen run", () => {
  const models = freshModels();
  const ep = createEpisode("continuity", 3001, "first-order", models).agent;
  let track = ep.tracks[0]!;
  const blob = ep.blobs[track.boundBlob!]!;
  const obs = { gazeX: track.x, gazeY: track.y, aperture: 20, cells: ep.cells, grid: ep.grid };
  for (let i = 1; i <= 25; i++) {
    track = matchTrack(ep, track, blob, obs);
    assert.equal(track.seenRun, Math.min(20, i));
  }
  const miss = predictMiss(track, obs, models);
  assert.equal(matchTrack(ep, miss, blob, obs).seenRun, 1);
  assert.equal(matchTrack(ep, track, blob, { ...obs, gazeX: track.x + 15 }).seenRun, 0);
});

test("confirmed absence requires three uninterrupted observations of the whole empty disk", () => {
  const models = freshModels();
  const ep = createEpisode("continuity", 3001, "first-order", models);
  const track = ep.agent.tracks[0]!;
  const obs = { gazeX: track.x, gazeY: track.y, aperture: 12, grid: 12,
    cells: Array.from({ length: 144 }, () => ({ hue: 0, saturation: 0, lightness: 0 })),
  };
  let missed = predictMiss(track, obs, models);
  assert.equal(missed.reason, "no-data");
  missed = predictMiss(missed, obs, models);
  assert.equal(missed.reason, "no-data");
  // The center remains visible, but the right edge of the disk leaves the window.
  missed = predictMiss(missed, { ...obs, gazeX: track.x - 12 + track.radius / 2 }, models);
  assert.equal(missed.emptyViews, 0);
  for (let i = 1; i <= 3; i++) {
    missed = predictMiss(missed, obs, models);
    assert.equal(missed.reason, i === 3 ? "absence" : "no-data");
  }
});


test("any failing control blocks the comparison and its verdict", () => {
  const valid = { frozen: [0.5, 0.75], "keep-all": [0.75, 0.75], ideal: [0.9], oracle: [0.5, 0.9] };
  assert.ok(permitsComparison(valid));
  for (const control of ["frozen", "keep-all"] as const) {
    assert.equal(permitsComparison({ ...valid, [control]: [0.751, 0.5] }), false);
  }
  for (const control of ["ideal", "oracle"] as const) {
    assert.equal(permitsComparison({ ...valid, [control]: [1, 0.899] }), false);
  }
});

test("the feasibility pipeline reproduces the grid choice, curves and final component evaluations", () => {
  const small = { firstRun: 701, runs: 2, firstEvalSeed: 3001, seeds: 2, stages: [0, 1] };
  const run = () => runFeasibility(["frozen", "keep-all", "ideal", "oracle"], small);
  const first = run();
  assert.equal(first.curves.get("oracle")!.at(-1)!.split!.length, 2);
  assert.deepEqual(run(), first);
});

test("the task and oracle settings selected on the design seeds are frozen", () => {
  assert.deepEqual(genesisTask, {
    trainingSteps: 80, checkpoints: [0, 4, 8, 16, 32, 64],
    curriculum: ["continuity", "vanish", "binding", "nursery", "continuity", "binding", "nursery"],
    uncertaintyGrowth: 0.03, emptyViews: 3, ghostDelay: 20, outdoorFirst: 12, outdoorLast: 35,
    keepAllExistence: { hidden: 0.001, open: 0.001, miss: 0.999 },
    idealExistence: { hidden: 0.001, open: 0.02, miss: 0.1 },
    oracleRates: { motion: 0.03, existence: 0.003 }, controlFloor: 0.9, controlCeiling: 0.75,
  });
});

test("nursery training losses occur at the first open step after each seeded deadline", () => {
  for (let episode = 1; episode <= 10; episode++) {
    const seed = 10000 + 100 * 701 + episode;
    const live = makeWorld("nursery", seed);
    const training = makeWorld("nursery", seed);
    scheduleOutdoorLosses(training);
    const expected = live.objects.map((): number | null => null);
    for (let step = 1; step <= 80; step++) {
      advanceWorld(live);
      for (const [i, object] of live.objects.entries()) {
        const deadline = 12 + Math.floor(24 * hashUnit(seed * 3 + i));
        if (step >= deadline && expected[i] == null && !hiddenByOccluder(object, live.occluder)) {
          expected[i] = step;
        }
      }
    }
    assert.deepEqual(training.objects.map((o) => o.vanishAt), expected);
    assert.ok(live.objects.every((o) => o.vanishAt == null));
  }
});
