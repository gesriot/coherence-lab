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

import { features, fitMotion, fitPermanence, freshModels, GapBook, hazard, motion, type Features } from "./learning.ts";
import type { Track } from "./types.ts";
import { matchRadius, predictMiss } from "./perception.ts";

test("normalized motion learning converges on a constant displacement", () => {
  const models = freshModels();
  const input: Features = [1.5, -0.5, 1.5, -0.5, 1];
  assert.deepEqual(motion(models, input), [0, 0]);
  for (let i = 0; i < 500; i++) fitMotion(models, { input, target: [1.5, -0.5] }, 0.1);
  assert.ok(Math.hypot(motion(models, input)[0] - 1.5, motion(models, input)[1] + 0.5) < 1e-8);
});

test("survival likelihood recovers cause-specific event rates", () => {
  const models = freshModels();
  for (let i = 0; i < 5000; i++) {
    for (const [c, duration] of [["occlusion", 20], ["absence", 2], ["no-data", 5]] as const) {
      fitPermanence(models, {
        exposure: { occlusion: 0, absence: 0, "no-data": 0, [c]: duration }, last: c, found: false,
      }, 0.01);
    }
  }
  assert.ok(Math.abs(hazard(models, "occlusion") - 0.05) < 1e-6);
  assert.ok(Math.abs(hazard(models, "absence") - 0.5) < 1e-6);
  assert.ok(Math.abs(hazard(models, "no-data") - 0.2) < 1e-6);
});

test("gap outcomes use live bindings, retrospective births, timeout and episode censoring", () => {
  const track = createEpisode("continuity", 1).agent.tracks[0]!;
  const advance = (t: Track): Track => ({ ...t, boundBlob: null, unseen: t.unseen + 1, reason: "absence" });
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
    assert.equal(record!.exposure.absence, outcome === "birth" ? 31 : 32);
  }
});

test("zero motion and equal initial hazards lose the target behind the wall", () => {
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
  const learner = new Learner({ motion: 0.1, permanence: 0.01 }, "attention", 1);
  const example = { input: [1, 0, 1, 0, 1] as Features, target: [1, 0] as [number, number] };
  const gap = { found: true, last: "absence" as const, exposure: { absence: 1, occlusion: 0, "no-data": 0 } };
  learner.rememberMotion(example, null);
  learner.rememberGap(gap, null);
  assert.equal(learner.movements.entries.length + learner.gaps.entries.length, 0);
  for (let i = 0; i < 300; i++) {
    learner.rememberMotion({ ...example, target: [i, 0] }, 0);
    learner.rememberGap({ ...gap, exposure: { ...gap.exposure, absence: i + 1 } }, 0.3);
  }
  assert.equal(learner.movements.entries.length, 256);
  assert.equal(learner.movements.entries[0]!.value.target[0], 44);
  assert.equal(learner.gaps.entries.length, 32);
  assert.equal(learner.gaps.entries[0]!.value.exposure.absence, 269);
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
    const learner = new Learner({ motion: 0.1, permanence: 0.01 }, selection, 3);
    learner.update();
    assert.deepEqual(learner.updates, { motion: 0, permanence: 0 });
    learner.rememberMotion({ input: [0, 0, 0, 0, 1], target: [0, 1] }, 0);
    learner.update();
    learner.rememberGap({ exposure: { absence: 1, occlusion: 0, "no-data": 0 }, last: "absence", found: false }, 0);
    learner.update();
    assert.deepEqual(learner.updates, { motion: 2, permanence: 1 });
  }
});

test("only the oracle training condition requests privileged labels", () => {
  for (const condition of Object.keys(conditions) as Condition[]) {
    const learner = new Learner({ motion: 0.1, permanence: 0.01 }, conditions[condition].selection, 1);
    const run = () => trainEpisode(learner, condition, "vanish", 1, () => { throw Error("label hook"); });
    if (condition === "oracle") assert.throws(run, /label hook/);
    else run();
  }
});

import { bootstrap, effect, feasibility, evaluate, trainingCurve } from "./genesis.ts";

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

test("feasibility gates are inclusive and saturation is a separate annotation", () => {
  assert.deepEqual(feasibility(0.9, [0.75, 0.5], 0.9, 0.9), { positive: true, lower: true, saturated: true });
  assert.deepEqual(feasibility(0.899, [0.5], 0.89, 0.91), { positive: false, lower: true, saturated: false });
  assert.equal(feasibility(1, [0.5, 0.751], 0, 0).lower, false);
});

test("a seeded first-order training run reproduces its held-out learning curve", () => {
  const run = () => trainingCurve("first-order", 701, { motion: 0.03, permanence: 0.003 }, [0, 1],
    (models) => evaluate(models, "first-order", 3001, 1));
  assert.deepEqual(run(), run());
});

import { confirmsForecast } from "./genesis.ts";
import { controlCurve, referenceModel, requireControl } from "./feasibility.ts";

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
    const curve = controlCurve(id, 701, { motion: 1, permanence: 0.1 }, [0, 4], 3001, 2);
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
