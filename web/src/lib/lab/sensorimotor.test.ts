import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { createEpisode, defaultConfig, snapshot, stepEpisode } from "./agent.ts";
import { referenceModel } from "./feasibility.ts";
import { conditions } from "./genesis-training.ts";
import type { Vector } from "./learning.ts";
import { alignScene, sceneFrame, type AxisRatios } from "./scene-alignment.ts";
import { observe } from "./sensor.ts";
import { bodyForRun, oracleGainLabel } from "./sensorimotor-task.ts";
import { fitGain, GainLearner, trainGainEpisode, type TrainingCondition } from "./sensorimotor-training.ts";

test("episodes without a body reproduce every serialized reference frame", () => {
  const digest = createHash("sha256");
  for (const scenario of ["continuity", "vanish"] as const) {
    const ep = createEpisode(scenario, 3101, "first-order", referenceModel("ideal"));
    for (let step = 0; step <= 80; step++) {
      if (step) stepEpisode(ep, { ...defaultConfig(), ablation: "first-order" });
      digest.update(JSON.stringify(snapshot(ep, "first-order")));
    }
  }
  assert.equal(digest.digest("hex"), "39cd01bdff23f6aebf63328aac7617808cc5cdc6875627e373bf067cd3f31eb5");
});

test("the sensor samples the physical window but reports the agent's intended center", () => {
  const models = referenceModel("ideal");
  models.gain = [0.9, 1.1];
  const body: Vector = [1.15, 0.85];
  const ep = createEpisode("continuity", 3101, "first-order", models, body);
  for (let step = 0; step < 5; step++) {
    if (step) stepEpisode(ep, { ...defaultConfig(), ablation: "first-order" });
    const { x, y, aperture } = ep.agent.obsGaze;
    const unitWorld = { ...ep.world, bodyGain: [1, 1] as Vector };
    const physical = observe(unitWorld, body[0] / models.gain[0] * x, body[1] / models.gain[1] * y, aperture);
    assert.deepEqual(ep.agent.cells, physical.cells);
    if (step === 0) {
      assert.deepEqual([x, y], [16, 50]);
      assert.notDeepEqual(physical.cells, observe(unitWorld, x, y, aperture).cells);
    }
  }
});

function textureFrame(command: Vector, body: Vector, grid = 32) {
  const cells = Array.from({ length: grid * grid }, (_, i) => {
    const x = command[0] * body[0] + i % grid + 0.5 - grid / 2;
    const y = command[1] * body[1] + Math.floor(i / grid) + 0.5 - grid / 2;
    return { hue: 0, saturation: 0, lightness: 6 + 4 * Math.sin(0.37 * x + 1.3) * Math.sin(0.29 * y + 0.7) };
  });
  return sceneFrame(cells, grid, grid / 2, command);
}

test("static texture extends outside the world only for a body", () => {
  const ep = createEpisode("continuity", 3101, "first-order", referenceModel("ideal"), [1, 1]);
  for (const center of [-20, 120]) {
    const frame = observe(ep.world, center, center, 6);
    const x = center - 5.5;
    assert.equal(frame.cells[0]!.lightness, 6 + 4 * Math.sin(0.37 * x + 1.3) * Math.sin(0.29 * x + 0.7));
    assert.equal(observe({ ...ep.world, bodyGain: undefined }, center, center, 6).cells[0]!.lightness, 6);
  }
});

test("alignment recovers a synthetic translation and has a fixed point at the correct gain", () => {
  const body: Vector = [1.1, 0.9];
  const previous = textureFrame([0, 0], body);
  for (const axis of [0, 1] as const) {
    const command: Vector = [0, 0];
    command[axis] = 10;
    const pair = { previous, current: textureFrame(command, body) };
    assert.equal(alignScene(pair, [1, 1])[axis], body[axis]);
    assert.equal(alignScene(pair, body)[axis], 1);
    assert.equal(alignScene(pair, body)[axis === 0 ? 1 : 0], null);
  }
});

test("surface cells and all eight neighbors are masked before interpolation", () => {
  const cells = Array.from({ length: 25 }, () => ({ hue: 0, saturation: 0, lightness: 6 }));
  cells[12] = { hue: 30, saturation: 70, lightness: 50 };
  cells[0]!.lightness = 32;
  const frame = sceneFrame(cells, 5, 5, [0, 0]);
  assert.deepEqual([...frame.light.keys()].filter((i) => Number.isNaN(frame.light[i])), [6, 7, 8, 11, 12, 13, 16, 17, 18]);
  assert.equal(frame.light[0], 32);
});

test("uninformative draws do not update either gain or request an oracle label", () => {
  const learner = new GainLearner(referenceModel("ideal"), 0.1, "uniform", 801);
  for (const grid of [4, 32]) {
    const previous = textureFrame([0, 0], [1, 1], grid), current = textureFrame([1, 1], [1, 1], grid);
    if (grid === 32) { previous.light.fill(6); current.light.fill(6); }
    const pair = { previous, current };
    const ratios = alignScene(pair, [1, 1]);
    assert.deepEqual(ratios, [null, null]);
    learner.observe(pair, "pursue", false, 0, ratios);
    assert.equal(learner.update(() => assert.fail("uninformative label")), false);
  }
  assert.deepEqual(learner.models.gain, [1, 1]);
});

test("attention admits gaze switches and error priorities use informative ratios at insertion", () => {
  const models = referenceModel("ideal");
  const previous = textureFrame([0, 0], [1.1, 0.9]), current = textureFrame([10, 0], [1.1, 0.9]);
  const pair = { previous, current };
  const ratios = alignScene(pair, models.gain);
  for (const selection of ["attention", "error", "uniform"] as const) {
    const learner = new GainLearner(models, 0.1, selection, 801);
    learner.observe(pair, "pursue", false, 0.5, ratios);
    assert.equal(learner.memory.entries.length, selection === "attention" ? 0 : 1);
    learner.observe(pair, "pursue", true, 0.5, ratios);
    learner.observe(pair, "explore", false, 0.5, ratios);
    const entries = learner.memory.entries;
    assert.equal(entries.length, selection === "attention" ? 2 : 3);
    if (selection !== "uniform") assert.ok(Math.abs(entries[0]!.weight - (selection === "attention" ? 0.55 : 0.15)) < 1e-12);
  }
});

test("replay realigns the stored commands at the current estimate and only updates informative axes", () => {
  const models = referenceModel("ideal"), body: Vector = [1.1, 0.9];
  const learner = new GainLearner(models, 0.1, "uniform", 801);
  const pair = { previous: textureFrame([0, 0], body), current: textureFrame([10, 0], body) };
  learner.observe(pair, "explore", false, 0, alignScene(pair, models.gain));
  models.gain[0] = 1.1;
  assert.ok(learner.update());
  assert.deepEqual(models.gain, [1.1, 1]);
  const ratios: AxisRatios = [100, null];
  fitGain(models.gain, ratios, 1);
  assert.deepEqual(models.gain, [2, 1]);
  fitGain(models.gain, [0.01, null], 1);
  assert.deepEqual(models.gain, [0.5, 1]);
});

test("only oracle updates request a label and gain training preserves the first-stage models", () => {
  const body = bodyForRun(801);
  for (const id of ["oracle", "first-order", "strong-first-order", "no-protocol", "full"] satisfies TrainingCondition[]) {
    const models = referenceModel("ideal");
    const { weights, logits } = structuredClone(models);
    const learner = new GainLearner(models, 0.03, conditions[id].selection, 801);
    const ep = createEpisode("continuity", 3101, conditions[id].gaze, models, body);
    let calls = 0;
    trainGainEpisode(learner, ep, id, (estimate) => {
      calls++;
      return oracleGainLabel(body, estimate);
    });
    assert.equal(calls > 0, id === "oracle");
    assert.deepEqual(models.weights, weights);
    assert.deepEqual(models.logits, logits);
  }
});

import { designGainControls, designOracleRate } from "./sensorimotor-feasibility.ts";

test("the sensorimotor design runner rejects participant conditions before running controls", () => {
  for (const id of ["full", "first-order", "no-protocol", "strong-first-order", "keep-all", "constructor"]) {
    assert.throws(() => designGainControls(["ideal", id]), /Недопустимое условие/);
  }
});

import { compareGainConditions, type ComparisonControls } from "./sensorimotor-comparison.ts";
import { sensorimotorTask } from "./sensorimotor-task.ts";
import { calibrationError, calibrationPoint, gainArea, gainTrial, type GainRuns } from "./sensorimotor.ts";

test("the sensorimotor design, oracle rate and reserved seeds are frozen", () => {
  assert.deepEqual(sensorimotorTask, {
    bodyMinOffset: 0.1, bodyMaxOffset: 0.2, minGain: 0.5, maxGain: 2, memory: 256,
    metric: { baselineGain: [1, 1], firstScoredIndex: 1 }, saccadeDistance: 4,
    checkpoints: [0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120], rates: [0.01, 0.03, 0.1, 0.3], evaluationCount: 20,
    design: { firstRun: 801, runs: 10, firstEvalSeed: 3101 },
    tuning: { firstRun: 531, runs: 10, firstEvalSeed: 2061 },
    comparison: { firstRun: 91, runs: 30, firstEvalSeed: 1061 }, oracleTieTolerance: 0.005, oracleRate: 0.01,
  });
});

test("no participant or verdict is produced if a sensorimotor control fails", () => {
  const synthetic = (scores: number[], accuracy = 1): GainRuns => [{
    curve: scores.map((q, i) => ({ step: sensorimotorTask.checkpoints[i]!, q, error: 0, gainX: 1, gainY: 1 })),
    secondary: { accuracy, retention: accuracy, ghostFree: accuracy, untracked: 0, motionError: 0, hidden: 0.001, open: 0.02, miss: 0.1 },
    saccadesPerEpisode: 0,
  }];
  const zeros = sensorimotorTask.checkpoints.map((): number => 0);
  const ones = zeros.map(() => 1);
  const stable = [0, 0.9, 0.85, ...ones.slice(3)];
  const good = synthetic(stable);
  const valid = { frozen: synthetic(zeros, 0.75), ideal: synthetic(ones), oracle: good, participantOracle: good };
  for (const [key, invalid] of [
    ["frozen", synthetic([0.01, ...zeros.slice(1)], 0.75)], ["frozen", synthetic(zeros, 0.775)],
    ["ideal", synthetic([0.99, ...ones.slice(1)])], ["ideal", synthetic(ones, 0.875)],
    ["oracle", synthetic([0, 0.90, 0.84, ...ones.slice(3)])], ["oracle", synthetic(stable, 0.875)],
    ["participantOracle", synthetic([...stable.slice(0, -1), 0.899])], ["participantOracle", synthetic(stable, 0.875)],
  ] satisfies [keyof ComparisonControls, GainRuns][]) {
    assert.equal(compareGainConditions({ ...valid, [key]: invalid }, () => assert.fail("gate opened")), null);
  }
  const result = compareGainConditions(valid, () => good)!;
  assert.equal(result.participants.size, 4);
  assert.deepEqual(result.differences.map((d) => d.estimate.mean), [0, 0, 0]);
  assert.match(result.verdicts[2]!, /не применимо/);
  const precise = compareGainConditions(valid, (id) =>
    synthetic([0, ...Array(10).fill(id === "full" ? 0.600123 : 0.5)]))!;
  assert.ok(Math.abs(precise.differences[0]!.estimate.mean - 0.100123) < 1e-12);
  assert.equal(precise.differences[0]!.effect, "faster");

});

test("calibration progress is relative to the same body's unit estimate and can fall below zero", () => {
  const body: Vector = [0.875, 1.25];
  const initial = calibrationError([1, 1], body);
  assert.equal(initial, 0.1875);
  assert.equal(calibrationPoint(0, [1, 1], body, initial).q, 0);
  assert.equal(calibrationPoint(0, body, body, initial).q, 1);
  assert.equal(calibrationPoint(10, [1.125, 0.75], body, initial).q, -1);
});

test("area equally weights every checkpoint after zero and preserves continuous progress", () => {
  const values = [0, -0.56123, 0.11, 0.24, 0.35, 0.46, 0.57, 0.68, 0.79, 0.89, 1];
  assert.ok(Math.abs(gainArea(values.map((q) => ({ q }))) - 0.452877) < 1e-12);
  assert.equal(gainArea([{ q: 1 }, { q: 0.25 }, { q: 0.75 }]), 0.5);
});

test("calibration checkpoints capture completed environment steps across episode boundaries", () => {
  const body = bodyForRun(801), models = referenceModel("ideal");
  const learner = new GainLearner(models, 0.01, "uniform", 801);
  const { curve } = gainTrial("oracle", 801, 3101, 0.01);
  assert.deepEqual(curve.map((p) => p.step), [0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]);
  const cfg = { ...defaultConfig(), ablation: "first-order" as const };
  let steps = 0;
  for (const [index, scenario] of (["continuity", "vanish"] as const).entries()) {
    const ep = createEpisode(scenario, 10000 + 100 * 801 + index + 1, cfg.ablation, models, body);
    const capture = () => {
      const agent = ep.agent, gaze = agent.obsGaze;
      return sceneFrame(agent.cells, agent.grid, gaze.aperture, [gaze.x / models.gain[0], gaze.y / models.gain[1]]);
    };
    let previous = capture();
    for (let local = 0; local < 80; local++) {
      stepEpisode(ep, cfg);
      const current = capture(), pair = { previous, current };
      learner.observe(pair, "pursue", false, 0, alignScene(pair, models.gain));
      learner.update((estimate) => oracleGainLabel(body, estimate));
      previous = current;
      steps++;
      const point = curve.find((p) => p.step === steps);
      if (point) assert.deepEqual([point.gainX, point.gainY], models.gain);
    }
  }
});

test("oracle rate ties are measured against the highest unrounded final progress", () => {
  assert.equal(designOracleRate([
    { rate: 0.1, finalQ: 1 }, { rate: 0.03, finalQ: 0.999 }, { rate: 0.003, finalQ: 0.993 },
    { rate: 0.01, finalQ: 0.996 },
  ]), 0.01);
  assert.equal(designOracleRate([
    { rate: 0.003, finalQ: 0 }, { rate: 0.01, finalQ: 0.005 },
  ]), 0.01);
});
