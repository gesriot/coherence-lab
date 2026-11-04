import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { createEpisode, defaultConfig, snapshot, stepEpisode } from "./agent.ts";
import { referenceModel } from "./feasibility.ts";
import { conditions } from "./genesis-training.ts";
import type { Vector } from "./learning.ts";
import { observe } from "./sensor.ts";
import { bodyForRun, oracleGainLabel } from "./sensorimotor-task.ts";
import { fitGain, GainLearner, trainGainEpisode, type TrainingCondition } from "./sensorimotor-training.ts";

test("unit body and estimate reproduce every serialized reference frame", () => {
  const digest = createHash("sha256");
  for (const scenario of ["continuity", "vanish"] as const) {
    const ep = createEpisode(scenario, 3101, "first-order", referenceModel("ideal"), [1, 1]);
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
    const unitWorld = { ...ep.world, bodyGain: undefined };
    const physical = observe(unitWorld, body[0] / models.gain[0] * x, body[1] / models.gain[1] * y, aperture);
    assert.deepEqual(ep.agent.cells, physical.cells);
    if (step === 0) {
      assert.deepEqual([x, y], [16, 50]);
      assert.notDeepEqual(physical.cells, observe(unitWorld, x, y, aperture).cells);
    }
  }
});

test("calibration converges from opposite sides without moving a stationary axis", () => {
  const body: Vector = [0.85, 1.15], estimate: Vector = [1, 1], shift: Vector = [2, -3];
  for (let i = 0; i < 1000; i++) fitGain(estimate, shift, oracleGainLabel(body, shift, estimate), 0.03);
  assert.ok(Math.hypot(estimate[0] - body[0], estimate[1] - body[1]) < 1e-8);
  fitGain(estimate, [2, 0], [-1e4, 1e4], 0.1);
  assert.equal(estimate[0], 2);
  assert.ok(Math.abs(estimate[1] - body[1]) < 1e-8);
  fitGain(estimate, [2, 0], [1e4, 0], 0.1);
  assert.equal(estimate[0], 0.5);
});

test("gain replay uses consecutive bindings, executed focus and the previous forecast", () => {
  const models = referenceModel("ideal");
  const track = createEpisode("continuity", 3101, "first-order", models).agent.tracks[0]!;
  const prior = { ...track, vx: 1, vy: -2 };
  const now = { ...prior, x: prior.x + 4, y: prior.y + 2 };
  const gap = { ...now, boundBlob: null };
  for (const selection of ["attention", "error", "uniform"] as const) {
    const learner = new GainLearner(models, 0.03, selection, 801);
    learner.observe([gap], [now], [3, 0], prior.id, 0.5);
    learner.observe([prior], [gap], [3, 0], prior.id, 0.5);
    assert.equal(learner.memory.entries.length, 0);
    learner.observe([prior], [now], [3, 0], null, 0.5);
    assert.equal(learner.memory.entries.length, selection === "attention" ? 0 : 1);
    learner.observe([prior], [now], [3, 0], prior.id, 0.5);
    const last = learner.memory.entries.at(-1)!;
    assert.deepEqual(last.value, { shift: [3, 0], error: [3, 4] });
    if (selection !== "uniform") assert.equal(last.weight, selection === "attention" ? 0.55 : 5.05);
  }
});

test("only oracle updates request a label and gain training preserves the first-stage models", () => {
  const body = bodyForRun(801);
  for (const id of ["oracle", "first-order", "strong-first-order", "no-protocol", "full"] satisfies TrainingCondition[]) {
    const models = referenceModel("ideal");
    const { weights, logits } = structuredClone(models);
    const learner = new GainLearner(models, 0.03, conditions[id].selection, 801);
    const ep = createEpisode("continuity", 3101, conditions[id].gaze, models, body);
    let calls = 0;
    trainGainEpisode(learner, ep, id, (shift, estimate) => {
      calls++;
      return oracleGainLabel(body, shift, estimate);
    });
    assert.equal(calls, id === "oracle" ? 80 : 0);
    assert.deepEqual(models.weights, weights);
    assert.deepEqual(models.logits, logits);
  }
});

import { designGainControls } from "./sensorimotor-feasibility.ts";

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
    bodyMinOffset: 0.1, bodyMaxOffset: 0.2, minGazeShift: 0, minGain: 0.5, maxGain: 2, memory: 256,
    checkpoints: [0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120], rates: [0.003, 0.01, 0.03, 0.1], evaluationCount: 20,
    design: { firstRun: 801, runs: 10, firstEvalSeed: 3101 },
    tuning: { firstRun: 531, runs: 10, firstEvalSeed: 2061 },
    comparison: { firstRun: 91, runs: 30, firstEvalSeed: 1061 }, oracleRate: 0.003,
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
    for (let local = 0; local < 80; local++) {
      const agent = ep.agent, before = agent.tracks, gaze = agent.obsGaze;
      stepEpisode(ep, cfg);
      learner.observe(before, agent.tracks, [agent.obsGaze.x - gaze.x, agent.obsGaze.y - gaze.y], null, 0);
      learner.update((shift, estimate) => oracleGainLabel(body, shift, estimate));
      steps++;
      const point = curve.find((p) => p.step === steps);
      if (point) assert.deepEqual([point.gainX, point.gainY], models.gain);
    }
  }
});
