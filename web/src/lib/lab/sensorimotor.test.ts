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
