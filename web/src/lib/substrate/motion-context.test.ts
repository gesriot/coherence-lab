import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { createElement } from "./element.ts";
import { movingFrames, movingScene, substrateWorld } from "./input.ts";
import { contextGroup, fitContext } from "./motion-context.ts";
import { evaluateMotion, evaluateMotionWithContext } from "./motion-evaluation.ts";
import { train } from "./training.ts";

test("the default path preserves the B3 movies, commands, learned parameters and scores", () => {
  const hash = () => createHash("sha256");
  const movie = hash(), commands = hash(), weights = hash(), scores = hash();
  for (const copy of ["executed", "zero", "shuffled"] as const) {
    const motion = { gain: "fixed", copy } as const;
    for (const scenario of ["continuity", "vanish", "nursery", "binding"] as const) {
      for (const input of movingFrames(scenario, 8101, 40, motion)) {
        movie.update(input.frame);
        commands.update(Float64Array.from(input.copy));
      }
    }
    const make = () => createElement(7, true, 16, 3, "memory", true), model = make();
    train(model, 7, "per-cell", undefined, undefined, { motion, episodes: 4 });
    weights.update(model.weights);
    scores.update(JSON.stringify(evaluateMotion(model, make(), motion,
      [8001, 8002, 8003, 8004], [8101, 8102, 8103, 8104])));
  }
  assert.deepEqual([movie, commands, weights, scores].map((h) => h.digest("hex")), [
    "1ff01438d1bc8087ec437145fa191b486c77724dbfb216133b4bb1bd96a5f9ef",
    "edc10b7571e67ebe8b6c422bafa117fe42e9e8bbadd7be2f4c900be2aa52da79",
    "da6539f23d01ad1abadd15c2e5d5466dcd6fdd5e3bf79272fa631e062b88faba",
    "09504bf1162d0e60aa19b536491cf94d3ffd55b7f118b16338bf57b6d291a3bd",
  ]);
});

test("moving scenes use the seeded tracking path only in continuity", () => {
  const motion = { gain: "fixed", copy: "executed" } as const;
  for (const scenario of ["continuity", "vanish", "nursery", "binding", "conflict"] as const) {
    for (const seed of [13, 27, 382]) {
      const random = movingScene(scenario, seed, 16, { ...motion, path: "random" });
      const original = movingScene(scenario, seed, 16, motion);
      for (const a of random) {
        const b = original.next();
        assert.ok(!b.done);
        if (scenario === "continuity" && a.world.t === 0) {
          assert.deepEqual(a.world.objects, substrateWorld(scenario, seed, { path: "random" }).objects);
          assert.notDeepEqual(a.world.objects[0], b.value.world.objects[0]);
        }
        if (scenario !== "continuity") assert.deepEqual(a, b.value);
        assert.deepEqual(a.gaze, b.value.gaze);
        assert.deepEqual(a.command, b.value.command);
      }
    }
  }
});

test("context predicts the training label mean for scenario and eight-step bin with a pooled fallback", () => {
  const samples = [contextGroup("continuity", 1), contextGroup("continuity", 7),
    contextGroup("continuity", 8), contextGroup("vanish", 7)];
  const predict = fitContext(samples, [3, 5, -10, 6]);
  assert.equal(predict(contextGroup("continuity", 0)), 4);
  assert.equal(predict(contextGroup("continuity", 15)), -10);
  assert.equal(predict(contextGroup("vanish", 1)), 6);
  assert.equal(predict(contextGroup("vanish", 8)), 1);
  assert.equal(predict(contextGroup("continuity", 16)), 1);
});

test("context evaluation rejects overlap between probe training and test episodes", () => {
  const m = createElement(1, true, 16, 3, "memory", true);
  assert.throws(() => evaluateMotionWithContext(m, m, { gain: "fixed", copy: "zero", path: "random" },
    [4001, 4002], [3001, 4002]), /пересекаются/);
});
