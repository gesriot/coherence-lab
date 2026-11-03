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
