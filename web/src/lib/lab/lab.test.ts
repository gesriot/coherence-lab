import assert from "node:assert/strict";
import test from "node:test";
import { createEpisode, defaultConfig, snapshot, stepEpisode } from "./agent.ts";

test("the episode entry point advances one world step", () => {
  const ep = createEpisode("continuity", 1);
  stepEpisode(ep, defaultConfig());
  assert.equal(snapshot(ep, "full").step, 1);
});
