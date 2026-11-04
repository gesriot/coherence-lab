import assert from "node:assert/strict";
import test from "node:test";
import { partialPersistence } from "./persistence-report.ts";

test("partial persistence requires both exit bounds and an elevated, increasing expectation", () => {
  const score = (exit: number, at = 0.03, early = 0.02) =>
    ({ exit, curve: [early, early, early, 0, 0, 0, at, 0, 0] });
  assert.ok(partialPersistence(Array(5).fill(score(0.90))).passes);
  assert.ok(!partialPersistence(Array(5).fill(score(0.90001))).passes);
  assert.ok(!partialPersistence([...Array(4).fill(score(0.8)), score(0.95001)]).passes);
  assert.ok(!partialPersistence(Array(5).fill(score(0.85, 0.02999))).passes);
  assert.ok(!partialPersistence(Array(5).fill(score(0.85, 0.04, 0.04))).passes);
});
