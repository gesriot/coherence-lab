import assert from "node:assert/strict";
import test from "node:test";
import { GRID } from "./input.ts";
import { otherRowCells, scoreTrackingEpisode, trackingTotals, type TrackingEpisode } from "./tracking-evaluation.ts";
import { trackingCriterion } from "./tracking-report.ts";

test("target timing and spatial and temporal masks separate tracking from false expectations", () => {
  const size = GRID * GRID * 3;
  const ep: TrackingEpisode = {
    frames: Array.from({ length: 25 }, () => new Float64Array(size)),
    ghostCells: Array.from({ length: 25 }, () => []), events: [],
  };
  const predictions = ep.frames.map(() => new Float64Array(size));
  const cells = [5 * GRID + 2, 6 * GRID + 3], te = 8;
  ep.events.push({ time: te, length: 4, kind: "wall", cells });
  ep.events.push({ time: 4, length: 1, kind: "gap", cells: [0] });
  ep.frames[4]![1] = 1;
  predictions[4]![1] = 0.25;
  const others = [0, 1, 2, 9, 10, 11, 12, 13, 14, 15].flatMap((r) => [r * GRID + 2, r * GRID + 3]);
  assert.deepEqual(otherRowCells(cells), others);
  for (const i of cells) ep.frames[te]![3 * i + 1] = 1;
  for (const [t, p] of predictions.entries()) {
    for (const i of cells) p[3 * i + 1] = t / 100;
    for (const i of others) p[3 * i + 2] = t >= 2 && t <= 8 ? 0.125 : 1;
    p[3 * (8 * GRID + 2) + 1] = 1;
  }
  ep.frames[17]![3 * cells[0]! + 1] = 0.051;
  ep.frames[18]![3 * cells[1]! + 1] = 0.05;
  const totals = trackingTotals();
  scoreTrackingEpisode(totals, ep, predictions);
  assert.equal(totals.wall.count, 1);
  assert.equal(totals.wall.min, 4);
  assert.ok(Math.abs(totals.wall.squared / totals.wall.copy - 0.92 ** 2) < 1e-14);
  assert.equal(totals.gap.squared / totals.gap.copy, 0.75 ** 2);
  assert.deepEqual(totals.curve.map((s) => s.sum / s.count), [2,3,4,5,6,7,8,9,10].map((t) => t / 100));
  assert.equal(totals.other.count, others.length * 7);
  assert.equal(totals.other.sum / totals.other.count, 0.125);
  assert.equal(totals.after.count, 15);
  assert.ok(Math.abs(totals.after.sum - (2 * 8 * 0.195 - 0.17)) < 1e-12);
  ep.events = [{ time: 6, length: 2, kind: "wall", cells }];
  const early = trackingTotals();
  scoreTrackingEpisode(early, ep, predictions);
  assert.equal(early.wall.count, 1);
  assert.ok(early.curve.every((v) => v.count === 0));
});

test("the feasibility gate checks both exit bounds, both false expectations and the ordinary baseline", () => {
  const base = Array(5).fill(0.90);
  const control = Array.from({ length: 5 }, () => ({ wall: 0.50, other: 0.03, after: 0.08 }));
  assert.ok(trackingCriterion(base, control).passes);
  assert.ok(!trackingCriterion(base, control.map((s) => ({ ...s, wall: 0.50001 }))).passes);
  assert.ok(!trackingCriterion(base, [0.2,0.2,0.2,0.2,0.65001].map((wall) => ({ ...control[0]!, wall }))).passes);
  assert.ok(!trackingCriterion(base, control.map((s) => ({ ...s, other: 0.03001 }))).passes);
  assert.ok(!trackingCriterion(base, control.map((s) => ({ ...s, after: 0.08001 }))).passes);
  assert.ok(!trackingCriterion(Array(5).fill(0.89999), control).passes);
  assert.ok(!trackingCriterion(base.slice(1), control).passes);
});
