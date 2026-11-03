import { WORLD, wrapHue } from "./math";
import type { Occluder, ScenarioId, Shape, SimObject, World } from "./types";

function obj(
  id: string,
  shape: Shape,
  hue: number,
  x: number,
  y: number,
  vx: number,
  vy: number,
  r: number,
): SimObject {
  return { id, shape, hue, x, y, vx, vy, r };
}

export function hiddenByOccluder(o: SimObject, occ: Occluder): boolean {
  const cx = Math.max(occ.x, Math.min(o.x, occ.x + occ.w));
  const cy = Math.max(occ.y, Math.min(o.y, occ.y + occ.h));
  const dx = o.x - cx;
  const dy = o.y - cy;
  return dx * dx + dy * dy <= o.r * o.r;
}

export function contains(o: SimObject, x: number, y: number): boolean {
  const dx = x - o.x;
  const dy = y - o.y;
  if (o.shape === "disk") return dx * dx + dy * dy <= o.r * o.r;
  if (o.shape === "square") return Math.abs(dx) <= o.r && Math.abs(dy) <= o.r;
  const y0 = -o.r * 0.95;
  const y1 = o.r * 0.72;
  if (dy < y0 || dy > y1) return false;
  const span = y1 - y0;
  const half = o.r * ((dy - y0) / span);
  return Math.abs(dx) <= half;
}

export function makeWorld(scenario: ScenarioId, seed: number): World {
  const base = {
    seed,
    t: 0,
    light: 0,
    noise: 7,
    baseNoise: 7,
    swapAt: null as number | null,
    swapped: false,
  };

  if (scenario === "continuity") {
    return {
      ...base,
      occluder: { x: 40, y: 0, w: 20, h: WORLD },
      objects: [
        obj("A", "disk", 18, 16, 50, 1.65, 0, 4),
        obj("B", "square", 208, 82, 30, -0.04, 0.22, 6.5),
      ],
    };
  }

  if (scenario === "binding") {
    return {
      ...base,
      noise: 4,
      baseNoise: 4,
      swapAt: 16,
      occluder: { x: 70, y: 8, w: 12, h: 22 },
      objects: [
        obj("A", "disk", 16, 14, 38, 1.7, 0.35, 4.2),
        obj("B", "square", 206, 14, 64, 1.7, -0.35, 4.2),
      ],
    };
  }

  if (scenario === "conflict") {
    return {
      ...base,
      noise: 5,
      baseNoise: 5,
      occluder: { x: 8, y: 78, w: 16, h: 14 },
      objects: [
        obj("A", "disk", 18, 30, 50, 0.08, 0.05, 3.4),
        obj("B", "triangle", 34, 40, 50, -0.05, 0.06, 3.4),
        obj("C", "square", 200, 78, 50, 0.02, -0.08, 8),
      ],
    };
  }

  return {
    ...base,
    noise: 8,
    baseNoise: 8,
    occluder: { x: 40, y: 30, w: 18, h: 40 },
    objects: [
      obj("A", "disk", 12, 22, 58, 0.85, -0.35, 4.2),
      obj("B", "square", 200, 72, 24, -0.55, 0.4, 5),
      obj("C", "triangle", 128, 28, 22, 0.45, 0.55, 4.4),
    ],
  };
}

export function initialGaze(scenario: ScenarioId): { x: number; y: number; aperture: number } {
  if (scenario === "continuity") return { x: 16, y: 50, aperture: 20 };
  if (scenario === "binding") return { x: 24, y: 51, aperture: 28 };
  if (scenario === "conflict") return { x: 74, y: 50, aperture: 24 };
  return { x: 50, y: 48, aperture: 30 };
}

export function advanceWorld(world: World): void {
  world.t += 1;
  world.light = Math.sin(world.t * 0.08) * 20;
  for (const o of world.objects) {
    o.x += o.vx;
    o.y += o.vy;
    if (o.x < o.r) {
      o.x = o.r;
      o.vx = Math.abs(o.vx);
    } else if (o.x > WORLD - o.r) {
      o.x = WORLD - o.r;
      o.vx = -Math.abs(o.vx);
    }
    if (o.y < o.r) {
      o.y = o.r;
      o.vy = Math.abs(o.vy);
    } else if (o.y > WORLD - o.r) {
      o.y = WORLD - o.r;
      o.vy = -Math.abs(o.vy);
    }
  }
  if (world.swapAt != null && world.t === world.swapAt && world.objects.length >= 2) {
    const hue = world.objects[0]!.hue;
    world.objects[0]!.hue = world.objects[1]!.hue;
    world.objects[1]!.hue = hue;
    world.swapped = true;
  }
}

export function apparentHue(intrinsic: number, light: number): number {
  return wrapHue(intrinsic + light);
}
