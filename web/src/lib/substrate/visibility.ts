import { contains } from "../lab/world.ts";
import type { Occluder, World } from "../lab/types.ts";
import { CELL_SIZE, GRID } from "./input.ts";

export function inWall(wall: Occluder, x: number, y: number) {
  return x >= wall.x && x <= wall.x + wall.w && y >= wall.y && y <= wall.y + wall.h;
}

// Age zero is the first frame with no visible cell of the living object.
export function visibility(world: World, previousAge: number) {
  const a = world.objects[0]!, alive = a.vanishAt === null || world.t < a.vanishAt;
  const visible = new Uint8Array(GRID * GRID);
  let count = 0;
  if (alive) {
    for (let i = 0; i < visible.length; i++) {
      const x = (i % GRID + 0.5) * CELL_SIZE, y = (Math.floor(i / GRID) + 0.5) * CELL_SIZE;
      if (!inWall(world.occluder, x, y) && contains(a, x, y)) {
        visible[i] = 1;
        count++;
      }
    }
  }
  return { visible, age: alive && !count ? previousAge + 1 : -1, exit: count > 0 && previousAge >= 0 };
}
