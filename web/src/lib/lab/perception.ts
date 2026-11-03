import { WORLD, clamp, hueDelta, hypot, lerpHue, wrapHue } from "./math.ts";
import { classify, extractBlobs, locate, type Location } from "./observe.ts";
import type {
  Ablation,
  Agreement,
  AttentionReason,
  AttentionState,
  Blob,
  EnergyBreakdown,
  Expectation,
  LabConfig,
  Observation,
  Track,
  ViewState,
} from "./types.ts";
const BUDGET = 4;
const WORKSPACE = 3;
const MAX_TRACKS = 5;
const WIDE = 22;
const NARROW = 13;

interface Decision {
  x: number;
  y: number;
  aperture: number;
  reason: AttentionReason;
  focusId: number | null;
  note: string;
}

export interface Agent {
  tracks: Track[];
  nextId: number;
  gazeX: number;
  gazeY: number;
  aperture: number;
  lightEstimate: number;
  cycle: number;
  attention: AttentionState;
  energy: EnergyBreakdown;
  guesses: Expectation[];
  wallMap: Uint8Array;
  prevU: Record<number, number>;
  focusPrevId: number | null;
  predictedDrop: number;
  predictedObserved: boolean;
  selfError: number | null;
  iterationBudget: number;
  iterationsUsed: number;
  energyTrace: number[];
  history: { step: number; total: number }[];
  stepIndex: number;
  cells: ViewState["cells"];
  grid: number;
  blobs: Blob[];
  obsGaze: { x: number; y: number; aperture: number };
  workspaceCapacity: number;
  salienceFrames: (Blob | null)[];
  focusHold: number;
}

function emptyEnergy(): EnergyBreakdown {
  return {
    sensory: 0,
    binding: 0,
    temporal: 0,
    action: 0,
    self: null,
    regularization: 0,
    total: 0,
    overconfident: false,
  };
}

function emptyAttention(x: number, y: number, aperture: number): AttentionState {
  return {
    active: true,
    influencesAction: true,
    focusId: null,
    reason: "idle",
    agreement: "underdetermined",
    intendedX: x,
    intendedY: y,
    intendedAperture: aperture,
    executedX: x,
    executedY: y,
    executedAperture: aperture,
    note: "Контур только открыл глаза.",
    predictedObserved: false,
    predictedDrop: 0,
    selfError: null,
  };
}

function spawnFromBlob(ep: Agent, blob: Blob): void {
  if (ep.tracks.length >= MAX_TRACKS) return;
  if (ep.tracks.some((t) => hypot(t.x - blob.cx, t.y - blob.cy) < 9)) return;
  const hue = wrapHue(blob.hue - ep.lightEstimate);
  ep.tracks.push({
    id: ep.nextId++,
    x: blob.cx,
    y: blob.cy,
    vx: 0,
    vy: 0,
    hue,
    fill: blob.fill,
    evidence: 0.58,
    prediction: 0.3,
    uncertainty: 0.48,
    confidence: 0.3,
    source: "observed",
    reason: "none",
    unseen: 0,
    age: 0,
    boundBlob: blob.index,
    conflict: false,
    inWorkspace: false,
    radius: Math.sqrt(blob.area / Math.PI) * (2 * ep.obsGaze.aperture / ep.grid),
    flowError: 0,
  });
}

function explainMiss(obs: Observation, x: number, y: number): Location {
  const here = locate(obs, x, y);
  if (here.status === "cell" && here.kind === "empty") {
    for (const [dx, dy] of [
      [5, 0],
      [-5, 0],
      [0, 5],
      [0, -5],
      [7, 0],
      [-7, 0],
      [0, 7],
      [0, -7],
    ]) {
      const near = locate(obs, x + dx, y + dy);
      if (near.status === "cell" && near.kind === "occluder") return near;
    }
  }
  return here;
}

function finishTrack(t: Track): void {
  const src =
    t.source === "imagined" ? 0.55 : t.source === "predicted" ? 0.78 : t.source === "mixed" ? 0.7 : 1;
  t.confidence = clamp(t.evidence * (1 - t.uncertainty * 0.85) * src);
}

function agreementOf(tracks: Track[], blobs: Blob[]): Agreement {
  if (tracks.some((t) => t.conflict)) return "competing";
  const unbound = blobs.filter((b) => !tracks.some((t) => t.boundBlob === b.index)).length;
  const meanU = tracks.length
    ? tracks.reduce((s, t) => s + t.uncertainty, 0) / tracks.length
    : 1;
  if (unbound > 0 || meanU > 0.42) return "underdetermined";
  return "settled";
}

function perceive(ep: Agent, blobs: Blob[], obs: Observation): void {
  const trace: number[] = [];
  const origin = new Map<number, { x: number; y: number; vx: number; vy: number }>();
  for (const t of ep.tracks) {
    origin.set(t.id, { x: t.x, y: t.y, vx: t.vx, vy: t.vy });
    t.boundBlob = null;
    t.conflict = false;
    t.x += t.vx;
    t.y += t.vy;
  }
  const sampled = new Set<number>();

  let used = 0;
  for (let iter = 0; iter < ep.iterationBudget; iter++) {
    used++;
    const claimed = new Set<number>();
    const order = [...ep.tracks].sort((a, b) => b.evidence - a.evidence);
    for (const t of order) {
      let best = -1;
      let bestCost = 1e9;
      let runner = 1e9;
      for (const blob of blobs) {
        if (claimed.has(blob.index)) continue;
        const dist = hypot(blob.cx - t.x, blob.cy - t.y);
        if (dist > 16) continue;
        const expected = wrapHue(t.hue + ep.lightEstimate);
        const cost =
          (dist / 18) * 1.2 + (hueDelta(expected, blob.hue) / 180) * 0.55 + Math.abs(t.fill - blob.fill) * 0.2;
        if (cost < bestCost) {
          runner = bestCost;
          bestCost = cost;
          best = blob.index;
        } else if (cost < runner) runner = cost;
      }
      if (best >= 0 && bestCost < 1.28) {
        claimed.add(best);
        const blob = blobs.find((b) => b.index === best)!;
        const from = origin.get(t.id)!;
        const alpha = 0.42 + iter * 0.08;
        t.x = t.x * (1 - alpha) + blob.cx * alpha;
        t.y = t.y * (1 - alpha) + blob.cy * alpha;
        if (!sampled.has(t.id)) {
          const obsVx = blob.cx - from.x;
          const obsVy = blob.cy - from.y;
          t.vx = from.vx * 0.35 + obsVx * 0.65;
          t.vy = from.vy * 0.35 + obsVy * 0.65;
          const speed = hypot(t.vx, t.vy);
          if (speed > 3.2) {
            t.vx = (t.vx / speed) * 3.2;
            t.vy = (t.vy / speed) * 3.2;
          }
          t.flowError = hypot(obsVx - from.vx, obsVy - from.vy);
          const intrinsic = wrapHue(blob.hue - ep.lightEstimate);
          const hueResidual = hueDelta(wrapHue(t.hue + ep.lightEstimate), blob.hue);
          t.hue = lerpHue(t.hue, intrinsic, 0.32);
          t.fill = t.fill * 0.65 + blob.fill * 0.35;
          t.evidence = clamp(t.evidence * 0.45 + 0.55);
          t.prediction = clamp(1 - bestCost / 1.28);
          t.uncertainty = clamp(t.uncertainty * 0.62 + (bestCost / 1.28) * 0.22);
          t.reason = hueResidual > 32 ? "noise" : "none";
          t.radius = Math.sqrt(blob.area / Math.PI) * (2 * obs.aperture / obs.grid);
          sampled.add(t.id);
        }
        t.source = "observed";
        t.unseen = 0;
        t.boundBlob = best;
        t.conflict = runner < bestCost * 1.22 && runner < 1.28;
      }
    }

    const unexplained =
      blobs.length === 0
        ? 0
        : blobs.filter((b) => !ep.tracks.some((t) => t.boundBlob === b.index)).length / blobs.length;
    const meanU = ep.tracks.length
      ? ep.tracks.reduce((s, t) => s + t.uncertainty, 0) / ep.tracks.length
      : 1;
    trace.push(clamp(unexplained * 0.6 + meanU * 0.4));
    const settled =
      !ep.tracks.some((t) => t.conflict) && unexplained === 0 && meanU < 0.3 && ep.tracks.length > 0;
    if (settled && iter >= 1) break;
  }

  for (const t of ep.tracks) {
    if (t.boundBlob == null || t.uncertainty > 0.4) continue;
    const blob = blobs.find((b) => b.index === t.boundBlob);
    if (!blob) continue;
    let residual = wrapHue(blob.hue - t.hue) - ep.lightEstimate;
    residual = ((residual + 540) % 360) - 180;
    ep.lightEstimate += residual * 0.08;
    if (ep.lightEstimate > 40) ep.lightEstimate = 40;
    if (ep.lightEstimate < -40) ep.lightEstimate = -40;
  }

  ep.iterationsUsed = used;
  ep.energyTrace = trace;

  for (const t of ep.tracks) {
    if (t.boundBlob != null) continue;
    t.unseen += 1;
    const loc = explainMiss(obs, t.x, t.y);
    if (loc.status === "outside") {
      t.source = t.unseen > 10 ? "imagined" : "predicted";
      t.uncertainty = clamp(t.uncertainty + 0.08);
      t.evidence = clamp(t.evidence - 0.03);
      t.reason = "no-data";
      t.prediction = clamp(t.prediction * 0.94);
    } else if (loc.kind === "occluder") {
      t.source = "predicted";
      t.uncertainty = clamp(t.uncertainty + 0.02);
      t.evidence = clamp(t.evidence - 0.005);
      t.reason = "occlusion";
      t.prediction = Math.max(t.prediction, 0.55);
    } else if (loc.kind === "empty") {
      t.source = "predicted";
      t.uncertainty = clamp(t.uncertainty + 0.16);
      t.evidence = clamp(t.evidence * 0.72);
      t.reason = "absence";
      t.prediction = clamp(t.prediction * 0.65);
    } else {
      t.source = "mixed";
      t.uncertainty = clamp(t.uncertainty + 0.06);
      t.reason = "conflict";
      t.conflict = true;
    }
  }

  for (const blob of blobs) {
    if (ep.tracks.some((t) => t.boundBlob === blob.index)) continue;
    if (ep.tracks.some((t) => hypot(t.x - blob.cx, t.y - blob.cy) < 9)) continue;
    spawnFromBlob(ep, blob);
  }

  ep.tracks = ep.tracks.filter((t) => !(t.age > 2 && (t.uncertainty > 0.92 || t.evidence < 0.08)));
  for (const t of ep.tracks) {
    t.age += 1;
    finishTrack(t);
  }

}

function selectWorkspace(ep: Agent): void {
  const ranked = [...ep.tracks].sort((a, b) => {
    const score = (t: Track) =>
      t.confidence + (t.conflict ? 0.22 : 0) + (t.reason === "occlusion" ? 0.18 : 0);
    return score(b) - score(a);
  });
  const keep = new Set(ranked.slice(0, ep.workspaceCapacity).map((t) => t.id));
  for (const t of ep.tracks) t.inWorkspace = keep.has(t.id);
}

const WAYPOINTS = [
  { x: 82, y: 32 },
  { x: 34, y: 50 },
  { x: 50, y: 18 },
  { x: 18, y: 78 },
];

function explore(cycle: number): Decision {
  const point = WAYPOINTS[Math.floor(cycle / 4) % WAYPOINTS.length]!;
  return {
    x: point.x,
    y: point.y,
    aperture: WIDE,
    reason: "explore",
    focusId: null,
    note: "В окне пусто. Обход по четырём точкам, без модели того, что искать.",
  };
}

function priority(t: Track, gazeX: number, gazeY: number, aperture: number): number {
  let score = t.uncertainty;
  if (t.reason === "occlusion" || t.conflict) score += 1.05;
  if (t.reason === "absence") score += 0.55;
  if (t.reason === "no-data") score += 0.22;
  if (t.source === "predicted" || t.source === "imagined") score += 0.18;
  const inside = hypot(t.x - gazeX, t.y - gazeY) <= aperture;
  if (inside && (t.uncertainty > 0.3 || t.conflict)) score += 0.35;
  if (t.source === "observed" && t.uncertainty < 0.22 && t.unseen === 0 && !t.conflict) score -= 0.2;
  return score;
}

function schemaDecision(ep: Agent): Decision {
  const workspace = ep.tracks.filter((t) => t.inWorkspace);
  const held = ep.focusHold > 0 ? workspace.find((t) => t.id === ep.attention.focusId) : undefined;
  if (held && (held.reason === "occlusion" || held.conflict || held.uncertainty > 0.18)) {
    ep.focusHold -= 1;
    const reason: AttentionReason = held.reason === "occlusion" || held.source !== "observed" ? "verify" : held.conflict ? "conflict" : "uncertainty";
    return {
      x: clamp(held.x, 4, 96),
      y: clamp(held.y, 4, 96),
      aperture: held.conflict ? NARROW : 18,
      reason,
      focusId: held.id,
      note: "Проверка ещё не закрыта – не переключаюсь на более яркое.",
    };
  }

  const conflicts = workspace.filter((t) => t.conflict);
  if (conflicts.length > 0) {
    const x = conflicts.reduce((s, t) => s + t.x, 0) / conflicts.length;
    const y = conflicts.reduce((s, t) => s + t.y, 0) / conflicts.length;
    return {
      x: clamp(x, 4, 96),
      y: clamp(y, 4, 96),
      aperture: NARROW,
      reason: "conflict",
      focusId: conflicts[0]!.id,
      note: "Конфликт гипотез. Узкое окно – проверить, одна это сущность или две.",
    };
  }

  let best: Track | null = null;
  let bestScore = -1;
  for (const t of workspace) {
    const score = priority(t, ep.gazeX, ep.gazeY, ep.aperture);
    if (score > bestScore) {
      bestScore = score;
      best = t;
    }
  }
  if (best && bestScore > 0.42) {
    const reason: AttentionReason =
      best.reason === "occlusion" || best.source === "predicted" || best.source === "imagined"
        ? "verify"
        : "uncertainty";
    return {
      x: clamp(best.x, 4, 96),
      y: clamp(best.y, 4, 96),
      aperture: best.reason === "noise" || best.conflict ? NARROW : 18,
      reason,
      focusId: best.id,
      note:
        reason === "verify"
          ? "Прогноз не подтверждён взглядом. Иду туда, где гипотеза должна проявиться."
          : "Неопределённость выше порога. Смотрю на неё, а не на самый яркий кусок.",
    };
  }

  const settled = workspace.length > 0 && workspace.every((t) => t.unseen === 0 && t.uncertainty < 0.34 && !t.conflict);
  if (ep.cycle > 0 && ep.cycle % 8 === 0 && settled) {
    const p = WAYPOINTS[Math.floor(ep.cycle / 8) % WAYPOINTS.length]!;
    return {
      x: p.x,
      y: p.y,
      aperture: WIDE,
      reason: "explore",
      focusId: null,
      note: "Текущая сцена устойчива. Короткий обход – не пропущена ли другая сущность.",
    };
  }

  const pursued = workspace.filter((t) => t.source === "observed" && t.boundBlob != null);
  if (pursued.length > 0) {
    const sticky = pursued.find((t) => t.id === ep.attention.focusId) ?? pursued[0]!;
    return {
      x: clamp(sticky.x, 4, 96),
      y: clamp(sticky.y, 4, 96),
      aperture: 18,
      reason: "pursue",
      focusId: sticky.id,
      note: "Гипотеза устойчива. Сопровождаю её, пока не появится дисгармония.",
    };
  }
  return explore(ep.cycle);
}

function firstOrderDecision(ep: Agent): Decision {
  const focus = ep.tracks.find((t) => t.id === ep.attention.focusId)
    ?? [...ep.tracks].sort((a, b) => b.evidence - a.evidence)[0];
  if (!focus) return { ...explore(ep.cycle), aperture: 18 };
  return {
    x: clamp(focus.x + focus.vx, 4, 96),
    y: clamp(focus.y + focus.vy, 4, 96),
    aperture: 18,
    reason: "pursue",
    focusId: focus.id,
    note: "Взгляд следует прогнозу положения выбранного трека.",
  };
}

function salienceDecision(ep: Agent, useMemory: boolean): Decision {
  const largest = [...ep.blobs].sort((a, b) => b.area - a.area)[0] ?? null;
  if (useMemory) {
    // One salient patch per frame, with the same slot budget as the workspace.
    ep.salienceFrames.push(largest);
    if (ep.salienceFrames.length > ep.workspaceCapacity) ep.salienceFrames.shift();
  }
  const blob = largest ?? [...ep.salienceFrames].reverse().find((patch) => patch != null);
  if (!blob) {
    const wander = explore(ep.cycle);
    return { ...wander, reason: "salience", note: "Без модели внимания: в окне пусто, иду по дуге обхода." };
  }
  return {
    x: clamp(blob.cx, 4, 96),
    y: clamp(blob.cy, 4, 96),
    aperture: WIDE,
    reason: "salience",
    focusId: largest ? ep.tracks.find((t) => t.boundBlob === largest.index)?.id ?? null : null,
    note: largest ? "Взгляд направлен к самому крупному пятну." : "В окне пусто. Проверяю место последнего крупного пятна.",

  };
}

function gradeAction(ep: Agent, obs: Observation): number {
  let bad = 0;
  for (const [i, cell] of obs.cells.entries()) {
    if (ep.guesses[i] !== classify(cell)) bad++;
  }
  return bad / obs.cells.length;
}

function energies(ep: Agent, blobs: Blob[], action: number, selfActive: boolean): EnergyBreakdown {
  const unbound =
    blobs.length === 0
      ? 0
      : blobs.filter((b) => !ep.tracks.some((t) => t.boundBlob === b.index)).length / blobs.length;
  let hueAcc = 0;
  let hueN = 0;
  let flowAcc = 0;
  let flowN = 0;
  for (const t of ep.tracks) {
    if (t.boundBlob == null) continue;
    const blob = blobs.find((b) => b.index === t.boundBlob);
    if (!blob) continue;
    hueAcc += hueDelta(wrapHue(t.hue + ep.lightEstimate), blob.hue) / 180;
    hueN++;
    flowAcc += Math.min(1, t.flowError / 4);
    flowN++;
  }
  const silence = blobs.length > 0 && ep.tracks.length === 0 ? 0.85 : 0;
  const sensory = clamp(unbound * 0.7 + (hueN ? hueAcc / hueN : 0) * 0.45 + silence);
  const binding = clamp(ep.tracks.filter((t) => t.conflict).length / 2);
  const temporal = flowN ? clamp(flowAcc / flowN) : ep.tracks.some((t) => t.unseen > 0) ? 0.2 : 0;
  const meanConf = ep.tracks.length
    ? ep.tracks.reduce((s, t) => s + t.confidence, 0) / ep.tracks.length
    : 0;
  const overconfident = meanConf > 0.82 && sensory > 0.38;
  const regularization = clamp(
    (ep.tracks.length / MAX_TRACKS) * 0.45 +
      (ep.iterationsUsed / BUDGET) * 0.2 +
      (overconfident ? 0.45 : 0) +
      (silence > 0 ? 0.25 : 0),
  );
  const self = selfActive ? ep.selfError : null;
  const wS = 0.3;
  const wB = 0.16;
  const wT = 0.16;
  const wA = 0.18;
  const wM = 0.12;
  const wR = 0.08;
  const partial = wS * sensory + wB * binding + wT * temporal + wA * action + wR * regularization;
  const total = self == null ? partial / (1 - wM) : partial + wM * self;
  return {
    sensory,
    binding,
    temporal,
    action,
    self,
    regularization,
    total: clamp(total),
    overconfident,
  };
}

/** Project only the samples just seen into a persistent map of the stationary wall. */
function rememberWall(ep: Agent, obs: Observation): void {
  const size = 2 * obs.aperture / obs.grid;
  for (const [i, cell] of obs.cells.entries()) {
    const left = obs.gazeX - obs.aperture + (i % obs.grid) * size;
    const top = obs.gazeY - obs.aperture + Math.floor(i / obs.grid) * size;
    const x0 = clamp(Math.ceil(left - 0.5), 0, WORLD);
    const x1 = clamp(Math.ceil(left + size - 0.5), 0, WORLD);
    const bottom = Math.min(WORLD, Math.ceil(top + size - 0.5));
    const wall = classify(cell) === "occluder" ? 1 : 0;
    for (let y = Math.max(0, Math.ceil(top - 0.5)); y < bottom; y++) {
      ep.wallMap.fill(wall, y * WORLD + x0, y * WORLD + x1);
    }
  }
}

function buildGuesses(ep: Agent, gazeX: number, gazeY: number, aperture: number): Expectation[] {
  const size = 2 * aperture / ep.grid;
  return Array.from({ length: ep.grid ** 2 }, (_, i) => {
    const x = gazeX - aperture + ((i % ep.grid) + 0.5) * size;
    const y = gazeY - aperture + (Math.floor(i / ep.grid) + 0.5) * size;
    if (x < 0 || y < 0 || x >= WORLD || y >= WORLD) return "empty";
    if (ep.wallMap[Math.floor(y) * WORLD + Math.floor(x)]) return "occluder";
    const covered = ep.tracks.some((t) => hypot(x - t.x - t.vx, y - t.y - t.vy) <= t.radius);
    return covered ? "surface" : "empty";
  });
}

function selfPrediction(ep: Agent, decision: Decision): { observed: boolean; drop: number } {
  const focus = ep.tracks.find((t) => t.id === decision.focusId);
  if (!focus) return { observed: false, drop: 0 };
  const inside = hypot(focus.x - decision.x, focus.y - decision.y) <= decision.aperture;
  const observed = inside && focus.reason !== "occlusion" && focus.reason !== "absence";
  const drop = !inside ? -0.07 : focus.reason === "occlusion" ? -0.02 : focus.uncertainty * 0.45;
  return { observed, drop };
}

export function createAgent(obs: Observation): Agent {
  const gaze = { x: obs.gazeX, y: obs.gazeY, aperture: obs.aperture };
  const ep: Agent = {
    tracks: [],
    nextId: 1,
    gazeX: gaze.x,
    gazeY: gaze.y,
    aperture: gaze.aperture,
    lightEstimate: 0,
    cycle: 0,
    attention: emptyAttention(gaze.x, gaze.y, gaze.aperture),
    energy: emptyEnergy(),
    guesses: [],
    wallMap: new Uint8Array(WORLD ** 2),
    prevU: {},
    focusPrevId: null,
    predictedDrop: 0,
    predictedObserved: false,
    selfError: null,
    iterationBudget: BUDGET,
    iterationsUsed: 0,
    energyTrace: [],
    history: [],
    stepIndex: 0,
    cells: [],
    grid: 12,
    blobs: [],
    obsGaze: gaze,
    workspaceCapacity: WORKSPACE,
    salienceFrames: [],
    focusHold: 0,
  };
  ep.cells = obs.cells;
  ep.grid = obs.grid;
  ep.blobs = extractBlobs(obs);
  perceive(ep, ep.blobs, obs);
  selectWorkspace(ep);
  rememberWall(ep, obs);
  ep.guesses = buildGuesses(ep, gaze.x, gaze.y, gaze.aperture);
  ep.energy = energies(ep, ep.blobs, 0, true);
  ep.attention.agreement = agreementOf(ep.tracks, ep.blobs);
  return ep;
}

export type AgentControls = Pick<LabConfig, "ablation" | "pinGaze" | "pinX" | "pinY" | "injectConfidence">;

export function stepAgent(ep: Agent, obs: Observation, cfg: AgentControls): void {
  const actionError = gradeAction(ep, obs);
  ep.blobs = extractBlobs(obs);
  ep.cells = obs.cells;
  ep.grid = obs.grid;
  ep.obsGaze = { x: obs.gazeX, y: obs.gazeY, aperture: obs.aperture };
  rememberWall(ep, obs);

  const prevFocus = ep.focusPrevId;
  const beforeU = prevFocus == null ? null : (ep.prevU[prevFocus] ?? null);
  perceive(ep, ep.blobs, obs);

  const schemaOn = cfg.ablation === "full" || cfg.ablation === "record-only";
  if (schemaOn && prevFocus != null && beforeU != null) {
    const now = ep.tracks.find((t) => t.id === prevFocus);
    const actualDrop = now ? beforeU - now.uncertainty : -0.45;
    const actualObserved = !!now && now.boundBlob != null && now.source === "observed";
    const miss = ep.predictedObserved === actualObserved ? 0 : 0.7;
    ep.selfError = clamp(miss + 0.3 * Math.min(1, Math.abs(ep.predictedDrop - actualDrop)));

  } else {
    ep.selfError = null;
  }

  if (cfg.injectConfidence && schemaOn) {
    for (const t of ep.tracks) {
      t.uncertainty = 0.02;
      t.confidence = 0.98;
      t.conflict = false;
      t.reason = "none";
    }
  }
  selectWorkspace(ep);
  ep.energy = energies(ep, ep.blobs, actionError, schemaOn);
  ep.history.push({ step: ep.stepIndex, total: ep.energy.total });
  if (ep.history.length > 48) ep.history.shift();

  const ablation: Ablation = cfg.ablation;
  if (!schemaOn || cfg.pinGaze) ep.focusHold = 0;
  if (ablation !== "dummy-memory") ep.salienceFrames = [];
  const schema = schemaOn ? schemaDecision(ep) : null;
  let executed: Decision;
  if (cfg.pinGaze) {
    executed = {
      x: cfg.pinX,
      y: cfg.pinY,
      aperture: ep.aperture,
      reason: "pinned",
      focusId: schema?.focusId ?? null,
      note: "Вмешательство: взгляд зафиксирован и не следует модели.",
    };
  } else if (ablation === "full" && schema) {
    executed = schema;
  } else if (ablation === "first-order") {
    executed = firstOrderDecision(ep);
  } else {
    executed = salienceDecision(ep, ablation === "dummy-memory");
  }

  if (schema && schema.focusId !== ep.attention.focusId) ep.focusHold = 4;
  const pred = schemaOn ? selfPrediction(ep, executed) : { observed: false, drop: 0 };
  ep.predictedObserved = pred.observed;
  ep.predictedDrop = pred.drop;
  ep.focusPrevId = schemaOn ? executed.focusId : null;
  ep.prevU = {};
  for (const t of ep.tracks) ep.prevU[t.id] = t.uncertainty;

  const influences = ablation === "full" && !cfg.pinGaze;
  ep.attention = {
    active: schemaOn,
    influencesAction: influences,
    focusId: schema ? schema.focusId : executed.focusId,
    reason: influences ? executed.reason : (schema?.reason ?? executed.reason),
    agreement: agreementOf(ep.tracks, ep.blobs),
    intendedX: schema?.x ?? executed.x,
    intendedY: schema?.y ?? executed.y,
    intendedAperture: schema?.aperture ?? executed.aperture,
    executedX: executed.x,
    executedY: executed.y,
    executedAperture: executed.aperture,
    note: executed.note,
    predictedObserved: pred.observed,
    predictedDrop: pred.drop,
    selfError: ep.selfError,
  };

  ep.gazeX = clamp(executed.x, 2, 98);
  ep.gazeY = clamp(executed.y, 2, 98);
  ep.aperture = executed.aperture;
  ep.guesses = buildGuesses(ep, ep.gazeX, ep.gazeY, ep.aperture);
  ep.cycle += 1;
  ep.stepIndex += 1;
}
