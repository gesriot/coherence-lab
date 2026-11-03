import { clamp, hueDelta, hypot, lerpHue, wrapHue } from "./math";
import { extractBlobs, locate, observe, reinterpret, type Location } from "./observe";
import type {
  Ablation,
  Agreement,
  AttentionReason,
  AttentionState,
  Blob,
  EnergyBreakdown,
  Expectation,
  LabConfig,
  ScenarioId,
  Track,
  ViewState,
} from "./types";
import { advanceWorld, hiddenByOccluder, initialGaze, makeWorld } from "./world";

const BUDGET = 4;
const WORKSPACE = 3;
const MAX_TRACKS = 5;
const WIDE = 22;
const NARROW = 13;

interface Guess {
  id: number;
  x: number;
  y: number;
  expect: Expectation;
}

interface Decision {
  x: number;
  y: number;
  aperture: number;
  reason: AttentionReason;
  focusId: number | null;
  note: string;
}

export interface Episode {
  scenario: ScenarioId;
  seed: number;
  world: ReturnType<typeof makeWorld>;
  tracks: Track[];
  nextId: number;
  gazeX: number;
  gazeY: number;
  aperture: number;
  lightEstimate: number;
  cycle: number;
  attention: AttentionState;
  energy: EnergyBreakdown;
  guesses: Guess[];
  prevU: Record<number, number>;
  focusPrevId: number | null;
  predictedDrop: number;
  predictedObserved: boolean;
  selfError: number | null;
  iterationsUsed: number;
  energyTrace: number[];
  history: { step: number; total: number }[];
  recording: ReturnType<typeof observe>[];
  clip: ReturnType<typeof observe>[] | null;
  replayIndex: number;
  initialTrackId: number | null;
  stepIndex: number;
  cells: ViewState["cells"];
  grid: number;
  blobs: Blob[];
  obsGaze: { x: number; y: number; aperture: number };
  coupling: "intact" | "broken";
  memoryFrames: number;
  focusHold: number;
  configEcho: ViewState["configEcho"];
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

function spawnFromBlob(ep: Episode, blob: Blob): void {
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
    flowError: 0,
  });
}

function explainMiss(obs: ReturnType<typeof observe>, x: number, y: number): Location {
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

function perceive(ep: Episode, blobs: Blob[], obs: ReturnType<typeof observe>): void {
  const trace: number[] = [];
  const origin = new Map<number, { x: number; y: number; vx: number; vy: number }>();
  for (const t of ep.tracks) {
    origin.set(t.id, { x: t.x, y: t.y, vx: t.vx, vy: t.vy });
    t.boundBlob = null;
    t.conflict = false;
    t.x += t.vx;
    t.y += t.vy;
  }
  const velocityLatched = new Set<number>();

  let used = 0;
  for (let iter = 0; iter < BUDGET; iter++) {
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
        if (!velocityLatched.has(t.id)) {
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
          velocityLatched.add(t.id);
        }
        const intrinsic = wrapHue(blob.hue - ep.lightEstimate);
        const hueResidual = hueDelta(wrapHue(t.hue + ep.lightEstimate), blob.hue);
        t.hue = lerpHue(t.hue, intrinsic, 0.32);
        t.fill = t.fill * 0.65 + blob.fill * 0.35;
        t.evidence = clamp(t.evidence * 0.45 + 0.55);
        t.prediction = clamp(1 - bestCost / 1.28);
        t.uncertainty = clamp(t.uncertainty * 0.62 + (bestCost / 1.28) * 0.22);
        t.source = "observed";
        t.unseen = 0;
        t.boundBlob = best;
        t.reason = hueResidual > 32 ? "noise" : "none";
        t.conflict = runner < bestCost * 1.22 && runner < 1.28;
      }
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

  const ranked = [...ep.tracks].sort((a, b) => {
    const score = (t: Track) =>
      t.confidence + (t.conflict ? 0.22 : 0) + (t.reason === "occlusion" ? 0.18 : 0);
    return score(b) - score(a);
  });
  const keep = new Set(ranked.slice(0, WORKSPACE).map((t) => t.id));
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

function schemaDecision(ep: Episode): Decision {
  const held = ep.focusHold > 0 ? ep.tracks.find((t) => t.id === ep.attention.focusId) : undefined;
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

  const conflicts = ep.tracks.filter((t) => t.conflict);
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
  for (const t of ep.tracks) {
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

  const settled = ep.tracks.length > 0 && ep.tracks.every((t) => t.unseen === 0 && t.uncertainty < 0.34 && !t.conflict);
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

  const pursued = ep.tracks.filter((t) => t.source === "observed" && t.boundBlob != null);
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

function salienceDecision(ep: Episode): Decision {
  if (ep.blobs.length === 0) {
    const wander = explore(ep.cycle);
    return { ...wander, reason: "salience", note: "Без модели внимания: в окне пусто, иду по дуге обхода." };
  }
  const blob = [...ep.blobs].sort((a, b) => b.area - a.area)[0]!;
  return {
    x: clamp(blob.cx, 4, 96),
    y: clamp(blob.cy, 4, 96),
    aperture: WIDE,
    reason: "salience",
    focusId: null,
    note: "Без влияния модели: взгляд прилип к самой большой видимой области.",
  };
}

function retinalSlip(obs: ReturnType<typeof observe>): number {
  const slip = hypot(obs.gazeX - obs.capturedGazeX, obs.gazeY - obs.capturedGazeY);
  if (slip < 5) return 0;
  return clamp((slip - 5) / 18);
}

function gradeAction(ep: Episode, obs: ReturnType<typeof observe>): number {
  if (ep.guesses.length === 0) return 0;
  let bad = 0;
  for (const g of ep.guesses) {
    const loc = locate(obs, g.x, g.y);
    const actual: Expectation =
      loc.status === "outside"
        ? "outside"
        : loc.kind === "occluder"
          ? "occluder"
          : loc.kind === "surface"
            ? "surface"
            : "empty";
    if (actual !== g.expect) bad++;
  }
  return bad / ep.guesses.length;
}

function energies(ep: Episode, blobs: Blob[], action: number, selfActive: boolean): EnergyBreakdown {
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

function buildGuesses(ep: Episode, gazeX: number, gazeY: number, aperture: number): Guess[] {
  return ep.tracks.map((t) => {
    const x = t.x + t.vx;
    const y = t.y + t.vy;
    const inside = hypot(x - gazeX, y - gazeY) <= aperture * 0.96;
    let expect: Expectation = "outside";
    if (inside) {
      if (t.reason === "occlusion") expect = "occluder";
      else if (t.reason === "absence") expect = "empty";
      else expect = "surface";
    }
    return { id: t.id, x, y, expect };
  });
}

function selfPrediction(ep: Episode, decision: Decision): { observed: boolean; drop: number } {
  const focus = ep.tracks.find((t) => t.id === decision.focusId);
  if (!focus) return { observed: false, drop: 0 };
  const inside = hypot(focus.x - decision.x, focus.y - decision.y) <= decision.aperture;
  const observed = inside && focus.reason !== "occlusion" && focus.reason !== "absence";
  const drop = !inside ? -0.07 : focus.reason === "occlusion" ? -0.02 : focus.uncertainty * 0.45;
  return { observed, drop };
}

export function createEpisode(scenario: ScenarioId, seed: number): Episode {
  const world = makeWorld(scenario, seed);
  const gaze = initialGaze(scenario);
  const ep: Episode = {
    scenario,
    seed,
    world,
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
    prevU: {},
    focusPrevId: null,
    predictedDrop: 0,
    predictedObserved: false,
    selfError: null,
    iterationsUsed: 0,
    energyTrace: [],
    history: [],
    recording: [],
    clip: null,
    replayIndex: 0,
    initialTrackId: null,
    stepIndex: 0,
    cells: [],
    grid: 12,
    blobs: [],
    obsGaze: gaze,
    coupling: "intact",
    memoryFrames: 0,
    focusHold: 0,
    configEcho: { pinGaze: false, injectConfidence: false, replay: false, extraNoise: 0 },
  };
  const obs = observe(world, gaze.x, gaze.y, gaze.aperture);
  ep.recording.push(obs);
  ep.cells = obs.cells;
  ep.grid = obs.grid;
  ep.blobs = extractBlobs(obs);
  perceive(ep, ep.blobs, obs);
  const target = world.objects[0];
  if (target) {
    let best: Track | null = null;
    let bestD = 1e9;
    for (const t of ep.tracks) {
      const d = hypot(t.x - target.x, t.y - target.y);
      if (d < bestD) {
        bestD = d;
        best = t;
      }
    }
    if (best && bestD < 14) ep.initialTrackId = best.id;
  }
  ep.energy = energies(ep, ep.blobs, 0, true);
  ep.attention.agreement = agreementOf(ep.tracks, ep.blobs);
  return ep;
}

export const defaultConfig = (): LabConfig => ({
  ablation: "full",
  pinGaze: false,
  pinX: 50,
  pinY: 50,
  injectConfidence: false,
  replay: false,
  extraNoise: 0,
});

export function stepEpisode(ep: Episode, cfg: LabConfig): void {
  ep.configEcho = {
    pinGaze: cfg.pinGaze,
    injectConfidence: cfg.injectConfidence,
    replay: cfg.replay,
    extraNoise: cfg.extraNoise,
  };
  ep.world.noise = ep.world.baseNoise + cfg.extraNoise;
  advanceWorld(ep.world);

  const obsGaze = { x: ep.gazeX, y: ep.gazeY, aperture: ep.aperture };
  let obs = observe(ep.world, obsGaze.x, obsGaze.y, obsGaze.aperture);
  let coupling: "intact" | "broken" = "intact";
  if (cfg.replay) {
    if (!ep.clip) ep.clip = ep.recording.slice(-28);
    if (ep.clip.length > 0) {
      const frame = ep.clip[ep.replayIndex % ep.clip.length]!;
      ep.replayIndex += 1;
      obs = reinterpret(frame, obsGaze.x, obsGaze.y, obsGaze.aperture);
      coupling = "broken";
    }
  } else {
    ep.clip = null;
    ep.replayIndex = 0;
    ep.recording.push(obs);
    if (ep.recording.length > 36) ep.recording.shift();
  }

  const actionError = Math.max(gradeAction(ep, obs), retinalSlip(obs));
  ep.blobs = extractBlobs(obs);
  ep.cells = obs.cells;
  ep.grid = obs.grid;
  ep.obsGaze = obsGaze;
  ep.coupling = coupling;

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
  } else if (schemaOn) {
    ep.selfError = ep.stepIndex === 0 ? 0 : ep.selfError;
  } else {
    ep.selfError = null;
  }

  ep.energy = energies(ep, ep.blobs, actionError, schemaOn);
  ep.history.push({ step: ep.stepIndex, total: ep.energy.total });
  if (ep.history.length > 48) ep.history.shift();

  const ablation: Ablation = cfg.ablation;
  if (cfg.ablation !== "full" || cfg.injectConfidence || cfg.pinGaze) ep.focusHold = 0;
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
    if (cfg.injectConfidence) {
      executed = {
        ...explore(ep.cycle),
        note: "Уверенность подменена. Контур считает сцену решённой и перестаёт проверять.",
      };
      executed.focusId = schema.focusId;
    }
  } else {
    executed = salienceDecision(ep);
  }

  const predictFrom = schema ?? executed;
  if (schema && schema.focusId !== ep.attention.focusId) ep.focusHold = 4;
  const pred = selfPrediction(ep, predictFrom);
  ep.predictedObserved = pred.observed;
  ep.predictedDrop = pred.drop;
  ep.focusPrevId = predictFrom.focusId;
  ep.prevU = {};
  for (const t of ep.tracks) ep.prevU[t.id] = t.uncertainty;

  const influences = ablation === "full" && !cfg.pinGaze && !cfg.injectConfidence;
  ep.attention = {
    active: schemaOn,
    influencesAction: influences,
    focusId: schema?.focusId ?? null,
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
  ep.memoryFrames = ablation === "dummy-memory" ? ep.recording.length : 0;
  ep.cycle += 1;
  ep.stepIndex += 1;
}

export function snapshot(ep: Episode, ablation: Ablation): ViewState {
  const objects = ep.world.objects.map((o) => ({ ...o }));
  const reports = objects.map((o) => {
    let nearest: Track | null = null;
    let dist = 1e9;
    for (const t of ep.tracks) {
      const d = hypot(t.x - o.x, t.y - o.y);
      if (d < dist) {
        dist = d;
        nearest = t;
      }
    }
    const occluded = hiddenByOccluder(o, ep.world.occluder);
    const close = nearest && dist < 12 ? nearest : null;
    return {
      id: o.id,
      bodyHue: o.hue,
      nearestTrack: close ? close.id : null,
      trackHue: close ? close.hue : null,
      dist: close ? dist : null,
      occluded,
      visible: !occluded,
    };
  });
  const target = reports[0];
  let retained: boolean | null = null;
  if (ep.initialTrackId != null && target) {
    if (target.occluded) retained = ep.tracks.some((t) => t.id === ep.initialTrackId);
    else if (ep.stepIndex > 2) retained = target.nearestTrack === ep.initialTrackId;
  }
  return {
    step: ep.stepIndex,
    scenario: ep.scenario,
    seed: ep.seed,
    ablation,
    objects,
    occluder: { ...ep.world.occluder },
    light: ep.world.light,
    noise: ep.world.noise,
    swapped: ep.world.swapped,
    obsGaze: ep.obsGaze,
    nextGaze: { x: ep.gazeX, y: ep.gazeY, aperture: ep.aperture },
    cells: ep.cells.map((c) => ({ ...c })),
    grid: ep.grid,
    blobs: ep.blobs.map((b) => ({ ...b })),
    tracks: ep.tracks.map((t) => ({ ...t })),
    attention: { ...ep.attention },
    energy: { ...ep.energy },
    energyTrace: [...ep.energyTrace],
    history: ep.history.map((h) => ({ ...h })),
    coupling: ep.coupling,
    memoryFrames: ep.memoryFrames,
    iterationsUsed: ep.iterationsUsed,
    iterationBudget: BUDGET,
    evaluator: {
      initialTrackId: ep.initialTrackId,
      retained,
      objects: reports,
      lightTrue: ep.world.light,
      lightEstimated: ep.lightEstimate,
    },
    configEcho: { ...ep.configEcho },
  };
}
