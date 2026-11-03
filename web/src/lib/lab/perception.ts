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
  focusPrevId: number | null;
  predictedDrop: number;
  predictedObserved: boolean;
  selfError: number | null;
  weights: typeof DEFAULT_WEIGHTS;
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

function newTrack(ep: Agent, blob: Blob): Track {
  const hue = wrapHue(blob.hue - ep.lightEstimate);
  const t: Track = {
    id: ep.nextId + blob.index,
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
    age: 1,
    boundBlob: blob.index,
    conflict: false,
    inWorkspace: false,
    radius: Math.sqrt(blob.area / Math.PI) * (2 * ep.obsGaze.aperture / ep.grid),
    flowError: 0,
  };
  finishTrack(t);
  return t;
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

function predictMiss(prior: Track, obs: Observation): Track {
  const t: Track = {
    ...prior,
    x: prior.x + prior.vx,
    y: prior.y + prior.vy,
    age: prior.age + 1,
    conflict: false,
    boundBlob: null,
  };
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
  finishTrack(t);
  return t;
}

/** Fixed frame priors: a different binding always starts from the same evidence. */
function matchTrack(ep: Agent, prior: Track, blob: Blob, obs: Observation): Track {
  const dx = blob.cx - prior.x;
  const dy = blob.cy - prior.y;
  const flowError = hypot(dx - prior.vx, dy - prior.vy);
  const hueError = hueDelta(prior.hue + ep.lightEstimate, blob.hue);
  const mismatch = clamp((flowError / 18 * 1.2 + hueError / 180 * 0.55
    + Math.abs(prior.fill - blob.fill) * 0.2) / 1.28);
  const vx = prior.vx * 0.35 + dx * 0.65;
  const vy = prior.vy * 0.35 + dy * 0.65;
  const scale = 3.2 / Math.max(3.2, hypot(vx, vy));
  const t: Track = {
    ...prior,
    x: blob.cx,
    y: blob.cy,
    vx: vx * scale,
    vy: vy * scale,
    hue: lerpHue(prior.hue, blob.hue - ep.lightEstimate, 0.32),
    fill: prior.fill * 0.65 + blob.fill * 0.35,
    evidence: prior.evidence * 0.45 + 0.55,
    prediction: 1 - mismatch,
    uncertainty: clamp(prior.uncertainty * 0.62 + mismatch * 0.22),
    reason: hueError > 32 ? "noise" : "none",
    source: "observed",
    age: prior.age + 1,
    boundBlob: blob.index,
    unseen: 0,
    conflict: false,
    flowError,
    radius: Math.sqrt(blob.area / Math.PI) * (2 * obs.aperture / obs.grid),
  };
  finishTrack(t);
  return t;
}

function perceive(ep: Agent, blobs: Blob[], obs: Observation, action: number, schemaOn: boolean): void {
  const priors = ep.tracks;
  const misses = priors.map((t) => predictMiss(t, obs));
  const matches = priors.map((prior, i) => {
    const options = blobs.map((blob) => {
      if (hypot(blob.cx - misses[i]!.x, blob.cy - misses[i]!.y) > 16) return null;
      return matchTrack(ep, prior, blob, obs);
    });
    for (const t of options) {
      if (!t) continue;
      t.conflict = options.some((alt) =>
        alt && alt !== t && Math.abs(alt.prediction - t.prediction) < 0.1);
      if (t.conflict) t.reason = "conflict";
    }
    return options;
  });
  // Grade the executed look against the new frame before testing scene interpretations.
  // This measures available sensory support even if the optimizer later rejects a binding.
  const focusIndex = priors.findIndex((t) => t.id === ep.focusPrevId);
  ep.selfError = null;
  if (schemaOn && focusIndex >= 0) {
    const support = matches[focusIndex]!.reduce<Track | null>((best, t) =>
      t && (!best || t.prediction > best.prediction) ? t : best, null);
    const observed = support != null;
    const actual = support ?? misses[focusIndex]!;
    const drop = priors[focusIndex]!.uncertainty - actual.uncertainty;
    const miss = ep.predictedObserved === observed ? 0 : 0.7;
    ep.selfError = clamp(miss + 0.3 * Math.min(1, Math.abs(ep.predictedDrop - drop)));
  }
  const births = blobs.map((blob) => newTrack(ep, blob));
  // Slots keep prior identities stable while testing deletions, rebindings and swaps.
  type Slot = Track | null;
  let slots: Slot[] = [...misses, ...births.map(() => null)];
  const score = (candidate: Slot[]) =>
    energies(ep, candidate.filter((t): t is Track => t != null), misses, blobs, action);
  let current = score(slots);
  const trace = [current.total];
  const epsilon = 1e-5;

  for (let iter = 0; iter < ep.iterationBudget; iter++) {
    let best = slots;
    let lowest = current;
    const consider = (candidate: Slot[]) => {
      const candidateEnergy = score(candidate);
      if (candidateEnergy.total < lowest.total) {
        lowest = candidateEnergy;
        best = candidate;
      }
    };

    for (let i = 0; i < priors.length; i++) {
      if (slots[i] && slots[i]!.boundBlob == null) {
        const removed = slots.slice();
        removed[i] = null;
        consider(removed);
      }
      for (const match of matches[i]!) {
        if (!match || slots[i] === match) continue;
        const candidate = slots.slice();
        const owner = slots.findIndex((t) => t && t.boundBlob === match.boundBlob);
        if (owner >= 0) {
          const previous = slots[i]?.boundBlob;
          candidate[owner] = misses[owner] ?? null;
          if (owner < priors.length && previous != null && matches[owner]![previous]) {
            candidate[owner] = matches[owner]![previous]!;
          }
        }
        candidate[i] = match;
        if (candidate.filter(Boolean).length <= MAX_TRACKS) consider(candidate);
      }
    }

    if (slots.filter(Boolean).length < MAX_TRACKS) {
      for (const birth of births) {
        if (slots.some((t) => t?.boundBlob === birth.boundBlob)) continue;
        const candidate = slots.slice();
        candidate[priors.length + birth.boundBlob!] = birth;
        consider(candidate);
      }
    }

    const improvement = current.total - lowest.total;
    slots = best;
    current = lowest;
    trace.push(current.total);
    if (improvement < epsilon) break;
  }

  ep.tracks = slots.filter((t): t is Track => t != null);
  for (const t of ep.tracks) ep.nextId = Math.max(ep.nextId, t.id + 1);
  ep.energy = current;
  ep.energyTrace = trace;
  ep.iterationsUsed = trace.length - 1;
  for (const t of ep.tracks) {
    if (t.boundBlob == null || t.uncertainty > 0.4) continue;
    const residual = ((blobs[t.boundBlob]!.hue - t.hue - ep.lightEstimate + 540) % 360) - 180;
    ep.lightEstimate = clamp(ep.lightEstimate + residual * 0.08, -40, 40);
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

const DEFAULT_WEIGHTS = {
  sensory: 0.3, binding: 0.16, temporal: 0.16, action: 0.18, self: 0.12, regularization: 0.08,
};

function weightedEnergy(weights: typeof DEFAULT_WEIGHTS, terms: EnergyBreakdown): number {
  let sum = 0;
  let norm = 0;
  for (const key of Object.keys(weights) as (keyof typeof weights)[]) {
    const value = terms[key];
    if (value == null) continue;
    sum += weights[key] * value;
    norm += weights[key];
  }
  return norm > 0 ? sum / norm : 0;
}

function energies(ep: Agent, tracks: Track[], misses: Track[], blobs: Blob[], action: number): EnergyBreakdown {
  let unexplained = blobs.length;
  let hueError = 0;
  let bindingError = 0;
  let motionError = 0;
  for (const t of tracks) {
    if (t.boundBlob == null) continue;
    unexplained--;
    const blob = blobs[t.boundBlob]!;
    hueError += hueDelta(t.hue + ep.lightEstimate, blob.hue) / 180;
    const prior = ep.tracks.find((p) => p.id === t.id);
    if (prior) {
      bindingError += (Math.abs(prior.fill - blob.fill) + hueDelta(prior.hue + ep.lightEstimate, blob.hue) / 180) / 2;
      motionError += clamp(t.flowError / 16);
    } else {
      // A new identity explains a patch, but has no trajectory to support it.
      motionError += 0.5;
    }
  }
  for (const miss of misses) {
    if (!tracks.some((t) => t.id === miss.id)) motionError += miss.evidence * (1 - miss.uncertainty);
  }
  const patches = Math.max(1, blobs.length);
  const silence = blobs.length > 0 && tracks.length === 0;
  const sensory = clamp((unexplained + hueError) / patches);
  const binding = clamp((bindingError + tracks.filter((t) => t.conflict).length / 2) / patches);
  const temporal = clamp(motionError / Math.max(patches, misses.length));
  const overconfident = tracks.some((t) => t.confidence > 0.82) && sensory > 0.38;
  const regularization = clamp(tracks.length / MAX_TRACKS * 0.45 +
    (overconfident ? 0.45 : 0) + (silence ? 0.25 : 0));
  const terms = { sensory, binding, temporal, action, self: ep.selfError, regularization, total: 0, overconfident };
  terms.total = weightedEnergy(ep.weights, terms);
  return terms;
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
    focusPrevId: null,
    predictedDrop: 0,
    predictedObserved: false,
    selfError: null,
    weights: { ...DEFAULT_WEIGHTS },
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
  perceive(ep, ep.blobs, obs, 0, true);
  selectWorkspace(ep);
  rememberWall(ep, obs);
  ep.guesses = buildGuesses(ep, gaze.x, gaze.y, gaze.aperture);
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

  const schemaOn = cfg.ablation === "full" || cfg.ablation === "record-only";
  perceive(ep, ep.blobs, obs, actionError, schemaOn);

  if (cfg.injectConfidence && schemaOn) {
    for (const t of ep.tracks) {
      t.uncertainty = 0.02;
      t.confidence = 0.98;
      t.conflict = false;
      t.reason = "none";
    }
  }
  selectWorkspace(ep);
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
