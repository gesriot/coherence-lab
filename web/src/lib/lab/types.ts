export type Shape = "disk" | "square" | "triangle";

export type SourceKind = "observed" | "predicted" | "imagined" | "mixed";

export type UncertaintyReason =
  | "none"
  | "noise"
  | "occlusion"
  | "conflict"
  | "absence"
  | "no-data";

export type Ablation = "full" | "record-only" | "no-schema" | "dummy-memory";

export type ScenarioId = "continuity" | "binding" | "conflict" | "nursery";

export type Agreement = "settled" | "competing" | "underdetermined";

export type Expectation = "surface" | "occluder" | "empty";

export type AttentionReason =
  | "conflict"
  | "verify"
  | "uncertainty"
  | "explore"
  | "salience"
  | "pursue"
  | "pinned"
  | "idle";

export interface SimObject {
  id: string;
  shape: Shape;
  hue: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
}

export interface Occluder {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface World {
  seed: number;
  t: number;
  objects: SimObject[];
  occluder: Occluder;
  /** Degrees added to apparent hue. */
  light: number;
  /** Scale of per-cell hue noise, degrees. */
  noise: number;
  baseNoise: number;
  swapAt: number | null;
  swapped: boolean;
}

export interface Cell {
  saturation: number;
  lightness: number;
  hue: number;
}

export interface Observation {
  /** Frame the agent believes it is looking through. */
  gazeX: number;
  gazeY: number;
  aperture: number;
  grid: number;
  cells: Cell[];
}

export interface Blob {
  index: number;
  cx: number;
  cy: number;
  hue: number;
  area: number;
  fill: number;
}

export interface Track {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  hue: number;
  fill: number;
  evidence: number;
  prediction: number;
  uncertainty: number;
  confidence: number;
  source: SourceKind;
  reason: UncertaintyReason;
  unseen: number;
  age: number;
  boundBlob: number | null;
  conflict: boolean;
  inWorkspace: boolean;
  radius: number;
  flowError: number;
}

export interface AttentionState {
  /** Second-order model is being maintained. */
  active: boolean;
  /** That model is allowed to move gaze and aperture. */
  influencesAction: boolean;
  focusId: number | null;
  reason: AttentionReason;
  agreement: Agreement;
  intendedX: number;
  intendedY: number;
  intendedAperture: number;
  executedX: number;
  executedY: number;
  executedAperture: number;
  note: string;
  predictedObserved: boolean;
  predictedDrop: number;
  selfError: number | null;
}

export interface EnergyBreakdown {
  sensory: number;
  binding: number;
  temporal: number;
  action: number;
  self: number | null;
  regularization: number;
  total: number;
  overconfident: boolean;
}

export interface LabConfig {
  ablation: Ablation;
  pinGaze: boolean;
  pinX: number;
  pinY: number;
  injectConfidence: boolean;
  replay: boolean;
  /** Extra hue noise in degrees, added to the world's base. */
  extraNoise: number;
}

export interface ObjectReport {
  id: string;
  bodyHue: number;
  nearestTrack: number | null;
  trackHue: number | null;
  dist: number | null;
  occluded: boolean;
  visible: boolean;
}

export interface EvaluatorView {
  /** Track locked to the continuity target at birth. Not given to the agent. */
  initialTrackId: number | null;
  retained: boolean | null;
  objects: ObjectReport[];
  lightTrue: number;
  lightEstimated: number;
}

export interface ViewState {
  step: number;
  scenario: ScenarioId;
  seed: number;
  ablation: Ablation;
  objects: SimObject[];
  occluder: Occluder;
  light: number;
  noise: number;
  swapped: boolean;
  obsGaze: { x: number; y: number; aperture: number };
  nextGaze: { x: number; y: number; aperture: number };
  cells: Cell[];
  grid: number;
  blobs: Blob[];
  tracks: Track[];
  attention: AttentionState;
  energy: EnergyBreakdown;
  energyTrace: number[];
  history: { step: number; total: number }[];
  coupling: "intact" | "broken";
  memoryFrames: number;
  iterationsUsed: number;
  iterationBudget: number;
  evaluator: EvaluatorView;
  configEcho: Pick<LabConfig, "pinGaze" | "injectConfidence" | "replay" | "extraNoise">;
}
