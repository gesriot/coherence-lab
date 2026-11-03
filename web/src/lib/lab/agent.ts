import type { Models, Vector } from "./learning.ts";
import { hypot } from "./math.ts";
import { createAgent, stepAgent, type Agent } from "./perception.ts";
import { observe } from "./sensor.ts";
import type { Ablation, Cell, LabConfig, ScenarioId, Track, ViewState, World } from "./types.ts";
import { advanceWorld, hiddenByOccluder, initialGaze, makeWorld } from "./world.ts";

export interface Episode {
  agent: Agent;
  scenario: ScenarioId;
  seed: number;
  world: World;
  recording: Cell[][];
  clip: Cell[][] | null;
  replayIndex: number;
  initialTrackId: number | null;
  coupling: ViewState["coupling"];
  configEcho: ViewState["configEcho"];
}

export function createEpisode(
  scenario: ScenarioId, seed: number, ablation: Ablation = "full", learning?: Models, bodyGain?: Readonly<Vector>,
): Episode {
  const world = makeWorld(scenario, seed);
  if (bodyGain) world.bodyGain = bodyGain;
  const gaze = initialGaze(scenario);
  const obs = observe(world, gaze.x, gaze.y, gaze.aperture, learning?.gain);
  const ep: Episode = {
    agent: createAgent(obs, ablation, learning),
    scenario,
    seed,
    world,
    recording: [obs.cells], clip: null, replayIndex: 0,
    initialTrackId: null, coupling: "intact",
    configEcho: { pinGaze: false, injectConfidence: false, replay: false, extraNoise: 0 },
  };
  ep.initialTrackId = snapshot(ep, ablation).evaluator.objects[0]?.nearestTrack ?? null;
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

  const { gazeX, gazeY, aperture } = ep.agent;
  const obs = observe(ep.world, gazeX, gazeY, aperture, ep.agent.learning?.gain);
  let coupling: "intact" | "broken" = "intact";
  if (cfg.replay) {
    if (!ep.clip) ep.clip = ep.recording.slice(-28);
    if (ep.clip.length > 0) {
      const frame = ep.clip[ep.replayIndex % ep.clip.length]!;
      ep.replayIndex += 1;
      obs.cells = frame;
      coupling = "broken";
    }
  } else {
    ep.clip = null;
    ep.replayIndex = 0;
    ep.recording.push(obs.cells);
    if (ep.recording.length > 36) ep.recording.shift();
  }

  ep.coupling = coupling;
  const { ablation, pinGaze, pinX, pinY, injectConfidence } = cfg;
  stepAgent(ep.agent, obs, { ablation, pinGaze, pinX, pinY, injectConfidence });
}

export function snapshot(ep: Episode, ablation: Ablation): ViewState {
  const agent = ep.agent;
  const objects = ep.world.objects.map((o) => ({ ...o }));
  const reports = objects.map((o) => {
    let nearest: Track | null = null;
    let dist = 1e9;
    for (const t of agent.tracks) {
      const d = hypot(t.x - o.x, t.y - o.y);
      if (d < dist) {
        dist = d;
        nearest = t;
      }
    }
    const vanished = o.vanishAt != null && ep.world.t >= o.vanishAt;
    const occluded = hiddenByOccluder(o, ep.world.occluder);
    const close = nearest && dist < (agent.stepIndex === 0 ? 14 : 12) ? nearest : null;
    return {
      id: o.id,
      bodyHue: o.hue,
      nearestTrack: close ? close.id : null,
      trackHue: close ? close.hue : null,
      dist: close ? dist : null,
      occluded,
      vanished,
      visible: !vanished && !occluded,
    };
  });
  const target = reports[0];
  let retained: boolean | null = null;
  if (ep.initialTrackId != null && target) {
    if (target.occluded) retained = agent.tracks.some((t) => t.id === ep.initialTrackId);
    else if (agent.stepIndex > 2) retained = target.nearestTrack === ep.initialTrackId;
  }
  return {
    step: agent.stepIndex,
    scenario: ep.scenario,
    seed: ep.seed,
    ablation,
    objects,
    occluder: { ...ep.world.occluder },
    light: ep.world.light,
    noise: ep.world.noise,
    swapped: ep.world.swapped,
    obsGaze: agent.obsGaze,
    nextGaze: { x: agent.gazeX, y: agent.gazeY, aperture: agent.aperture },
    cells: agent.cells.map((c) => ({ ...c })),
    grid: agent.grid,
    blobs: agent.blobs.map((b) => ({ ...b })),
    tracks: agent.tracks.map((t) => ({ ...t, motionSamples: t.motionSamples.map((s) => ({ ...s })) })),
    attention: { ...agent.attention },
    energy: { ...agent.energy },
    energyTrace: [...agent.energyTrace],
    history: agent.history.map((h) => ({ ...h })),
    coupling: ep.coupling,
    memoryFrames: agent.salienceFrames.length,
    workspaceCapacity: agent.workspaceCapacity,
    iterationsUsed: agent.iterationsUsed,
    iterationBudget: agent.iterationBudget,
    evaluator: {
      initialTrackId: ep.initialTrackId,
      retained,
      objects: reports,
      lightTrue: ep.world.light,
      lightEstimated: agent.lightEstimate,
    },
    configEcho: { ...ep.configEcho },
  };
}
