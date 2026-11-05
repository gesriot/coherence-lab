use crate::{
    learning::Models,
    perception::Agent,
    sensor::observe,
    types::*,
    world::{World, advance, hidden, initial_gaze, make_world},
};
use jscompat::hypot;
use serde::Serialize;

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Episode {
    pub(crate) agent: Agent,
    scenario: Scenario,
    seed: u32,
    pub(crate) world: World,
    recording: Vec<Vec<Cell>>,
    clip: Option<Vec<Vec<Cell>>>,
    replay_index: usize,
    initial_track_id: Option<u32>,
    broken: bool,
    echo: Echo,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct Echo {
    pin_gaze: bool,
    inject_confidence: bool,
    replay: bool,
    extra_noise: f64,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct ObjectReport {
    pub id: &'static str,
    pub body_hue: f64,
    pub nearest_track: Option<u32>,
    pub track_hue: Option<f64>,
    pub dist: Option<f64>,
    pub occluded: bool,
    pub vanished: bool,
    pub visible: bool,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct Evaluator {
    pub initial_track_id: Option<u32>,
    pub retained: Option<bool>,
    pub objects: Vec<ObjectReport>,
    pub light_true: f64,
    pub light_estimated: f64,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct Snapshot<'a> {
    pub step: u32,
    pub scenario: Scenario,
    pub seed: u32,
    pub ablation: Policy,
    pub objects: &'a [Object],
    pub occluder: Wall,
    pub light: f64,
    pub noise: f64,
    pub swapped: bool,
    pub obs_gaze: Gaze,
    pub next_gaze: Gaze,
    pub cells: &'a [Cell],
    pub grid: usize,
    blobs: &'a [Blob],
    pub tracks: &'a [Track],
    pub attention: &'a Attention,
    pub energy: Energy,
    pub energy_trace: &'a [f64],
    pub history: &'a [History],
    pub coupling: &'static str,
    pub memory_frames: usize,
    pub workspace_capacity: usize,
    pub iterations_used: usize,
    pub iteration_budget: usize,
    pub evaluator: Evaluator,
    config_echo: Echo,
}
impl Episode {
    pub fn new(sc: Scenario, seed: u32, policy: Policy) -> Self {
        Self::with_models(sc, seed, policy, None, None)
    }
    pub fn with_models(
        sc: Scenario,
        seed: u32,
        policy: Policy,
        models: Option<Models>,
        body_gain: Option<[f64; 2]>,
    ) -> Self {
        let mut world = make_world(sc, seed);
        world.body_gain = body_gain;
        let obs = observe(&world, initial_gaze(sc), models.as_ref().map(|m| m.gain));
        let mut ep = Self {
            agent: Agent::new(&obs, policy, models),
            scenario: sc,
            seed,
            world,
            recording: vec![obs.cells],
            clip: None,
            replay_index: 0,
            initial_track_id: None,
            broken: false,
            echo: Echo::default(),
        };
        ep.initial_track_id = ep.object_reports()[0].nearest_track;
        ep
    }
    pub fn step(&mut self, cfg: &Config) {
        self.echo = Echo {
            pin_gaze: cfg.pin_gaze,
            inject_confidence: cfg.inject_confidence,
            replay: cfg.replay,
            extra_noise: cfg.extra_noise,
        };
        self.world.noise = self.world.base_noise + cfg.extra_noise;
        advance(&mut self.world);
        let mut obs = observe(
            &self.world,
            self.agent.gaze,
            self.agent.models.as_ref().map(|m| m.gain),
        );
        self.broken = false;
        if cfg.replay {
            let clip = self.clip.get_or_insert_with(|| {
                self.recording[self.recording.len().saturating_sub(28)..].to_vec()
            });
            if !clip.is_empty() {
                obs.cells.clone_from(&clip[self.replay_index % clip.len()]);
                self.replay_index += 1;
                self.broken = true;
            }
        } else {
            self.clip = None;
            self.replay_index = 0;
            self.recording.push(obs.cells.clone());
            if self.recording.len() > 36 {
                self.recording.remove(0);
            }
        }
        self.agent.step(obs, cfg);
    }
    fn object_reports(&self) -> Vec<ObjectReport> {
        self.world
            .objects
            .iter()
            .map(|o| {
                let (mut nearest, mut dist) = (None, 1e9);
                for t in &self.agent.tracks {
                    let d = hypot(t.x - o.x, t.y - o.y);
                    if d < dist {
                        dist = d;
                        nearest = Some(t);
                    }
                }
                let near = nearest.filter(|_| dist < if self.agent.step == 0 { 14. } else { 12. });
                let vanished = o.vanish_at.is_some_and(|t| self.world.t >= t);
                let occluded = hidden(o, self.world.wall);
                ObjectReport {
                    id: o.id,
                    body_hue: o.hue,
                    nearest_track: near.map(|t| t.id),
                    track_hue: near.map(|t| t.hue),
                    dist: near.map(|_| dist),
                    occluded,
                    vanished,
                    visible: !vanished && !occluded,
                }
            })
            .collect()
    }
    pub fn snapshot(&self, ablation: Policy) -> Snapshot<'_> {
        let a = &self.agent;
        let w = &self.world;
        let reports = self.object_reports();
        let retained = self.initial_track_id.and_then(|id| {
            if reports[0].occluded {
                Some(a.tracks.iter().any(|t| t.id == id))
            } else if a.step > 2 {
                Some(reports[0].nearest_track == Some(id))
            } else {
                None
            }
        });
        Snapshot {
            step: a.step,
            scenario: self.scenario,
            seed: self.seed,
            ablation,
            objects: &w.objects,
            occluder: w.wall,
            light: w.light,
            noise: w.noise,
            swapped: w.swapped,
            obs_gaze: a.obs_gaze,
            next_gaze: a.gaze,
            cells: &a.cells,
            grid: a.grid,
            blobs: &a.blobs,
            tracks: &a.tracks,
            attention: &a.attention,
            energy: a.energy,
            energy_trace: &a.energy_trace,
            history: &a.history,
            coupling: if self.broken { "broken" } else { "intact" },
            memory_frames: a.salience.len(),
            workspace_capacity: a.capacity,
            iterations_used: a.iterations,
            iteration_budget: a.budget,
            evaluator: Evaluator {
                initial_track_id: self.initial_track_id,
                retained,
                objects: reports,
                light_true: w.light,
                light_estimated: a.light_estimate,
            },
            config_echo: self.echo,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jscompat::number;
    use sha2::{Digest, Sha256};
    fn gaze(g: Gaze) -> String {
        format!(
            "{{\"x\":{},\"y\":{},\"aperture\":{}}}",
            number(g.x),
            number(g.y),
            number(g.aperture)
        )
    }
    #[test]
    fn reference_gaze_tracks_and_energy() {
        for (sc, seed, want) in [
            (
                Scenario::Continuity,
                1,
                "32a12ead2e7b036fbd8cea6b4560640c3b8fe9805ccde0ac4a8f06c8e187be72",
            ),
            (
                Scenario::Continuity,
                2,
                "abd24d99e3d86605f33f6ed274125a4daabdfdd7b37e1499de11419c739ac2b2",
            ),
            (
                Scenario::Conflict,
                1,
                "829e213dc168d7d96f869a350891ca215cb7e6d6e15012dd6b8d8c65bce85a72",
            ),
            (
                Scenario::Conflict,
                2,
                "c3e637300adaacffd0b477183be127fb5e2c2fe60cc3633f309199028f1ee98e",
            ),
        ] {
            let mut ep = Episode::new(sc, seed, Policy::Full);
            let mut hash = Sha256::new();
            for i in 0..=if sc == Scenario::Continuity { 72 } else { 200 } {
                if i != 0 {
                    ep.step(&Config::default());
                }
                let v = ep.snapshot(Policy::Full);
                let tracks = v
                    .tracks
                    .iter()
                    .map(|t| format!("[{},{},{}]", t.id, number(t.x), number(t.y)))
                    .collect::<Vec<_>>()
                    .join(",");
                hash.update(format!(
                    "[{},{},[{}],{}]",
                    gaze(v.obs_gaze),
                    gaze(v.next_gaze),
                    tracks,
                    number(v.energy.total)
                ));
            }
            assert_eq!(format!("{:x}", hash.finalize()), want, "{sc:?} seed {seed}");
        }
    }
}
