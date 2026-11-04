use crate::{
    attention,
    learning::{self, Context, Features, GapStep, Models},
    observation::{blobs, classify, locate},
    sensor::Observation,
    types::*,
};
use jscompat::{PI, SQRT_2, hypot};
const MAX_TRACKS: usize = 5;
const MAX_SPEED: f64 = 3.2;
const BASE_RADIUS: f64 = 16.;
const VELOCITY_INTERVALS: usize = 8;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Agent {
    pub(crate) models: Option<Models>,
    pub(crate) tracks: Vec<Track>,
    next_id: u32,
    pub(crate) gaze: Gaze,
    pub(crate) light_estimate: f64,
    pub(crate) cycle: u32,
    pub(crate) attention: Attention,
    pub(crate) energy: Energy,
    pub(crate) guesses: Vec<Expectation>,
    wall_map: Vec<u8>,
    pub(crate) last_viewed: [i32; 16],
    pub(crate) focus_prev_id: Option<u32>,
    predicted_drop: f64,
    pub(crate) predicted_observed: bool,
    pub(crate) self_error: Option<f64>,
    pub(crate) weights: [f64; 6],
    pub(crate) budget: usize,
    pub(crate) iterations: usize,
    pub(crate) energy_trace: Vec<f64>,
    pub(crate) history: Vec<History>,
    pub(crate) step: u32,
    pub(crate) cells: Vec<Cell>,
    pub(crate) grid: usize,
    pub(crate) blobs: Vec<Blob>,
    pub(crate) obs_gaze: Gaze,
    pub(crate) capacity: usize,
    pub(crate) salience: Vec<Option<Blob>>,
    pub(crate) hold: u32,
}
fn finish(t: &mut Track) {
    let src = match t.source {
        Source::Remembered => 0.55,
        Source::Predicted => 0.78,
        Source::Mixed => 0.7,
        Source::Observed => 1.,
    };
    t.confidence = unit(t.evidence * (1. - t.uncertainty * 0.85) * src);
}
fn forecast(t: &mut Track, m: &Models, input: Features) {
    let [vx, vy] = learning::motion(m, input);
    let scale = MAX_SPEED / MAX_SPEED.max(hypot(vx, vy));
    t.forecast_input = Some(input);
    t.vx = vx * scale;
    t.vy = vy * scale;
}
fn birth(a: &Agent, b: &Blob) -> Track {
    let width = 2. * a.obs_gaze.aperture / a.grid as f64;
    let mut t = Track {
        id: a.next_id + b.index as u32,
        x: b.cx,
        y: b.cy,
        vx: 0.,
        vy: 0.,
        hue: wrap(b.hue - a.light_estimate),
        fill: b.fill,
        evidence: 0.58,
        prediction: 0.3,
        uncertainty: 0.48,
        confidence: 0.3,
        source: Source::Observed,
        reason: Reason::None,
        unseen: 0,
        age: 1,
        bound_blob: Some(b.index),
        conflict: false,
        in_workspace: false,
        radius: (b.area as f64 / PI).sqrt() * width,
        flow_error: 0.,
        velocity_spread: MAX_SPEED,
        motion_samples: vec![MotionSample {
            step: 1,
            x: b.cx,
            y: b.cy,
            cell_width: width,
        }],
        seen_run: None,
        gap_step: None,
        empty_views: None,
        forecast_input: None,
    };
    if let Some(m) = &a.models {
        let input = learning::features(&t.motion_samples);
        forecast(&mut t, m, input);
    }
    finish(&mut t);
    t
}
fn explain_miss(obs: &Observation, x: f64, y: f64) -> Option<Expectation> {
    let here = locate(obs, x, y);
    if here == Some(Expectation::Empty) {
        for (dx, dy) in [
            (5., 0.),
            (-5., 0.),
            (0., 5.),
            (0., -5.),
            (7., 0.),
            (-7., 0.),
            (0., 7.),
            (0., -7.),
        ] {
            if locate(obs, x + dx, y + dy) == Some(Expectation::Occluder) {
                return Some(Expectation::Occluder);
            }
        }
    }
    here
}
fn disk_in_view(g: Gaze, x: f64, y: f64, r: f64) -> bool {
    (x - g.x).abs() + r <= g.aperture && (y - g.y).abs() + r <= g.aperture
}
fn empty_disk(obs: &Observation, x: f64, y: f64, r: f64, intersecting: bool) -> bool {
    let g = obs.gaze;
    if !disk_in_view(g, x, y, r) {
        return false;
    }
    let size = 2. * g.aperture / obs.grid as f64;
    let left = g.x - g.aperture;
    let top = g.y - g.aperture;
    let lo = |v: f64| (v / size).floor().max(0.) as usize;
    let hi = |v: f64| (v / size).floor().min((obs.grid - 1) as f64) as usize;
    let margin = if intersecting { size / 2. } else { 0. };
    for row in lo(y - r - top)..=hi(y + r - top) {
        for col in lo(x - r - left)..=hi(x + r - left) {
            let dx = ((left + (col as f64 + 0.5) * size - x).abs() - margin).max(0.);
            let dy = ((top + (row as f64 + 0.5) * size - y).abs() - margin).max(0.);
            if dx * dx + dy * dy <= r * r
                && classify(obs.cells[row * obs.grid + col]) != Expectation::Empty
            {
                return false;
            }
        }
    }
    true
}
pub(crate) fn match_radius(t: &Track) -> f64 {
    BASE_RADIUS + f64::from(t.unseen) * t.velocity_spread
}
pub(crate) fn miss(prior: &Track, obs: &Observation, models: Option<&Models>) -> Track {
    let mut t = prior.clone();
    t.x += prior.vx;
    t.y += prior.vy;
    t.age += 1;
    t.unseen += 1;
    t.conflict = false;
    t.bound_blob = None;
    let loc = explain_miss(obs, t.x, t.y);
    if models.is_some() {
        let empty = loc == Some(Expectation::Empty) && empty_disk(obs, t.x, t.y, t.radius, true);
        t.empty_views = Some(if empty {
            1 + if prior.bound_blob.is_none() {
                prior.empty_views.unwrap_or(0)
            } else {
                0
            }
        } else {
            0
        });
    }
    if loc.is_none()
        || (models.is_some() && loc == Some(Expectation::Empty) && t.empty_views.unwrap_or(0) < 3)
    {
        t.source = if t.unseen > 10 {
            Source::Remembered
        } else {
            Source::Predicted
        };
        t.uncertainty = unit(t.uncertainty + 0.08);
        t.evidence = unit(t.evidence - 0.03);
        t.reason = Reason::NoData;
        t.prediction = unit(t.prediction * 0.94);
    } else if loc == Some(Expectation::Occluder) {
        t.source = Source::Predicted;
        t.uncertainty = unit(t.uncertainty + 0.02);
        t.evidence = unit(t.evidence - 0.005);
        t.reason = Reason::Occlusion;
        t.prediction = t.prediction.max(0.55);
    } else if loc == Some(Expectation::Empty) {
        t.source = Source::Predicted;
        t.uncertainty = unit(t.uncertainty + 0.16);
        t.evidence = unit(t.evidence * 0.72);
        t.reason = Reason::Absence;
        t.prediction = unit(t.prediction * 0.65);
    } else {
        t.source = Source::Mixed;
        t.uncertainty = unit(t.uncertainty + 0.06);
        t.reason = Reason::Conflict;
        t.conflict = true;
    }
    if let Some(m) = models {
        t.seen_run = Some(0);
        let step = GapStep::new(
            if t.reason == Reason::Occlusion {
                Context::Hidden
            } else {
                Context::Open
            },
            t.reason != Reason::Conflict
                && empty_disk(obs, t.x, t.y, 0.5 * match_radius(prior), false),
        );
        t.gap_step = Some(step);
        if t.reason != Reason::Conflict {
            t.evidence = learning::filter(m, prior.evidence, step);
            t.uncertainty = unit(prior.uncertainty + 0.03);
        }
        let last = prior
            .forecast_input
            .expect("a track with learned motion has forecast features");
        forecast(&mut t, m, [prior.vx, prior.vy, last[2], last[3], 1.]);
    }
    finish(&mut t);
    t
}
pub(crate) fn matched(a: &Agent, prior: &Track, b: &Blob, obs: &Observation) -> Track {
    let dx = b.cx - prior.x;
    let dy = b.cy - prior.y;
    let flow = hypot(dx - prior.vx, dy - prior.vy);
    let color = hue_delta(prior.hue + a.light_estimate, b.hue);
    let mismatch = unit(
        (flow / match_radius(prior) * (BASE_RADIUS / 18.) * 1.2
            + color / 180. * 0.55
            + (prior.fill - b.fill).abs() * 0.2)
            / 1.28,
    );
    let sample = MotionSample {
        step: prior.age + 1,
        x: b.cx,
        y: b.cy,
        cell_width: 2. * obs.gaze.aperture / obs.grid as f64,
    };
    let mut samples = prior.motion_samples[prior
        .motion_samples
        .len()
        .saturating_sub(VELOCITY_INTERVALS)..]
        .to_vec();
    samples.push(sample);
    let first = samples[0];
    let elapsed = f64::from(sample.step - first.step);
    let vx = (sample.x - first.x) / elapsed;
    let vy = (sample.y - first.y) / elapsed;
    let scale = MAX_SPEED / MAX_SPEED.max(hypot(vx, vy));
    let mut t = prior.clone();
    t.x = b.cx;
    t.y = b.cy;
    t.vx = vx * scale;
    t.vy = vy * scale;
    t.velocity_spread = SQRT_2 * (first.cell_width + sample.cell_width) / (2. * elapsed);
    t.motion_samples = samples;
    let delta = ((wrap(b.hue - a.light_estimate) - wrap(prior.hue) + 540.) % 360.) - 180.;
    t.hue = wrap(prior.hue + delta * 0.32);
    t.fill = prior.fill * 0.65 + b.fill * 0.35;
    t.evidence = prior.evidence * 0.45 + 0.55;
    t.prediction = 1. - mismatch;
    t.uncertainty = unit(prior.uncertainty * 0.62 + mismatch * 0.22);
    t.reason = if color > 32. {
        Reason::Noise
    } else {
        Reason::None
    };
    t.source = Source::Observed;
    t.age += 1;
    t.bound_blob = Some(b.index);
    t.unseen = 0;
    t.conflict = false;
    t.flow_error = flow;
    t.radius = (b.area as f64 / PI).sqrt() * sample.cell_width;
    if let Some(m) = &a.models {
        t.seen_run = Some(
            if disk_in_view(
                obs.gaze,
                prior.x + prior.vx,
                prior.y + prior.vy,
                0.5 * match_radius(prior),
            ) {
                (1 + if prior.bound_blob.is_some() {
                    prior.seen_run.unwrap_or(0)
                } else {
                    0
                })
                .min(20)
            } else {
                0
            },
        );
        t.gap_step = None;
        let input = learning::features(&t.motion_samples);
        forecast(&mut t, m, input);
    }
    finish(&mut t);
    t
}

fn energy(a: &Agent, slots: &[Option<&Track>], misses: &[Track], action: f64) -> Energy {
    let mut unexplained = a.blobs.len();
    let (mut hue_error, mut binding_error, mut motion_error) = (0., 0., 0.);
    let mut count = 0;
    let mut conflicts = 0;
    let mut confident = false;
    for &t in slots.iter().flatten() {
        count += 1;
        conflicts += usize::from(t.conflict);
        confident |= t.confidence > 0.82;
        let Some(i) = t.bound_blob else { continue };
        let b = a.blobs[i];
        unexplained -= 1;
        hue_error += hue_delta(t.hue + a.light_estimate, b.hue) / 180.;
        if let Some(prior) = a.tracks.iter().find(|p| p.id == t.id) {
            binding_error += ((prior.fill - b.fill).abs()
                + hue_delta(prior.hue + a.light_estimate, b.hue) / 180.)
                / 2.;
            motion_error += unit(t.flow_error / match_radius(prior));
        } else {
            motion_error += 0.5;
        }
    }
    for t in misses {
        if !slots.iter().flatten().any(|p| p.id == t.id) {
            motion_error += t.evidence * (1. - t.uncertainty);
        }
    }
    let patches = a.blobs.len().max(1);
    let sensory = unit((unexplained as f64 + hue_error) / patches as f64);
    let binding = unit((binding_error + conflicts as f64 / 2.) / patches as f64);
    let temporal = unit(motion_error / patches.max(misses.len()) as f64);
    let overconfident = confident && sensory > 0.38;
    let silence = !a.blobs.is_empty() && count == 0;
    let regularization = unit(
        count as f64 / MAX_TRACKS as f64 * 0.45
            + if overconfident { 0.45 } else { 0. }
            + if silence { 0.25 } else { 0. },
    );
    let (mut sum, mut norm) = (0., 0.);
    for (w, v) in a.weights.into_iter().zip([
        Some(sensory),
        Some(binding),
        Some(temporal),
        Some(action),
        a.self_error,
        Some(regularization),
    ]) {
        if let Some(v) = v {
            sum += w * v;
            norm += w;
        }
    }
    Energy {
        sensory,
        binding,
        temporal,
        action,
        self_error: a.self_error,
        regularization,
        total: if norm > 0. { sum / norm } else { 0. },
        overconfident,
    }
}
fn perceive(a: &mut Agent, obs: &Observation, action: f64, schema: bool) {
    let misses: Vec<_> = a
        .tracks
        .iter()
        .map(|t| miss(t, obs, a.models.as_ref()))
        .collect();
    let matches: Vec<Vec<Option<Track>>> = a
        .tracks
        .iter()
        .zip(&misses)
        .map(|(prior, m)| {
            let mut options: Vec<_> = a
                .blobs
                .iter()
                .map(|b| {
                    (hypot(b.cx - m.x, b.cy - m.y) <= match_radius(prior))
                        .then(|| matched(a, prior, b, obs))
                })
                .collect();
            for i in 0..options.len() {
                if let Some(t) = &options[i] {
                    let conflict = options.iter().enumerate().any(|(j, alt)| {
                        j != i
                            && alt
                                .as_ref()
                                .is_some_and(|v| (v.prediction - t.prediction).abs() < 0.1)
                    });
                    if conflict {
                        let t = options[i].as_mut().expect("candidate exists at this index");
                        t.conflict = true;
                        t.reason = Reason::Conflict;
                    }
                }
            }
            options
        })
        .collect();
    a.self_error = None;
    if schema && let Some(i) = a.tracks.iter().position(|t| Some(t.id) == a.focus_prev_id) {
        let support = matches[i]
            .iter()
            .flatten()
            .reduce(|b, t| if t.prediction > b.prediction { t } else { b });
        let observed = support.is_some();
        let actual = support.unwrap_or(&misses[i]);
        let drop = a.tracks[i].uncertainty - actual.uncertainty;
        let miss = if a.predicted_observed == observed {
            0.
        } else {
            0.7
        };
        a.self_error = Some(unit(miss + 0.3 * (a.predicted_drop - drop).abs().min(1.)));
    }
    let births: Vec<_> = a.blobs.iter().map(|b| birth(a, b)).collect();
    let priors = a.tracks.len();
    let mut slots: Vec<Option<&Track>> = misses
        .iter()
        .map(Some)
        .chain(births.iter().map(|_| None))
        .collect();
    let mut current = energy(a, &slots, &misses, action);
    let mut trace = vec![current.total];
    for _ in 0..a.budget {
        let mut best = slots.clone();
        let mut lowest = current;
        let mut consider = |candidate: Vec<_>| {
            let e = energy(a, &candidate, &misses, action);
            if e.total < lowest.total {
                lowest = e;
                best = candidate;
            }
        };
        for i in 0..priors {
            if slots[i].is_some_and(|t| t.bound_blob.is_none()) {
                let mut candidate = slots.clone();
                candidate[i] = None;
                consider(candidate);
            }
            for m in matches[i].iter().flatten() {
                if slots[i].is_some_and(|t| std::ptr::eq(t, m)) {
                    continue;
                }
                let mut candidate = slots.clone();
                if let Some(owner) = slots
                    .iter()
                    .position(|t| t.is_some_and(|v| v.bound_blob == m.bound_blob))
                {
                    candidate[owner] = misses.get(owner);
                    if owner < priors
                        && let Some(b) = slots[i].and_then(|t| t.bound_blob)
                        && let Some(other) = &matches[owner][b]
                    {
                        candidate[owner] = Some(other);
                    }
                }
                candidate[i] = Some(m);
                if candidate.iter().flatten().count() <= MAX_TRACKS {
                    consider(candidate);
                }
            }
        }
        if slots.iter().flatten().count() < MAX_TRACKS {
            for b in &births {
                if slots.iter().flatten().any(|t| t.bound_blob == b.bound_blob) {
                    continue;
                }
                let mut candidate = slots.clone();
                candidate[priors + b.bound_blob.expect("birth is bound to its patch")] = Some(b);
                consider(candidate);
            }
        }
        let improvement = current.total - lowest.total;
        slots = best;
        current = lowest;
        trace.push(current.total);
        if improvement < 1e-5 {
            break;
        }
    }
    a.tracks = slots.into_iter().flatten().cloned().collect();
    for t in &a.tracks {
        a.next_id = a.next_id.max(t.id + 1);
        if let Some(i) = t.bound_blob
            && t.uncertainty <= 0.4
        {
            let residual = ((a.blobs[i].hue - t.hue - a.light_estimate + 540.) % 360.) - 180.;
            a.light_estimate = clamp(a.light_estimate + residual * 0.08, -40., 40.);
        }
    }
    a.energy = current;
    a.iterations = trace.len() - 1;
    a.energy_trace = trace;
}
fn workspace(a: &mut Agent) {
    let score = |t: &Track| {
        t.confidence
            + if t.conflict { 0.22 } else { 0. }
            + if t.reason == Reason::Occlusion {
                0.18
            } else {
                0.
            }
    };
    let mut ranked: Vec<_> = a.tracks.iter().collect();
    ranked.sort_by(|x, y| score(y).total_cmp(&score(x)));
    let keep: Vec<_> = ranked.into_iter().take(a.capacity).map(|t| t.id).collect();
    for t in &mut a.tracks {
        t.in_workspace = keep.contains(&t.id);
    }
}
fn remember_wall(a: &mut Agent, obs: &Observation) {
    let g = obs.gaze;
    let n = obs.grid;
    let size = 2. * g.aperture / n as f64;
    for (i, &cell) in obs.cells.iter().enumerate() {
        let left = g.x - g.aperture + (i % n) as f64 * size;
        let top = g.y - g.aperture + (i / n) as f64 * size;
        let start = clamp((left - 0.5).ceil(), 0., 100.) as usize;
        let end = clamp((left + size - 0.5).ceil(), 0., 100.) as usize;
        let wall = u8::from(classify(cell) == Expectation::Occluder);
        for row in (top - 0.5).ceil().max(0.) as usize..(top + size - 0.5).ceil().min(100.) as usize
        {
            a.wall_map[row * 100 + start..row * 100 + end].fill(wall);
        }
    }
}
fn guesses(a: &Agent) -> Vec<Expectation> {
    let n = a.grid;
    let g = a.gaze;
    let size = 2. * g.aperture / n as f64;
    (0..n * n)
        .map(|i| {
            let x = g.x - g.aperture + ((i % n) as f64 + 0.5) * size;
            let y = g.y - g.aperture + ((i / n) as f64 + 0.5) * size;
            if !(0.0..100.).contains(&x) || !(0.0..100.).contains(&y) {
                return Expectation::Empty;
            }
            if a.wall_map[y.floor() as usize * 100 + x.floor() as usize] != 0 {
                return Expectation::Occluder;
            }
            if a.tracks
                .iter()
                .any(|t| hypot(x - t.x - t.vx, y - t.y - t.vy) <= t.radius)
            {
                Expectation::Surface
            } else {
                Expectation::Empty
            }
        })
        .collect()
}
impl Agent {
    pub(crate) fn new(obs: &Observation, policy: Policy, models: Option<Models>) -> Self {
        let g = obs.gaze;
        let schema = matches!(policy, Policy::Full | Policy::RecordOnly);
        let mut a = Self {
            models,
            tracks: Vec::new(),
            next_id: 1,
            gaze: g,
            obs_gaze: g,
            light_estimate: 0.,
            cycle: 0,
            attention: attention::initial(g, policy),
            energy: Energy::default(),
            guesses: vec![],
            wall_map: vec![0; 10000],
            last_viewed: [-1; 16],
            focus_prev_id: None,
            predicted_drop: 0.,
            predicted_observed: false,
            self_error: None,
            weights: [0.3, 0.16, 0.16, 0.18, 0.12, 0.08],
            budget: 4,
            iterations: 0,
            energy_trace: vec![],
            history: vec![],
            step: 0,
            cells: obs.cells.clone(),
            grid: obs.grid,
            blobs: blobs(obs),
            capacity: 3,
            salience: vec![],
            hold: 0,
        };
        perceive(&mut a, obs, 0., schema);
        workspace(&mut a);
        attention::record(&mut a.last_viewed, g, 0);
        remember_wall(&mut a, obs);
        a.guesses = guesses(&a);
        a.attention.agreement = attention::agreement(&a);
        a
    }
    pub(crate) fn step(&mut self, obs: Observation, cfg: &Config) {
        let errors = obs
            .cells
            .iter()
            .enumerate()
            .filter(|(i, cell)| self.guesses.get(*i) != Some(&classify(**cell)))
            .count();
        let action = errors as f64 / obs.cells.len() as f64;
        self.blobs = blobs(&obs);
        self.grid = obs.grid;
        self.obs_gaze = obs.gaze;
        attention::record(&mut self.last_viewed, obs.gaze, self.step + 1);
        remember_wall(self, &obs);
        let schema = matches!(cfg.ablation, Policy::Full | Policy::RecordOnly);
        perceive(self, &obs, action, schema);
        self.cells = obs.cells;
        if cfg.inject_confidence && schema {
            for t in &mut self.tracks {
                t.uncertainty = 0.02;
                t.confidence = 0.98;
                t.conflict = false;
                t.reason = Reason::None;
            }
        }
        workspace(self);
        self.history.push(History {
            step: self.step,
            total: self.energy.total,
        });
        if self.history.len() > 48 {
            self.history.remove(0);
        }
        attention::decide(self, cfg);
        self.predicted_observed = self.attention.predicted_observed;
        self.predicted_drop = self.attention.predicted_drop;
        self.focus_prev_id = if schema {
            self.attention.executed_focus_id
        } else {
            None
        };
        self.guesses = guesses(self);
        self.cycle += 1;
        self.step += 1;
    }
}

#[cfg(test)]
mod tests;
