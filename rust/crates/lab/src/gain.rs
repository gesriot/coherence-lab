use crate::task::{object, sensorimotor as task};
use crate::{
    Episode,
    alignment::{Pair, Ratios, align, frame},
    experience::{Generator, Replay, Selection},
    feasibility::{Control, reference_model, stable},
    genesis::{EVAL_COUNT, Point, SeedRange, TRAINING_STEPS, average, curriculum, evaluate},
    learning::Models,
    training::Condition,
    types::*,
    world::schedule_outdoor_losses,
};
use jscompat::{abs, hash_unit, hypot, max, min, round, sign};

pub const GAIN_STAGES: [u32; 11] = [0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120];
pub const GAIN_RATES: [f64; 4] = [0.01, 0.03, 0.1, 0.3];
pub const GAIN_ORACLE: f64 = 0.03;
pub const GAIN_SELECTED: f64 = 0.3;
pub const GAIN_DESIGN: SeedRange = SeedRange::new(801, 10, 3101);
pub const GAIN_TUNING: SeedRange = SeedRange::new(541, 10, 2081);
pub const GAIN_COMPARISON: SeedRange = SeedRange::new(121, 30, 1081);
const EPISODES: u32 = GAIN_STAGES[10] / TRAINING_STEPS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum GainCondition {
    Training(Condition),
    Ideal,
    Neutral,
}
pub const GAIN_CONTROLS: [GainCondition; 4] = [
    GainCondition::Training(Condition::Frozen),
    GainCondition::Ideal,
    GainCondition::Training(Condition::Oracle),
    GainCondition::Neutral,
];
impl GainCondition {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ideal => "Идеальные параметры",
            Self::Neutral => "Самообучение при нейтральном взгляде",
            Self::Training(c) => c.label(),
        }
    }
    fn policy(self) -> Policy {
        match self {
            Self::Neutral => Policy::NoSchema,
            Self::Ideal => Policy::FirstOrder,
            Self::Training(c) => c.policy(),
        }
    }
    fn selection(self) -> Selection {
        match self {
            Self::Training(c) => c.selection(),
            _ => Selection::Uniform,
        }
    }
    pub fn trains(self) -> bool {
        !matches!(self, Self::Ideal | Self::Training(Condition::Frozen))
    }
}
impl std::str::FromStr for GainCondition {
    type Err = LabError;
    fn from_str(s: &str) -> Result<Self, LabError> {
        match s {
            "frozen" => Ok(GAIN_CONTROLS[0]),
            "ideal" => Ok(Self::Ideal),
            "oracle" => Ok(GAIN_CONTROLS[2]),
            "neutral" => Ok(Self::Neutral),
            _ => Err(LabError::Control(s.into())),
        }
    }
}
fn body_for_run(seed: u32) -> [f64; 2] {
    [0., 2.].map(|index| {
        let key = 4. * f64::from(seed) + index;
        let sign = if hash_unit(key) < 0.5 { -1. } else { 1. };
        1. + sign * (task::MIN_OFFSET + (task::MAX_OFFSET - task::MIN_OFFSET) * hash_unit(key + 1.))
    })
}
fn fit_gain(estimate: &mut [f64; 2], ratios: Ratios, rate: f64) {
    for (a, r) in ratios.into_iter().enumerate() {
        if let Some(r) = r {
            estimate[a] = clamp(
                estimate[a] * (1. + rate * (r - 1.)),
                task::MIN_GAIN,
                task::MAX_GAIN,
            );
        }
    }
}
struct GainLearner {
    memory: Replay<Pair>,
    selection: Selection,
    rate: f64,
    random: Generator,
}
impl GainLearner {
    fn new(rate: f64, selection: Selection, seed: u32) -> Self {
        Self {
            memory: Replay::new(task::MEMORY),
            rate,
            selection,
            random: Generator::new(seed),
        }
    }
    fn observe(&mut self, pair: Pair, reason: Look, changed: bool, error: f64, ratios: Ratios) {
        if self.selection == Selection::Attention && reason == Look::Pursue && !changed {
            return;
        }
        let priority = if self.selection == Selection::Error {
            abs(ratios[0].unwrap_or(1.) - 1.) + abs(ratios[1].unwrap_or(1.) - 1.)
        } else {
            error
        };
        self.memory.add(pair, 0.05 + priority);
    }
    fn update(
        &mut self,
        estimate: &mut [f64; 2],
        label: Option<&mut dyn FnMut([f64; 2]) -> [f64; 2]>,
    ) -> bool {
        let Some(pair) = self
            .memory
            .sample(&mut self.random, self.selection == Selection::Uniform)
        else {
            return false;
        };
        let mut ratios = align(pair, *estimate);
        if ratios == [None, None] {
            return false;
        }
        if let Some(label) = label {
            let exact = label(*estimate);
            for (a, r) in ratios.iter_mut().enumerate() {
                if r.is_some() {
                    *r = Some(exact[a]);
                }
            }
        }
        fit_gain(estimate, ratios, self.rate);
        true
    }
}
#[derive(Clone, Copy)]
struct GainStep {
    shift: [f64; 2],
    observed: Ratios,
    informative: bool,
    gain: [f64; 2],
}
fn train_episode(
    ep: &mut Episode,
    id: GainCondition,
    mut learner: Option<&mut GainLearner>,
    label: &mut dyn FnMut([f64; 2]) -> [f64; 2],
    after: &mut dyn FnMut(GainStep),
) {
    let cfg = Config::new(id.policy());
    let capture = |ep: &Episode| {
        let a = &ep.agent;
        let g = a.obs_gaze;
        let estimate = a
            .models
            .as_ref()
            .expect("gain episodes carry a sensor model")
            .gain;
        frame(
            &a.cells,
            a.grid,
            g.aperture,
            [g.x / estimate[0], g.y / estimate[1]],
        )
    };
    let mut previous = capture(ep);
    let mut last_focus = None;
    for _ in 0..TRAINING_STEPS {
        let gaze = ep.agent.obs_gaze;
        let focus = ep.agent.attention.executed_focus_id;
        let reason = ep.agent.attention.executed_reason;
        ep.step(&cfg);
        let current = capture(ep);
        let pair = [previous, current.clone()];
        let a = &mut ep.agent;
        let estimate = &mut a
            .models
            .as_mut()
            .expect("gain episodes carry a sensor model")
            .gain;
        let observed = align(&pair, *estimate);
        let informative = learner.as_deref_mut().is_some_and(|l| {
            l.observe(
                pair,
                reason,
                focus != last_focus,
                a.self_error.unwrap_or(0.),
                observed,
            );
            if id == GainCondition::Training(Condition::Oracle) {
                l.update(estimate, Some(label))
            } else {
                l.update(estimate, None)
            }
        });
        after(GainStep {
            shift: [a.obs_gaze.x - gaze.x, a.obs_gaze.y - gaze.y],
            observed,
            informative,
            gain: *estimate,
        });
        previous = current;
        last_focus = focus;
    }
}
fn run_episodes(
    id: GainCondition,
    seed: u32,
    body: [f64; 2],
    m: &mut Models,
    mut learner: Option<&mut GainLearner>,
    mut after: impl FnMut(GainStep),
) {
    for n in 1..=EPISODES {
        let sc = curriculum(n);
        let mut ep = Episode::with_models(
            sc,
            10000 + 100 * seed + n,
            id.policy(),
            Some(m.clone()),
            Some(body),
        );
        if sc == Scenario::Nursery {
            schedule_outdoor_losses(&mut ep.world);
        }
        train_episode(
            &mut ep,
            id,
            learner.as_deref_mut(),
            &mut |g| [body[0] / g[0], body[1] / g[1]],
            &mut after,
        );
        m.gain = ep
            .agent
            .models
            .expect("completed gain episode has a calibrated model")
            .gain;
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Calibration {
    pub step: u32,
    pub error: f64,
    pub q: f64,
    pub gain: [f64; 2],
}
fn calibration_error(g: [f64; 2], b: [f64; 2]) -> f64 {
    (abs(g[0] - b[0]) + abs(g[1] - b[1])) / 2.
}
fn point(step: u32, gain: [f64; 2], b: [f64; 2], initial: f64) -> Calibration {
    let error = calibration_error(gain, b);
    Calibration {
        step,
        error,
        q: 1. - error / initial,
        gain,
    }
}
pub fn gain_area(curve: &[Calibration]) -> f64 {
    average(curve.iter().skip(task::FIRST_SCORED).map(|p| p.q))
}
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct GainRun {
    pub curve: Vec<Calibration>,
    pub secondary: Point,
    pub saccades: f64,
    pub pairs: f64,
    pub draws: f64,
}
fn gain_trial(id: GainCondition, seed: u32, eval: u32, rate: f64) -> GainRun {
    let body = body_for_run(seed);
    let mut m = reference_model(Control::Ideal);
    let initial = calibration_error(task::BASELINE, body);
    if id == GainCondition::Ideal {
        m.gain = body;
    }
    let mut learner = id
        .trains()
        .then(|| GainLearner::new(rate, id.selection(), seed));
    let mut curve = vec![point(0, m.gain, body, initial)];
    let (mut step, mut saccades, mut pairs, mut draws) = (0, 0, 0, 0);
    run_episodes(id, seed, body, &mut m, learner.as_mut(), |s| {
        step += 1;
        pairs += u32::from(s.observed != [None, None]);
        draws += u32::from(s.informative);
        saccades += u32::from(hypot(s.shift[0], s.shift[1]) > task::SACCADE);
        if GAIN_STAGES.get(curve.len()) == Some(&step) {
            curve.push(point(step, s.gain, body, initial));
        }
    });
    let per = |v| f64::from(v) / f64::from(EPISODES);
    GainRun {
        curve,
        secondary: evaluate(&m, id.policy(), eval, EVAL_COUNT, Some(body)),
        saccades: per(saccades),
        pairs: per(pairs),
        draws: per(draws),
    }
}
pub fn gain_runs(id: GainCondition, rate: f64, seeds: SeedRange) -> Vec<GainRun> {
    seeds
        .seeds()
        .map(|s| gain_trial(id, s, seeds.eval, rate))
        .collect()
}
#[non_exhaustive]
pub struct GainSummary {
    pub curve: Vec<Calibration>,
    pub accuracy: f64,
    pub retention: f64,
    pub ghost_free: f64,
    pub saccades: f64,
    pub pairs: f64,
    pub draws: f64,
}
pub fn gain_summary(runs: &[GainRun]) -> GainSummary {
    let units = f64::from(2 * EVAL_COUNT);
    let curve = GAIN_STAGES
        .iter()
        .enumerate()
        .map(|(i, &step)| {
            let avg = |f: fn(Calibration) -> f64| average(runs.iter().map(|r| f(r.curve[i])));
            Calibration {
                step,
                q: avg(|p| p.q),
                error: avg(|p| p.error),
                gain: [avg(|p| p.gain[0]), avg(|p| p.gain[1])],
            }
        })
        .collect();
    let avg = |f: fn(&GainRun) -> f64| average(runs.iter().map(f));
    GainSummary {
        curve,
        accuracy: average(runs.iter().map(|r| round(r.secondary.accuracy * units))) / units,
        retention: avg(|r| r.secondary.retention),
        ghost_free: avg(|r| r.secondary.ghost_free),
        saccades: avg(|r| r.saccades),
        pairs: avg(|r| r.pairs),
        draws: avg(|r| r.draws),
    }
}
pub fn controls_pass(frozen: &GainSummary, ideal: &GainSummary, oracles: &[&GainSummary]) -> bool {
    let complete = |s: &GainSummary| s.curve.len() == GAIN_STAGES.len();
    complete(frozen)
        && frozen.curve.iter().all(|p| p.q == 0.)
        && frozen.accuracy <= object::CEILING
        && complete(ideal)
        && ideal.curve.iter().all(|p| p.q == 1.)
        && ideal.accuracy >= object::FLOOR
        && !oracles.is_empty()
        && oracles.iter().all(|s| {
            complete(s)
                && stable(&s.curve.iter().map(|p| p.q).collect::<Vec<_>>())
                && s.accuracy >= object::FLOOR
        })
}
pub fn design_gain_rate(scores: &[(f64, f64)]) -> f64 {
    let best = scores.iter().fold(f64::NEG_INFINITY, |s, x| max(s, x.1));
    scores
        .iter()
        .filter(|x| best - x.1 < task::TIE)
        .fold(f64::INFINITY, |s, x| min(s, x.0))
}
#[non_exhaustive]
pub struct FixedPoint {
    pub bias: f64,
    pub limit: f64,
    pub direction: f64,
    pub correct: u32,
    pub directional: u32,
    pub informative: u32,
}
impl FixedPoint {
    pub fn passes(&self) -> bool {
        abs(self.bias) <= self.limit && self.direction >= task::MIN_DIRECTION
    }
}
pub fn fixed_point(range: SeedRange) -> FixedPoint {
    let (mut bias_sum, mut offset_sum) = (0., 0.);
    let (mut correct, mut directional, mut informative) = (0, 0, 0);
    for seed in range.seeds() {
        let body = body_for_run(seed);
        offset_sum += abs(body[0] - 1.) + abs(body[1] - 1.);
        for calibrated in [true, false] {
            let mut m = reference_model(Control::Ideal);
            if calibrated {
                m.gain = body;
            }
            run_episodes(GainCondition::Neutral, seed, body, &mut m, None, |s| {
                for (a, &r) in s.observed.iter().enumerate() {
                    if let Some(r) = r {
                        if calibrated {
                            bias_sum += r - 1.;
                            informative += 1;
                        } else if r != 1. {
                            directional += 1;
                            correct += u32::from(sign(r - 1.) == sign(body[a] - 1.));
                        }
                    }
                }
            });
        }
    }
    FixedPoint {
        bias: bias_sum / f64::from(informative),
        limit: task::MAX_BIAS * offset_sum / (2. * f64::from(range.runs)),
        direction: f64::from(correct) / f64::from(directional),
        correct,
        directional,
        informative,
    }
}

#[non_exhaustive]
pub struct GainComparison {
    pub participants: Vec<(Condition, Vec<GainRun>)>,
    pub differences: Vec<(Condition, crate::Estimate, crate::Effect)>,
    pub saturated: bool,
}
pub fn compare_gain(
    controls: [&GainSummary; 4],
    fixed: &FixedPoint,
    mut measure: impl FnMut(Condition) -> Vec<GainRun>,
) -> Option<GainComparison> {
    let [frozen, ideal, oracle, participant_oracle] = controls;
    if !fixed.passes() || !controls_pass(frozen, ideal, &[oracle, participant_oracle]) {
        return None;
    }
    let participants: Vec<_> = crate::PARTICIPANTS
        .iter()
        .map(|&id| (id, measure(id)))
        .collect();
    let full = &participants[0].1;
    let differences = participants[1..]
        .iter()
        .map(|(id, runs)| {
            let delta: Vec<_> = full
                .iter()
                .zip(runs)
                .map(|(a, b)| gain_area(&a.curve) - gain_area(&b.curve))
                .collect();
            let estimate = crate::bootstrap(&delta);
            (*id, estimate, crate::effect(estimate))
        })
        .collect();
    let saturated = participants[..2]
        .iter()
        .all(|(_, r)| gain_summary(r).curve[1].q >= object::FLOOR);
    Some(GainComparison {
        participants,
        differences,
        saturated,
    })
}

#[cfg(test)]
mod tests;
