use crate::task::object as task;
use crate::{
    Episode,
    experience::{Draw, Generator, Learner, Rates},
    learning::{Models, probabilities},
    training::{Condition, nearest, train},
    types::*,
    world::hidden,
};
use jscompat::{ceil, floor, hypot, round};
use serde::Serialize;

pub(crate) const TRAINING_STEPS: u32 = 80;
pub const STAGES: [u32; 6] = [0, 4, 8, 16, 32, 64];
pub const MOTION_RATES: [f64; 4] = [0.03, 0.1, 0.3, 1.];
pub const EXISTENCE_RATES: [f64; 4] = [0.003, 0.01, 0.03, 0.1];
pub const ORACLE_RATES: Rates = Rates::new(0.03, 0.01);
pub const SELECTED_RATES: Rates = Rates::new(0.03, 0.1);
pub const EVAL_COUNT: u32 = 20;
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct SeedRange {
    pub first_run: u32,
    pub runs: u32,
    pub eval: u32,
}
impl SeedRange {
    pub const fn new(first_run: u32, runs: u32, eval: u32) -> Self {
        Self {
            first_run,
            runs,
            eval,
        }
    }
    pub fn seeds(self) -> std::ops::Range<u32> {
        self.first_run..self.first_run + self.runs
    }
}
pub const GENESIS_TUNING: SeedRange = SeedRange::new(521, 10, 2041);
pub const GENESIS_COMPARISON: SeedRange = SeedRange::new(61, 30, 1041);
pub(crate) fn curriculum(episode: u32) -> Scenario {
    use Scenario::*;
    [
        Continuity, Vanish, Binding, Nursery, Continuity, Binding, Nursery,
    ][(episode as usize - 1) % 7]
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct Point {
    pub accuracy: f64,
    pub retention: f64,
    pub ghost_free: f64,
    pub untracked: u32,
    pub motion_error: f64,
    pub hidden: f64,
    pub open: f64,
    pub miss: f64,
}
#[derive(Debug, PartialEq)]
enum Disappearance {
    Untracked,
    Ghost,
    Cleared,
}
fn disappearance(m: &Models, seed: u32, cfg: &Config, body: Option<[f64; 2]>) -> Disappearance {
    let mut ep = Episode::with_models(Scenario::Vanish, seed, cfg.ablation, Some(m.clone()), body);
    let at = ep.world.objects[0]
        .vanish_at
        .expect("the vanish scenario has a scheduled disappearance");
    while ep.world.t < at - 1 {
        ep.step(cfg);
    }
    let Some(id) = ep.snapshot(cfg.ablation).evaluator.objects[0].nearest_track else {
        return Disappearance::Untracked;
    };
    while ep.world.t < at + task::GHOST_DELAY {
        ep.step(cfg);
    }
    if ep.agent.tracks.iter().any(|t| t.id == id) {
        Disappearance::Ghost
    } else {
        Disappearance::Cleared
    }
}
pub(crate) fn evaluate(
    m: &Models,
    policy: Policy,
    first: u32,
    seeds: u32,
    body: Option<[f64; 2]>,
) -> Point {
    let cfg = Config::new(policy);
    let (mut held, mut clean, mut untracked, mut predictions) = (0, 0, 0, 0);
    let mut error = 0.;
    for seed in first..first + seeds {
        let mut ep =
            Episode::with_models(Scenario::Continuity, seed, policy, Some(m.clone()), body);
        let mut was_hidden = false;
        let mut exit = 0;
        for step in 1..=72 {
            let forecasts: Vec<_> = ep
                .agent
                .tracks
                .iter()
                .filter(|t| t.bound_blob.is_some())
                .map(|t| (nearest(&ep.world.objects, t.x, t.y), t.x + t.vx, t.y + t.vy))
                .collect();
            ep.step(&cfg);
            for (i, x, y) in forecasts {
                let o = &ep.world.objects[i];
                error += hypot(x - o.x, y - o.y);
                predictions += 1;
            }
            if hidden(&ep.world.objects[0], ep.world.wall) {
                was_hidden = true;
            } else if was_hidden && exit == 0 {
                exit = step;
            }
            if exit != 0 && step == exit + 1 && ep.snapshot(policy).evaluator.retained == Some(true)
            {
                held += 1;
            }
        }
        match disappearance(m, seed, &cfg, body) {
            Disappearance::Untracked => untracked += 1,
            Disappearance::Cleared => clean += 1,
            _ => (),
        }
    }
    let [hidden, open, miss] = probabilities(m);
    Point {
        accuracy: f64::from(held + clean) / f64::from(2 * seeds),
        retention: f64::from(held) / f64::from(seeds),
        ghost_free: f64::from(clean) / f64::from(seeds),
        untracked,
        motion_error: error / f64::from(predictions),
        hidden,
        open,
        miss,
    }
}
pub(crate) fn train_curve<T>(
    id: Condition,
    seed: u32,
    rates: Rates,
    stages: &[u32],
    mut measure: impl FnMut(&Models, u32) -> T,
) -> Vec<T> {
    let mut learner = Learner::new(rates, id.selection(), seed);
    let mut curve = Vec::new();
    for n in 0..=stages.last().copied().unwrap_or(0) {
        if n > 0 {
            train(
                &mut learner,
                id,
                curriculum(n),
                10000 + 100 * seed + n,
                None,
            );
        }
        if stages.contains(&n) {
            curve.push(measure(&learner.models, n));
        }
    }
    curve
}
pub fn trial(id: Condition, seed: u32, eval: u32, rates: Rates) -> Vec<Point> {
    train_curve(id, seed, rates, &STAGES, |m, _| {
        evaluate(m, id.policy(), eval, EVAL_COUNT, None)
    })
}
#[derive(Serialize)]
#[non_exhaustive]
pub struct TrainingFrame {
    episode: u32,
    weights: [[f64; 5]; 2],
    logits: [f64; 3],
    draws: Vec<Draw>,
    point: Point,
}
pub fn training_trace(id: Condition, seed: u32, count: u32) -> Vec<TrainingFrame> {
    let mut learner = Learner::new(SELECTED_RATES, id.selection(), seed);
    (0..=count)
        .map(|episode| {
            let mut draws = Vec::new();
            if episode > 0 {
                train(
                    &mut learner,
                    id,
                    curriculum(episode),
                    10000 + 100 * seed + episode,
                    Some(&mut draws),
                );
            }
            TrainingFrame {
                episode,
                draws,
                weights: learner.models.weights,
                logits: learner.models.logits,
                point: evaluate(&learner.models, id.policy(), 3001, 2, None),
            }
        })
        .collect()
}
pub fn average(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = values.fold((0., 0u32), |(s, n), x| (s + x, n + 1));
    sum / f64::from(n)
}
pub fn area(curve: &[Point]) -> f64 {
    average(curve.iter().skip(1).map(|p| p.accuracy))
}
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Estimate {
    pub mean: f64,
    pub low: f64,
    pub high: f64,
}
pub fn bootstrap(values: &[f64]) -> Estimate {
    let mut random = Generator::new(12345);
    let mut samples: Vec<_> = (0..2000)
        .map(|_| {
            average(
                values
                    .iter()
                    .map(|_| values[floor(random.next() * values.len() as f64) as usize]),
            )
        })
        .collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let quantile = |p: f64| {
        let index = p * (samples.len() - 1) as f64;
        let lo = floor(index);
        samples[lo as usize] + (index - lo) * (samples[ceil(index) as usize] - samples[lo as usize])
    };
    Estimate {
        mean: average(values.iter().copied()),
        low: quantile(0.025),
        high: quantile(0.975),
    }
}
pub fn score_interval(values: &[f64], units: f64) -> Estimate {
    let e = bootstrap(&values.iter().map(|x| round(x * units)).collect::<Vec<_>>());
    Estimate {
        mean: e.mean / units,
        low: e.low / units,
        high: e.high / units,
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Effect {
    Faster,
    Slower,
    None,
}
pub fn effect(e: Estimate) -> Effect {
    if e.mean > 0.10 && e.low > 0. {
        Effect::Faster
    } else if e.mean < -0.10 && e.high < 0. {
        Effect::Slower
    } else {
        Effect::None
    }
}
impl Effect {
    pub fn acceleration(self) -> &'static str {
        match self {
            Self::Faster => "подтверждено на этих конфигурациях",
            Self::None => "не подтверждено",
            Self::Slower => "контур замедляет обучение",
        }
    }
    pub fn protocol(self) -> &'static str {
        match self {
            Self::Faster => "подтверждён на этих конфигурациях",
            Self::None => "не подтверждён",
            Self::Slower => "протокол замедляет обучение",
        }
    }
}
pub fn replacement(first: Effect, alt: Effect) -> &'static str {
    if first != Effect::Faster {
        "не применимо: ускорения относительно первого порядка нет"
    } else if alt == Effect::Faster {
        "исключено"
    } else {
        "не исключено"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feasibility::{Control, reference_model};
    #[test]
    fn last_visible_identity_distinguishes_early_clearing_untracked_and_ghost() {
        let mut m = reference_model(Control::Ideal);
        m.logits[2] = jscompat::log(0.0005 / (1. - 0.0005));
        let cfg = Config::new(Policy::FirstOrder);
        let mut ep =
            Episode::with_models(Scenario::Vanish, 3001, cfg.ablation, Some(m.clone()), None);
        let at = ep.world.objects[0]
            .vanish_at
            .expect("vanish world schedules a loss");
        while ep.world.t < at - 1 {
            ep.step(&cfg);
        }
        let id = ep.snapshot(cfg.ablation).evaluator.objects[0]
            .nearest_track
            .expect("A is tracked on the last visible frame");
        ep.step(&cfg);
        assert!(!ep.agent.tracks.iter().any(|t| t.id == id));
        let p = evaluate(&m, cfg.ablation, 3001, 1, None);
        assert_eq!((p.untracked, p.ghost_free), (0, 1.));
        assert_eq!(
            disappearance(
                &Models::default(),
                3001,
                &Config {
                    pin_gaze: true,
                    pin_x: 90.,
                    pin_y: 90.,
                    ..cfg
                },
                None
            ),
            Disappearance::Untracked
        );
        let keep = evaluate(
            &reference_model(Control::KeepAll),
            cfg.ablation,
            3001,
            1,
            None,
        );
        assert_eq!((keep.untracked, keep.ghost_free), (0, 0.));
        assert!(evaluate(&Models::default(), cfg.ablation, 1, 8, None).retention <= 5. / 8.);
    }
    #[test]
    fn disk_vanishes_on_schedule_from_sensor_but_stays_in_world() {
        let cfg = Config {
            pin_gaze: true,
            pin_x: 26.,
            pin_y: 34.,
            ..Config::default()
        };
        for seed in 1..=20 {
            let mut ep = Episode::new(Scenario::Vanish, seed, Policy::Full);
            let at = ep.world.objects[0]
                .vanish_at
                .expect("vanish world has a deadline");
            assert!((20..=23).contains(&at));
            for i in 1..=40 {
                ep.step(&cfg);
                assert_eq!(
                    ep.snapshot(cfg.ablation).evaluator.objects[0].vanished,
                    i >= at
                );
                assert_eq!(ep.agent.cells.iter().any(|c| c.saturation > 0.), i < at);
            }
        }
    }
    #[test]
    fn post_training_area_bootstrap_and_effect_boundaries() {
        let mut points = [Point::default(); 6];
        points[0].accuracy = 1.;
        points[1].accuracy = 3. / 40.;
        assert_eq!(round(area(&points) * 200.), 3.);
        let sample = [-0.1, 0.2, 0.4, 0.25];
        assert_eq!(bootstrap(&sample), bootstrap(&sample));
        use Effect::*;
        for (mean, low, high, want) in [
            (0.10, 0.01, 0.25, None),
            (0.11, 0., 0.25, None),
            (0.11, 0.01, 0.25, Faster),
            (-0.10, -0.25, -0.01, None),
            (-0.11, -0.25, 0., None),
            (-0.11, -0.25, -0.01, Slower),
        ] {
            assert_eq!(effect(Estimate { mean, low, high }), want);
        }
        assert_eq!(replacement(Faster, Faster), "исключено");
        assert_eq!(replacement(Faster, None), "не исключено");
        for e in [None, Slower] {
            assert!(replacement(e, Faster).starts_with("не применимо"));
        }
    }
    #[test]
    fn seeded_training_reproduces_held_out_curve_and_seed_ranges() {
        let run = || {
            train_curve(
                Condition::FirstOrder,
                701,
                Rates::new(0.03, 0.003),
                &[0, 1],
                |m, _| evaluate(m, Policy::FirstOrder, 3001, 1, None),
            )
        };
        assert_eq!(run(), run());
        assert_eq!(STAGES, [0, 4, 8, 16, 32, 64]);
        assert_eq!(TRAINING_STEPS, 80);
        assert_eq!(EVAL_COUNT, 20);
        assert_eq!(
            (
                GENESIS_TUNING.first_run,
                GENESIS_TUNING.runs,
                GENESIS_TUNING.eval
            ),
            (521, 10, 2041)
        );
        assert_eq!(
            (
                GENESIS_COMPARISON.first_run,
                GENESIS_COMPARISON.runs,
                GENESIS_COMPARISON.eval
            ),
            (61, 30, 1041)
        );
        assert_eq!(ORACLE_RATES, Rates::new(0.03, 0.01));
        assert_eq!(EXISTENCE_RATES, [0.003, 0.01, 0.03, 0.1]);
    }
}
