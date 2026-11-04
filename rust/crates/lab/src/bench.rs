use crate::{
    Episode,
    types::{Config, Look, Policy, Scenario},
};
use jscompat::{round, sqrt};

const CONFIGURATIONS: u32 = 200;
const TASK_MARGIN: f64 = 0.2;
const AUC_THRESHOLD: f64 = 0.9;
const CONDITIONS: [(Policy, &str, &str); 6] = [
    (
        Policy::Full,
        "Полный контур",
        "Модель восприятия двигает взгляд",
    ),
    (
        Policy::FirstOrder,
        "Первый порядок",
        "Прогноз движения и обход по карте просмотра",
    ),
    (
        Policy::Novelty,
        "Новизна",
        "Приоритет самого молодого трека, если его возраст не превышает шести шагов",
    ),
    (
        Policy::RecordOnly,
        "Только запись",
        "Модель есть, на действие не влияет",
    ),
    (
        Policy::NoSchema,
        "Без модели",
        "Взгляд направлен к самому крупному пятну",
    ),
    (
        Policy::DummyMemory,
        "Лишняя память",
        "Буфер кадров без схемы внимания",
    ),
];

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Proportion {
    pub k: u32,
    pub n: u32,
    pub rate: f64,
    pub low: f64,
    pub high: f64,
}
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct BenchRow {
    pub id: Policy,
    pub label: &'static str,
    pub detail: &'static str,
    pub retention: Proportion,
    pub pair: Proportion,
    pub explore_rate: f64,
    pub self_error: Option<f64>,
    pub action_error: f64,
}
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ReplayResult {
    pub label: &'static str,
    pub live_error: f64,
    pub replay_error: f64,
    pub auc: f64,
    pub median_difference: f64,
    pub higher: Proportion,
    pub detected: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Verdict {
    Supported,
    Unsupported,
    Mixed,
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct BenchReport {
    pub configurations: u32,
    pub rows: Vec<BenchRow>,
    pub replay: Vec<ReplayResult>,
    pub saturation: Vec<String>,
    pub unequal_exploration: bool,
    pub alternative_excluded: bool,
    pub causal: Verdict,
}

fn wilson(k: u32, n: u32) -> Proportion {
    let size = f64::from(n);
    let z = 1.959963984540054;
    let rate = f64::from(k) / size;
    let denominator = 1. + z * z / size;
    let center = (rate + z * z / (2. * size)) / denominator;
    let half = z / denominator * sqrt(rate * (1. - rate) / size + z * z / (4. * (size * size)));
    Proportion {
        k,
        n,
        rate,
        low: if k == 0 { 0. } else { center - half },
        high: if k == n { 1. } else { center + half },
    }
}
fn mean(values: &[f64]) -> f64 {
    values.iter().fold(0., |sum, v| sum + v) / values.len() as f64
}
fn replay_stats(label: &'static str, live: &[f64], replay: &[f64]) -> ReplayResult {
    assert!(
        !live.is_empty() && live.len() == replay.len(),
        "live and replay scores must come from paired episodes"
    );
    let mut wins = 0.;
    for positive in replay {
        for negative in live {
            if positive > negative {
                wins += 1.;
            } else if positive == negative {
                wins += 0.5;
            }
        }
    }
    let auc = wins / (live.len() * replay.len()) as f64;
    let mut differences: Vec<_> = replay.iter().zip(live).map(|(r, l)| r - l).collect();
    differences.sort_by(f64::total_cmp);
    let n = differences.len();
    ReplayResult {
        label,
        live_error: mean(live),
        replay_error: mean(replay),
        auc,
        median_difference: (differences[n / 2] + differences[(n - 1) / 2]) / 2.,
        higher: wilson(
            differences.iter().filter(|v| **v > 0.).count() as u32,
            n as u32,
        ),
        detected: auc >= AUC_THRESHOLD,
    }
}
fn beats_control(full: &BenchRow, control: &BenchRow) -> bool {
    [
        (full.retention, control.retention),
        (full.pair, control.pair),
    ]
    .iter()
    .any(|(a, b)| f64::from(a.k) - f64::from(b.k) > TASK_MARGIN * f64::from(b.n))
}
fn assess(full: &BenchRow, first: &BenchRow, record: &BenchRow, memory: &BenchRow) -> Verdict {
    if !beats_control(full, first) {
        Verdict::Unsupported
    } else if beats_control(full, record) && beats_control(full, memory) {
        Verdict::Supported
    } else {
        Verdict::Mixed
    }
}
fn saturation_notes(full: &BenchRow, first: &BenchRow) -> Vec<String> {
    let mut notes = Vec::new();
    for (name, a, b) in [
        ("Удержание A", full.retention.rate, first.retention.rate),
        ("Пара", full.pair.rate, first.pair.rate),
    ] {
        if (a <= 0.05 && b <= 0.05) || (a >= 0.95 && b >= 0.95) {
            notes.push(format!("Задача «{name}» не различает полный контур и первый порядок: оба результата у края шкалы."));
        }
    }
    notes
}

// Keep the evaluator's measurements after each step, while the episode owns the current frame.
struct Frame {
    occluded: bool,
    retained: bool,
    distinct_pair: bool,
    explored: bool,
    self_error: Option<f64>,
    action: f64,
    cells: usize,
}
fn roll(scenario: Scenario, policy: Policy, seed: u32, steps: usize, replay: bool) -> Vec<Frame> {
    let mut ep = Episode::new(scenario, seed, policy);
    let mut cfg = Config::new(policy);
    let mut frames = Vec::with_capacity(steps);
    for i in 0..steps {
        cfg.replay = replay && i >= 14;
        ep.step(&cfg);
        let evaluator = ep.snapshot(policy).evaluator;
        let first = &evaluator.objects[0];
        let second = &evaluator.objects[1];
        let distinct_pair = matches!(
            (first.nearest_track, second.nearest_track),
            (Some(a), Some(b)) if a != b
        );
        frames.push(Frame {
            occluded: first.occluded,
            retained: evaluator.retained == Some(true),
            distinct_pair,
            explored: ep.agent.attention.executed_reason == Look::Explore,
            self_error: ep.agent.energy.self_error,
            action: ep.agent.energy.action,
            cells: ep.agent.grid.pow(2),
        });
    }
    frames
}
fn retained(views: &[Frame]) -> bool {
    let Some(hidden) = views.iter().position(|v| v.occluded) else {
        return false;
    };
    let exit = views
        .iter()
        .enumerate()
        .skip(hidden + 1)
        .find(|(_, v)| !v.occluded);
    exit.is_some_and(|(i, _)| views[(i + 1).min(views.len() - 1)].retained)
}
fn pair_resolved(views: &[Frame]) -> bool {
    let mut streak = 0;
    for v in views {
        streak = if v.distinct_pair { streak + 1 } else { 0 };
        if streak >= 6 {
            return true;
        }
    }
    false
}
fn action_score(views: &[Frame]) -> f64 {
    let window = &views[16..40];
    let cells = window[0].cells;
    let mismatches = window
        .iter()
        .fold(0., |sum, v| sum + round(v.action * cells as f64));
    mismatches / (window.len() * cells) as f64
}
#[derive(Default)]
struct Exploration {
    explored: usize,
    steps: usize,
}
impl Exploration {
    fn add(&mut self, explored: usize, steps: usize) {
        self.explored += explored;
        self.steps += steps;
    }
    fn rate(&self) -> f64 {
        if self.steps == 0 {
            0.
        } else {
            self.explored as f64 / self.steps as f64
        }
    }
}
pub fn run_bench() -> BenchReport {
    let mut replay = Vec::new();
    let rows = CONDITIONS.map(|(id, label, detail)| {
        let (mut holds, mut pairs) = (0, 0);
        let mut exploration = Exploration::default();
        let mut selves = Vec::new();
        let mut live = Vec::new();
        let mut recordings = Vec::new();
        for seed in 1..=CONFIGURATIONS {
            let continuity = roll(Scenario::Continuity, id, seed, 72, false);
            let conflict = roll(Scenario::Conflict, id, seed, 200, false);
            holds += u32::from(retained(&continuity));
            pairs += u32::from(pair_resolved(&conflict));
            for episode in [&continuity, &conflict] {
                exploration.add(episode.iter().filter(|v| v.explored).count(), episode.len());
            }
            let errors: Vec<_> = continuity[..60]
                .iter()
                .filter_map(|v| v.self_error)
                .collect();
            if !errors.is_empty() {
                selves.push(mean(&errors));
            }
            live.push(action_score(&continuity));
            if matches!(id, Policy::Full | Policy::FirstOrder) {
                recordings.push(action_score(&roll(
                    Scenario::Continuity,
                    id,
                    seed,
                    40,
                    true,
                )));
            }
        }
        if !recordings.is_empty() {
            replay.push(replay_stats(label, &live, &recordings));
        }
        BenchRow {
            id,
            label,
            detail,
            retention: wilson(holds, CONFIGURATIONS),
            pair: wilson(pairs, CONFIGURATIONS),
            explore_rate: exploration.rate(),
            self_error: (!selves.is_empty()).then(|| mean(&selves)),
            action_error: mean(&live),
        }
    });
    let [full, first, novelty, record, _, memory] = &rows;
    BenchReport {
        configurations: CONFIGURATIONS,
        causal: assess(full, first, record, memory),
        alternative_excluded: beats_control(full, novelty),
        saturation: saturation_notes(full, first),
        unequal_exploration: full.explore_rate.max(first.explore_rate)
            > 1.5 * full.explore_rate.min(first.explore_rate),
        rows: rows.into(),
        replay,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(id: Policy, holds: u32, pairs: u32) -> BenchRow {
        BenchRow {
            id,
            label: "",
            detail: "",
            retention: wilson(holds, 200),
            pair: wilson(pairs, 200),
            explore_rate: 0.,
            self_error: None,
            action_error: 0.,
        }
    }
    #[test]
    fn wilson_matches_known_binomial_examples() {
        for (k, low, high) in [
            (0, 0., 0.161125),
            (10, 0.299298, 0.700702),
            (20, 0.838875, 1.),
        ] {
            let p = wilson(k, 20);
            assert!((p.low - low).abs() < 1e-6 && (p.high - high).abs() < 1e-6);
        }
    }
    #[test]
    fn replay_auc_counts_all_pairs_splits_ties_and_includes_threshold() {
        let mixed = replay_stats("", &[0., 1.], &[1., 2.]);
        assert_eq!(mixed.auc, 0.875);
        assert!(!mixed.detected);
        assert_eq!(mixed.median_difference, 1.);
        assert_eq!(mixed.higher.k, 2);
        assert_eq!(replay_stats("", &[1.], &[1.]).auc, 0.5);
        assert_eq!(replay_stats("", &[2.], &[1.]).auc, 0.);
        let boundary = replay_stats("", &[0., 1., 2., 3., 4.], &[4.; 5]);
        assert_eq!(boundary.auc, 0.9);
        assert!(boundary.detected);
    }
    #[test]
    fn causal_role_requires_a_strict_task_advantage() {
        let first = row(Policy::FirstOrder, 0, 0);
        let record = row(Policy::RecordOnly, 0, 0);
        let mut memory = row(Policy::DummyMemory, 0, 0);
        let mut full = row(Policy::Full, 40, 0);
        assert_eq!(
            assess(&full, &first, &record, &memory),
            Verdict::Unsupported
        );
        full.retention = wilson(41, 200);
        memory.retention = wilson(41, 200);
        assert_eq!(assess(&full, &first, &record, &memory), Verdict::Mixed);
        full.pair = wilson(41, 200);
        assert_eq!(assess(&full, &first, &record, &memory), Verdict::Supported);
    }
    #[test]
    fn saturation_requires_the_same_inclusive_edge() {
        let notes =
            |a, b, c, d| saturation_notes(&row(Policy::Full, a, b), &row(Policy::FirstOrder, c, d));
        let both = notes(10, 190, 0, 200);
        assert_eq!(both.len(), 2);
        assert!(both[0].contains("Удержание A") && both[1].contains("Пара"));
        assert!(notes(10, 190, 11, 189).is_empty());
        assert!(notes(0, 200, 200, 0).is_empty());
    }
    #[test]
    fn exploration_pools_steps_across_episodes() {
        let mut e = Exploration::default();
        e.add(1, 2);
        e.add(0, 8);
        assert_eq!(e.rate(), 0.1);
    }
    #[test]
    fn functional_replacement_uses_a_separate_strict_margin() {
        let full = row(Policy::Full, 170, 180);
        let novelty = |a, b| row(Policy::Novelty, a, b);
        assert!(!beats_control(&full, &novelty(130, 140)));
        assert!(beats_control(&full, &novelty(129, 180)));
        assert!(beats_control(&full, &novelty(170, 139)));
        assert!(!beats_control(&full, &novelty(200, 200)));
        assert_eq!(
            assess(
                &full,
                &row(Policy::FirstOrder, 0, 0),
                &row(Policy::RecordOnly, 0, 0),
                &row(Policy::DummyMemory, 0, 0)
            ),
            Verdict::Supported
        );
    }
}
