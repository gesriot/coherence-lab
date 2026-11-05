use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn assert_view(state: &State, episode: &Episode, policy: Policy) {
    assert_eq!(
        serde_json::to_vec(&state.view()).expect("snapshot serializes"),
        serde_json::to_vec(&episode.snapshot(policy)).expect("snapshot serializes")
    );
}

// Independent transcription of the event handlers; never calls State's configuration builder.
struct Reference {
    episode: Episode,
    scenario: Scenario,
    seed: u32,
    cfg: Config,
    shown: Policy,
    running: bool,
    speed: Speed,
    truth: bool,
    hypotheses: bool,
}
impl Reference {
    fn new() -> Self {
        Self {
            episode: Episode::new(Scenario::Continuity, 4, Policy::Full),
            scenario: Scenario::Continuity,
            seed: 4,
            cfg: Config::default(),
            shown: Policy::Full,
            running: true,
            speed: Speed::One,
            truth: true,
            hypotheses: true,
        }
    }
    fn act(&mut self, action: &Action) {
        let reset = match action {
            Action::ToggleRunning => {
                self.running = !self.running;
                false
            }
            Action::Step => {
                self.running = false;
                self.episode.step(&self.cfg);
                self.shown = self.cfg.ablation;
                false
            }
            Action::Reset => true,
            Action::Speed(s) => {
                self.speed = *s;
                false
            }
            Action::Truth => {
                self.truth = !self.truth;
                false
            }
            Action::Hypotheses => {
                self.hypotheses = !self.hypotheses;
                false
            }
            Action::Pick(x, y) => {
                if (0. ..=100.).contains(x) && (0. ..=100.).contains(y) {
                    self.cfg.pin_x = *x;
                    self.cfg.pin_y = *y;
                    self.cfg.pin_gaze = true;
                }
                false
            }
            Action::Scenario(s) => {
                self.scenario = *s;
                self.cfg.replay = false;
                self.cfg.pin_gaze = false;
                self.cfg.inject_confidence = false;
                true
            }
            Action::Seed(s) => match s.parse::<f64>() {
                Ok(n) if n.is_finite() => {
                    self.seed = n.trunc().clamp(1., 999.) as u32;
                    true
                }
                _ => false,
            },
            Action::Policy(p) => {
                self.cfg.ablation = *p;
                false
            }
            Action::Pin => {
                self.cfg.pin_gaze = !self.cfg.pin_gaze;
                false
            }
            Action::Inject => {
                if matches!(self.cfg.ablation, Policy::Full | Policy::RecordOnly) {
                    self.cfg.inject_confidence = !self.cfg.inject_confidence;
                }
                false
            }
            Action::Replay => {
                self.cfg.replay = !self.cfg.replay;
                false
            }
            Action::Noise(n) => {
                self.cfg.extra_noise = f64::from((*n).min(48));
                false
            }
        };
        if reset {
            self.episode = Episode::new(self.scenario, self.seed, self.cfg.ablation);
            self.shown = self.cfg.ablation;
        }
    }
    fn check(&self, s: &State) {
        assert_view(s, &self.episode, self.shown);
        assert_eq!(
            (s.running, s.speed, s.show_truth, s.show_hypotheses),
            (self.running, self.speed, self.truth, self.hypotheses)
        );
        assert_eq!(
            (s.scenario, s.seed, s.policy),
            (self.scenario, self.seed, self.cfg.ablation)
        );
        assert_eq!(
            (s.pin_gaze, s.inject, s.replay),
            (
                self.cfg.pin_gaze,
                self.cfg.inject_confidence,
                self.cfg.replay
            )
        );
        assert_eq!(s.pin, [self.cfg.pin_x, self.cfg.pin_y]);
        assert_eq!(f64::from(s.extra_noise), self.cfg.extra_noise);
    }
}
fn check_actions(actions: impl IntoIterator<Item = Action>) {
    let mut state = State::default();
    let mut reference = Reference::new();
    reference.check(&state);
    for (i, action) in actions.into_iter().enumerate() {
        reference.act(&action);
        state.act(action, Duration::from_millis(i as u64));
        reference.check(&state);
    }
}
#[test]
fn every_action_preserves_or_replaces_the_expected_episode() {
    check_actions([
        Action::Step,
        Action::ToggleRunning,
        Action::Speed(Speed::Two),
        Action::Speed(Speed::Four),
        Action::Truth,
        Action::Hypotheses,
        Action::Pick(30., 70.),
        Action::Inject,
        Action::Replay,
        Action::Noise(48),
        Action::Step,
        Action::Reset,
        Action::Step,
        Action::Scenario(Scenario::Conflict),
        Action::Seed("2".into()),
        Action::Step,
        Action::Policy(Policy::FirstOrder),
        Action::Inject,
        Action::Step,
        Action::Policy(Policy::RecordOnly),
        Action::Inject,
        Action::Pin,
        Action::Replay,
        Action::Step,
        Action::Pick(-1., 50.),
        Action::Pick(101., 50.),
        Action::Pick(f64::NAN, 50.),
        Action::Pick(0., 100.),
    ]);
}
#[test]
fn sixty_mixed_actions_match_direct_episode_calls_after_every_event() {
    let actions = (0..60).map(|i| match i % 12 {
        0 => Action::Scenario(
            [
                Scenario::Continuity,
                Scenario::Vanish,
                Scenario::Binding,
                Scenario::Conflict,
                Scenario::Nursery,
            ][i / 12],
        ),
        1 => Action::Seed(format!("{}", i + 2)),
        2 => Action::Step,
        3 => Action::Policy(
            [
                Policy::Full,
                Policy::FirstOrder,
                Policy::Novelty,
                Policy::RecordOnly,
                Policy::DummyMemory,
            ][i / 12],
        ),
        4 => Action::Pick(30., 70.),
        5 => Action::Inject,
        6 => Action::Step,
        7 => Action::Replay,
        8 => Action::Noise((i % 49) as u8),
        9 => Action::Step,
        10 => Action::Reset,
        _ => Action::Step,
    });
    check_actions(actions);
}
#[test]
fn seed_clamps_truncates_ignores_invalid_and_resets_even_when_unchanged() {
    for (input, want, reset) in [
        ("0", 1, true),
        ("1", 1, true),
        ("999", 999, true),
        ("1000", 999, true),
        ("4.7", 4, true),
        ("4", 4, true),
        ("", 4, false),
        ("abc", 4, false),
        ("NaN", 4, false),
        ("inf", 4, false),
    ] {
        let mut s = State::default();
        s.act(Action::Step, Duration::ZERO);
        s.act(Action::Seed(input.into()), Duration::ZERO);
        assert_eq!(s.seed, want);
        assert_eq!(s.view().step, if reset { 0 } else { 1 });
    }
}
#[test]
fn timer_uses_exact_delays_rearms_only_for_running_or_speed_and_never_catches_up() {
    for (speed, delay) in [(Speed::One, 460), (Speed::Two, 230), (Speed::Four, 110)] {
        let mut s = State::default();
        s.act(Action::Speed(speed), Duration::ZERO);
        s.tick(Duration::from_millis(delay - 1));
        assert_eq!(s.view().step, 0);
        s.tick(Duration::from_millis(delay));
        assert_eq!(s.view().step, 1);
        s.act(Action::Reset, Duration::from_millis(delay + 1));
        s.tick(Duration::from_millis(2 * delay));
        assert_eq!(s.view().step, 1);
        s.tick(Duration::from_millis(10 * delay));
        assert_eq!(s.view().step, 2);
        s.act(Action::ToggleRunning, Duration::from_millis(10 * delay));
        s.tick(Duration::from_millis(100 * delay));
        assert_eq!(s.view().step, 2);
        s.act(Action::ToggleRunning, Duration::from_millis(100 * delay));
        s.tick(Duration::from_millis(101 * delay));
        assert_eq!(s.view().step, 3);
    }
}
#[test]
fn worker_starts_once_and_reports_errors_and_panics() {
    for panic in [false, true] {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let mut s = State::default();
        s.start_worker(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            assert!(!panic, "test worker failure");
            Err(())
        });
        s.start_worker(|| panic!("worker must not start twice"));
        let start = std::time::Instant::now();
        while matches!(s.bench, BenchState::Pending) && start.elapsed() < Duration::from_secs(2) {
            s.poll_bench();
            thread::yield_now();
        }
        assert!(matches!(s.bench, BenchState::Failed));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(s.bench.pending_label(), "Не удалось рассчитать бенчмарк.");
    }
}
