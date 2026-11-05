use lab::{BenchReport, Config, Episode, Policy, Scenario, Snapshot};
use std::{panic::catch_unwind, sync::mpsc, thread, time::Duration};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Speed {
    One,
    Two,
    Four,
}
impl Speed {
    pub(crate) fn delay(self) -> Duration {
        Duration::from_millis(match self {
            Self::One => 460,
            Self::Two => 230,
            Self::Four => 110,
        })
    }
}

pub enum BenchState {
    Pending,
    Ready(BenchReport),
    Failed,
}
impl BenchState {
    pub fn pending_label(&self) -> &'static str {
        if matches!(self, Self::Failed) {
            "Не удалось рассчитать бенчмарк."
        } else {
            "Идёт расчёт…"
        }
    }
}

#[derive(Clone, Debug)]
pub enum Action {
    ToggleRunning,
    Step,
    Reset,
    Speed(Speed),
    Truth,
    Hypotheses,
    Pick(f64, f64),
    Scenario(Scenario),
    Seed(String),
    Policy(Policy),
    Pin,
    Inject,
    Replay,
    Noise(u8),
}

pub struct State {
    pub(crate) scenario: Scenario,
    pub(crate) seed: u32,
    pub(crate) policy: Policy,
    pub(crate) pin_gaze: bool,
    pub(crate) pin: [f64; 2],
    pub(crate) inject: bool,
    pub(crate) replay: bool,
    pub(crate) extra_noise: u8,
    pub(crate) show_truth: bool,
    pub(crate) show_hypotheses: bool,
    pub(crate) running: bool,
    pub(crate) speed: Speed,
    pub(crate) bench: BenchState,
    episode: Episode,
    snapshot_policy: Policy,
    next_tick: Duration,
    receiver: Option<mpsc::Receiver<Result<BenchReport, ()>>>,
    bench_started: bool,
}
impl Default for State {
    fn default() -> Self {
        Self {
            scenario: Scenario::Continuity,
            seed: 4,
            policy: Policy::Full,
            pin_gaze: false,
            pin: [50., 50.],
            inject: false,
            replay: false,
            extra_noise: 0,
            show_truth: true,
            show_hypotheses: true,
            running: true,
            speed: Speed::One,
            bench: BenchState::Pending,
            episode: Episode::new(Scenario::Continuity, 4, Policy::Full),
            snapshot_policy: Policy::Full,
            next_tick: Speed::One.delay(),
            receiver: None,
            bench_started: false,
        }
    }
}
impl State {
    pub fn view(&self) -> Snapshot<'_> {
        self.episode.snapshot(self.snapshot_policy)
    }
    pub(crate) fn schema_enabled(&self) -> bool {
        matches!(self.policy, Policy::Full | Policy::RecordOnly)
    }
    fn step(&mut self) {
        let mut cfg = Config::default();
        cfg.ablation = self.policy;
        cfg.pin_gaze = self.pin_gaze;
        cfg.pin_x = self.pin[0];
        cfg.pin_y = self.pin[1];
        cfg.inject_confidence = self.inject;
        cfg.replay = self.replay;
        cfg.extra_noise = f64::from(self.extra_noise);
        self.episode.step(&cfg);
        self.snapshot_policy = self.policy;
    }
    fn reset(&mut self) {
        self.episode = Episode::new(self.scenario, self.seed, self.policy);
        self.snapshot_policy = self.policy;
    }
    pub fn act(&mut self, action: Action, now: Duration) {
        match action {
            Action::ToggleRunning => {
                self.running = !self.running;
                if self.running {
                    self.next_tick = now + self.speed.delay();
                }
            }
            Action::Step => {
                self.running = false;
                self.step();
            }
            Action::Reset => self.reset(),
            Action::Speed(speed) => {
                if self.speed != speed {
                    self.speed = speed;
                    self.next_tick = now + speed.delay();
                }
            }
            Action::Truth => self.show_truth = !self.show_truth,
            Action::Hypotheses => self.show_hypotheses = !self.show_hypotheses,
            Action::Pick(x, y) => {
                if x.is_finite()
                    && y.is_finite()
                    && (0. ..=100.).contains(&x)
                    && (0. ..=100.).contains(&y)
                {
                    self.pin = [x, y];
                    self.pin_gaze = true;
                }
            }
            Action::Scenario(scenario) => {
                self.scenario = scenario;
                self.replay = false;
                self.pin_gaze = false;
                self.inject = false;
                self.reset();
            }
            Action::Seed(value) => {
                if let Ok(value) = value.trim().parse::<f64>()
                    && value.is_finite()
                {
                    self.seed = value.trunc().clamp(1., 999.) as u32;
                    self.reset();
                }
            }
            Action::Policy(policy) => self.policy = policy,
            Action::Pin => self.pin_gaze = !self.pin_gaze,
            Action::Inject => {
                if self.schema_enabled() {
                    self.inject = !self.inject;
                }
            }
            Action::Replay => self.replay = !self.replay,
            Action::Noise(noise) => self.extra_noise = noise.min(48),
        }
    }
    pub fn tick(&mut self, now: Duration) {
        if self.running && now >= self.next_tick {
            self.step();
            // A delayed event loop delivers one interval callback, not a burst of old frames.
            let elapsed = now.saturating_sub(self.next_tick).as_millis();
            let delay = self.speed.delay();
            self.next_tick += delay.mul_f64((elapsed / delay.as_millis() + 1) as f64);
        }
        self.poll_bench();
    }
    pub fn repaint_after(&self, now: Duration) -> Option<Duration> {
        self.running.then(|| self.next_tick.saturating_sub(now))
    }
    pub(crate) fn worker_pending(&self) -> bool {
        self.receiver.is_some()
    }
    pub fn start_bench(&mut self) {
        self.start_worker(|| Ok(lab::run_bench()));
    }
    fn start_worker(&mut self, work: impl FnOnce() -> Result<BenchReport, ()> + Send + 'static) {
        if self.bench_started {
            return;
        }
        self.bench_started = true;
        let (sender, receiver) = mpsc::channel();
        match thread::Builder::new()
            .name("benchmark".into())
            .spawn(move || {
                let result = catch_unwind(std::panic::AssertUnwindSafe(work)).unwrap_or(Err(()));
                let _ = sender.send(result);
            }) {
            Ok(_) => self.receiver = Some(receiver),
            Err(_) => self.bench = BenchState::Failed,
        }
    }
    fn poll_bench(&mut self) {
        let Some(receiver) = &self.receiver else {
            return;
        };
        match receiver.try_recv() {
            Ok(Ok(report)) => {
                self.bench = BenchState::Ready(report);
                self.receiver = None;
            }
            Ok(Err(())) | Err(mpsc::TryRecvError::Disconnected) => {
                self.bench = BenchState::Failed;
                self.receiver = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }
}

#[cfg(test)]
mod tests;
