mod element;
mod evaluation;
mod experiment;
mod gradient;
mod input;
mod matrix;
mod motion;
mod persistence;
mod probes;
mod report;
mod signals;
mod tracking;
mod training;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Недопустимые размеры решётки или длина обучения")]
    Dimensions,
    #[error("Отсутствует копия команды")]
    MissingCommand,
    #[error("Для положительного контроля нужны метки")]
    MissingHint,
    #[error("Обучающие и проверочные зёрна зондов пересекаются")]
    SeedOverlap,
    #[error("Нет опорной ошибки копии в выборке")]
    NoBaseline,
    #[error("Нет примеров для зонда")]
    NoExamples,
    #[error("Поток прогона прерван")]
    Worker,
}
#[derive(Clone, Copy, Debug)]
pub struct EvaluationSettings {
    pub episodes: u32,
    pub steps: usize,
    pub first_probe_seed: u32,
    pub first_test_seed: u32,
}
impl Default for EvaluationSettings {
    fn default() -> Self {
        Self {
            episodes: 30,
            steps: 72,
            first_probe_seed: 4001,
            first_test_seed: 3001,
        }
    }
}
impl EvaluationSettings {
    pub(crate) fn probe_seeds(self) -> Vec<u32> {
        (self.first_probe_seed..self.first_probe_seed + self.episodes).collect()
    }
    pub(crate) fn test_seeds(self) -> Vec<u32> {
        (self.first_test_seed..self.first_test_seed + self.episodes).collect()
    }
    pub(crate) fn probe_range(self) -> String {
        format!(
            "{}–{}",
            self.first_probe_seed,
            self.first_probe_seed + self.episodes - 1
        )
    }
    pub(crate) fn test_range(self) -> String {
        format!(
            "{}–{}",
            self.first_test_seed,
            self.first_test_seed + self.episodes - 1
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Scale {
    pub grid: usize,
    pub depth: usize,
    pub episodes: usize,
    pub runs: usize,
    pub steps: usize,
    pub memory: usize,
    pub evaluation: EvaluationSettings,
}
impl Default for Scale {
    fn default() -> Self {
        Self {
            grid: 16,
            depth: 16,
            episodes: 2000,
            runs: 5,
            steps: 80,
            memory: 30,
            evaluation: EvaluationSettings::default(),
        }
    }
}
impl Scale {
    pub(crate) fn validate(self) -> Result<(), Error> {
        if [
            self.grid,
            self.depth,
            self.episodes,
            self.runs,
            self.steps,
            self.memory,
            self.evaluation.steps,
            self.evaluation.episodes as usize,
        ]
        .contains(&0)
            || self
                .evaluation
                .first_probe_seed
                .checked_add(self.evaluation.episodes)
                .is_none()
            || self
                .evaluation
                .first_test_seed
                .checked_add(self.evaluation.episodes)
                .is_none()
        {
            Err(Error::Dimensions)
        } else {
            Ok(())
        }
    }
}
pub use element::Kind;
pub use experiment::{TrainingOptions, TrainingTrace, training_trace};
pub use input::{CopyMode, Gain, Motion, WorldSettings};

pub use persistence::{PersistenceSettings, b0, b1};

pub use motion::{MotionExperiment, motion_experiment};
pub use tracking::{TrackingSettings, b1b};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn eight_by_eight_trains_and_evaluates_every_experiment() -> Result<(), Error> {
        let scale = Scale {
            grid: 8,
            depth: 4,
            runs: 1,
            episodes: 2,
            steps: 32,
            evaluation: EvaluationSettings {
                episodes: 6,
                ..Default::default()
            },
            ..Default::default()
        };
        let p = PersistenceSettings {
            scale,
            control_runs: 1,
            wide_depth: 4,
        };
        let b0 = b0(p, |_| {})?;
        assert!(b0.contains("2 независимых обучений по 2 эпизодов × 32 шагов"));
        assert!(b0.contains("Жизнеспособность B0:"));
        let b1 = b1(p, |_| {})?;
        assert!(b1.contains("Память 4"));
        assert!(b1.contains("Оценка: 3001–3006; обучение зондов: 4001–4006"));
        let tracking = b1b(
            TrackingSettings {
                scale,
                wide_runs: 1,
                wide_weighted_episodes: 3,
                checkpoint: 2,
            },
            |_| {},
        )?;
        assert!(tracking.contains("4 · Вес ×50 · среднее\t3\t"));
        for random_path in [false, true] {
            let text = motion_experiment(MotionExperiment { scale, random_path }, |_| {})?;
            assert!(text.contains("Подвижное окно 8 × 8"));
            assert_eq!(text.contains("Только контекст"), random_path);
        }
        Ok(())
    }
    #[test]
    fn prediction_and_gradient_modules_have_no_access_to_experiment_data() {
        for (name, source) in [
            ("element", include_str!("element.rs")),
            ("gradient", include_str!("gradient.rs")),
            ("matrix", include_str!("matrix.rs")),
            ("training", include_str!("training.rs")),
            ("signals", include_str!("signals.rs")),
        ] {
            let (core, _) = source.split_once("#[cfg(test)]").unwrap_or((source, ""));
            // Check dependency paths, independent of local variable names.
            let compact: String = core.split_whitespace().collect();
            for layer in [
                "lab",
                "input",
                "experiment",
                "evaluation",
                "probes",
                "motion",
                "tracking",
                "persistence",
            ] {
                assert!(
                    !compact.contains(&format!("{layer}::"))
                        && !compact.contains(&format!("crate::{layer}")),
                    "{name} imports {layer}"
                );
            }
        }
    }
}
