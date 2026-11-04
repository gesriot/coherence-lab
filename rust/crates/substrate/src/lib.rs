mod element;
mod experiment;
mod gradient;
mod input;
mod matrix;
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
    #[error("Для веса выхода нужен хук оценщика")]
    MissingWeight,
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
pub struct Scale {
    pub grid: usize,
    pub depth: usize,
    pub episodes: usize,
    pub runs: usize,
    pub steps: usize,
    pub memory: usize,
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
        ]
        .contains(&0)
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

mod evaluation;
mod persistence;
mod probes;
mod report;
pub use persistence::{PersistenceSettings, b0, b1};

mod motion;
mod tracking;
pub use motion::{MotionExperiment, motion_experiment};
pub use tracking::{TrackingSettings, b1b};
