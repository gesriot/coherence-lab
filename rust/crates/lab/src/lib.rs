mod attention;
mod bench;
mod episode;
mod learning;
mod observation;
mod perception;
mod sensor;
mod types;
mod world;

pub use bench::{BenchReport, BenchRow, Proportion, ReplayResult, Verdict, run_bench};
pub use episode::{Episode, Snapshot};
pub use learning::{Context, GapObservation, GapStep, Models, fit_existence, fit_motion};
pub use types::{Config, LabError, Policy, Scenario};

mod experience;
mod feasibility;
mod genesis;
mod training;
pub use experience::Rates;
pub use feasibility::{
    CONTROLS, Control, DesignPoint, DesignReport, control_runs, genesis_design, permits, stable,
};
pub use genesis::{
    EVAL_COUNT, EXISTENCE_RATES, Effect, Estimate, GENESIS_COMPARISON, GENESIS_TUNING,
    MOTION_RATES, ORACLE_RATES, Point, SELECTED_RATES, STAGES, SeedRange, TrainingFrame, area,
    average, bootstrap, effect, replacement, score_interval, training_trace, trial,
};
pub use training::{Condition, PARTICIPANTS};
