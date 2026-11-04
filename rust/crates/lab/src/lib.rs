mod alignment;
mod attention;
mod bench;
mod episode;
mod experience;
mod feasibility;
mod gain;
mod genesis;
mod learning;
mod observation;
mod perception;
mod sensor;
mod task;
mod training;
mod types;
mod world;

pub use bench::{BenchReport, BenchRow, Proportion, ReplayResult, Verdict, run_bench};
pub use episode::{Episode, Snapshot};
pub use learning::{Context, GapObservation, GapStep, Models, fit_existence, fit_motion};
pub use types::{Config, LabError, Policy, Scenario};

pub use experience::Rates;
pub use feasibility::{
    CONTROLS, Control, DesignPoint, DesignReport, control_runs, genesis_design, ideal_model,
    permits, stable,
};
pub use gain::{
    Calibration, FixedPoint, GAIN_COMPARISON, GAIN_CONTROLS, GAIN_DESIGN, GAIN_ORACLE, GAIN_RATES,
    GAIN_SELECTED, GAIN_STAGES, GAIN_TUNING, GainComparison, GainCondition, GainRun, GainSummary,
    compare_gain, controls_pass, design_gain_rate, fixed_point, gain_area, gain_runs, gain_summary,
};
pub use genesis::{
    EVAL_COUNT, EXISTENCE_RATES, Effect, Estimate, GENESIS_COMPARISON, GENESIS_TUNING,
    MOTION_RATES, ORACLE_RATES, Point, SELECTED_RATES, STAGES, SeedRange, TrainingFrame, area,
    average, bootstrap, effect, replacement, score_interval, training_trace, trial,
};
pub use training::{Condition, PARTICIPANTS};
