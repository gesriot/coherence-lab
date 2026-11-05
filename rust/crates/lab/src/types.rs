use crate::learning::{Features, GapStep};
use serde::Serialize;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LabError {
    #[error("Недопустимое условие проверки выполнимости: {0}")]
    Control(String),
    #[error("unknown scenario {0}")]
    Scenario(String),
    #[error("unknown policy {0}")]
    Policy(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum Scenario {
    Continuity,
    Binding,
    Conflict,
    Nursery,
    Vanish,
}
impl std::str::FromStr for Scenario {
    type Err = LabError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "continuity" => Ok(Self::Continuity),
            "binding" => Ok(Self::Binding),
            "conflict" => Ok(Self::Conflict),
            "nursery" => Ok(Self::Nursery),
            "vanish" => Ok(Self::Vanish),
            _ => Err(LabError::Scenario(s.into())),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum Policy {
    Full,
    FirstOrder,
    Novelty,
    RecordOnly,
    NoSchema,
    DummyMemory,
}
impl std::str::FromStr for Policy {
    type Err = LabError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "full" => Ok(Self::Full),
            "first-order" => Ok(Self::FirstOrder),
            "novelty" => Ok(Self::Novelty),
            "record-only" => Ok(Self::RecordOnly),
            "no-schema" => Ok(Self::NoSchema),
            "dummy-memory" => Ok(Self::DummyMemory),
            _ => Err(LabError::Policy(s.into())),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum Shape {
    Disk,
    Square,
    Triangle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum Source {
    Observed,
    Predicted,
    Remembered,
    Mixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum Reason {
    None,
    Noise,
    Occlusion,
    Absence,
    Conflict,
    NoData,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum Agreement {
    Settled,
    Competing,
    Underdetermined,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Expectation {
    Surface,
    Occluder,
    Empty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum Look {
    Verify,
    Conflict,
    Uncertainty,
    Explore,
    Pursue,
    Salience,
    Pinned,
    Idle,
}

#[derive(Clone, Debug, PartialEq, Serialize, Copy, Default)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct Gaze {
    pub x: f64,
    pub y: f64,
    pub aperture: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Copy, Default)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct Cell {
    pub hue: f64,
    pub saturation: f64,
    pub lightness: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Copy)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct Wall {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct Object {
    pub id: &'static str,
    pub shape: Shape,
    pub hue: f64,
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub r: f64,
    pub vanish_at: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Copy)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Blob {
    pub(crate) index: usize,
    pub(crate) cx: f64,
    pub(crate) cy: f64,
    pub(crate) hue: f64,
    pub(crate) area: usize,
    pub(crate) fill: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Copy)]
#[serde(rename_all = "camelCase")]
pub(crate) struct MotionSample {
    pub(crate) step: u32,
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) cell_width: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct Track {
    pub id: u32,
    pub x: f64,
    pub y: f64,
    pub(crate) vx: f64,
    pub(crate) vy: f64,
    pub hue: f64,
    pub(crate) fill: f64,
    pub evidence: f64,
    pub(crate) prediction: f64,
    pub uncertainty: f64,
    pub confidence: f64,
    pub source: Source,
    pub reason: Reason,
    pub(crate) unseen: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) empty_views: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) seen_run: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) gap_step: Option<GapStep>,
    pub(crate) age: u32,
    pub(crate) bound_blob: Option<usize>,
    pub conflict: bool,
    pub in_workspace: bool,
    pub(crate) radius: f64,
    pub(crate) flow_error: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) forecast_input: Option<Features>,
    pub(crate) velocity_spread: f64,
    pub(crate) motion_samples: Vec<MotionSample>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct Attention {
    pub active: bool,
    pub influences_action: bool,
    pub focus_id: Option<u32>,
    pub reason: Look,
    pub executed_reason: Look,
    pub executed_focus_id: Option<u32>,
    pub agreement: Agreement,
    pub intended_x: f64,
    pub intended_y: f64,
    pub intended_aperture: f64,
    pub executed_x: f64,
    pub executed_y: f64,
    pub executed_aperture: f64,
    pub note: &'static str,
    pub predicted_observed: bool,
    pub predicted_drop: f64,
    pub self_error: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Copy, Default)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct Energy {
    pub sensory: f64,
    pub binding: f64,
    pub temporal: f64,
    pub action: f64,
    #[serde(rename = "self")]
    pub self_error: Option<f64>,
    pub regularization: f64,
    pub total: f64,
    pub overconfident: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Copy)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct History {
    pub step: u32,
    pub total: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Copy)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct Config {
    pub ablation: Policy,
    pub pin_gaze: bool,
    pub pin_x: f64,
    pub pin_y: f64,
    pub inject_confidence: bool,
    pub replay: bool,
    pub extra_noise: f64,
}

impl Config {
    pub fn new(ablation: Policy) -> Self {
        Self {
            ablation,
            pin_gaze: false,
            pin_x: 50.,
            pin_y: 50.,
            inject_confidence: false,
            replay: false,
            extra_noise: 0.,
        }
    }
}
impl Default for Config {
    fn default() -> Self {
        Self::new(Policy::Full)
    }
}
impl Gaze {
    pub(crate) fn new(x: f64, y: f64, aperture: f64) -> Self {
        Self { x, y, aperture }
    }
}
pub(crate) fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    jscompat::max(lo, jscompat::min(hi, x))
}
pub(crate) fn unit(x: f64) -> f64 {
    clamp(x, 0., 1.)
}
pub(crate) fn wrap(h: f64) -> f64 {
    ((h % 360.) + 360.) % 360.
}
pub(crate) fn hue_delta(a: f64, b: f64) -> f64 {
    let d = (wrap(a) - wrap(b)).abs() % 360.;
    if d > 180. { 360. - d } else { d }
}
