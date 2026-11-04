use crate::types::{MotionSample, clamp};
use jscompat::{exp, log, log1p};
use serde::Serialize;
pub type Features = [f64; 5];
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[non_exhaustive]
#[serde(rename_all = "lowercase")]
pub enum Context {
    Hidden,
    Open,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[non_exhaustive]
#[serde(rename_all = "lowercase")]
pub enum GapObservation {
    Empty,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[non_exhaustive]
pub struct GapStep {
    pub context: Context,
    pub observation: GapObservation,
}
impl GapStep {
    pub fn new(context: Context, empty: bool) -> Self {
        Self {
            context,
            observation: if empty {
                GapObservation::Empty
            } else {
                GapObservation::None
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Models {
    pub gain: [f64; 2],
    pub weights: [Features; 2],
    pub logits: [f64; 3],
}
impl Default for Models {
    fn default() -> Self {
        Self {
            gain: [1., 1.],
            weights: [[0.; 5]; 2],
            logits: [0.28, 0.28, 0.5].map(|p| log(p / (1. - p))),
        }
    }
}
pub(crate) fn features(samples: &[MotionSample]) -> Features {
    if samples.len() < 2 {
        return [0., 0., 0., 0., 1.];
    }
    let first = samples[0];
    let last = samples[samples.len() - 1];
    let prev = samples[samples.len() - 2];
    let recent = f64::from(last.step - prev.step);
    let total = f64::from(last.step - first.step);
    [
        (last.x - prev.x) / recent,
        (last.y - prev.y) / recent,
        (last.x - first.x) / total,
        (last.y - first.y) / total,
        1.,
    ]
}
pub(crate) fn motion(m: &Models, input: Features) -> [f64; 2] {
    m.weights
        .map(|w| w.iter().zip(input).fold(0., |s, (v, x)| s + v * x))
}
pub fn fit_motion(m: &mut Models, input: Features, target: [f64; 2], rate: f64) {
    let pred = motion(m, input);
    let norm = 1. + input.iter().fold(0., |s, x| s + x * x);
    for a in 0..2 {
        let gain = rate * (target[a] - pred[a]) / norm;
        for (w, x) in m.weights[a].iter_mut().zip(input) {
            *w += gain * x;
        }
    }
}
fn probabilities(m: &Models) -> [f64; 3] {
    m.logits.map(|v| 1. / (1. + exp(-v)))
}
pub(crate) fn filter(m: &Models, belief: f64, step: GapStep) -> f64 {
    let p = probabilities(m);
    let pred = belief * (1. - p[step.context as usize]);
    if step.observation == GapObservation::None {
        return pred;
    }
    let missed = pred * p[2];
    missed / (missed + (1. - pred))
}
fn likelihood(m: &Models, seen: u32, steps: &[GapStep], found: bool) -> f64 {
    let [hidden, open, miss] = probabilities(m);
    let detected = f64::from(seen) * (log1p(-open) + log1p(-miss));
    let p = [hidden, open];
    let decay = p.map(|v| log1p(-v));
    let log_miss = log(miss);
    let (mut survival, mut lost) = (0., 0.);
    for step in steps {
        let k = step.context as usize;
        if !found {
            lost += exp(survival) * p[k];
        }
        survival += decay[k]
            + if step.observation == GapObservation::Empty {
                log_miss
            } else {
                0.
            };
    }
    detected + if found { survival } else { log(lost) }
}
pub fn fit_existence(m: &mut Models, seen: u32, steps: &[GapStep], found: bool, rate: f64) {
    const DELTA: f64 = 1e-4;
    let mut next = m.logits;
    for (k, out) in next.iter_mut().enumerate() {
        let theta = m.logits[k];
        m.logits[k] = theta + DELTA;
        let plus = likelihood(m, seen, steps, found);
        m.logits[k] = theta - DELTA;
        let minus = likelihood(m, seen, steps, found);
        m.logits[k] = theta;
        *out = clamp(theta + rate * (plus - minus) / (2. * DELTA), -9., 9.);
    }
    m.logits = next;
}
