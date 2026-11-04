use crate::task::object as task;
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
            logits: task::INITIAL.map(|p| log(p / (1. - p))),
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
pub(crate) fn probabilities(m: &Models) -> [f64; 3] {
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
pub(crate) fn likelihood(m: &Models, seen: u32, steps: &[GapStep], found: bool) -> f64 {
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
    use task::{DELTA, LOGIT_LIMIT};
    let mut next = m.logits;
    for (k, out) in next.iter_mut().enumerate() {
        let theta = m.logits[k];
        m.logits[k] = theta + DELTA;
        let plus = likelihood(m, seen, steps, found);
        m.logits[k] = theta - DELTA;
        let minus = likelihood(m, seen, steps, found);
        m.logits[k] = theta;
        *out = clamp(
            theta + rate * (plus - minus) / (2. * DELTA),
            -LOGIT_LIMIT,
            LOGIT_LIMIT,
        );
    }
    m.logits = next;
}

#[cfg(test)]
mod tests {
    use super::*;
    use jscompat::{abs, hypot, sqrt};
    #[test]
    fn motion_converges_and_features_use_elapsed_steps() {
        let mut m = Models::default();
        let input = [1.5, -0.5, 1.5, -0.5, 1.];
        assert_eq!(motion(&m, input), [0.; 2]);
        for _ in 0..500 {
            fit_motion(&mut m, input, [1.5, -0.5], 0.1);
        }
        let [x, y] = motion(&m, input);
        assert!(hypot(x - 1.5, y + 0.5) < 1e-8);
        let samples = [(1, 3., 6.), (5, 9., 6.), (7, 15., 0.)].map(|(step, x, y)| MotionSample {
            step,
            x,
            y,
            cell_width: 3.,
        });
        assert_eq!(features(&samples[..1]), [0., 0., 0., 0., 1.]);
        assert_eq!(features(&samples), [3., -3., 2., -1., 1.]);
    }
    #[test]
    fn posterior_and_likelihood_account_for_missing_detections() {
        let m = Models {
            logits: [0.25, 0.5, 0.2].map(|p| log(p / (1. - p))),
            ..Models::default()
        };
        assert_eq!(filter(&m, 0.8, GapStep::new(Context::Open, false)), 0.4);
        assert!(
            abs(filter(&m, 0.8, GapStep::new(Context::Hidden, true)) - 0.12 / (0.12 + 0.4)) < 1e-12
        );
        let steps = [
            GapStep::new(Context::Open, true),
            GapStep::new(Context::Hidden, false),
        ];
        let detected = (0.5 * 0.8) * (0.5 * 0.8);
        assert!(abs(likelihood(&m, 2, &steps, true) - log(detected * 0.5 * 0.2 * 0.75)) < 1e-12);
        assert!(
            abs(likelihood(&m, 2, &steps, false) - log(detected * (0.5 + 0.5 * 0.2 * 0.25)))
                < 1e-12
        );
    }
    #[test]
    fn synthetic_gaps_recover_both_hazards_and_missed_detections() {
        let mut m = Models::default();
        for epoch in 1..=150 {
            let rate = 0.04 / sqrt(f64::from(epoch));
            for i in 0..100 {
                fit_existence(
                    &mut m,
                    1,
                    &[GapStep::new(Context::Hidden, false)],
                    i % 4 != 0,
                    rate,
                );
                fit_existence(
                    &mut m,
                    1,
                    &[GapStep::new(Context::Open, i % 10 < 8)],
                    i % 10 >= 3,
                    rate,
                );
            }
        }
        for (x, want) in probabilities(&m).into_iter().zip([0.25, 0.1, 0.2]) {
            assert!(abs(x - want) < 0.002);
        }
    }
}
