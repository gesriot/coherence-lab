use crate::{
    Error,
    element::{Element, Kind, Step},
    gradient::Gradient,
    input::Observation,
};
use jscompat::{max, pow_u32, sqrt};
struct Adam {
    step: u32,
    first: Vec<f64>,
    second: Vec<f64>,
}
impl Adam {
    fn update(&mut self, w: &mut [f64], grad: &[f64]) {
        let squared = grad.iter().fold(0., |s, &x| s + x * x);
        let clip = 1. / max(1., sqrt(squared));
        self.step += 1;
        let m_scale = 1. - pow_u32(0.9, self.step);
        let v_scale = 1. - pow_u32(0.999, self.step);
        for (i, weight) in w.iter_mut().enumerate() {
            let g = grad[i] * clip;
            self.first[i] = 0.9 * self.first[i] + 0.1 * g;
            self.second[i] = 0.999 * self.second[i] + 0.001 * g * g;
            *weight -= 0.003 * (self.first[i] / m_scale) / (sqrt(self.second[i] / v_scale) + 1e-8);
        }
    }
}
pub(crate) struct Trainer {
    tape: Vec<Step>,
    work: Gradient,
    adam: Adam,
    carry: Vec<f64>,
}
type MapHook<'a> = Option<&'a mut dyn FnMut(usize) -> Vec<f64>>;
impl Trainer {
    pub(crate) fn new(m: &Element, memory: usize) -> Self {
        Self {
            tape: (0..memory).map(|_| m.buffers()).collect(),
            work: Gradient::new(m),
            adam: Adam {
                step: 0,
                first: vec![0.; m.weights.len()],
                second: vec![0.; m.weights.len()],
            },
            carry: m.blank(),
        }
    }
    pub(crate) fn episode(
        &mut self,
        m: &mut Element,
        mut movie: impl Iterator<Item = Observation>,
        mut hint: MapHook<'_>,
        mut weight: MapHook<'_>,
        mut trace: Option<&mut Vec<Vec<f64>>>,
    ) -> Result<(), Error> {
        if m.kind == Kind::Hinted && hint.is_none() {
            return Err(Error::MissingHint);
        }
        self.carry.fill(0.);
        let Some(mut prev) = movie.next() else {
            return Ok(());
        };
        let mut time = 0;
        loop {
            let start = time;
            let mut targets = Vec::new();
            let mut labels = Vec::new();
            let mut maps = Vec::new();
            for t in 0..self.tape.len() {
                let Some(next) = movie.next() else {
                    break;
                };
                let (earlier, rest) = self.tape.split_at_mut(t);
                let out = &mut rest[0];
                let state = earlier.last().map_or(&self.carry, |s| &s.state);
                m.forward(state, &prev.x, out, prev.copy)?;
                if let Some(v) = trace.as_deref_mut() {
                    v.push(out.prediction.clone());
                }
                if m.kind == Kind::Hinted {
                    labels.push(hint.as_mut().ok_or(Error::MissingHint)?(time));
                }
                time += 1;
                if let Some(map) = weight.as_mut() {
                    maps.push(map(time));
                }
                // Move the previous frame into the targets once its next prediction is made.
                if t > 0 {
                    targets.push(prev.x);
                }
                prev = next;
            }
            if time == start {
                break;
            }
            // A complete or partial segment ends with the frame carried into the next one.
            targets.push(prev.x.clone());
            self.work.backward(m, &self.tape, &targets, &labels, &maps);
            self.carry
                .copy_from_slice(&self.tape[targets.len() - 1].state);
            self.adam.update(&mut m.weights, &self.work.grad);
            if targets.len() < self.tape.len() {
                break;
            }
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{WorldSettings, movie};
    use lab::{Occlusion, Scenario};
    use sha2::{Digest, Sha256};
    #[test]
    fn command_features_and_adam_use_the_outgoing_command() -> Result<(), Error> {
        let make = || Element::new(19, 2, 2, true, Kind::Memory, true);
        let mut m = make();
        let mut reference = make();
        let x: Vec<_> = (0..12).map(|i| f64::from(i) / 20.).collect();
        let next: Vec<_> = x.iter().map(|v| v + 0.1).collect();
        let obs = [
            Observation {
                x: x.clone(),
                copy: Some([1., -1.]),
            },
            Observation {
                x: next.clone(),
                copy: Some([-1., 0.]),
            },
        ];
        Trainer::new(&m, 30).episode(&mut m, obs.into_iter(), None, None, None)?;
        let mut out = reference.buffers();
        reference.forward(&reference.blank(), &x, &mut out, Some([1., -1.]))?;
        for (i, f) in out.features.chunks_exact(reference.f).enumerate() {
            assert_eq!(
                &f[reference.f - 5..],
                &[1., -1., x[3 * i], x[3 * i + 1], x[3 * i + 2]]
            );
        }
        let mut grad = Gradient::new(&reference);
        grad.backward(&reference, &[out], &[next], &[], &[]);
        let mut adam = Trainer::new(&reference, 30).adam;
        adam.update(&mut reference.weights, &grad.grad);
        assert_eq!(m.weights, reference.weights);
        Ok(())
    }
    #[test]
    fn feature_and_parameter_reference_bytes() -> Result<(), Error> {
        let m = Element::new(29, 16, 4, true, Kind::Memory, false);
        let mut state = m.blank();
        let mut h = Sha256::new();
        for o in movie(
            Scenario::Continuity,
            31,
            16,
            5,
            Occlusion::WholeObject,
            WorldSettings::default(),
            None,
        ) {
            let mut out = m.buffers();
            m.forward(&state, &o.x, &mut out, None)?;
            for x in out.features {
                h.update(x.to_le_bytes());
            }
            state = out.state;
        }
        assert_eq!(
            format!("{:x}", h.finalize()),
            "5dc38b15278b17301acbc858bd0b04f948a3cb867429473a6b7de464bbded3f1"
        );
        let mut m = Element::new(23, 16, 4, true, Kind::Memory, false);
        let mut coach = Trainer::new(&m, 30);
        for (i, sc) in [
            Scenario::Continuity,
            Scenario::Vanish,
            Scenario::Nursery,
            Scenario::Binding,
        ]
        .into_iter()
        .enumerate()
        {
            coach.episode(
                &mut m,
                movie(
                    sc,
                    29000 + i as u32,
                    16,
                    32,
                    Occlusion::PerCell,
                    WorldSettings::default(),
                    None,
                ),
                None,
                None,
                None,
            )?;
        }
        let mut h = Sha256::new();
        for w in m.weights {
            h.update(w.to_le_bytes());
        }
        assert_eq!(
            format!("{:x}", h.finalize()),
            "6921fe85eeb9ffbc79dd208e5c04752a501741f002dfadc5e66e63df69d68bca"
        );
        Ok(())
    }
    #[test]
    fn hooks_receive_current_labels_and_next_frame_weights_across_segments() -> Result<(), Error> {
        for kind in [Kind::Plain, Kind::Memory, Kind::Hinted] {
            let mut m = Element::new(3, 4, 2, true, kind, false);
            let mut t = Trainer::new(&m, 30);
            let (mut labels, mut maps) = (vec![], vec![]);
            t.episode(
                &mut m,
                movie(
                    Scenario::Continuity,
                    15,
                    4,
                    32,
                    Occlusion::PerCell,
                    WorldSettings::default(),
                    None,
                ),
                Some(&mut |n| {
                    labels.push(n);
                    vec![0.; 16]
                }),
                Some(&mut |n| {
                    maps.push(n);
                    vec![1.; 16]
                }),
                None,
            )?;
            assert_eq!(
                labels,
                if kind == Kind::Hinted {
                    (0..32).collect()
                } else {
                    vec![]
                }
            );
            assert_eq!(maps, (1..=32).collect::<Vec<_>>());
        }
        Ok(())
    }
}
