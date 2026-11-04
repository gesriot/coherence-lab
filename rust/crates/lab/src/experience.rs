use crate::learning::{Features, GapStep, Models, fit_existence, fit_motion, likelihood, motion};
use jscompat::{floor, hash_unit, hypot, imul};
use serde::Serialize;
use std::collections::VecDeque;

pub(crate) struct Generator {
    seed: u32,
    draw: u32,
}
impl Generator {
    pub(crate) fn new(seed: u32) -> Self {
        Self { seed, draw: 0 }
    }
    pub(crate) fn next(&mut self) -> f64 {
        self.draw = self.draw.wrapping_add(1);
        hash_unit(f64::from(self.seed) + f64::from(imul(self.draw.into(), 0x9e3779b9u32.into())))
    }
}
pub(crate) struct Replay<T> {
    entries: VecDeque<(T, f64)>,
    capacity: usize,
}
impl<T> Replay<T> {
    #[cfg(test)]
    pub(crate) fn weights(&self) -> Vec<f64> {
        self.entries.iter().map(|e| e.1).collect()
    }
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            capacity,
        }
    }
    pub(crate) fn add(&mut self, value: T, weight: f64) {
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back((value, weight));
    }
    pub(crate) fn sample(&self, random: &mut Generator, uniform: bool) -> Option<&T> {
        if self.entries.is_empty() {
            return None;
        }
        let u = random.next();
        if uniform {
            return Some(&self.entries[floor(u * self.entries.len() as f64) as usize].0);
        }
        let mut threshold = u * self.entries.iter().fold(0., |s, e| s + e.1);
        for (v, w) in &self.entries {
            threshold -= w;
            if threshold < 0. {
                return Some(v);
            }
        }
        self.entries.back().map(|e| &e.0)
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Selection {
    Uniform,
    Attention,
    Error,
}
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Rates {
    pub motion: f64,
    pub existence: f64,
}
impl Rates {
    pub const fn new(motion: f64, existence: f64) -> Self {
        Self { motion, existence }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct Movement {
    pub(crate) input: Features,
    pub(crate) target: [f64; 2],
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct GapExample {
    pub(crate) seen: u32,
    pub(crate) steps: Vec<GapStep>,
    pub(crate) found: bool,
}
#[derive(Clone, Debug, Serialize)]
pub(crate) struct Draw {
    movement: Option<Movement>,
    gap: Option<GapExample>,
}
pub(crate) struct Learner {
    pub(crate) models: Models,
    selection: Selection,
    rates: Rates,
    random: Generator,
    movements: Replay<Movement>,
    gaps: Replay<GapExample>,
}
impl Learner {
    pub(crate) fn new(rates: Rates, selection: Selection, seed: u32) -> Self {
        Self {
            models: Models::default(),
            selection,
            rates,
            random: Generator::new(seed),
            movements: Replay::new(256),
            gaps: Replay::new(32),
        }
    }
    pub(crate) fn remember_motion(&mut self, m: Movement, error: Option<f64>) {
        if self.selection == Selection::Attention && error.is_none() {
            return;
        }
        let priority = if self.selection == Selection::Error {
            let [x, y] = motion(&self.models, m.input);
            hypot(m.target[0] - x, m.target[1] - y)
        } else {
            error.unwrap_or(0.)
        };
        self.movements.add(m, 0.05 + priority);
    }
    pub(crate) fn remember_gap(&mut self, g: GapExample, error: Option<f64>) {
        if self.selection == Selection::Attention && error.is_none() {
            return;
        }
        let priority = if self.selection == Selection::Error {
            -likelihood(&self.models, g.seen, &g.steps, g.found)
                / (g.steps.len() + g.seen as usize) as f64
        } else {
            error.unwrap_or(0.)
        };
        self.gaps.add(g, 0.05 + priority);
    }
    pub(crate) fn update(&mut self, trace: Option<&mut Vec<Draw>>) {
        let uniform = self.selection == Selection::Uniform;
        let movement = self.movements.sample(&mut self.random, uniform);
        let gap = self.gaps.sample(&mut self.random, uniform);
        if let Some(out) = trace {
            out.push(Draw {
                movement: movement.cloned(),
                gap: gap.cloned(),
            });
        }
        if let Some(m) = movement {
            fit_motion(&mut self.models, m.input, m.target, self.rates.motion);
        }
        if let Some(g) = gap {
            fit_existence(
                &mut self.models,
                g.seen,
                &g.steps,
                g.found,
                self.rates.existence,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::Context;
    #[test]
    fn weights_are_reproducible_and_proportional() {
        let mut r = Replay::new(2);
        r.add(0, 1.);
        r.add(1, 3.);
        let sample = || {
            let mut rng = Generator::new(3);
            (0..10000)
                .map(|_| r.sample(&mut rng, false).copied())
                .collect::<Vec<_>>()
        };
        let draws = sample();
        assert_eq!(draws, sample());
        assert!(
            (draws.iter().filter(|v| **v == Some(1)).count() as f64 / 10000. - 0.75).abs() < 0.02
        );
    }
    #[test]
    fn bounded_memories_admit_attended_samples_and_update_once_per_step() {
        for selection in [Selection::Attention, Selection::Uniform, Selection::Error] {
            let mut l = Learner::new(Rates::new(0.1, 0.01), selection, 3);
            let mut draws = vec![];
            let m = Movement {
                input: [0., 0., 0., 0., 1.],
                target: [0., 1.],
            };
            let g = GapExample {
                seen: 1,
                steps: vec![GapStep::new(Context::Open, true)],
                found: false,
            };
            if selection == Selection::Attention {
                l.remember_motion(m.clone(), None);
                l.remember_gap(g.clone(), None);
            }
            l.update(Some(&mut draws));
            l.remember_motion(m.clone(), Some(0.));
            l.update(Some(&mut draws));
            l.remember_gap(g.clone(), Some(0.));
            l.update(Some(&mut draws));
            assert_eq!(
                draws
                    .iter()
                    .map(|d| (d.movement.is_some(), d.gap.is_some()))
                    .collect::<Vec<_>>(),
                [(false, false), (true, false), (true, true)]
            );
            for i in 0..300 {
                l.remember_motion(
                    Movement {
                        target: [i as f64, 0.],
                        ..m
                    },
                    Some(0.),
                );
                l.remember_gap(
                    GapExample {
                        seen: i + 1,
                        ..g.clone()
                    },
                    Some(0.3),
                );
            }
            assert_eq!(l.movements.entries.len(), 256);
            assert_eq!(l.movements.entries[0].0.target[0], 44.);
            assert_eq!(l.gaps.entries.len(), 32);
            assert_eq!(l.gaps.entries[0].0.seen, 269);
        }
    }
}
