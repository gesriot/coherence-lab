use crate::task::object as task;
use crate::{
    experience::Rates,
    genesis::*,
    learning::Models,
    training::Condition,
    types::{LabError, Policy},
};
use jscompat::{log, round};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Control {
    Frozen,
    KeepAll,
    Ideal,
    Oracle,
}
pub const CONTROLS: [Control; 4] = [
    Control::Frozen,
    Control::KeepAll,
    Control::Ideal,
    Control::Oracle,
];
impl std::str::FromStr for Control {
    type Err = LabError;
    fn from_str(s: &str) -> Result<Self, LabError> {
        match s {
            "frozen" => Ok(Self::Frozen),
            "keep-all" => Ok(Self::KeepAll),
            "ideal" => Ok(Self::Ideal),
            "oracle" => Ok(Self::Oracle),
            _ => Err(LabError::Control(s.into())),
        }
    }
}
impl Control {
    pub fn label(self) -> &'static str {
        match self {
            Self::Frozen => "Без обучения",
            Self::KeepAll => "Хранить всё",
            Self::Ideal => "Идеальные параметры",
            Self::Oracle => "Оракул",
        }
    }
}
pub fn ideal_model() -> Models {
    reference_model(Control::Ideal)
}
pub(crate) fn reference_model(id: Control) -> Models {
    let mut m = Models::default();
    if matches!(id, Control::Ideal | Control::KeepAll) {
        m.weights[0][2] = 1.;
        m.weights[1][3] = 1.;
        m.logits = if id == Control::Ideal {
            task::IDEAL
        } else {
            task::KEEP_ALL
        }
        .map(|p| log(p / (1. - p)));
    }
    m
}
#[derive(Debug, Clone, PartialEq)]
struct ControlPoint {
    point: Point,
    weights: [[f64; 5]; 2],
    split: Option<[Point; 2]>,
}
#[derive(Clone, Copy)]
struct Design {
    range: SeedRange,
    seeds: u32,
    stages: &'static [u32],
}
const DESIGN: Design = Design {
    range: SeedRange::new(701, 10, 3001),
    seeds: 20,
    stages: &STAGES,
};
fn control_curve(
    id: Control,
    seed: u32,
    rates: Rates,
    d: Design,
    split_final: bool,
) -> Vec<ControlPoint> {
    let eval = |m: &Models| evaluate(m, Policy::FirstOrder, d.range.eval, d.seeds, None);
    let measure = |m: &Models, step| {
        let split =
            (split_final && id == Control::Oracle && Some(&step) == d.stages.last()).then(|| {
                let ideal = reference_model(Control::Ideal);
                [
                    eval(&Models {
                        weights: m.weights,
                        ..ideal.clone()
                    }),
                    eval(&Models {
                        weights: ideal.weights,
                        ..m.clone()
                    }),
                ]
            });
        ControlPoint {
            point: eval(m),
            weights: m.weights,
            split,
        }
    };
    if id == Control::Oracle {
        train_curve(Condition::Oracle, seed, rates, d.stages, measure)
    } else {
        vec![measure(&reference_model(id), 0); d.stages.len()]
    }
}
pub fn control_runs(id: Control, rates: Rates, range: SeedRange) -> Vec<Vec<Point>> {
    let run = |seed| {
        control_curve(id, seed, rates, Design { range, ..DESIGN }, false)
            .into_iter()
            .map(|p| p.point)
            .collect::<Vec<_>>()
    };
    if id == Control::Oracle {
        range.seeds().map(run).collect()
    } else {
        vec![run(range.first_run); range.runs as usize]
    }
}
fn design_rate(d: Design) -> Rates {
    let mut best = -1.;
    let mut chosen = ORACLE_RATES;
    for motion in MOTION_RATES {
        for existence in EXISTENCE_RATES {
            let rates = Rates::new(motion, existence);
            let mut successes = 0.;
            for seed in d.range.seeds() {
                let stages = d
                    .stages
                    .last_chunk::<1>()
                    .expect("design has a final checkpoint");
                let points =
                    control_curve(Control::Oracle, seed, rates, Design { stages, ..d }, false);
                successes += round(points[0].point.accuracy * 2. * f64::from(d.seeds));
            }
            if successes > best {
                best = successes;
                chosen = rates;
            }
        }
    }
    chosen
}
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct DesignPoint {
    pub step: u32,
    pub accuracy: f64,
    pub retention: f64,
    pub ghost_free: f64,
    pub existence: [f64; 3],
    pub weights: [f64; 10],
    pub split: Option<[[f64; 3]; 2]>,
}
fn design_curve(id: Control, rates: Rates, d: Design) -> Vec<DesignPoint> {
    let runs = if id == Control::Oracle {
        d.range.runs
    } else {
        1
    };
    let curves: Vec<_> = (0..runs)
        .map(|i| control_curve(id, d.range.first_run + i, rates, d, true))
        .collect();
    let units = f64::from(2 * d.seeds);
    d.stages
        .iter()
        .enumerate()
        .map(|(i, &step)| {
            let avg = |f: fn(Point) -> f64| average(curves.iter().map(|c| f(c[i].point)));
            let split = curves[0][i].split.map(|_| {
                std::array::from_fn(|j| {
                    let mean = |f: fn(Point) -> f64| {
                        average(curves.iter().map(|c| {
                            f(c[i]
                                .split
                                .expect("all final oracle points measure components")[j])
                        }))
                    };
                    [
                        mean(|p| p.accuracy),
                        mean(|p| p.retention),
                        mean(|p| p.ghost_free),
                    ]
                })
            });
            DesignPoint {
                step,
                accuracy: curves
                    .iter()
                    .fold(0., |s, c| s + round(c[i].point.accuracy * units))
                    / (f64::from(runs) * units),
                retention: avg(|p| p.retention),
                ghost_free: avg(|p| p.ghost_free),
                existence: [avg(|p| p.hidden), avg(|p| p.open), avg(|p| p.miss)],
                weights: std::array::from_fn(|j| {
                    average(curves.iter().map(|c| c[i].weights[j / 5][j % 5]))
                }),
                split,
            }
        })
        .collect()
}
pub fn stable(values: &[f64]) -> bool {
    values
        .iter()
        .position(|&x| x >= task::FLOOR)
        .is_some_and(|i| {
            values.last().is_some_and(|x| *x >= task::FLOOR)
                && values[i..].iter().all(|&x| x >= task::STABILITY)
        })
}
pub fn permits(
    frozen: &[f64],
    keep: &[f64],
    ideal: &[f64],
    oracle: &[f64],
    extra: &[Vec<f64>],
) -> bool {
    [frozen, keep]
        .iter()
        .all(|v| !v.is_empty() && v.iter().all(|&x| x <= task::CEILING))
        && ideal.last().is_some_and(|x| *x >= task::FLOOR)
        && stable(oracle)
        && extra.iter().all(|v| stable(v))
}
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub struct DesignReport {
    pub rates: Rates,
    pub curves: Vec<(Control, Vec<DesignPoint>)>,
    pub stability: Vec<(f64, Vec<f64>)>,
}
fn design(ids: &[Control], d: Design) -> DesignReport {
    let rates = if ids.contains(&Control::Oracle) {
        design_rate(d)
    } else {
        Rates::new(0., 0.)
    };
    let curves: Vec<_> = ids
        .iter()
        .map(|&id| (id, design_curve(id, rates, d)))
        .collect();
    let mut stability = vec![];
    if let Some((_, oracle)) = curves.iter().find(|(id, _)| *id == Control::Oracle) {
        for r in EXISTENCE_RATES {
            let other;
            let curve = if r == rates.existence {
                oracle
            } else {
                other = design_curve(Control::Oracle, Rates::new(rates.motion, r), d);
                &other
            };
            stability.push((r, curve.iter().map(|p| p.accuracy).collect()));
        }
    }
    DesignReport {
        rates,
        curves,
        stability,
    }
}
pub fn genesis_design(ids: &[Control]) -> DesignReport {
    design(ids, DESIGN)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn design_gate_rejects_each_failure_and_comparison_conditions() {
        for id in [
            "full",
            "first-order",
            "no-protocol",
            "strong-first-order",
            "constructor",
        ] {
            assert!(id.parse::<Control>().is_err());
        }
        let pass = |f, k, i, o, extra: &[Vec<f64>]| permits(f, k, i, o, extra);
        let bottom = &[0.5, 0.75][..];
        let top = &[0.5, 0.9][..];
        assert!(pass(bottom, bottom, top, top, &[vec![0.4, 0.9, 0.85, 0.9]]));
        assert!(!pass(bottom, bottom, top, &[0.9, 0.849, 0.99], &[]));
        for v in [vec![0.99, 0.86], vec![0.9, 0.849, 0.99]] {
            assert!(!pass(bottom, bottom, top, top, &[v]));
        }
        let bad = &[0.751, 0.5][..];
        assert!(!pass(bad, bottom, top, top, &[]));
        assert!(!pass(bottom, bad, top, top, &[]));
        let bad = &[1., 0.899][..];
        assert!(!pass(bottom, bottom, bad, top, &[]));
        assert!(!pass(bottom, bottom, top, bad, &[]));
    }
    #[test]
    fn fixed_controls_predict_baseline_and_design_reproduces_grid_and_components() {
        let d = Design {
            range: SeedRange::new(701, 2, 3001),
            seeds: 2,
            stages: &[0, 1],
        };
        for id in [Control::KeepAll, Control::Ideal] {
            let m = reference_model(id);
            assert_eq!(
                crate::learning::motion(&m, [1., 2., -3., 4., 1.]),
                [-3., 4.]
            );
            let c = control_curve(id, 701, Rates::new(1., 0.1), d, false);
            assert_eq!(c[0], c[1]);
            assert_eq!(m.weights, c[1].weights);
        }
        let a = design(&CONTROLS, d);
        assert_eq!(a, design(&CONTROLS, d));
        assert!(a.curves[3].1[1].split.is_some());
        assert_eq!(
            a.stability.iter().map(|s| s.0).collect::<Vec<_>>(),
            EXISTENCE_RATES
        );
    }
}
