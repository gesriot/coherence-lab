use super::*;
use crate::{Effect, PARTICIPANTS, alignment::tests::texture};
#[test]
fn uninformative_draws_preserve_gain_and_skip_oracle_queries() {
    let mut l = GainLearner::new(0.1, Selection::Uniform, 801);
    let mut estimate = [1.; 2];
    for n in [4, 32] {
        let make = |command| {
            if n == 4 {
                texture(command, [1.; 2], n)
            } else {
                frame(
                    &vec![
                        Cell {
                            lightness: 6.,
                            ..Cell::default()
                        };
                        n * n
                    ],
                    n,
                    n as f64 / 2.,
                    command,
                )
            }
        };
        let pair = [make([0.; 2]), make([1.; 2])];
        let r = align(&pair, estimate);
        assert_eq!(r, [None, None]);
        l.observe(pair, Look::Pursue, false, 0., r);
        assert!(!l.update(
            &mut estimate,
            Some(&mut |_| panic!("uninformative sample requested a label"))
        ));
    }
    assert_eq!(estimate, [1.; 2]);
}
#[test]
fn admission_priorities_and_replay_use_current_gain() {
    let body = [1.1, 0.9];
    let pair = [texture([0.; 2], body, 32), texture([10., 0.], body, 32)];
    let r = align(&pair, [1.; 2]);
    for selection in [Selection::Attention, Selection::Uniform, Selection::Error] {
        let mut l = GainLearner::new(0.1, selection, 801);
        l.observe(pair.clone(), Look::Pursue, false, 0.5, r);
        assert_eq!(
            l.memory.weights().len(),
            usize::from(selection != Selection::Attention)
        );
        l.observe(pair.clone(), Look::Pursue, true, 0.5, r);
        l.observe(pair.clone(), Look::Explore, false, 0.5, r);
        let weights = l.memory.weights();
        assert_eq!(
            weights.len(),
            if selection == Selection::Attention {
                2
            } else {
                3
            }
        );
        assert!(
            abs(weights[0]
                - if selection == Selection::Error {
                    0.15
                } else {
                    0.55
                })
                < 1e-12
        );
    }
    let mut l = GainLearner::new(0.1, Selection::Uniform, 801);
    l.observe(pair, Look::Explore, false, 0., r);
    let mut gain = [1.1, 1.];
    assert!(l.update(&mut gain, None));
    assert_eq!(gain, [1.1, 1.]);
    fit_gain(&mut gain, [Some(100.), None], 1.);
    assert_eq!(gain, [2., 1.]);
    fit_gain(&mut gain, [Some(0.01), None], 1.);
    assert_eq!(gain, [0.5, 1.]);
}
#[test]
fn labels_are_oracle_only_and_object_models_stay_frozen() {
    for c in PARTICIPANTS.into_iter().chain([Condition::Oracle]) {
        let body = body_for_run(801);
        let m = reference_model(Control::Ideal);
        let id = GainCondition::Training(c);
        let mut ep = Episode::with_models(
            Scenario::Continuity,
            3101,
            id.policy(),
            Some(m.clone()),
            Some(body),
        );
        let mut l = GainLearner::new(0.03, id.selection(), 801);
        let mut calls = 0;
        train_episode(
            &mut ep,
            id,
            Some(&mut l),
            &mut |g| {
                calls += 1;
                [body[0] / g[0], body[1] / g[1]]
            },
            &mut |_| {},
        );
        assert_eq!(calls > 0, c == Condition::Oracle);
        let after = ep.agent.models.expect("gain model");
        assert_eq!(after.weights, m.weights);
        assert_eq!(after.logits, m.logits);
    }
}
#[test]
fn design_ids_rates_and_reserved_seeds() {
    for id in [
        "full",
        "first-order",
        "no-protocol",
        "strong-first-order",
        "keep-all",
        "constructor",
    ] {
        assert!(id.parse::<GainCondition>().is_err());
    }
    assert_eq!(GAIN_RATES, [0.01, 0.03, 0.1, 0.3]);
    assert_eq!(GAIN_ORACLE, 0.03);
    for (range, want) in [
        (GAIN_DESIGN, (801, 10, 3101)),
        (GAIN_TUNING, (541, 10, 2081)),
        (GAIN_COMPARISON, (121, 30, 1081)),
    ] {
        assert_eq!((range.first_run, range.runs, range.eval), want);
    }
    assert_eq!(
        crate::alignment::RATIOS,
        [
            0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1., 1.01, 1.02, 1.05, 1.1, 1.15, 1.2
        ]
    );
    assert_eq!(
        design_gain_rate(&[(0.1, 1.), (0.03, 0.999), (0.003, 0.993), (0.01, 0.996)]),
        0.01
    );
    assert_eq!(design_gain_rate(&[(0.003, 0.), (0.01, 0.005)]), 0.01);
}
fn synthetic(scores: &[f64], accuracy: f64) -> Vec<GainRun> {
    vec![GainRun {
        curve: scores
            .iter()
            .zip(GAIN_STAGES)
            .map(|(&q, step)| Calibration {
                step,
                q,
                error: 0.,
                gain: [1.; 2],
            })
            .collect(),
        secondary: Point {
            accuracy,
            retention: accuracy,
            ghost_free: accuracy,
            ..Point::default()
        },
        saccades: 0.,
        pairs: 0.,
        draws: 0.,
    }]
}
#[test]
fn controls_gate_participants_and_verdict_before_any_measurement() {
    let ones = [1.; 11];
    let zeros = [0.; 11];
    let mut good = ones;
    good[..3].copy_from_slice(&[0., 0.9, 0.85]);
    let p = FixedPoint {
        bias: -0.03,
        limit: 0.03,
        direction: 0.8,
        correct: 4,
        directional: 5,
        informative: 1,
    };
    let valid = [(&zeros, 0.75), (&ones, 1.), (&good, 1.), (&good, 1.)]
        .map(|(q, a)| gain_summary(&synthetic(q, a)));
    let reject = |controls: [&GainSummary; 4], fixed: &FixedPoint| {
        assert!(
            compare_gain(controls, fixed, |_| panic!(
                "failed control opened comparison"
            ))
            .is_none()
        );
    };
    for invalid in [
        FixedPoint { bias: -0.031, ..p },
        FixedPoint {
            direction: 0.799,
            ..p
        },
        FixedPoint {
            direction: f64::NAN,
            ..p
        },
        FixedPoint {
            bias: f64::NAN,
            ..p
        },
    ] {
        reject(valid.each_ref(), &invalid);
    }
    for (i, step, value, accuracy) in [
        (0, 0, 0.01, 0.75),
        (0, 0, 0., 0.775),
        (1, 0, 0.99, 1.),
        (1, 0, 1., 0.875),
        (2, 2, 0.84, 1.),
        (2, 2, 0.85, 0.875),
        (3, 10, 0.899, 1.),
        (3, 2, 0.85, 0.875),
    ] {
        let mut scores = if i == 0 {
            zeros
        } else if i == 1 {
            ones
        } else {
            good
        };
        scores[step] = value;
        let invalid = gain_summary(&synthetic(&scores, accuracy));
        let mut controls = valid.each_ref();
        controls[i] = &invalid;
        reject(controls, &p);
    }
    let equal = compare_gain(valid.each_ref(), &p, |_| synthetic(&good, 1.))
        .expect("passing controls admit participants");
    assert_eq!(equal.participants.len(), 4);
    assert!(
        equal
            .differences
            .iter()
            .all(|x| x.1.mean == 0. && x.2 == Effect::None)
    );
    assert!(
        crate::replacement(equal.differences[0].2, equal.differences[2].2)
            .starts_with("не применимо")
    );
    let precise = compare_gain(valid.each_ref(), &p, |id| {
        let mut q = [if id == Condition::Full { 0.600123 } else { 0.5 }; 11];
        q[0] = 0.;
        synthetic(&q, 1.)
    })
    .expect("passing controls");
    assert!(abs(precise.differences[0].1.mean - 0.100123) < 1e-12);
    assert_eq!(precise.differences[0].2, Effect::Faster);
}
#[test]
fn calibration_uses_unit_baseline_and_equal_post_initial_weights() {
    let b = [0.875, 1.25];
    let initial = calibration_error([1.; 2], b);
    assert_eq!(initial, 0.1875);
    assert_eq!(point(0, [1.; 2], b, initial).q, 0.);
    assert_eq!(point(0, b, b, initial).q, 1.);
    assert_eq!(point(10, [1.125, 0.75], b, initial).q, -1.);
    let v = synthetic(
        &[
            0., -0.56123, 0.11, 0.24, 0.35, 0.46, 0.57, 0.68, 0.79, 0.89, 1.,
        ],
        1.,
    );
    assert!(abs(gain_area(&v[0].curve) - 0.452877) < 1e-12);
    assert_eq!(gain_area(&synthetic(&[1., 0.25, 0.75], 1.)[0].curve), 0.5);
}
#[test]
fn checkpoints_count_completed_environment_steps_across_episodes() {
    let body = body_for_run(801);
    let id = GainCondition::Training(Condition::Oracle);
    let trial = gain_trial(id, 801, 3101, 0.01);
    assert_eq!(
        trial.curve.iter().map(|p| p.step).collect::<Vec<_>>(),
        [0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
    );
    let mut m = reference_model(Control::Ideal);
    let mut l = GainLearner::new(0.01, id.selection(), 801);
    let mut step = 0;
    for (i, sc) in [Scenario::Continuity, Scenario::Vanish]
        .into_iter()
        .enumerate()
    {
        let mut ep = Episode::with_models(
            sc,
            90101 + i as u32,
            id.policy(),
            Some(m.clone()),
            Some(body),
        );
        let cfg = Config::new(id.policy());
        let capture = |ep: &Episode, g: [f64; 2]| {
            let a = &ep.agent;
            let v = a.obs_gaze;
            frame(&a.cells, a.grid, v.aperture, [v.x / g[0], v.y / g[1]])
        };
        let mut prev = capture(&ep, m.gain);
        for _ in 0..80 {
            ep.step(&cfg);
            let next = capture(&ep, m.gain);
            let pair = [prev, next.clone()];
            let r = align(&pair, m.gain);
            l.observe(pair, Look::Pursue, false, 0., r);
            l.update(&mut m.gain, Some(&mut |g| [body[0] / g[0], body[1] / g[1]]));
            ep.agent.models = Some(m.clone());
            prev = next;
            step += 1;
            if let Some(p) = trial.curve.iter().find(|p| p.step == step) {
                assert_eq!(p.gain, m.gain);
            }
        }
    }
}
