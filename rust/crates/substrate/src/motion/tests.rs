use super::*;
use crate::input::{WorldSettings, movie, world};
use jscompat::{abs, number};
use sha2::{Digest, Sha256};
#[test]
fn transition_pools_by_executed_command_and_both_time_boundaries() {
    let mut e = [Ratio::default(); 5];
    for t in [0, 23, 24, 47, 48, 71] {
        let c = match t {
            0 | 71 => [0., 0.],
            24 => [0., -1.],
            _ => [1., 0.],
        };
        transition(&mut e, t, c, &[f64::from(t + 1)], &[0.], &[1.]);
    }
    let moved = 23. * 23. + 24. * 24. + 47. * 47. + 48. * 48.;
    assert_eq!(
        e.map(|s| [s.sum, s.copy]),
        [
            [moved + 71. * 71., 6.],
            [moved, 4.],
            [71. * 71., 2.],
            [23. * 23., 1.],
            [48. * 48., 1.]
        ]
    );
}
#[test]
fn only_visible_centers_supply_incoming_targets_and_patches_use_previous_copy() {
    let mut w = world(Scenario::Continuity, 17, WorldSettings::default());
    let set = |w: &World| target(w, [39., 46.], [36., 36.375], 16);
    w.objects[0].x = 34.3125;
    w.objects[0].y = 47.5625;
    w.objects[0].r = 4.;
    assert!(set(&w).is_none());
    w.t = 1;
    assert_eq!(set(&w), Some((134, [2.625, -1.6875, -4.3125])));
    w.objects[0].r = 0.1;
    w.objects[0].y += 1.;
    assert!(set(&w).is_none());
    w.objects[0].r = 4.;
    for x in [42., 13.] {
        w.objects[0].x = x;
        assert!(set(&w).is_none());
    }
    w.objects[0].x = 34.;
    w.objects[0].vanish_at = Some(1);
    assert!(set(&w).is_none());
    let p = Observation {
        x: (1..=768).map(f64::from).collect(),
        copy: Some([-1., 1.]),
    };
    let c = Observation {
        x: p.x.iter().map(|x| x + 1000.).collect(),
        copy: Some([1., -1.]),
    };
    let values = patch(0, &p, &c, 16);
    assert_eq!(values.len(), 56);
    assert_eq!(values[..12], [0.; 12]);
    assert_eq!(values[12..18], [1., 2., 3., 4., 5., 6.]);
    assert_eq!(values[39..42], [1001., 1002., 1003.]);
    assert_eq!(values[54..], [-1., 1.]);
}
#[test]
fn context_uses_training_bin_mean_or_global_fallback() -> Result<(), Error> {
    use Scenario::{Continuity as C, Vanish as V};
    let key = |sc, t: u32| (sc, t / 8);
    let ctx = Context::fit(
        &[key(C, 1), key(C, 7), key(C, 8), key(V, 7)],
        &[3., 5., -10., 6.],
    )?;
    assert_eq!(
        [key(C, 0), key(C, 15), key(V, 1), key(V, 8), key(C, 16)].map(|k| ctx.predict(k)),
        [4., -10., 6., 1., 1.]
    );
    Ok(())
}
#[test]
fn criteria_use_strict_pairs_and_each_runs_best_baseline() {
    let gate = |a: &[f64], b: &[f64], c: &[f64]| command_gate(a, b, c, 5).3;
    assert!(gate(&[0.85; 5], &[1.; 5], &[1.; 5]));
    assert!(!gate(&[0.85001; 5], &[1.; 5], &[2.; 5]));
    assert!(!gate(&[0.85001; 5], &[2.; 5], &[1.; 5]));
    for x in [([1.; 5], [2.; 5]), ([2.; 5], [1.; 5])] {
        assert!(!gate(&[0.5, 0.5, 0.5, 0.5, 1.], &x.0, &x.1));
    }
    assert!(!gate(&[0.5; 4], &[1.; 5], &[1.; 5]));
    let p = Probe {
        learned: 0.1,
        initial: 0.,
        raw: -0.2,
        context: -0.1,
        ..Default::default()
    };
    assert_eq!(representation(&[p; 5], 5), (0.1, 5, true));
    assert!(!representation(&[p; 4], 5).2);
    for j in 0..3 {
        let mut ps = [Probe { learned: 0.5, ..p }; 5];
        match j {
            0 => ps[2].initial = 0.5,
            1 => ps[2].raw = 0.5,
            _ => ps[2].context = 0.5,
        };
        let (avg, wins, pass) = representation(&ps, 5);
        assert!(avg > 0.1);
        assert_eq!(wins, 4);
        assert!(!pass);
    }
    let ps: Vec<_> = (0..5)
        .map(|i| Probe {
            learned: 0.5,
            initial: if i == 0 { 0.45 } else { 0. },
            raw: if i % 2 == 1 { 0.45 } else { 0. },
            context: if i % 2 == 0 { 0.45 } else { 0. },
            ..p
        })
        .collect();
    let (avg, wins, pass) = representation(&ps, 5);
    assert!(abs(avg - 0.05) < 1e-12);
    assert_eq!(wins, 5);
    assert!(!pass);
}
#[test]
fn moving_random_paths_change_only_continuity_and_leave_commands_unchanged() {
    let settings = Motion {
        gain: Gain::Fixed,
        copy: CopyMode::Executed,
        random: false,
    };
    for sc in [
        Scenario::Continuity,
        Scenario::Vanish,
        Scenario::Nursery,
        Scenario::Binding,
        Scenario::Conflict,
    ] {
        for seed in [13, 27, 382] {
            let mut a = MovingScene::new(sc, seed, settings, 16);
            let mut b = MovingScene::new(
                sc,
                seed,
                Motion {
                    random: true,
                    ..settings
                },
                16,
            );
            if sc == Scenario::Continuity {
                assert_ne!(a.world.objects, b.world.objects);
                assert_eq!(
                    b.world.objects,
                    world(
                        sc,
                        seed,
                        WorldSettings {
                            random: true,
                            columns: None
                        }
                    )
                    .objects
                );
            }
            for _ in 0..=16 {
                let (ga, ca, oa) = a.read();
                let (gb, cb, ob) = b.read();
                assert_eq!(ga, gb);
                assert_eq!(ca, cb);
                if sc != Scenario::Continuity {
                    assert_eq!(oa.x, ob.x);
                    assert_eq!(a.world, b.world);
                }
                advance(&mut a.world);
                advance(&mut b.world);
            }
        }
    }
}
fn score_json(s: &MotionScore) -> String {
    let ratios = ["all", "moved", "still", "early", "late"]
        .iter()
        .zip(s.ratios)
        .map(|(k, v)| format!("\"{k}\":{}", number(v)))
        .collect::<Vec<_>>()
        .join(",");
    let probes = ["own", "world", "apparent"]
        .iter()
        .zip(s.probes)
        .map(|(key, p)| {
            format!(
                "\"{key}\":{{\"learned\":{},\"initial\":{},\"raw\":{},\"shuffled\":{}}}",
                number(p.learned),
                number(p.initial),
                number(p.raw),
                number(p.shuffled)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"ratios\":{{{ratios}}},\"probes\":{{{probes}}},\"trainCount\":{},\"testCount\":{}}}",
        s.train, s.test
    )
}
#[test]
fn movies_commands_weights_and_scores_reproduce_the_reference_digests() -> Result<(), Error> {
    let mut hashes: [Sha256; 4] = Default::default();
    for copy in COPIES {
        let mo = Motion {
            gain: Gain::Fixed,
            copy,
            random: false,
        };
        let opt = TrainingOptions {
            kind: Kind::Memory,
            motion: Some(mo),
            occlusion: Occlusion::PerCell,
            ..Default::default()
        };
        for sc in [
            Scenario::Continuity,
            Scenario::Vanish,
            Scenario::Nursery,
            Scenario::Binding,
        ] {
            for obs in movie(sc, 8101, 16, 40, opt.occlusion, opt.world, opt.motion) {
                for x in obs.x {
                    hashes[0].update(x.to_le_bytes());
                }
                for x in obs.copy.expect("command channel") {
                    hashes[1].update(x.to_le_bytes());
                }
            }
        }
        let make = || Element::new(7, 16, 3, true, Kind::Memory, true);
        let mut m = make();
        train(
            &mut m,
            Scale {
                depth: 3,
                episodes: 4,
                ..Scale::default()
            },
            7,
            opt,
            |_, _| Ok(()),
            None,
        )?;
        for x in &m.weights {
            hashes[2].update(x.to_le_bytes());
        }
        let initial = make();
        let saved = m.weights.clone();
        let trajectory = || -> Result<Vec<f64>, Error> {
            let mut p = Predictor::new(&m);
            let mut state = vec![];
            for o in movie(
                Scenario::Continuity,
                71,
                16,
                3,
                opt.occlusion,
                opt.world,
                opt.motion,
            ) {
                state = p.step(&o.x, o.copy)?.state.clone();
            }
            Ok(state)
        };
        let pre = trajectory()?;
        for random in [true, false] {
            assert!(matches!(
                evaluate(&m, &initial, Motion { random, ..mo }, 72, &[7, 8], &[8, 9]),
                Err(Error::SeedOverlap)
            ));
        }
        let s = evaluate(
            &m,
            &initial,
            mo,
            72,
            &[8001, 8002, 8003, 8004],
            &[8101, 8102, 8103, 8104],
        )?;
        if copy == CopyMode::Executed {
            assert!(s.probes[0].raw > 0.999);
        }
        assert_eq!(m.weights, saved);
        assert_eq!(pre, trajectory()?);
        hashes[3].update(score_json(&s));
    }
    assert_eq!(
        hashes.map(|h| format!("{:x}", h.finalize())),
        [
            "1ff01438d1bc8087ec437145fa191b486c77724dbfb216133b4bb1bd96a5f9ef",
            "edc10b7571e67ebe8b6c422bafa117fe42e9e8bbadd7be2f4c900be2aa52da79",
            "da6539f23d01ad1abadd15c2e5d5466dcd6fdd5e3bf79272fa631e062b88faba",
            "09504bf1162d0e60aa19b536491cf94d3ffd55b7f118b16338bf57b6d291a3bd"
        ]
    );
    Ok(())
}
