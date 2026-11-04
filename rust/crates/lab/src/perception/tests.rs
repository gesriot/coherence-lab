use super::*;
use crate::{
    Episode,
    world::{advance, hidden, make_world},
};

fn patches(columns: &[usize]) -> Observation {
    Observation {
        grid: 8,
        gaze: Gaze::new(50., 50., 24.),
        cells: (0..64)
            .map(|i| {
                let surface = i / 8 == 4 && columns.contains(&(i % 8));
                Cell {
                    hue: 20.,
                    saturation: if surface { 70. } else { 0. },
                    lightness: if surface { 50. } else { 6. },
                }
            })
            .collect(),
    }
}
fn agent(obs: &Observation) -> Agent {
    Agent::new(obs, Policy::Full, None)
}

#[test]
fn entry_point_advances_and_equal_seeds_reproduce_the_episode() {
    let run = || {
        let mut ep = Episode::new(Scenario::Continuity, 9, Policy::Full);
        ep.step(&Config::default());
        assert_eq!(ep.world.t, 1);
        assert_eq!(ep.agent.step, 1);
        for _ in 1..40 {
            ep.step(&Config::default());
        }
        ep
    };
    assert_eq!(run(), run());
}
#[test]
fn connected_reds_have_a_red_circular_mean() {
    let obs = Observation {
        grid: 2,
        gaze: Gaze::new(0., 0., 1.),
        cells: [355., 5., 355., 5.]
            .map(|hue| Cell {
                hue,
                saturation: 70.,
                lightness: 50.,
            })
            .to_vec(),
    };
    let blobs = blobs(&obs);
    assert_eq!(blobs.len(), 1);
    assert!(hue_delta(blobs[0].hue, 0.) < 1e-9);
}
#[test]
fn revisiting_the_same_frame_adds_no_evidence() {
    let run = |budget| {
        let mut ep = Episode::new(Scenario::Continuity, 4, Policy::Full);
        ep.agent.budget = budget;
        ep.step(&Config::default());
        ep.agent.tracks.remove(0)
    };
    let t = run(1);
    let u = run(4);
    assert_eq!(t.bound_blob, Some(0));
    let values = |t: Track| [t.evidence, t.prediction, t.uncertainty, t.fill, t.hue];
    assert_eq!(values(t), values(u));
}
#[test]
fn seeds_vary_geometry_and_preserve_events() {
    assert_ne!(
        make_world(Scenario::Binding, 1).objects,
        make_world(Scenario::Binding, 2).objects
    );
    for seed in 1..=20 {
        let mut w = make_world(Scenario::Continuity, seed);
        let (mut seen, mut emerged) = (false, false);
        for _ in 0..60 {
            advance(&mut w);
            if hidden(&w.objects[0], w.wall) {
                seen = true
            } else if seen {
                emerged = true;
            }
        }
        assert!(emerged);
        let mut w = make_world(Scenario::Binding, seed);
        let hue = w.objects[0].hue;
        for i in 1..=16 {
            advance(&mut w);
            assert_eq!(w.objects[0].hue == hue, i < 16);
        }
        let w = make_world(Scenario::Conflict, seed);
        let (a, b) = (&w.objects[0], &w.objects[1]);
        assert!(hypot(a.x - b.x, a.y - b.y) < 15.);
    }
}
#[test]
fn replay_grades_raw_pixels_against_the_previous_command() {
    let mut ep = Episode::new(Scenario::Continuity, 3, Policy::Full);
    let frame = ep.agent.cells.clone();
    for _ in 0..8 {
        ep.step(&Config::default());
    }
    let guesses = &ep.agent.guesses;
    assert_eq!(guesses.len(), frame.len());
    let mismatches = frame
        .iter()
        .zip(guesses)
        .filter(|(c, g)| classify(**c) != **g)
        .count();
    ep.step(&Config {
        replay: true,
        ..Config::default()
    });
    assert_eq!(ep.agent.cells, frame);
    assert_eq!(
        ep.agent.energy.action,
        mismatches as f64 / frame.len() as f64
    );
}
#[test]
fn forecast_projects_the_wall_into_the_commanded_window() {
    let obs = Observation {
        grid: 2,
        gaze: Gaze::new(50., 50., 4.),
        cells: [6., 40., 6., 40.]
            .map(|lightness| Cell {
                lightness,
                ..Cell::default()
            })
            .to_vec(),
    };
    let mut a = agent(&obs);
    a.step(
        obs,
        &Config {
            pin_gaze: true,
            pin_x: 54.,
            ..Config::default()
        },
    );
    use Expectation::{Empty, Occluder};
    assert_eq!(a.guesses, [Occluder, Empty, Occluder, Empty]);
    assert_eq!(a.energy.action, 0.);
}
#[test]
fn first_order_keeps_focus_and_follows_its_forecast() {
    let mut obs = patches(&[1, 6]);
    let mut a = agent(&obs);
    a.attention.focus_id = Some(a.tracks[1].id);
    obs.gaze.x = 52.;
    a.step(obs, &Config::new(Policy::FirstOrder));
    let t = &a.tracks[1];
    assert_eq!(a.attention.focus_id, Some(t.id));
    assert_eq!(a.gaze.x, t.x + t.vx);
    assert_eq!(a.gaze.aperture, 18.);
    assert!(!a.attention.active);
}
#[test]
fn only_salience_memory_returns_after_a_blank_frame() {
    for (p, returns) in [(Policy::DummyMemory, true), (Policy::NoSchema, false)] {
        let mut a = agent(&patches(&[1]));
        let cfg = Config::new(p);
        a.step(patches(&[1]), &cfg);
        let last = a.gaze.x;
        a.step(patches(&[]), &cfg);
        assert_eq!(a.gaze.x == last, returns);
    }
}
#[test]
fn record_only_grades_the_executed_focus() {
    let obs = patches(&[1, 5, 6]);
    let mut a = agent(&obs);
    a.attention.focus_id = Some(a.tracks[0].id);
    a.hold = 2;
    a.step(obs, &Config::new(Policy::RecordOnly));
    assert_ne!(a.attention.intended_x, a.attention.executed_x);
    assert_eq!(a.focus_prev_id, Some(a.tracks[1].id));
    assert!(a.predicted_observed);
}
#[test]
fn workspace_and_confidence_change_the_schema_decision() {
    let run = |capacity, inject_confidence| {
        let mut a = agent(&patches(&[1, 6]));
        a.capacity = capacity;
        for _ in 0..3 {
            a.step(patches(&[1, 6]), &Config::default());
        }
        a.step(
            patches(&[1]),
            &Config {
                inject_confidence,
                ..Config::default()
            },
        );
        a
    };
    let full = run(3, false);
    let narrow = run(1, false);
    let injected = run(3, true);
    assert_eq!(narrow.tracks.iter().filter(|t| t.in_workspace).count(), 1);
    assert_ne!(full.gaze.x, narrow.gaze.x);
    assert!(injected.tracks.iter().all(|t| t.uncertainty < 0.05
        && t.confidence > 0.95
        && !t.conflict
        && t.reason == Reason::None));
    assert!(injected.attention.influences_action);
    assert_eq!(injected.attention.intended_x, injected.attention.executed_x);
    assert_ne!(injected.gaze.x, full.gaze.x);
}
#[test]
fn binding_and_motion_weights_resolve_an_ambiguous_crossing() {
    let frame = |swapped| {
        let mut obs = patches(&[3, 4]);
        obs.cells[35].hue = if swapped { 200. } else { 20. };
        obs.cells[36].hue = if swapped { 20. } else { 200. };
        obs
    };
    let assignment = |binding, temporal| {
        let mut a = agent(&frame(false));
        let ids: Vec<_> = a.tracks.iter().map(|t| t.id).collect();
        a.weights[1] = binding;
        a.weights[2] = temporal;
        a.step(frame(true), &Config::default());
        ids.iter()
            .map(|id| {
                a.tracks
                    .iter()
                    .find(|t| t.id == *id)
                    .and_then(|t| t.bound_blob)
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(assignment(0.1, 2.), [Some(0), Some(1)]);
    assert_eq!(assignment(2., 0.1), [Some(1), Some(0)]);
}
#[test]
fn energies_descend_and_weak_hypotheses_expire() {
    let mut a = agent(&patches(&[1, 6]));
    assert_eq!(a.tracks.len(), 2);
    assert!(a.energy_trace[0] > a.energy.total);
    for _ in 0..20 {
        a.step(patches(&[]), &Config::default());
        assert_eq!(a.energy_trace.len(), a.iterations + 1);
        assert!(a.iterations <= a.budget);
        assert!(a.energy_trace.windows(2).all(|w| w[1] <= w[0]));
        assert_eq!(a.energy_trace.last(), Some(&a.energy.total));
    }
    assert!(a.tracks.is_empty());
}
#[test]
fn injecting_confidence_changes_later_samples_and_bindings() {
    let mut control = Episode::new(Scenario::Binding, 1, Policy::Full);
    let mut injected = control.clone();
    let (mut samples, mut bindings) = (false, false);
    for i in 0..16 {
        control.step(&Config::default());
        injected.step(&Config {
            inject_confidence: true,
            ..Config::default()
        });
        let (a, b) = (&control.agent, &injected.agent);
        if i == 0 {
            assert_eq!(a.cells, b.cells);
        }
        samples |= a
            .cells
            .iter()
            .zip(&b.cells)
            .any(|(x, y)| classify(*x) != classify(*y));
        bindings |= samples
            && a.tracks.iter().any(|t| {
                !b.tracks
                    .iter()
                    .any(|u| u.id == t.id && u.bound_blob == t.bound_blob)
            });
    }
    assert!(samples && bindings);
}
#[test]
fn exploration_uses_window_history_and_row_order() {
    let frame = |x, y, aperture| Observation {
        gaze: Gaze::new(x, y, aperture),
        ..patches(&[])
    };
    let run = |initial| {
        let mut a = agent(&frame(initial, 12.5, 12.5));
        a.step(frame(87.5, 87.5, 12.5), &Config::default());
        assert_eq!(a.attention.executed_reason, Look::Explore);
        a.gaze
    };
    assert_eq!(run(12.5), Gaze::new(37.5, 12.5, 22.));
    assert_eq!(run(87.5), Gaze::new(12.5, 12.5, 22.));
    assert_eq!(run(12.5), run(12.5));
    for (x, r) in [(87.5, 12.5), (50., 50.)] {
        let mut a = agent(&frame(x, x, r));
        a.step(frame(25., 25., 12.5), &Config::default());
        if r == 12.5 {
            assert_eq!(a.last_viewed[..6], [1, 1, -1, -1, 1, 1]);
        }
        assert_eq!(a.gaze, Gaze::new(62.5, 12.5, 22.));
    }
}
#[test]
fn first_order_and_novelty_ignore_second_order_fields() {
    for p in [Policy::FirstOrder, Policy::Novelty] {
        let mut a = Agent::new(&patches(&[1, 6]), p, None);
        if p == Policy::Novelty {
            a.step(patches(&[1, 6]), &Config::new(p));
            assert!(!a.attention.active);
            assert_eq!(a.energy.self_error, None);
        }
        for cycle in [0, 1, 8, 9] {
            a.cycle = cycle;
            let before = attention::first_order(&a, p);
            if p == Policy::Novelty {
                assert_eq!(
                    before.reason,
                    if cycle == 8 {
                        Look::Explore
                    } else {
                        Look::Pursue
                    }
                );
                assert_eq!(before.gaze.aperture, if cycle == 8 { 22. } else { 18. });
            }
            for t in &mut a.tracks {
                t.source = if t.source == Source::Observed {
                    Source::Remembered
                } else {
                    Source::Observed
                };
                t.reason = if t.reason == Reason::None {
                    Reason::Occlusion
                } else {
                    Reason::None
                };
                t.uncertainty = 1. - t.uncertainty;
                t.conflict = !t.conflict;
            }
            a.attention.agreement = if a.attention.agreement == Agreement::Settled {
                Agreement::Competing
            } else {
                Agreement::Settled
            };
            assert_eq!(attention::first_order(&a, p), before);
        }
    }
}
#[test]
fn scheduled_exploration_requires_a_bound_sticky_focus() {
    let mut a = agent(&patches(&[1, 6]));
    a.attention.focus_id = Some(a.tracks[1].id);
    for cycle in [0, 7, 8, 9, 16] {
        a.cycle = cycle;
        let d = attention::first_order(&a, Policy::FirstOrder);
        let due = cycle == 8 || cycle == 16;
        assert_eq!(d.reason, if due { Look::Explore } else { Look::Pursue });
        assert_eq!(d.gaze.aperture, if due { 22. } else { 18. });
        assert_eq!(d.focus, a.attention.focus_id);
    }
    a.tracks[1].bound_blob = None;
    assert_eq!(
        attention::first_order(&a, Policy::FirstOrder).reason,
        Look::Pursue
    );
}
#[test]
fn a_quantized_jump_survives_eighteen_misses_and_rebinds() {
    let frame = |y, hidden: bool| Observation {
        grid: 12,
        gaze: Gaze::new(48.5, y - 1.5, 18.),
        cells: (0..144)
            .map(|i| {
                let spot = !hidden && i == 78;
                Cell {
                    hue: if hidden { 0. } else { 20. },
                    saturation: if spot { 70. } else { 0. },
                    lightness: if spot || hidden { 50. } else { 6. },
                }
            })
            .collect(),
    };
    let cfg = Config::default();
    let mut a = agent(&frame(50., false));
    for _ in 0..7 {
        a.step(frame(50., false), &cfg);
    }
    a.step(frame(53., false), &cfg);
    let id = a.tracks[0].id;
    assert_eq!(a.tracks[0].vy, 3. / 8.);
    for _ in 0..18 {
        a.step(frame(50., true), &cfg);
    }
    let t = &a.tracks[0];
    assert_eq!(t.unseen, 18);
    assert!((t.y - 50.).abs() < 16.);
    assert!((t.y + t.vy - 50.).abs() / 16. > 0.5);
    a.budget = 1;
    a.step(frame(50., false), &cfg);
    assert_eq!(a.tracks.len(), 1);
    assert_eq!(a.tracks[0].id, id);
    assert_eq!(a.tracks[0].bound_blob, Some(0));
    assert!(a.energy.temporal < 0.5);
}
#[test]
fn novelty_uses_youngest_then_id_then_sticky_focus() {
    let mut a = Agent::new(&patches(&[1, 4, 6]), Policy::Novelty, None);
    a.tracks.reverse();
    let [c, b, first] = [a.tracks[0].id, a.tracks[1].id, a.tracks[2].id];
    a.attention.focus_id = Some(c);
    for (t, age) in a.tracks.iter_mut().zip([7, 6, 6]) {
        t.age = age;
    }
    let focus = |a: &Agent| attention::first_order(a, Policy::Novelty).focus;
    assert_eq!(focus(&a), Some(first));
    a.tracks[1].age = 5;
    assert_eq!(focus(&a), Some(b));
    for t in &mut a.tracks {
        t.age = 7;
    }
    a.tracks[1].evidence = 0.99;
    assert_eq!(focus(&a), Some(c));
    a.attention.focus_id = None;
    assert_eq!(focus(&a), Some(b));
}
