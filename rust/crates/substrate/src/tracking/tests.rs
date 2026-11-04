use super::*;
#[test]
fn target_index_and_masks_separate_returns_and_false_predictions() {
    let grid = 16;
    let mut e = Episode {
        frames: vec![vec![0.; 768]; 25],
        events: vec![],
        ghost: vec![vec![]; 25],
    };
    let mut pred = e.frames.clone();
    let cells = vec![82, 99];
    e.events.push(Event {
        time: 8,
        wall: true,
        length: 4,
        cells: cells.clone(),
    });
    e.events.push(Event {
        time: 4,
        wall: false,
        length: 1,
        cells: vec![0],
    });
    e.frames[4][1] = 1.;
    pred[4][1] = 0.25;
    let others: Vec<_> = [0, 1, 2, 9, 10, 11, 12, 13, 14, 15]
        .iter()
        .flat_map(|r| [r * grid + 2, r * grid + 3])
        .collect();
    assert_eq!(other_rows(&cells, grid), others);
    for &i in &cells {
        e.frames[8][3 * i + 1] = 1.;
    }
    for (t, p) in pred.iter_mut().enumerate() {
        for &i in &cells {
            p[3 * i + 1] = t as f64 / 100.;
        }
        for &i in &others {
            p[3 * i + 2] = if (2..=8).contains(&t) { 0.125 } else { 1. };
        }
        p[3 * 130 + 1] = 1.;
    }
    e.frames[17][3 * cells[0] + 1] = 0.051;
    e.frames[18][3 * cells[1] + 1] = 0.05;
    let mut t = Totals::default();
    t.score(&e, &pred, grid);
    assert_eq!((t.span[0].count, t.span[0].min), (1, 4));
    assert!(abs(t.event[0].value() - 0.92 * 0.92) < 1e-14);
    assert_eq!(t.event[1].value(), 0.75 * 0.75);
    assert_eq!(
        t.curve.map(Average::value),
        [2, 3, 4, 5, 6, 7, 8, 9, 10].map(|n| f64::from(n) / 100.)
    );
    assert_eq!(t.other.n, others.len() * 7);
    assert_eq!(t.other.value(), 0.125);
    assert_eq!(t.after.n, 15);
    assert!(abs(t.after.sum - (2. * 8. * 0.195 - 0.17)) < 1e-12);
    e.events = vec![Event {
        time: 6,
        length: 2,
        wall: true,
        cells,
    }];
    let mut t = Totals::default();
    t.score(&e, &pred, grid);
    assert_eq!(t.span[0].count, 1);
    assert!(t.curve.iter().all(|s| s.n == 0));
}
#[test]
fn tracking_gate_checks_mean_worst_false_expectations_and_base() {
    let good = TrackingScore {
        wall: 0.5,
        other: 0.03,
        after: 0.08,
        ..Default::default()
    };
    let base = [0.9; 5];
    let test = |b: &[f64], s: &[&TrackingScore]| criterion(b, s, 5).1;
    assert!(test(&base, &[&good; 5]));
    for s in [
        TrackingScore {
            wall: 0.50001,
            ..good
        },
        TrackingScore {
            other: 0.03001,
            ..good
        },
        TrackingScore {
            after: 0.08001,
            ..good
        },
    ] {
        assert!(!test(&base, &[&s; 5]));
    }
    let lo = TrackingScore { wall: 0.2, ..good };
    let hi = TrackingScore {
        wall: 0.65001,
        ..good
    };
    assert!(!test(&base, &[&lo, &lo, &hi, &lo, &lo]));
    assert!(!test(&[0.89999; 5], &[&good; 5]));
    assert!(!test(&base[..4], &[&good; 5]));
}
