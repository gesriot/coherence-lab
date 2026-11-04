use crate::{perception::Agent, types::*};
use jscompat::hypot;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Decision {
    pub(crate) gaze: Gaze,
    pub(crate) reason: Look,
    pub(crate) focus: Option<u32>,
    note: &'static str,
}
fn decision(
    x: f64,
    y: f64,
    aperture: f64,
    reason: Look,
    focus: Option<u32>,
    note: &'static str,
) -> Decision {
    Decision {
        gaze: Gaze::new(x, y, aperture),
        reason,
        focus,
        note,
    }
}
pub(crate) fn initial(g: Gaze, p: Policy) -> Attention {
    Attention {
        active: matches!(p, Policy::RecordOnly | Policy::Full),
        influences_action: p == Policy::Full,
        focus_id: None,
        reason: Look::Idle,
        executed_reason: Look::Idle,
        executed_focus_id: None,
        agreement: Agreement::Underdetermined,
        intended_x: g.x,
        intended_y: g.y,
        intended_aperture: g.aperture,
        executed_x: g.x,
        executed_y: g.y,
        executed_aperture: g.aperture,
        note: "Контур только открыл глаза.",
        predicted_observed: false,
        predicted_drop: 0.,
        self_error: None,
    }
}
pub(crate) fn agreement(a: &Agent) -> Agreement {
    if a.tracks.iter().any(|t| t.conflict) {
        return Agreement::Competing;
    }
    let unbound = a
        .blobs
        .iter()
        .any(|b| !a.tracks.iter().any(|t| t.bound_blob == Some(b.index)));
    let mean = if a.tracks.is_empty() {
        1.
    } else {
        a.tracks.iter().fold(0., |s, t| s + t.uncertainty) / a.tracks.len() as f64
    };
    if unbound || mean > 0.42 {
        Agreement::Underdetermined
    } else {
        Agreement::Settled
    }
}
fn center(i: usize) -> (f64, f64) {
    (((i % 4) as f64 + 0.5) * 25., ((i / 4) as f64 + 0.5) * 25.)
}
pub(crate) fn record(last: &mut [i32; 16], g: Gaze, step: u32) {
    for (i, v) in last.iter_mut().enumerate() {
        let (x, y) = center(i);
        if (x - g.x).abs() <= g.aperture && (y - g.y).abs() <= g.aperture {
            *v = step as i32;
        }
    }
}
fn explore(last: &[i32; 16]) -> Decision {
    let (i, &step) = last
        .iter()
        .enumerate()
        .min_by_key(|(_, v)| *v)
        .expect("region grid has sixteen cells");
    let (x, y) = center(i);
    decision(
        x,
        y,
        22.,
        Look::Explore,
        None,
        if step < 0 {
            "Осматриваю область, центр которой ещё не попадал в окно."
        } else {
            "Возвращаюсь в область, которую дольше всего не проверял."
        },
    )
}
fn due(cycle: u32) -> bool {
    cycle > 0 && cycle.is_multiple_of(8)
}
fn priority(t: &Track, g: Gaze) -> f64 {
    let mut s = t.uncertainty;
    if t.reason == Reason::Occlusion || t.conflict {
        s += 1.05;
    }
    if t.reason == Reason::Absence {
        s += 0.55;
    }
    if t.reason == Reason::NoData {
        s += 0.22;
    }
    if matches!(t.source, Source::Predicted | Source::Remembered) {
        s += 0.18;
    }
    if hypot(t.x - g.x, t.y - g.y) <= g.aperture && (t.uncertainty > 0.3 || t.conflict) {
        s += 0.35;
    }
    if t.source == Source::Observed && t.uncertainty < 0.22 && t.unseen == 0 && !t.conflict {
        s -= 0.2;
    }
    s
}
fn focused(t: &Track, aperture: f64, reason: Look, note: &'static str) -> Decision {
    decision(
        clamp(t.x, 4., 96.),
        clamp(t.y, 4., 96.),
        aperture,
        reason,
        Some(t.id),
        note,
    )
}
fn schema(a: &mut Agent) -> Decision {
    let ws: Vec<_> = a.tracks.iter().filter(|t| t.in_workspace).collect();
    if a.hold > 0
        && let Some(t) = ws.iter().find(|t| Some(t.id) == a.attention.focus_id)
        && (t.reason == Reason::Occlusion || t.conflict || t.uncertainty > 0.18)
    {
        a.hold -= 1;
        let reason = if t.reason == Reason::Occlusion || t.source != Source::Observed {
            Look::Verify
        } else if t.conflict {
            Look::Conflict
        } else {
            Look::Uncertainty
        };
        return focused(
            t,
            if t.conflict { 13. } else { 18. },
            reason,
            "Проверка ещё не закончена. Удерживаю взгляд на выбранном треке.",
        );
    }
    let conflicts: Vec<_> = ws.iter().filter(|t| t.conflict).collect();
    if let Some(t) = conflicts.first() {
        let x = conflicts.iter().fold(0., |s, t| s + t.x) / conflicts.len() as f64;
        let y = conflicts.iter().fold(0., |s, t| s + t.y) / conflicts.len() as f64;
        return decision(
            clamp(x, 4., 96.),
            clamp(y, 4., 96.),
            13.,
            Look::Conflict,
            Some(t.id),
            "Конфликт гипотез. Узкое окно – проверить, одна это сущность или две.",
        );
    }
    let (mut best, mut best_score) = (None, -1.);
    for &t in &ws {
        let score = priority(t, a.gaze);
        if score > best_score {
            best_score = score;
            best = Some(t);
        }
    }
    if let Some(t) = best
        && best_score > 0.42
    {
        let verify = t.reason == Reason::Occlusion
            || matches!(t.source, Source::Predicted | Source::Remembered);
        return focused(
            t,
            if t.reason == Reason::Noise || t.conflict {
                13.
            } else {
                18.
            },
            if verify {
                Look::Verify
            } else {
                Look::Uncertainty
            },
            if verify {
                "Прогноз не подтверждён взглядом. Иду туда, где гипотеза должна проявиться."
            } else {
                "Неопределённость выше порога. Перевожу взгляд на этот трек."
            },
        );
    }
    if due(a.cycle)
        && !ws.is_empty()
        && ws
            .iter()
            .all(|t| t.unseen == 0 && t.uncertainty < 0.34 && !t.conflict)
    {
        return explore(&a.last_viewed);
    }
    let pursued: Vec<_> = ws
        .into_iter()
        .filter(|t| t.source == Source::Observed && t.bound_blob.is_some())
        .collect();
    if let Some(t) = pursued
        .iter()
        .find(|t| Some(t.id) == a.attention.focus_id)
        .or(pursued.first())
    {
        return focused(
            t,
            18.,
            Look::Pursue,
            "Гипотеза устойчива. Сопровождаю её, пока не появится дисгармония.",
        );
    }
    explore(&a.last_viewed)
}
pub(crate) fn first_order(a: &Agent, p: Policy) -> Decision {
    let young = if p == Policy::Novelty {
        a.tracks
            .iter()
            .filter(|t| t.age <= 6)
            .min_by_key(|t| (t.age, t.id))
    } else {
        None
    };
    let focus = young
        .or_else(|| a.tracks.iter().find(|t| Some(t.id) == a.attention.focus_id))
        .or_else(|| {
            a.tracks
                .iter()
                .reduce(|b, t| if t.evidence > b.evidence { t } else { b })
        });
    let Some(t) = focus else {
        return explore(&a.last_viewed);
    };
    if due(a.cycle) && t.bound_blob.is_some() {
        return Decision {
            focus: Some(t.id),
            ..explore(&a.last_viewed)
        };
    }
    decision(
        clamp(t.x + t.vx, 4., 96.),
        clamp(t.y + t.vy, 4., 96.),
        18.,
        Look::Pursue,
        Some(t.id),
        "Взгляд следует прогнозу положения выбранного трека.",
    )
}
fn salience(a: &mut Agent, memory: bool) -> Decision {
    let largest = a
        .blobs
        .iter()
        .copied()
        .reduce(|b, t| if t.area > b.area { t } else { b });
    if memory {
        a.salience.push(largest);
        if a.salience.len() > a.capacity {
            a.salience.remove(0);
        }
    }
    let blob = largest.or_else(|| a.salience.iter().rev().flatten().next().copied());
    let Some(b) = blob else {
        return explore(&a.last_viewed);
    };
    let focus = largest.and_then(|b| {
        a.tracks
            .iter()
            .find(|t| t.bound_blob == Some(b.index))
            .map(|t| t.id)
    });
    decision(
        clamp(b.cx, 4., 96.),
        clamp(b.cy, 4., 96.),
        22.,
        Look::Salience,
        focus,
        if largest.is_some() {
            "Взгляд направлен к самому крупному пятну."
        } else {
            "В окне пусто. Проверяю место последнего крупного пятна."
        },
    )
}
fn predict(a: &Agent, d: Decision) -> (bool, f64) {
    let Some(t) = a.tracks.iter().find(|t| Some(t.id) == d.focus) else {
        return (false, 0.);
    };
    let inside = hypot(t.x - d.gaze.x, t.y - d.gaze.y) <= d.gaze.aperture;
    let observed = inside && !matches!(t.reason, Reason::Occlusion | Reason::Absence);
    let drop = if !inside {
        -0.07
    } else if t.reason == Reason::Occlusion {
        -0.02
    } else {
        t.uncertainty * 0.45
    };
    (observed, drop)
}
pub(crate) fn decide(a: &mut Agent, c: &Config) {
    let policy = c.ablation;
    let active = matches!(policy, Policy::Full | Policy::RecordOnly);
    if !active || c.pin_gaze {
        a.hold = 0;
    }
    if policy != Policy::DummyMemory {
        a.salience.clear();
    }
    let intended = active.then(|| schema(a));
    let executed = if c.pin_gaze {
        decision(
            c.pin_x,
            c.pin_y,
            a.gaze.aperture,
            Look::Pinned,
            intended.and_then(|d| d.focus),
            "Вмешательство: взгляд зафиксирован и не следует модели.",
        )
    } else if policy == Policy::Full {
        intended.expect("full policy maintains its attention schema")
    } else if matches!(policy, Policy::FirstOrder | Policy::Novelty) {
        first_order(a, policy)
    } else {
        salience(a, policy == Policy::DummyMemory)
    };
    if intended.is_some_and(|d| d.focus != a.attention.focus_id) {
        a.hold = 4;
    }
    let (observed, drop) = if active {
        predict(a, executed)
    } else {
        (false, 0.)
    };
    let influences = policy == Policy::Full && !c.pin_gaze;
    let goal = intended.unwrap_or(executed);
    a.attention = Attention {
        active,
        influences_action: influences,
        focus_id: goal.focus,
        executed_reason: executed.reason,
        executed_focus_id: executed.focus,
        reason: if influences {
            executed.reason
        } else {
            goal.reason
        },
        agreement: agreement(a),
        intended_x: goal.gaze.x,
        intended_y: goal.gaze.y,
        intended_aperture: goal.gaze.aperture,
        executed_x: executed.gaze.x,
        executed_y: executed.gaze.y,
        executed_aperture: executed.gaze.aperture,
        note: executed.note,
        predicted_observed: observed,
        predicted_drop: drop,
        self_error: a.self_error,
    };
    a.gaze = Gaze::new(
        clamp(executed.gaze.x, 2., 98.),
        clamp(executed.gaze.y, 2., 98.),
        executed.gaze.aperture,
    );
}
