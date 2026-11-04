use crate::{
    Episode,
    experience::{Draw, GapExample, Learner, Movement, Selection},
    perception::{match_radius, miss},
    sensor::Observation,
    types::*,
    world::schedule_outdoor_losses,
};
use jscompat::{hypot, max};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Condition {
    Full,
    FirstOrder,
    NoProtocol,
    StrongFirstOrder,
    Frozen,
    Oracle,
}
pub const PARTICIPANTS: [Condition; 4] = [
    Condition::Full,
    Condition::FirstOrder,
    Condition::NoProtocol,
    Condition::StrongFirstOrder,
];
impl Condition {
    pub fn label(self) -> &'static str {
        match self {
            Self::Full => "Полный контур",
            Self::FirstOrder => "Первый порядок",
            Self::NoProtocol => "Без протокола",
            Self::StrongFirstOrder => "Сильный первый порядок",
            Self::Frozen => "Без обучения",
            Self::Oracle => "Оракул",
        }
    }
    pub(crate) fn policy(self) -> Policy {
        match self {
            Self::Full | Self::NoProtocol => Policy::Full,
            Self::StrongFirstOrder => Policy::Novelty,
            _ => Policy::FirstOrder,
        }
    }
    pub(crate) fn selection(self) -> Selection {
        match self {
            Self::Full => Selection::Attention,
            Self::StrongFirstOrder => Selection::Error,
            _ => Selection::Uniform,
        }
    }
}

struct Gap {
    start: [f64; 2],
    step: u32,
    trace: Track,
    example: GapExample,
    deleted: Option<u32>,
}
struct ClosedGap {
    id: u32,
    start: [f64; 2],
    step: u32,
    forecast: [f64; 3],
    example: GapExample,
}
#[derive(Default)]
struct GapBook(Vec<Gap>);
impl GapBook {
    fn step(
        &mut self,
        step: u32,
        before: &[Track],
        after: &[Track],
        advance: impl Fn(&Track) -> Track,
    ) -> Vec<ClosedGap> {
        for prior in before {
            if prior.bound_blob.is_none()
                || after
                    .iter()
                    .any(|t| t.id == prior.id && t.bound_blob.is_some())
            {
                continue;
            }
            let g = Gap {
                start: [prior.x + prior.vx, prior.y + prior.vy],
                step,
                trace: prior.clone(),
                deleted: None,
                example: GapExample {
                    seen: prior.seen_run.unwrap_or(0),
                    steps: vec![],
                    found: false,
                },
            };
            if let Some(old) = self.0.iter_mut().find(|x| x.trace.id == prior.id) {
                *old = g;
            } else {
                self.0.push(g);
            }
        }
        let mut births: Vec<_> = after.iter().filter(|t| t.age == 1).collect();
        let mut closed = Vec::new();
        self.0.retain_mut(|g| {
            let prev = &g.trace;
            let live = after.iter().find(|t| t.id == prev.id);
            let predicted = live.cloned().unwrap_or_else(|| advance(prev));
            let mut found = live.is_some_and(|t| t.bound_blob.is_some());
            if live.is_none() {
                g.deleted.get_or_insert(step);
                if let Some(i) = births.iter().position(|b| {
                    hypot(b.x - predicted.x, b.y - predicted.y) <= match_radius(prev)
                        && hue_delta(b.hue, predicted.hue) <= 40.
                }) {
                    found = true;
                    births.remove(i);
                }
            }
            if !found {
                g.example.steps.push(
                    predicted
                        .gap_step
                        .expect("unobserved learned tracks have a gap step"),
                );
            }
            if found || g.deleted.is_some_and(|at| step - at >= 30) {
                if !g.example.steps.is_empty() {
                    g.example.found = found;
                    closed.push(ClosedGap {
                        id: prev.id,
                        start: g.start,
                        step: g.step,
                        forecast: [prev.x + prev.vx, prev.y + prev.vy, match_radius(prev)],
                        example: g.example.clone(),
                    });
                }
                false
            } else {
                g.trace = predicted;
                true
            }
        });
        closed
    }
}
pub(crate) fn nearest(objects: &[Object], x: f64, y: f64) -> usize {
    let mut best = 0;
    let mut distance = f64::INFINITY;
    for (i, o) in objects.iter().enumerate() {
        let dx = o.x - x;
        let dy = o.y - y;
        let d = dx * dx + dy * dy;
        if d < distance {
            distance = d;
            best = i;
        }
    }
    best
}
fn confirms(o: &Object, step: u32, [x, y, r]: [f64; 3]) -> bool {
    o.vanish_at.is_none_or(|at| at > step) && hypot(o.x - x, o.y - y) <= r
}
// Only this driver creates privileged labels. The learner consumes examples.
struct OracleFrames(Vec<Vec<Object>>);
impl OracleFrames {
    fn for_condition(id: Condition, initial_objects: impl FnOnce() -> Vec<Object>) -> Option<Self> {
        (id == Condition::Oracle).then(|| Self(vec![initial_objects()]))
    }
    fn displacement(&self, step: u32, t: &Track) -> [f64; 2] {
        let before = &self.0[step as usize - 1];
        let i = nearest(before, t.x, t.y);
        let now = &self.0[step as usize][i];
        [now.x - before[i].x, now.y - before[i].y]
    }
    fn found(&self, step: u32, g: &ClosedGap) -> bool {
        let i = nearest(&self.0[g.step as usize], g.start[0], g.start[1]);
        confirms(&self.0[step as usize][i], step, g.forecast)
    }
}
pub(crate) fn train(
    learner: &mut Learner,
    condition: Condition,
    scenario: Scenario,
    seed: u32,
    mut trace: Option<&mut Vec<Draw>>,
) {
    let cfg = Config::new(condition.policy());
    let mut ep = Episode::with_models(
        scenario,
        seed,
        cfg.ablation,
        Some(learner.models.clone()),
        None,
    );
    if scenario == Scenario::Nursery {
        schedule_outdoor_losses(&mut ep.world);
    }
    let mut labels = OracleFrames::for_condition(condition, || ep.world.objects.clone());
    let mut book = GapBook::default();
    let mut errors: Vec<(u32, f64)> = vec![];
    for step in 1..=crate::genesis::TRAINING_STEPS {
        let before = ep.agent.tracks.clone();
        let focus = ep.agent.attention.executed_focus_id;
        ep.step(&cfg);
        if condition == Condition::Frozen {
            continue;
        }
        if let Some(l) = &mut labels {
            l.0.push(ep.world.objects.clone());
        }
        let a = &mut ep.agent;
        let error = a.self_error.unwrap_or(0.);
        if let Some(id) = focus {
            if let Some((_, v)) = errors.iter_mut().find(|(i, _)| *i == id) {
                *v = max(*v, error);
            } else {
                errors.push((id, error));
            }
        }
        for t in &a.tracks {
            let [.., old, middle, _] = t.motion_samples.as_slice() else {
                continue;
            };
            if t.bound_blob.is_none() || middle.step + 1 != t.age || old.step + 2 != t.age {
                continue;
            }
            let prior = before
                .iter()
                .find(|p| p.id == t.id)
                .expect("consecutive matches retain a prior track");
            let target = labels.as_ref().map_or_else(
                || [t.x - prior.x, t.y - prior.y],
                |l| l.displacement(step, prior),
            );
            learner.remember_motion(
                Movement {
                    input: prior
                        .forecast_input
                        .expect("a learned track stores its forecast input"),
                    target,
                },
                (Some(t.id) == focus).then_some(error),
            );
        }
        let obs = Observation {
            gaze: a.obs_gaze,
            grid: a.grid,
            cells: a.cells.clone(),
        };
        for g in book.step(step, &before, &a.tracks, |t| {
            miss(t, &obs, Some(&learner.models))
        }) {
            let found = labels
                .as_ref()
                .map_or(g.example.found, |l| l.found(step, &g));
            learner.remember_gap(
                GapExample { found, ..g.example },
                errors.iter().find(|(id, _)| *id == g.id).map(|x| x.1),
            );
        }
        learner.update(trace.as_deref_mut());
        a.models = Some(learner.models.clone());
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::{Context, GapStep, Models};
    use crate::world::{advance, hidden, make_world};
    #[test]
    fn gap_book_closes_on_binding_birth_or_timeout_and_censors_pending() {
        let mut track = Episode::new(Scenario::Continuity, 1, Policy::Full)
            .agent
            .tracks
            .remove(0);
        track.seen_run = Some(6);
        let advance = |t: &Track| Track {
            bound_blob: None,
            unseen: t.unseen + 1,
            reason: Reason::Conflict,
            gap_step: Some(GapStep::new(Context::Open, false)),
            ..t.clone()
        };
        let missed = advance(&track);
        for outcome in ["live", "birth", "timeout", "censored"] {
            let mut book = GapBook::default();
            assert!(
                book.step(
                    1,
                    std::slice::from_ref(&track),
                    std::slice::from_ref(&missed),
                    advance
                )
                .is_empty()
            );
            if outcome == "live" {
                assert!(
                    book.step(
                        2,
                        std::slice::from_ref(&missed),
                        std::slice::from_ref(&track),
                        advance
                    )[0]
                    .example
                    .found
                );
                continue;
            }
            for i in 2..=31 {
                assert!(book.step(i, &[], &[], advance).is_empty());
            }
            if outcome == "censored" {
                continue;
            }
            let births = if outcome == "birth" {
                vec![Track {
                    id: 2,
                    age: 1,
                    ..track.clone()
                }]
            } else {
                vec![]
            };
            let r = book.step(32, &[], &births, advance);
            let g = &r[0].example;
            assert_eq!(g.found, outcome == "birth");
            assert_eq!(g.seen, 6);
            assert_eq!(
                g.steps,
                vec![GapStep::new(Context::Open, false); if g.found { 31 } else { 32 }]
            );
        }
    }
    #[test]
    fn privileged_labels_are_requested_only_for_oracle() {
        for id in PARTICIPANTS
            .into_iter()
            .chain([Condition::Frozen, Condition::Oracle])
        {
            let mut called = false;
            let frames = OracleFrames::for_condition(id, || {
                called = true;
                vec![]
            });
            assert_eq!(called, id == Condition::Oracle);
            assert_eq!(frames.is_some(), called);
        }
    }
    #[test]
    fn oracle_confirms_only_a_surviving_object_inside_forecast() {
        let mut o = make_world(Scenario::Vanish, 1).objects.remove(0);
        o.x = 8.;
        o.y = 11.;
        o.vanish_at = None;
        assert!(confirms(&o, 10, [5., 7., 5.]));
        o.vanish_at = Some(10);
        assert!(!confirms(&o, 10, [5., 7., 5.]));
        o.vanish_at = None;
        o.y = 12.;
        assert!(!confirms(&o, 10, [5., 7., 5.]));
    }
    #[test]
    fn nursery_losses_wait_for_first_open_step() {
        for ep in 1..=10 {
            let seed = 80100 + ep;
            let mut live = make_world(Scenario::Nursery, seed);
            let mut training = live.clone();
            schedule_outdoor_losses(&mut training);
            let mut expected = vec![None; live.objects.len()];
            for step in 1..=80 {
                advance(&mut live);
                for (i, o) in live.objects.iter().enumerate() {
                    let at = 12
                        + jscompat::floor(
                            24. * jscompat::hash_unit(f64::from(seed) * 3. + i as f64),
                        ) as u32;
                    if step >= at && expected[i].is_none() && !hidden(o, live.wall) {
                        expected[i] = Some(step);
                    }
                }
            }
            assert_eq!(
                training
                    .objects
                    .iter()
                    .map(|o| o.vanish_at)
                    .collect::<Vec<_>>(),
                expected
            );
            assert!(live.objects.iter().all(|o| o.vanish_at.is_none()));
        }
    }
    #[test]
    fn missing_and_matched_tracks_use_visible_evidence_and_consecutive_runs() {
        let m = Models::default();
        let ep = Episode::with_models(
            Scenario::Continuity,
            3001,
            Policy::FirstOrder,
            Some(m.clone()),
            None,
        );
        let a = &ep.agent;
        let prior = &a.tracks[0];
        let mut obs = Observation {
            gaze: Gaze::new(prior.x, prior.y, 12.),
            grid: 12,
            cells: vec![Cell::default(); 144],
        };
        let empty = GapStep::new(Context::Open, true);
        let mut t = miss(prior, &obs, Some(&m));
        assert_eq!(t.reason, Reason::NoData);
        assert_eq!(t.gap_step, Some(empty));
        t = miss(&t, &obs, Some(&m));
        assert_eq!(t.reason, Reason::NoData);
        obs.gaze.x = prior.x - 12. + prior.radius / 2.;
        t = miss(&t, &obs, Some(&m));
        assert_eq!(t.empty_views, Some(0));
        obs.gaze.x = prior.x;
        for i in 1..=3 {
            t = miss(&t, &obs, Some(&m));
            assert_eq!(
                t.reason,
                if i == 3 {
                    Reason::Absence
                } else {
                    Reason::NoData
                }
            );
        }
        obs.gaze.x -= 5.;
        assert_eq!(
            miss(prior, &obs, Some(&m)).gap_step,
            Some(GapStep::new(Context::Open, false))
        );
        obs.gaze.x += 5.;
        obs.cells[8 * 12 + 9].lightness = 50.;
        assert_eq!(miss(prior, &obs, Some(&m)).gap_step, Some(empty));
        obs.cells[6 * 12 + 6].lightness = 50.;
        assert_eq!(
            miss(prior, &obs, Some(&m)).gap_step,
            Some(GapStep::new(Context::Hidden, false))
        );
        obs.cells[6 * 12 + 6].saturation = 50.;
        let conflict = miss(prior, &obs, Some(&m));
        assert_eq!(conflict.reason, Reason::Conflict);
        assert_eq!(conflict.evidence, prior.evidence);
        let mut t = prior.clone();
        t.forecast_input = Some([3., -3., 2., -1., 1.]);
        t.vx = 1.;
        t.vy = -2.;
        obs.gaze.x = 90.;
        let t = miss(&t, &obs, Some(&m));
        assert_eq!(t.forecast_input, Some([1., -2., 2., -1., 1.]));
        assert_eq!(
            miss(&t, &obs, Some(&m)).forecast_input,
            Some([0., 0., 2., -1., 1.])
        );
        let mut t = prior.clone();
        let blob = &a.blobs[prior.bound_blob.expect("initial track is bound")];
        obs.gaze = Gaze::new(t.x, t.y, 20.);
        for i in 1..=25 {
            t = crate::perception::matched(a, &t, blob, &obs);
            assert_eq!(t.seen_run, Some(i.min(20)));
        }
        assert_eq!(
            crate::perception::matched(a, &miss(&t, &obs, Some(&m)), blob, &obs).seen_run,
            Some(1)
        );
        obs.gaze.x = t.x + 15.;
        assert_eq!(
            crate::perception::matched(a, &t, blob, &obs).seen_run,
            Some(0)
        );
    }
}
