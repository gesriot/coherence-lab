use crate::{
    Error,
    element::{Element, Step},
    input::{Vector, WorldSettings, center, frame, in_wall, occupancy, visible, world},
    probes::{Rows, probe},
};
use jscompat::{abs, floor, hypot, sign};
use lab::{Object, Occlusion, Scenario, World, advance, contains, hidden};
pub(crate) const OFFSETS: [i32; 9] = [-6, -5, -4, -3, -2, -1, 0, 1, 2];
pub(crate) struct Predictor<'a> {
    m: &'a Element,
    prev: Vec<f64>,
    out: Step,
}
impl<'a> Predictor<'a> {
    pub(crate) fn prediction(&self) -> &[f64] {
        &self.out.prediction
    }
    pub(crate) fn new(m: &'a Element) -> Self {
        Self {
            m,
            prev: m.blank(),
            out: m.buffers(),
        }
    }
    pub(crate) fn step(&mut self, x: &[f64], copy: Option<Vector>) -> Result<&Step, Error> {
        std::mem::swap(&mut self.prev, &mut self.out.state);
        self.m.forward(&self.prev, x, &mut self.out, copy)?;
        Ok(&self.out)
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Ratio {
    pub(crate) sum: f64,
    pub(crate) copy: f64,
}
impl Ratio {
    pub(crate) fn add(&mut self, s: f64, c: f64) {
        self.sum += s;
        self.copy += c;
    }
    pub(crate) fn value(self) -> f64 {
        self.sum / self.copy
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Average {
    pub(crate) sum: f64,
    pub(crate) n: usize,
}
impl Average {
    pub(crate) fn add(&mut self, x: f64) {
        self.sum += x;
        self.n += 1;
    }
    pub(crate) fn value(self) -> f64 {
        self.sum / self.n as f64
    }
}
#[derive(Default)]
struct Totals {
    all: Ratio,
    changed: Ratio,
    exit: Ratio,
    exit_cells: usize,
    wall: Ratio,
    events: usize,
    wall_events: usize,
    ghost: Average,
    hint: [Average; 2],
    curve: [Average; 9],
}
#[derive(Clone, Debug, Default)]
pub(crate) struct Scores {
    pub(crate) all: f64,
    pub(crate) changed: f64,
    pub(crate) exit: f64,
    pub(crate) wall: f64,
    pub(crate) events: usize,
    pub(crate) wall_events: usize,
    pub(crate) ghost: f64,
    pub(crate) curve: [f64; 9],
    pub(crate) hint: [f64; 2],
    pub(crate) probes: [[f64; 2]; 3],
}
pub(crate) fn saturation(x: &[f64], i: usize) -> f64 {
    hypot(x[3 * i + 1], x[3 * i + 2])
}
pub(crate) fn near(o: &Object, x: f64, y: f64) -> bool {
    let dx = x - o.x;
    let dy = y - o.y;
    let r = o.r + 4.;
    dx * dx + dy * dy <= r * r
}
fn wall_cells(w: &World, grid: usize) -> Vec<usize> {
    let r = &w.wall;
    let size = 100. / grid as f64;
    (0..grid * grid)
        .filter(|&i| {
            let x = (i % grid) as f64 * size;
            let y = (i / grid) as f64 * size;
            x < r.x + r.w && x + size > r.x && y < r.y + r.h && y + size > r.y
        })
        .collect()
}
struct Replay<'a> {
    m: Option<&'a Element>,
    grid: usize,
    occ: Occlusion,
    steps: usize,
}
impl Replay<'_> {
    fn episode(
        &self,
        sc: Scenario,
        seed: u32,
        rows: &mut [Rows; 3],
        mut totals: Option<&mut Totals>,
    ) -> Result<(), Error> {
        let grid = self.grid;
        let mut w = world(sc, seed, WorldSettings::default());
        let wall = wall_cells(&w, grid);
        let mut predictor = self.m.map(Predictor::new);
        let depth = self.m.map_or(3, |m| m.k);
        let mut prev = vec![];
        let mut predictions = vec![vec![]];
        let mut exits = vec![];
        let mut was_hidden = false;
        let mut last_seen: Option<Object> = None;
        let mut age = -1;
        let mut crossed = false;
        for t in 0..=self.steps {
            let x = frame(&w, grid, self.occ, None);
            let a = &w.objects[0];
            let alive = a.vanish_at.is_none_or(|at| w.t < at);
            let local = self.occ == Occlusion::PerCell;
            let cells = if local { visible(&w, grid) } else { vec![] };
            let exit = local && !cells.is_empty() && age >= 0;
            if local {
                age = if alive && cells.is_empty() {
                    age + 1
                } else {
                    -1
                };
                if age >= 0 && hidden(a, w.wall) {
                    crossed = true;
                }
            }
            let invisible = if local { age >= 0 } else { hidden(a, w.wall) };
            if let Some(v) = totals.as_deref_mut()
                && t > 0
            {
                let emergence = if local {
                    exit
                } else {
                    sc == Scenario::Continuity && was_hidden && !invisible
                };
                let ghost = a.vanish_at.is_some_and(|at| w.t > at && w.t - at <= 10);
                let mut exit_cells = vec![];
                let pred = &predictions[t];
                for i in 0..grid * grid {
                    let (mut squared, mut copy, mut absolute, mut changed) = (0., 0., 0., false);
                    for d in 0..3 {
                        let j = 3 * i + d;
                        let e: f64 = pred[j] - x[j];
                        let delta: f64 = prev[j] - x[j];
                        squared += e * e;
                        copy += delta * delta;
                        absolute += abs(e);
                        changed |= abs(delta) > 0.05;
                    }
                    v.all.add(squared, copy);
                    if changed {
                        v.changed.add(squared, copy);
                    }
                    let [cx, cy] = center(i, grid);
                    if emergence
                        && if local {
                            cells.binary_search(&i).is_ok()
                        } else {
                            contains(a, cx, cy)
                        }
                    {
                        v.exit.add(squared, copy);
                        v.exit_cells += 1;
                        if crossed {
                            v.wall.add(squared, copy);
                        }
                        exit_cells.push(i);
                    }
                    if ghost && last_seen.as_ref().is_some_and(|o| near(o, cx, cy)) {
                        v.ghost.sum += absolute;
                        v.ghost.n += 3;
                    }
                }
                if exit {
                    v.events += 1;
                    v.wall_events += usize::from(crossed);
                }
                if !exit_cells.is_empty() {
                    exits.push((t, exit_cells));
                }
            }
            let state = if let Some(p) = &mut predictor {
                let out = p.step(&x, None)?;
                if let Some(v) = totals.as_deref_mut()
                    && local
                    && invisible
                    && !out.occupancy.is_empty()
                {
                    for (i, label) in occupancy(&w, grid).into_iter().enumerate() {
                        let [cx, cy] = center(i, grid);
                        if contains(a, cx, cy) {
                            v.hint[0].add(out.occupancy[i]);
                        } else if label == 0. && in_wall(&w, cx, cy) {
                            v.hint[1].add(out.occupancy[i]);
                        }
                    }
                }
                if totals.is_some() && t < self.steps {
                    predictions.push(out.prediction.clone());
                }
                &out.state
            } else {
                if totals.is_some() && t < self.steps {
                    predictions.push(x.clone());
                }
                &x
            };
            if sc == Scenario::Continuity {
                if invisible {
                    let features: Vec<_> = wall
                        .iter()
                        .flat_map(|&i| state[i * depth..(i + 1) * depth].iter().copied())
                        .collect();
                    if age >= 4 {
                        rows[2].x.push(features.clone());
                        rows[2].y.push(a.x);
                    }
                    rows[1].x.push(features);
                    rows[1].y.push(a.x);
                } else {
                    let cell = 100. / grid as f64;
                    let i = floor(a.y / cell) as usize * grid + floor(a.x / cell) as usize;
                    rows[0].x.push(state[i * depth..(i + 1) * depth].to_vec());
                    rows[0].y.push(sign(a.vx));
                }
            }
            if !invisible && alive {
                last_seen = Some(a.clone());
            }
            if age < 0 {
                crossed = false;
            }
            was_hidden = invisible;
            prev = x;
            if t < self.steps {
                advance(&mut w);
            }
        }
        if let Some(v) = totals {
            for (time, cells) in exits {
                for (j, &offset) in OFFSETS.iter().enumerate() {
                    let t = time as i32 + offset;
                    if t > 0
                        && let Some(p) = predictions.get(t as usize)
                    {
                        for &i in &cells {
                            v.curve[j].add(saturation(p, i));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
pub(crate) fn evaluate(
    m: Option<&Element>,
    grid: usize,
    occ: Occlusion,
    steps: usize,
    training: &[u32],
    test: &[u32],
) -> Result<Scores, Error> {
    if training.iter().any(|s| test.contains(s)) {
        return Err(Error::SeedOverlap);
    }
    let replay = Replay {
        m,
        grid,
        occ,
        steps,
    };
    let (mut tr, mut te): ([Rows; 3], [Rows; 3]) = Default::default();
    let mut v = Totals::default();
    for &s in training {
        replay.episode(Scenario::Continuity, s, &mut tr, None)?;
    }
    for &s in test {
        for sc in [Scenario::Continuity, Scenario::Vanish] {
            replay.episode(sc, s, &mut te, Some(&mut v))?;
        }
    }
    // Coarse sensors may have no cell on the first exposed frame. The
    // empty exit ratio remains NaN, while the other metrics are measurable.
    if v.all.copy == 0. || v.changed.copy == 0. || (v.exit_cells > 0 && v.exit.copy == 0.) {
        return Err(Error::NoBaseline);
    }
    let mut probes = [[0.; 2]; 3];
    let count = if occ == Occlusion::WholeObject { 2 } else { 3 };
    for i in 0..count {
        probes[i] = probe(&tr[i], &te[i], i == 0)?;
    }
    Ok(Scores {
        all: v.all.value(),
        changed: v.changed.value(),
        exit: v.exit.value(),
        wall: v.wall.value(),
        events: v.events,
        wall_events: v.wall_events,
        ghost: v.ghost.value(),
        hint: v.hint.map(Average::value),
        curve: v.curve.map(Average::value),
        probes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::Kind;
    #[test]
    fn copy_forecasts_and_disjoint_frozen_probes() -> Result<(), Error> {
        let copy = evaluate(
            None,
            16,
            Occlusion::WholeObject,
            72,
            &[4111, 4112],
            &[3111, 3112],
        )?;
        assert_eq!(
            [copy.all, copy.changed, copy.exit, copy.ghost],
            [1., 1., 1., 0.]
        );
        assert_eq!(&copy.curve[..7], &[0.; 7]);
        assert!(abs(copy.curve[7] - 0.7) < 1e-12);
        for (kind, occ) in [
            (Kind::Plain, Occlusion::WholeObject),
            (Kind::Hinted, Occlusion::PerCell),
        ] {
            let m = Element::new(4, 16, 3, true, kind, false);
            let saved = m.weights.clone();
            let x = frame(
                &world(Scenario::Continuity, 8101, WorldSettings::default()),
                16,
                occ,
                None,
            );
            let trajectory = || {
                let mut p = Predictor::new(&m);
                p.step(&x, None)?;
                Ok::<_, Error>(p.step(&x, None)?.state.clone())
            };
            let before = trajectory()?;
            assert!(matches!(
                evaluate(Some(&m), 16, occ, 72, &[8101], &[8101]),
                Err(Error::SeedOverlap)
            ));
            let s = evaluate(Some(&m), 16, occ, 72, &[8201, 8202], &[8101, 8102])?;
            assert!(s.probes.iter().flatten().all(|x| x.is_finite()));
            if kind == Kind::Hinted {
                assert!(s.hint.iter().all(|x| x.is_finite()));
            }
            assert_eq!(saved, m.weights);
            assert_eq!(trajectory()?, before);
        }
        Ok(())
    }
}
