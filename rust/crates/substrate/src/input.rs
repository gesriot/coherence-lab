use jscompat::{PI, cos, floor, hash_unit, sin};
use lab::{
    Occlusion, Scenario, World, advance, contains, hidden, make_world, sample,
    schedule_outdoor_losses,
};
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct WorldSettings {
    pub random: bool,
    pub columns: Option<u8>,
}
pub(crate) fn world(s: Scenario, seed: u32, setting: WorldSettings) -> World {
    let mut w = make_world(s, seed);
    if s == Scenario::Nursery {
        schedule_outdoor_losses(&mut w);
    }
    if s == Scenario::Continuity {
        if setting.random {
            let u = |k| hash_unit(7919. * f64::from(seed) + 31. * f64::from(k) + 5.);
            let right = u(1) < 0.5;
            let speed = 1.2 + 0.9 * u(2);
            let a = &mut w.objects[0];
            a.y = 12. + 76. * u(3);
            a.x = (if right { 8. } else { 76. }) + 16. * u(4);
            a.vx = if right { speed } else { -speed };
            a.vy = 0.05 * (2. * u(5) - 1.);
        }
        if let Some(columns @ (1 | 2)) = setting.columns {
            w.wall.x = 44.;
            w.wall.w = 6. * f64::from(columns);
        }
    }
    w
}
pub(crate) type Vector = [f64; 2];
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CopyMode {
    Executed,
    Zero,
    Shuffled,
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Gain {
    Fixed,
    Episodic,
}
#[derive(Clone, Copy, Debug)]
pub struct Motion {
    pub gain: Gain,
    pub copy: CopyMode,
    pub random: bool,
}
pub(crate) struct GazePath {
    seed: u32,
    gaze: Vector,
    distance: f64,
    t: u32,
}
impl GazePath {
    pub(crate) fn new(seed: u32, mode: Gain, grid: usize) -> Self {
        let u = |k| hash_unit(104729. * f64::from(seed) + 7. * f64::from(k) + 3.);
        let gain = if mode == Gain::Fixed {
            1.2
        } else {
            0.7 + 0.7 * u(0)
        };
        Self {
            seed,
            gaze: [35. + 30. * u(1), 35. + 30. * u(2)],
            distance: gain * 0.7 * (50. / grid as f64),
            t: 0,
        }
    }
    fn next(&mut self) -> (Vector, Vector) {
        let current = self.gaze;
        let command = [0, 1].map(|axis| {
            let k = 11 + axis as u32 + 3 * self.t;
            let c =
                floor(3. * hash_unit(104729. * f64::from(self.seed) + 7. * f64::from(k) + 3.)) - 1.;
            let next = current[axis] + c * self.distance;
            if !(25. ..=75.).contains(&next) { -c } else { c }
        });
        for (i, c) in command.iter().enumerate() {
            self.gaze[i] += c * self.distance;
        }
        self.t += 1;
        (current, command)
    }
}
pub(crate) fn center(i: usize, grid: usize) -> Vector {
    let size = 100. / grid as f64;
    [
        ((i % grid) as f64 + 0.5) * size,
        ((i / grid) as f64 + 0.5) * size,
    ]
}
pub(crate) fn frame(w: &World, grid: usize, occ: Occlusion, gaze: Option<Vector>) -> Vec<f64> {
    let mut out = Vec::with_capacity(grid * grid * 3);
    for i in 0..grid * grid {
        let [x, y] = if let Some([x, y]) = gaze {
            let half = grid as f64 / 2.;
            let size = 50. / grid as f64;
            [
                x + ((i % grid) as f64 + 0.5 - half) * size,
                y + ((i / grid) as f64 + 0.5 - half) * size,
            ]
        } else {
            center(i, grid)
        };
        let c = sample(w, x, y, i, occ);
        let h = c.hue * PI / 180.;
        out.extend([
            c.lightness / 100.,
            c.saturation / 100. * cos(h),
            c.saturation / 100. * sin(h),
        ]);
    }
    out
}
pub(crate) struct Observation {
    pub(crate) x: Vec<f64>,
    pub(crate) copy: Option<Vector>,
}
pub(crate) struct MovingScene {
    pub(crate) world: World,
    path: GazePath,
    alt: Option<GazePath>,
    mode: CopyMode,
    grid: usize,
}
impl MovingScene {
    pub(crate) fn new(sc: Scenario, seed: u32, m: Motion, grid: usize) -> Self {
        let mut w = world(
            sc,
            seed,
            WorldSettings {
                random: m.random,
                columns: None,
            },
        );
        w.body_gain = Some([1.; 2]);
        Self {
            world: w,
            path: GazePath::new(seed, m.gain, grid),
            alt: (m.copy == CopyMode::Shuffled).then(|| GazePath::new(seed + 7777, m.gain, grid)),
            mode: m.copy,
            grid,
        }
    }
    pub(crate) fn read(&mut self) -> (Vector, Vector, Observation) {
        let (g, c) = self.path.next();
        let copy = if let Some(path) = &mut self.alt {
            path.next().1
        } else if self.mode == CopyMode::Zero {
            [0.; 2]
        } else {
            c
        };
        (
            g,
            c,
            Observation {
                x: frame(&self.world, self.grid, Occlusion::PerCell, Some(g)),
                copy: Some(copy),
            },
        )
    }
}
pub(crate) fn movie(
    sc: Scenario,
    seed: u32,
    grid: usize,
    steps: usize,
    occ: Occlusion,
    settings: WorldSettings,
    motion: Option<Motion>,
) -> impl Iterator<Item = Observation> {
    let mut moving = motion.map(|m| MovingScene::new(sc, seed, m, grid));
    let mut fixed = motion.is_none().then(|| world(sc, seed, settings));
    (0..=steps).map(move |t| {
        if let Some(m) = &mut moving {
            if t > 0 {
                advance(&mut m.world);
            }
            m.read().2
        } else {
            let w = fixed.as_mut().expect("stationary movie owns a world");
            if t > 0 {
                advance(w);
            }
            Observation {
                x: frame(w, grid, occ, None),
                copy: None,
            }
        }
    })
}
pub(crate) fn in_wall(w: &World, x: f64, y: f64) -> bool {
    let r = &w.wall;
    x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h
}
pub(crate) fn occupancy(w: &World, grid: usize) -> Vec<f64> {
    (0..grid * grid)
        .map(|i| {
            let [x, y] = center(i, grid);
            f64::from(
                w.objects
                    .iter()
                    .any(|o| o.vanish_at.is_none_or(|at| w.t < at) && contains(o, x, y)),
            )
        })
        .collect()
}
pub(crate) fn visible(w: &World, grid: usize) -> Vec<usize> {
    let a = &w.objects[0];
    if a.vanish_at.is_some_and(|at| w.t >= at) {
        return vec![];
    }
    (0..grid * grid)
        .filter(|&i| {
            let [x, y] = center(i, grid);
            !in_wall(w, x, y) && contains(a, x, y)
        })
        .collect()
}
#[derive(Clone, Debug)]
pub(crate) struct Event {
    pub(crate) wall: bool,
    pub(crate) time: usize,
    pub(crate) length: usize,
    pub(crate) cells: Vec<usize>,
}
pub(crate) struct Detector {
    pub(crate) age: i32,
    touched: bool,
}
impl Default for Detector {
    fn default() -> Self {
        Self {
            age: -1,
            touched: false,
        }
    }
}
impl Detector {
    pub(crate) fn read(&mut self, w: &World, grid: usize) -> (Option<Event>, Vec<usize>) {
        let cells = visible(w, grid);
        let a = &w.objects[0];
        let event = (!cells.is_empty() && self.age >= 0).then(|| Event {
            wall: self.touched,
            time: w.t as usize,
            length: (self.age + 1) as usize,
            cells: cells.clone(),
        });
        self.age = if cells.is_empty() && a.vanish_at.is_none_or(|at| w.t < at) {
            self.age + 1
        } else {
            -1
        };
        if self.age < 0 {
            self.touched = false;
        } else if hidden(a, w.wall) {
            self.touched = true;
        }
        (event, cells)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    #[test]
    fn fixed_channels_encode_light_wall_and_circular_object_color() {
        let mut w = make_world(Scenario::Continuity, 23);
        w.objects[0].x = 2.5 * 6.25;
        w.objects[0].y = 8.5 * 6.25;
        let x = frame(&w, 16, Occlusion::WholeObject, None);
        assert_eq!(x[..3], [0.06, 0., 0.]);
        assert_eq!(x[21..24], [0.54, 0., 0.]);
        let j = 3 * (8 * 16 + 2);
        assert_eq!(x[j], 0.5);
        assert!(jscompat::abs(jscompat::hypot(x[j + 1], x[j + 2]) - 0.7) < 1e-14);
        assert!(
            jscompat::abs(jscompat::atan2(x[j + 2], x[j + 1]) * 180. / PI - w.objects[0].hue)
                <= w.noise
        );
        w.objects[0].hue += 90.;
        let y = frame(&w, 16, Occlusion::WholeObject, None);
        assert!(
            jscompat::abs(y[j + 1] + x[j + 2]) < 1e-14
                && jscompat::abs(y[j + 2] - x[j + 1]) < 1e-14
        );
    }
    #[test]
    fn exit_mask_uses_first_partially_exposed_frame_and_grid_gaps() {
        let mut w = make_world(Scenario::Continuity, 17);
        w.objects[0].r = 6.;
        w.objects[0].y = 8.5 * 6.25;
        let mut d = Detector::default();
        for (t, x) in [33., 44., 47., 50., 53., 56., 60., 65.]
            .into_iter()
            .enumerate()
        {
            w.objects[0].x = x;
            let event = d.read(&w, 16).0;
            assert_eq!(
                d.age,
                if (1..=5).contains(&t) {
                    t as i32 - 1
                } else {
                    -1
                }
            );
            assert_eq!(
                event.map(|e| e.cells),
                if t == 6 { Some(vec![138]) } else { None }
            );
        }
        w.objects[0].vanish_at = Some(0);
        d.age = 5;
        assert!(d.read(&w, 16).0.is_none());
        let mut w = make_world(Scenario::Continuity, 19);
        w.objects[0].x = 12.5;
        w.objects[0].y = 12.5;
        w.objects[0].r = 4.;
        d.read(&w, 16);
        assert_eq!(d.age, 0);
        w.objects[0].x += 3.125;
        assert!(!d.read(&w, 16).0.expect("grid gap closes").wall);
    }
    #[test]
    fn stationary_sensor_and_partial_occlusion() {
        let mut h = Sha256::new();
        for o in movie(
            Scenario::Continuity,
            41,
            16,
            72,
            Occlusion::WholeObject,
            WorldSettings::default(),
            None,
        ) {
            for x in o.x {
                h.update(x.to_le_bytes());
            }
        }
        assert_eq!(
            format!("{:x}", h.finalize()),
            "99c5264c857356a642d44d1e89f394e8fb3dcb16ab55a7decee8a85bb49f42e4"
        );
        let mut w = make_world(Scenario::Continuity, 41);
        let a = &mut w.objects[0];
        a.x = 40.;
        a.y = 8.5 * 6.25;
        a.r = 6.;
        let whole = frame(&w, 16, Occlusion::WholeObject, None);
        let local = frame(&w, 16, Occlusion::PerCell, None);
        let occ = occupancy(&w, 16);
        let i = 8 * 16 + 5;
        assert_eq!(whole[3 * i], 0.06);
        assert_eq!(local[3 * i], 0.5);
        assert_eq!(whole[3 * i + 3..3 * i + 6], local[3 * i + 3..3 * i + 6]);
        assert_eq!(&occ[i..i + 2], &[1., 1.]);
        w.objects[0].vanish_at = Some(0);
        assert_eq!(occupancy(&w, 16)[i + 1], 0.);
    }
    #[test]
    fn seeded_paths_and_wall_returns() {
        for seed in 1..=40 {
            for columns in [1, 2, 4] {
                let s = WorldSettings {
                    random: true,
                    columns: Some(columns),
                };
                let w = world(Scenario::Continuity, seed, s);
                let a = &w.objects[0];
                let u = |k| hash_unit(7919. * f64::from(seed) + 31. * f64::from(k) + 5.);
                assert_eq!(a.x, (if u(1) < 0.5 { 8. } else { 76. }) + 16. * u(4));
                assert_eq!(
                    a.vx,
                    (if u(1) < 0.5 { 1. } else { -1. }) * (1.2 + 0.9 * u(2))
                );
                assert_eq!(a.y, 12. + 76. * u(3));
                assert_eq!(a.vy, 0.05 * (2. * u(5) - 1.));
                assert_eq!(
                    (0..16)
                        .filter(|&i| in_wall(&w, (i as f64 + 0.5) * 6.25, 50.))
                        .count(),
                    columns as usize
                );
                for sc in [
                    Scenario::Vanish,
                    Scenario::Nursery,
                    Scenario::Binding,
                    Scenario::Conflict,
                ] {
                    assert_eq!(
                        world(sc, seed, s),
                        world(sc, seed, WorldSettings::default())
                    );
                }
            }
        }
        let mut w = world(
            Scenario::Continuity,
            1,
            WorldSettings {
                random: false,
                columns: Some(1),
            },
        );
        let mut det = Detector::default();
        for (t, (x, y)) in [
            (20., 50.),
            (25., 50.),
            (28.125, 50.),
            (46.875, 46.875),
            (47.5, 46.875),
            (53.125, 46.875),
            (46.875, 46.875),
            (53.125, 46.875),
            (53.125, 46.875),
        ]
        .into_iter()
        .enumerate()
        {
            w.t = t as u32;
            w.objects[0].x = x;
            w.objects[0].y = y;
            if t == 7 {
                w.objects[0].vanish_at = Some(7);
            }
            let e = det.read(&w, 16).0;
            match t {
                2 | 5 => {
                    let e = e.expect("trajectory has a visible return");
                    assert_eq!(e.wall, t == 5);
                    assert_eq!(e.time, t);
                    assert_eq!(e.length, if t == 2 { 1 } else { 2 });
                    assert_eq!(e.cells, if t == 2 { vec![116, 132] } else { vec![120] });
                }
                _ => assert!(e.is_none()),
            }
        }
    }
    #[test]
    fn gaze_reflects_and_copy_controls_preserve_pixels() {
        let mut reflections = 0;
        for mode in [Gain::Fixed, Gain::Episodic] {
            for seed in 1..=20 {
                let mut path = GazePath::new(seed, mode, 16);
                let u = |k: u32| hash_unit(104729. * f64::from(seed) + 7. * f64::from(k) + 3.);
                let distance = if mode == Gain::Fixed {
                    1.2
                } else {
                    0.7 + 0.7 * u(0)
                } * 0.7
                    * 3.125;
                let mut position = [35. + 30. * u(1), 35. + 30. * u(2)];
                for t in 0..140 {
                    let (g, c) = path.next();
                    assert_eq!(g, position);
                    for a in 0..2 {
                        let raw = floor(3. * u(11 + 3 * t + a as u32)) - 1.;
                        let flip = !(25. ..=75.).contains(&(g[a] + raw * distance));
                        reflections += usize::from(flip);
                        assert_eq!(c[a], if flip { -raw } else { raw });
                        position[a] += c[a] * distance;
                    }
                }
            }
        }
        assert!(reflections > 0);
        let make = |copy, seed| {
            MovingScene::new(
                Scenario::Continuity,
                seed,
                Motion {
                    copy,
                    gain: Gain::Episodic,
                    random: false,
                },
                16,
            )
        };
        let mut sims = [
            make(CopyMode::Executed, 23),
            make(CopyMode::Zero, 23),
            make(CopyMode::Shuffled, 23),
            make(CopyMode::Executed, 7800),
        ];
        for _ in 0..=80 {
            let [a, b, c, d] = sims.each_mut().map(|s| {
                let v = s.read();
                advance(&mut s.world);
                v
            });
            assert_eq!(a.2.x, b.2.x);
            assert_eq!(a.2.x, c.2.x);
            assert_eq!(a.2.copy, Some(a.1));
            assert_eq!(b.2.copy, Some([0.; 2]));
            assert_eq!(c.2.copy, d.2.copy);
        }
        let mut w = make_world(Scenario::Continuity, 31);
        w.objects.clear();
        w.body_gain = Some([1.; 2]);
        let a = frame(&w, 16, Occlusion::PerCell, Some([44., 41.]));
        let b = frame(&w, 16, Occlusion::PerCell, Some([47.125, 41.]));
        for i in 0..256 {
            if i % 16 < 15 {
                assert_eq!(b[3 * i..3 * i + 3], a[3 * i + 3..3 * i + 6]);
            }
        }
    }
}
