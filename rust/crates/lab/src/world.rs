use crate::task::object::{OUTDOOR_FIRST, OUTDOOR_LAST};
use crate::types::*;
use jscompat::{hash_unit, sin};

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct World {
    pub objects: Vec<Object>,
    pub wall: Wall,
    pub(crate) seed: u32,
    pub t: u32,
    pub(crate) light: f64,
    pub noise: f64,
    pub(crate) base_noise: f64,
    swap_at: Option<u32>,
    pub(crate) swapped: bool,
    pub body_gain: Option<[f64; 2]>,
}
pub fn hidden(o: &Object, w: Wall) -> bool {
    let dx = o.x - clamp(o.x, w.x, w.x + w.w);
    let dy = o.y - clamp(o.y, w.y, w.y + w.h);
    dx * dx + dy * dy <= o.r * o.r
}
pub fn contains(o: &Object, x: f64, y: f64) -> bool {
    let dx = x - o.x;
    let dy = y - o.y;
    match o.shape {
        Shape::Disk => dx * dx + dy * dy <= o.r * o.r,
        Shape::Square => dx.abs() <= o.r && dy.abs() <= o.r,
        Shape::Triangle => {
            let y0 = -o.r * 0.95;
            let y1 = o.r * 0.72;
            dy >= y0 && dy <= y1 && dx.abs() <= o.r * ((dy - y0) / (y1 - y0))
        }
    }
}
pub(crate) fn initial_gaze(sc: Scenario) -> Gaze {
    match sc {
        Scenario::Continuity => Gaze::new(16., 50., 20.),
        Scenario::Binding => Gaze::new(24., 51., 28.),
        Scenario::Conflict => Gaze::new(74., 50., 24.),
        Scenario::Vanish => Gaze::new(22., 32., 20.),
        Scenario::Nursery => Gaze::new(50., 48., 30.),
    }
}
pub fn make_world(sc: Scenario, seed: u32) -> World {
    use Shape::*;
    let mut draw = 0;
    let mut jitter = || {
        draw += 1;
        2. * hash_unit(f64::from(seed) + 104729. * f64::from(draw)) - 1.
    };
    let mut obj = |id, shape, hue, x, y, vx, vy, r| Object {
        id,
        shape,
        hue,
        r,
        vanish_at: None,
        x: x + 2. * jitter(),
        y: y + 2. * jitter(),
        vx: vx * (1. + 0.08 * jitter()) + 0.025 * jitter(),
        vy: vy * (1. + 0.08 * jitter()) + 0.025 * jitter(),
    };
    let (objects, [x, y, w, h], noise) = match sc {
        Scenario::Continuity => (
            vec![
                obj("A", Disk, 18., 16., 50., 1.65, 0., 4.),
                obj("B", Square, 208., 82., 30., -0.04, 0.22, 6.5),
            ],
            [40., 0., 20., 100.],
            7.,
        ),
        Scenario::Vanish => {
            let mut a = obj("A", Disk, 18., 22., 32., 0.35, 0.12, 4.);
            a.vanish_at = Some(20 + (4. * hash_unit(f64::from(seed))).floor() as u32);
            (
                vec![a, obj("B", Square, 208., 78., 72., -0.12, 0.08, 6.)],
                [60., 6., 12., 16.],
                7.,
            )
        }
        Scenario::Binding => (
            vec![
                obj("A", Disk, 16., 14., 38., 1.7, 0.35, 4.2),
                obj("B", Square, 206., 14., 64., 1.7, -0.35, 4.2),
            ],
            [70., 8., 12., 22.],
            4.,
        ),
        Scenario::Conflict => (
            vec![
                obj("A", Disk, 18., 30., 50., 0.08, 0.05, 3.4),
                obj("B", Triangle, 34., 40., 50., -0.05, 0.06, 3.4),
                obj("C", Square, 200., 78., 50., 0.02, -0.08, 8.),
            ],
            [8., 78., 16., 14.],
            5.,
        ),
        Scenario::Nursery => (
            vec![
                obj("A", Disk, 12., 22., 58., 0.85, -0.35, 4.2),
                obj("B", Square, 200., 72., 24., -0.55, 0.4, 5.),
                obj("C", Triangle, 128., 28., 22., 0.45, 0.55, 4.4),
            ],
            [40., 30., 18., 40.],
            8.,
        ),
    };
    World {
        objects,
        wall: Wall { x, y, w, h },
        seed,
        t: 0,
        light: 0.,
        noise,
        base_noise: noise,
        swap_at: (sc == Scenario::Binding).then_some(16),
        swapped: false,
        body_gain: None,
    }
}
pub fn advance(world: &mut World) {
    world.t += 1;
    world.light = sin(f64::from(world.t) * 0.08) * 20.;
    for o in &mut world.objects {
        if o.vanish_at.is_some_and(|t| world.t > t) {
            continue;
        }
        for (pos, v) in [(&mut o.x, &mut o.vx), (&mut o.y, &mut o.vy)] {
            *pos += *v;
            if *pos < o.r {
                *pos = o.r;
                *v = v.abs();
            } else if *pos > 100. - o.r {
                *pos = 100. - o.r;
                *v = -v.abs();
            }
        }
    }
    if Some(world.t) == world.swap_at {
        let (a, b) = world.objects.split_at_mut(1);
        std::mem::swap(&mut a[0].hue, &mut b[0].hue);
        world.swapped = true;
    }
}

pub fn schedule_outdoor_losses(world: &mut World) {
    let mut preview = world.clone();
    let deadlines: Vec<_> = (0..world.objects.len())
        .map(|i| {
            OUTDOOR_FIRST
                + jscompat::floor(
                    f64::from(OUTDOOR_LAST - OUTDOOR_FIRST + 1)
                        * hash_unit(f64::from(world.seed) * 3. + i as f64),
                ) as u32
        })
        .collect();
    for step in 1..=crate::genesis::TRAINING_STEPS {
        advance(&mut preview);
        for (i, obj) in world.objects.iter_mut().enumerate() {
            if obj.vanish_at.is_none()
                && step >= deadlines[i]
                && !hidden(&preview.objects[i], world.wall)
            {
                obj.vanish_at = Some(step);
            }
        }
    }
}
