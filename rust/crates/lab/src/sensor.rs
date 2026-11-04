use crate::types::*;
use crate::world::{World, contains, hidden};
use jscompat::{hue_noise, sin};
#[derive(Clone, Debug)]
pub(crate) struct Observation {
    pub(crate) gaze: Gaze,
    pub(crate) grid: usize,
    pub(crate) cells: Vec<Cell>,
}

fn sample(world: &World, x: f64, y: f64, i: usize) -> Cell {
    if (0.0..=100.).contains(&x) && (0.0..=100.).contains(&y) {
        let w = world.wall;
        if x >= w.x && x <= w.x + w.w && y >= w.y && y <= w.y + w.h {
            return Cell {
                lightness: if ((x + y) / 4.).floor() % 2. != 0. {
                    32.
                } else {
                    54.
                },
                ..Cell::default()
            };
        }
        for o in world.objects.iter().rev() {
            if o.vanish_at.is_some_and(|t| world.t >= t) || hidden(o, w) || !contains(o, x, y) {
                continue;
            }
            return Cell {
                hue: wrap(
                    o.hue
                        + world.light
                        + hue_noise(world.seed.into(), world.t.into(), i as f64) * world.noise,
                ),
                saturation: 70.,
                lightness: 50.,
            };
        }
    }
    let lightness = if world.body_gain.is_some() {
        6. + 4. * sin(0.37 * x + 1.3) * sin(0.29 * y + 0.7)
    } else {
        6.
    };
    Cell {
        lightness,
        ..Cell::default()
    }
}
pub(crate) fn observe(world: &World, gaze: Gaze, estimate: Option<[f64; 2]>) -> Observation {
    let grid = 12;
    let size = gaze.aperture * 2. / grid as f64;
    let gain = world.body_gain.unwrap_or([1., 1.]);
    let estimate = estimate.unwrap_or([1., 1.]);
    let left = gain[0] / estimate[0] * gaze.x - gaze.aperture;
    let top = gain[1] / estimate[1] * gaze.y - gaze.aperture;
    let cells = (0..grid * grid)
        .map(|i| {
            sample(
                world,
                left + ((i % grid) as f64 + 0.5) * size,
                top + ((i / grid) as f64 + 0.5) * size,
                i,
            )
        })
        .collect();
    Observation { gaze, grid, cells }
}
