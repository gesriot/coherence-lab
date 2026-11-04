use crate::task::sensorimotor as task;
use crate::{
    observation::classify,
    types::{Cell, Expectation},
};
use jscompat::{abs, floor, max};
use std::rc::Rc;
pub(crate) type Ratios = [Option<f64>; 2];
pub(crate) const RATIOS: [f64; 13] = [
    0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1., 1.01, 1.02, 1.05, 1.1, 1.15, 1.2,
];
pub(crate) struct SceneFrame {
    grid: usize,
    aperture: f64,
    command: [f64; 2],
    light: Vec<f64>,
}
pub(crate) type Pair = [Rc<SceneFrame>; 2];
pub(crate) fn frame(
    cells: &[Cell],
    grid: usize,
    aperture: f64,
    command: [f64; 2],
) -> Rc<SceneFrame> {
    let mut light: Vec<_> = cells.iter().map(|c| c.lightness).collect();
    for (i, &c) in cells.iter().enumerate() {
        if classify(c) != Expectation::Surface {
            continue;
        }
        let (row, col) = (i / grid, i % grid);
        for y in row.saturating_sub(1)..=(row + 1).min(grid - 1) {
            light[y * grid + col.saturating_sub(1)..y * grid + (col + 2).min(grid)].fill(f64::NAN);
        }
    }
    Rc::new(SceneFrame {
        grid,
        aperture,
        command,
        light,
    })
}
fn score([previous, current]: &Pair, dx: f64, dy: f64) -> Option<f64> {
    let n = previous.grid;
    let size = 2. * previous.aperture / n as f64;
    let scale = (2. * current.aperture / current.grid as f64) / size;
    let x0 = (dx + previous.aperture - current.aperture) / size + (scale - 1.) / 2.;
    let y0 = (dy + previous.aperture - current.aperture) / size + (scale - 1.) / 2.;
    let old = &previous.light;
    let mut sum = 0.;
    let mut count = 0;
    for row in 0..current.grid {
        let y = y0 + row as f64 * scale;
        let iy = floor(y);
        let fy = y - iy;
        if iy < 0. || iy + 1. >= n as f64 {
            continue;
        }
        for col in 0..current.grid {
            let x = x0 + col as f64 * scale;
            let ix = floor(x);
            let fx = x - ix;
            if ix < 0. || ix + 1. >= n as f64 {
                continue;
            }
            let k = iy as usize * n + ix as usize;
            let top = old[k] + fx * (old[k + 1] - old[k]);
            let bottom = old[k + n] + fx * (old[k + n + 1] - old[k + n]);
            let residual = current.light[row * current.grid + col] - (top + fy * (bottom - top));
            if residual.is_nan() {
                continue;
            }
            sum += residual * residual;
            count += 1;
        }
    }
    (count >= task::MIN_CELLS).then(|| sum / f64::from(count))
}
pub(crate) fn align(pair: &Pair, estimate: [f64; 2]) -> Ratios {
    let [dx, dy] = [0, 1].map(|a| estimate[a] * (pair[1].command[a] - pair[0].command[a]));
    [0, 1].map(|a| {
        if [dx, dy][a] == 0. {
            return None;
        }
        let mut best = f64::INFINITY;
        let mut worst = f64::NEG_INFINITY;
        let mut sum = 0.;
        let mut winner = 1.;
        for r in RATIOS {
            let score = if a == 0 {
                score(pair, dx * r, dy)
            } else {
                score(pair, dx, dy * r)
            }?;
            if score < best || (score == best && abs(r - 1.) < abs(winner - 1.)) {
                best = score;
                winner = r;
            }
            worst = max(worst, score);
            sum += score;
        }
        (best < worst && best <= task::MIN_TO_MEAN * (sum / RATIOS.len() as f64)).then_some(winner)
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use jscompat::sin;
    pub(crate) fn texture(command: [f64; 2], body: [f64; 2], n: usize) -> Rc<SceneFrame> {
        let cells = (0..n * n)
            .map(|i| {
                let x = command[0] * body[0] + (i % n) as f64 + 0.5 - n as f64 / 2.;
                let y = command[1] * body[1] + (i / n) as f64 + 0.5 - n as f64 / 2.;
                Cell {
                    lightness: 6. + 4. * sin(0.37 * x + 1.3) * sin(0.29 * y + 0.7),
                    ..Cell::default()
                }
            })
            .collect::<Vec<_>>();
        frame(&cells, n, n as f64 / 2., command)
    }
    #[test]
    fn informative_translation_recovers_gain_and_has_a_fixed_point() {
        let body = [1.1, 0.9];
        for axis in [0, 1] {
            let mut c = [0.; 2];
            c[axis] = 10.;
            let pair = [texture([0.; 2], body, 32), texture(c, body, 32)];
            assert_eq!(align(&pair, [1.; 2])[axis], Some(body[axis]));
            let r = align(&pair, body);
            assert_eq!(r[axis], Some(1.));
            assert_eq!(r[1 - axis], None);
        }
    }
    #[test]
    fn mask_excludes_nine_cells_and_keeps_wall_lightness() {
        let mut cells = [Cell {
            hue: 0.,
            saturation: 0.,
            lightness: 6.,
        }; 25];
        cells[12] = Cell {
            hue: 30.,
            saturation: 70.,
            lightness: 50.,
        };
        cells[0].lightness = 32.;
        let f = frame(&cells, 5, 5., [0.; 2]);
        assert_eq!(f.light[0], 32.);
        assert_eq!(
            f.light
                .iter()
                .enumerate()
                .filter(|(_, x)| x.is_nan())
                .map(|(i, _)| i)
                .collect::<Vec<_>>(),
            [6, 7, 8, 11, 12, 13, 16, 17, 18]
        );
    }
}
