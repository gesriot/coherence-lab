use crate::sensor::Observation;
use crate::types::*;
use jscompat::{PI, atan2, cos, sin};
pub(crate) fn classify(c: Cell) -> Expectation {
    if c.lightness < 15. {
        Expectation::Empty
    } else if c.saturation < 20. {
        Expectation::Occluder
    } else {
        Expectation::Surface
    }
}
pub(crate) fn locate(obs: &Observation, x: f64, y: f64) -> Option<Expectation> {
    let g = obs.gaze;
    let size = g.aperture * 2. / obs.grid as f64;
    let c = ((x - (g.x - g.aperture)) / size).floor();
    let r = ((y - (g.y - g.aperture)) / size).floor();
    if x < g.x - g.aperture
        || x > g.x + g.aperture
        || y < g.y - g.aperture
        || y > g.y + g.aperture
        || c < 0.
        || r < 0.
        || c >= obs.grid as f64
        || r >= obs.grid as f64
    {
        return None;
    }
    Some(classify(obs.cells[r as usize * obs.grid + c as usize]))
}
pub(crate) fn blobs(obs: &Observation) -> Vec<Blob> {
    let n = obs.grid;
    let g = obs.gaze;
    let size = 2. * g.aperture / n as f64;
    let (left, top) = (g.x - g.aperture, g.y - g.aperture);
    let mut seen = vec![false; n * n];
    let mut result = Vec::new();
    for i in 0..n * n {
        if seen[i] || classify(obs.cells[i]) != Expectation::Surface {
            continue;
        }
        let seed = obs.cells[i].hue;
        let mut stack = vec![i];
        seen[i] = true;
        let (mut sx, mut sy, mut sh, mut ch) = (0., 0., 0., 0.);
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (n, 0, n, 0);
        let mut area = 0;
        while let Some(j) = stack.pop() {
            let (row, col) = (j / n, j % n);
            sx += left + (col as f64 + 0.5) * size;
            sy += top + (row as f64 + 0.5) * size;
            let angle = obs.cells[j].hue * PI / 180.;
            sh += sin(angle);
            ch += cos(angle);
            xmin = xmin.min(col);
            xmax = xmax.max(col);
            ymin = ymin.min(row);
            ymax = ymax.max(row);
            area += 1;
            for nb in [
                j.checked_sub(n),
                (row + 1 < n).then_some(j + n),
                (col > 0).then(|| j - 1),
                (col + 1 < n).then_some(j + 1),
            ]
            .into_iter()
            .flatten()
            {
                let d = (seed - obs.cells[nb].hue).abs() % 360.;
                if !seen[nb]
                    && classify(obs.cells[nb]) == Expectation::Surface
                    && (if d > 180. { 360. - d } else { d }) < 38.
                {
                    seen[nb] = true;
                    stack.push(nb);
                }
            }
        }
        result.push(Blob {
            index: result.len(),
            cx: sx / area as f64,
            cy: sy / area as f64,
            hue: wrap(atan2(sh, ch) * 180. / PI),
            area,
            fill: area as f64 / ((xmax - xmin + 1) * (ymax - ymin + 1)) as f64,
        });
    }
    result
}
