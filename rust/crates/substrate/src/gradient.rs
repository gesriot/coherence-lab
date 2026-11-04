use crate::{
    element::{Element, Kind, Step},
    matrix::{product, transpose},
};
pub(crate) struct Gradient {
    pub(crate) grad: Vec<f64>,
    ds: Vec<f64>,
    df: Vec<f64>,
    dz: Vec<f64>,
    retained: Vec<f64>,
    ft: Vec<f64>,
    wt: Vec<f64>,
    zt: Vec<f64>,
}
impl Gradient {
    pub(crate) fn new(m: &Element) -> Self {
        let (n, k, f) = (m.n, m.k, m.f);
        Self {
            grad: vec![0.; m.weights.len()],
            ds: vec![0.; n * k],
            df: vec![0.; n * f],
            dz: vec![0.; n * k],
            retained: vec![0.; n * k],
            ft: vec![0.; n * f],
            wt: vec![0.; f * k],
            zt: vec![0.; f * k],
        }
    }
    pub(crate) fn backward(
        &mut self,
        m: &Element,
        tape: &[Step],
        targets: &[Vec<f64>],
        labels: &[Vec<f64>],
        maps: &[Vec<f64>],
    ) -> f64 {
        let (n, k, f) = (m.n, m.k, m.f);
        let norm = 1. / (targets.len() * n * 3) as f64;
        let mut loss = 0.;
        let w = &m.weights;
        let grad = &mut self.grad;
        let ds = &mut self.ds;
        grad.fill(0.);
        ds.fill(0.);
        transpose(&w[..m.bias], &mut self.wt, f, k);
        if m.kind != Kind::Plain {
            transpose(&w[m.gate..m.gate_bias], &mut self.zt, f, k);
        }
        for t in (0..targets.len()).rev() {
            let s = &tape[t];
            let target = &targets[t];
            for i in 0..n {
                let weight = maps.get(t).map_or(1., |v| v[i]);
                let mut residual = [0.; 3];
                for (d, e) in residual.iter_mut().enumerate() {
                    let j = 3 * i + d;
                    let delta = s.prediction[j] - target[j];
                    loss += weight * (delta * delta);
                    *e = 2. * norm * delta * weight;
                    grad[m.end + d] += *e;
                }
                let mut hint_error = 0.;
                if m.kind == Kind::Hinted {
                    let e = s.occupancy[i] - labels[t][i];
                    loss += 3. * e * e;
                    hint_error = 6. * norm * e;
                    grad[m.hint + k] += hint_error;
                }
                for h in 0..k {
                    let j = i * k + h;
                    let state = s.state[j];
                    let mut v = ds[j];
                    for (d, &e) in residual.iter().enumerate() {
                        let ix = m.readout + 3 * h + d;
                        grad[ix] += state * e;
                        v += w[ix] * e;
                    }
                    if m.kind == Kind::Hinted {
                        grad[m.hint + h] += hint_error * state;
                        v += hint_error * w[m.hint + h];
                    }
                    ds[j] = if m.kind != Kind::Plain {
                        let z = s.gate[j];
                        let c = s.candidate[j];
                        self.dz[j] = v * (c - s.features[i * f + h]) * z * (1. - z);
                        grad[m.gate_bias + h] += self.dz[j];
                        self.retained[j] = v * (1. - z);
                        v * z * (1. - c * c)
                    } else {
                        v * (1. - state * state)
                    };
                    grad[m.bias + h] += ds[j];
                }
            }
            transpose(&s.features, &mut self.ft, n, f);
            product(&self.ft, ds, &mut grad[..m.bias], n, k, true);
            if m.kind != Kind::Plain {
                product(
                    &self.ft,
                    &self.dz,
                    &mut grad[m.gate..m.gate_bias],
                    n,
                    k,
                    true,
                );
            }
            if t > 0 {
                product(ds, &self.wt, &mut self.df, k, f, false);
                if m.kind != Kind::Plain {
                    product(&self.dz, &self.zt, &mut self.df, k, f, true);
                }
                scatter(m, &self.df, ds);
                if m.kind != Kind::Plain {
                    for (d, r) in ds.iter_mut().zip(&self.retained) {
                        *d += r;
                    }
                }
            }
        }
        loss * norm
    }
}
fn scatter(m: &Element, df: &[f64], ds: &mut [f64]) {
    let (k, f) = (m.k, m.f);
    for (dst, row) in ds.chunks_exact_mut(k).zip(df.chunks_exact(f)) {
        dst.copy_from_slice(&row[..k]);
    }
    for (i, row) in df.chunks_exact(f).enumerate() {
        for (j, &other) in m.neighbors[i].iter().enumerate() {
            if other == m.n {
                continue;
            }
            let index = if j < 4 { j } else { j + 1 };
            let x = (index % 3) as f64 - 1.;
            let y = (index / 3) as f64 - 1.;
            let sx = x * (if y == 0. { 2. } else { 1. }) / 8.;
            let sy = y * (if x == 0. { 2. } else { 1. }) / 8.;
            for (h, dst) in ds[other * k..(other + 1) * k].iter_mut().enumerate() {
                let mut v = m.inverse[i] * row[k + h];
                if m.directed {
                    v += sx * row[2 * k + h] + sy * row[3 * k + h];
                }
                *dst += v;
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use jscompat::{abs, max, sin};
    #[test]
    fn each_parameter_matches_central_differences() -> Result<(), crate::Error> {
        for (kind, directed, command, weighted) in [
            (Kind::Plain, false, false, false),
            (Kind::Plain, true, false, false),
            (Kind::Memory, true, false, false),
            (Kind::Memory, true, true, false),
            (Kind::Memory, true, false, true),
            (Kind::Hinted, true, false, false),
        ] {
            let plain = kind == Kind::Plain;
            let grid = if plain { 3 } else { 2 };
            let k = grid;
            let mut m = Element::new(if plain { 7 } else { 13 }, grid, k, directed, kind, command);
            let n = m.n;
            let sequence: Vec<Vec<_>> = (0..4)
                .map(|t| {
                    (0..3 * n)
                        .map(|i| {
                            if plain {
                                sin(0.71 * i as f64 + t as f64) / 2.
                            } else {
                                0.3 + sin(0.3 * i as f64 + t as f64) / 4.
                            }
                        })
                        .collect()
                })
                .collect();
            let mut initial = m.blank();
            if !plain {
                for (i, x) in initial[..n * k].iter_mut().enumerate() {
                    *x = 0.2 + sin(i as f64) / 3.;
                }
                assert!(m.weights[m.gate_bias..m.hint].iter().all(|x| *x == -1.));
            }
            let labels: Vec<_> = (0..3).map(|t| vec![0., 1., (t % 2) as f64, 1.]).collect();
            let weights: Vec<Vec<f64>> = if weighted {
                (0..3)
                    .map(|t| (0..n).map(|i| if i == t { 50. } else { 1. }).collect())
                    .collect()
            } else {
                vec![]
            };
            let mut tape: Vec<_> = (0..3).map(|_| m.buffers()).collect();
            let loss = |m: &Element, tape: &mut [Step]| -> Result<f64, crate::Error> {
                let mut sum = 0.;
                for t in 0..3 {
                    let (before, after) = tape.split_at_mut(t);
                    let out = &mut after[0];
                    m.forward(
                        before.last().map_or(&initial, |s| &s.state),
                        &sequence[t],
                        out,
                        command.then_some([1. - t as f64, (t % 2) as f64]),
                    )?;
                    for (i, (&p, &x)) in out.prediction.iter().zip(&sequence[t + 1]).enumerate() {
                        let e = p - x;
                        sum += weights.get(t).map_or(1., |w| w[i / 3]) * (e * e);
                    }
                    for (i, o) in out.occupancy.iter().enumerate() {
                        let e = o - labels[t][i];
                        sum += 3. * (e * e);
                    }
                }
                Ok(sum / (3 * n * 3) as f64)
            };
            let expected = loss(&m, &mut tape)?;
            let mut work = Gradient::new(&m);
            assert!(
                abs(work.backward(&m, &tape, &sequence[1..], &labels, &weights) - expected) < 1e-14
            );
            for i in 0..m.weights.len() {
                let saved = m.weights[i];
                m.weights[i] = saved + 1e-5;
                let above = loss(&m, &mut tape)?;
                m.weights[i] = saved - 1e-5;
                let numeric = (above - loss(&m, &mut tape)?) / 2e-5;
                m.weights[i] = saved;
                let a = work.grad[i];
                assert!(
                    abs(a - numeric) <= 1e-6 * max(abs(a), abs(numeric)),
                    "{kind:?} {i} {a} {numeric}"
                );
            }
        }
        Ok(())
    }
}
