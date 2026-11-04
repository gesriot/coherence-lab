use crate::{Error, matrix::product, signals::Vector};
use jscompat::{exp, hash_unit, tanh};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Plain,
    Memory,
    Hinted,
}
pub(crate) struct Element {
    pub(crate) grid: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) f: usize,
    pub(crate) directed: bool,
    pub(crate) command: bool,
    pub(crate) kind: Kind,
    pub(crate) weights: Vec<f64>,
    pub(crate) neighbors: Vec<[usize; 8]>,
    pub(crate) inverse: Vec<f64>,
    pub(crate) bias: usize,
    pub(crate) readout: usize,
    pub(crate) end: usize,
    pub(crate) gate: usize,
    pub(crate) gate_bias: usize,
    pub(crate) hint: usize,
}
pub(crate) struct Step {
    pub(crate) features: Vec<f64>,
    pub(crate) state: Vec<f64>,
    pub(crate) gate: Vec<f64>,
    pub(crate) candidate: Vec<f64>,
    pub(crate) occupancy: Vec<f64>,
    pub(crate) prediction: Vec<f64>,
}
impl Element {
    pub(crate) fn new(
        seed: u32,
        grid: usize,
        k: usize,
        directed: bool,
        kind: Kind,
        command: bool,
    ) -> Self {
        let n = grid * grid;
        let f = (if directed { 4 } else { 2 }) * k + usize::from(command) * 2 + 3;
        let bias = f * k;
        let readout = bias + k;
        let end = readout + 3 * k;
        let gate = end + 3;
        let gate_bias = gate + bias;
        let hint = gate_bias + k;
        let size = match kind {
            Kind::Plain => gate,
            Kind::Memory => hint,
            Kind::Hinted => hint + k + 1,
        };
        let mut weights: Vec<_> = (0..size)
            .map(|i| {
                let index = if !directed && i >= 2 * k * k {
                    i + 2 * k * k
                } else {
                    i
                };
                (2. * hash_unit(7919. * f64::from(seed) + index as f64) - 1.) * 0.3
            })
            .collect();
        if kind != Kind::Plain {
            weights[gate_bias..hint].fill(-1.);
        }
        let mut neighbors = vec![[n; 8]; n];
        let mut inverse = vec![0.; n];
        for (i, ns) in neighbors.iter_mut().enumerate() {
            let mut count = 0;
            for (j, o) in ns.iter_mut().enumerate() {
                let p = if j < 4 { j } else { j + 1 };
                let x = (i % grid) as isize + (p % 3) as isize - 1;
                let y = (i / grid) as isize + (p / 3) as isize - 1;
                if (0..grid as isize).contains(&x) && (0..grid as isize).contains(&y) {
                    *o = y as usize * grid + x as usize;
                    count += 1;
                }
            }
            if count > 0 {
                inverse[i] = 1. / f64::from(count);
            }
        }
        Self {
            grid,
            n,
            k,
            f,
            directed,
            command,
            kind,
            weights,
            neighbors,
            inverse,
            bias,
            readout,
            end,
            gate,
            gate_bias,
            hint,
        }
    }
    pub(crate) fn blank(&self) -> Vec<f64> {
        vec![0.; (self.n + 1) * self.k]
    }
    pub(crate) fn buffers(&self) -> Step {
        let gated = if self.kind == Kind::Plain {
            0
        } else {
            self.n * self.k
        };
        Step {
            features: vec![0.; self.n * self.f],
            state: self.blank(),
            gate: vec![0.; gated],
            candidate: vec![0.; gated],
            occupancy: vec![0.; if self.kind == Kind::Hinted { self.n } else { 0 }],
            prediction: vec![0.; 3 * self.n],
        }
    }
    pub(crate) fn forward(
        &self,
        prev: &[f64],
        x: &[f64],
        out: &mut Step,
        command: Option<Vector>,
    ) -> Result<(), Error> {
        let (k, f, n) = (self.k, self.f, self.n);
        for (i, row) in out.features.chunks_exact_mut(f).enumerate() {
            let [nw, no, ne, we, ea, sw, so, se] =
                self.neighbors[i].map(|j| &prev[j * k..(j + 1) * k]);
            row[..k].copy_from_slice(&prev[i * k..(i + 1) * k]);
            for h in 0..k {
                let (a, b, c, d, e, g, v, z) =
                    (nw[h], no[h], ne[h], we[h], ea[h], sw[h], so[h], se[h]);
                row[k + h] = (a + b + c + d + e + g + v + z) * self.inverse[i];
                if self.directed {
                    row[2 * k + h] = (c + 2. * e + z - a - 2. * d - g) / 8.;
                    row[3 * k + h] = (g + 2. * v + z - a - 2. * b - c) / 8.;
                }
            }
            if self.command {
                row[f - 5..f - 3].copy_from_slice(&command.ok_or(Error::MissingCommand)?);
            }
            row[f - 3..].copy_from_slice(&x[i * 3..i * 3 + 3]);
        }
        let w = &self.weights;
        product(
            &out.features,
            &w[..self.bias],
            &mut out.state[..n * k],
            f,
            k,
            false,
        );
        if self.kind != Kind::Plain {
            product(
                &out.features,
                &w[self.gate..self.gate_bias],
                &mut out.gate,
                f,
                k,
                false,
            );
        }
        for (i, state) in out.state[..n * k].chunks_exact_mut(k).enumerate() {
            for (h, s) in state.iter_mut().enumerate() {
                let j = i * k + h;
                let candidate = tanh(*s + w[self.bias + h]);
                *s = if self.kind == Kind::Plain {
                    candidate
                } else {
                    let z = 1. / (1. + exp(-(out.gate[j] + w[self.gate_bias + h])));
                    out.gate[j] = z;
                    out.candidate[j] = candidate;
                    (1. - z) * prev[j] + z * candidate
                };
            }
        }
        if self.kind == Kind::Hinted {
            product(
                &out.state[..n * k],
                &w[self.hint..self.hint + k],
                &mut out.occupancy,
                k,
                1,
                false,
            );
            for o in &mut out.occupancy {
                *o += w[self.hint + k];
            }
        }
        product(
            &out.state[..n * k],
            &w[self.readout..self.end],
            &mut out.prediction,
            k,
            3,
            false,
        );
        for p in out.prediction.as_chunks_mut::<3>().0 {
            for d in 0..3 {
                p[d] += w[self.end + d];
            }
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sobel_impulse_is_antisymmetric_and_mean_excludes_self_and_padding() -> Result<(), Error> {
        let m = Element::new(1, 3, 1, true, Kind::Plain, false);
        let mut prev = m.blank();
        let mut out = m.buffers();
        let (mut sx, mut sy) = (vec![], vec![]);
        for i in 0..9 {
            prev.fill(0.);
            prev[i] = 1.;
            m.forward(&prev, &[0.; 27], &mut out, None)?;
            sx.push(out.features[4 * m.f + 2] * 8.);
            sy.push(out.features[4 * m.f + 3] * 8.);
            assert_eq!(
                out.features[1],
                if [1, 3, 4].contains(&i) { 1. / 3. } else { 0. }
            );
        }
        assert_eq!(sx, [-1., 0., 1., -2., 0., 2., -1., 0., 1.]);
        assert_eq!(sy, [-1., -2., -1., 0., 0., 0., 1., 2., 1.]);
        Ok(())
    }
}
