use crate::Error;
use jscompat::{floor, hash_unit, sqrt};
pub(crate) fn mean(values: impl IntoIterator<Item = f64>) -> f64 {
    let (s, n) = values
        .into_iter()
        .fold((0., 0usize), |(s, n), v| (s + v, n + 1));
    s / n as f64
}
fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).fold(0., |s, (x, y)| s + x * y)
}
pub(crate) struct Ridge {
    center: Vec<f64>,
    basis: Vec<Vec<f64>>,
    lower: Vec<f64>,
    dual: bool,
}
pub(crate) struct Fit {
    weights: Vec<f64>,
    bias: f64,
}
impl Fit {
    pub(crate) fn predict(&self, x: &[f64]) -> f64 {
        dot(&self.weights, x) + self.bias
    }
}
impl Ridge {
    pub(crate) fn new(rows: &[Vec<f64>]) -> Result<Self, Error> {
        let n = rows.len();
        let d = rows.first().ok_or(Error::NoExamples)?.len();
        let mut center = vec![0.; d];
        for row in rows {
            for (v, &x) in center.iter_mut().zip(row) {
                *v += x;
            }
        }
        for c in &mut center {
            *c /= n as f64;
        }
        let dual = n < d;
        let mut basis = if dual {
            vec![vec![0.; d]; n]
        } else {
            vec![vec![0.; n]; d]
        };
        for (i, row) in rows.iter().enumerate() {
            for (j, &x) in row.iter().enumerate() {
                let centered = x - center[j];
                if dual {
                    basis[i][j] = centered;
                } else {
                    basis[j][i] = centered;
                }
            }
        }
        let size = basis.len();
        let mut lower = vec![0.; size * size];
        for i in 0..size {
            for j in 0..=i {
                let mut v = dot(&basis[i], &basis[j]) + if i == j { 0.01 } else { 0. };
                for k in 0..j {
                    v -= lower[i * size + k] * lower[j * size + k];
                }
                lower[i * size + j] = if i == j {
                    sqrt(v)
                } else {
                    v / lower[j * size + j]
                };
            }
        }
        Ok(Self {
            center,
            basis,
            lower,
            dual,
        })
    }
    pub(crate) fn fit(&self, labels: &[f64]) -> Fit {
        let intercept = mean(labels.iter().copied());
        let y: Vec<_> = labels.iter().map(|v| v - intercept).collect();
        let mut sol = if self.dual {
            y
        } else {
            self.basis.iter().map(|x| dot(x, &y)).collect()
        };
        let n = sol.len();
        let l = &self.lower;
        for i in 0..n {
            let mut x = sol[i];
            for j in 0..i {
                x -= l[i * n + j] * sol[j];
            }
            sol[i] = x / l[i * n + i];
        }
        for i in (0..n).rev() {
            let mut x = sol[i];
            for j in i + 1..n {
                x -= l[j * n + i] * sol[j];
            }
            sol[i] = x / l[i * n + i];
        }
        let weights = if self.dual {
            let mut w = vec![0.; self.center.len()];
            for (&s, row) in sol.iter().zip(&self.basis) {
                for (v, x) in w.iter_mut().zip(row) {
                    *v += s * x;
                }
            }
            w
        } else {
            sol
        };
        let bias = intercept - dot(&weights, &self.center);
        Fit { weights, bias }
    }
}
pub(crate) fn shuffled(y: &[f64]) -> Vec<f64> {
    let mut v = y.to_vec();
    for i in (1..v.len()).rev() {
        let j = floor(hash_unit(4001. + i as f64) * (i + 1) as f64) as usize;
        v.swap(i, j);
    }
    v
}
pub(crate) fn r2(labels: &[f64], prediction: &[f64]) -> f64 {
    let avg = mean(labels.iter().copied());
    let (mut r, mut v) = (0., 0.);
    for (&y, &p) in labels.iter().zip(prediction) {
        r += (p - y) * (p - y);
        v += (y - avg) * (y - avg);
    }
    1. - r / v
}
#[derive(Default)]
pub(crate) struct Rows {
    pub(crate) x: Vec<Vec<f64>>,
    pub(crate) y: Vec<f64>,
}
impl Rows {
    fn score(&self, fit: Fit, classify: bool) -> f64 {
        let pred: Vec<_> = self.x.iter().map(|r| fit.predict(r)).collect();
        if classify {
            mean(
                pred.iter()
                    .zip(&self.y)
                    .map(|(&x, &y)| f64::from((if x >= 0. { 1. } else { -1. }) == y)),
            )
        } else {
            r2(&self.y, &pred)
        }
    }
}
pub(crate) fn probe(train: &Rows, test: &Rows, classify: bool) -> Result<[f64; 2], Error> {
    let ridge = Ridge::new(&train.x)?;
    Ok([
        test.score(ridge.fit(&train.y), classify),
        test.score(ridge.fit(&shuffled(&train.y)), classify),
    ])
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn primal_and_dual_recover_regularized_slope_and_free_intercept() -> Result<(), Error> {
        for d in [1, 5] {
            let encode = |x| {
                let mut v = vec![0.; d];
                v[0] = x;
                v
            };
            let x = [-1., 0., 1.].map(encode);
            let fit = Ridge::new(&x)?.fit(&[5., 7., 9.]);
            assert!(jscompat::abs(fit.predict(&encode(4.)) - (7. + 4. * 4. / 2.01)) < 1e-12);
        }
        Ok(())
    }
}
