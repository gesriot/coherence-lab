// Each lane is a separate output column. Accumulate its dot product in inner-index order.
pub(crate) fn product(a: &[f64], b: &[f64], out: &mut [f64], inner: usize, cols: usize, add: bool) {
    let mut sums = vec![0.; cols];
    for (left, dest) in a.chunks_exact(inner).zip(out.chunks_exact_mut(cols)) {
        sums.fill(0.);
        for (&v, row) in left.iter().zip(b.chunks_exact(cols)) {
            for (s, &w) in sums.iter_mut().zip(row) {
                *s += v * w;
            }
        }
        for (d, s) in dest.iter_mut().zip(&sums) {
            *d = (if add { *d } else { 0. }) + s;
        }
    }
}
pub(crate) fn transpose(a: &[f64], out: &mut [f64], rows: usize, cols: usize) {
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = a[i * cols + j];
        }
    }
}
