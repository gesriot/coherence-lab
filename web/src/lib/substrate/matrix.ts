// Row-major product. Four adjacent outputs share each load from the left matrix.
// In accumulation mode this sums the weight derivatives across time.
export function product(
  a: Float64Array, b: Float64Array, out: Float64Array,
  rows: number, inner: number, cols: number, add = false,
): void {
  for (let i = 0; i < rows; i++) {
    const ai = i * inner;
    const oi = i * cols;
    let j = 0;
    for (; j + 3 < cols; j += 4) {
      let p = 0, q = 0, r = 0, s = 0;
      for (let h = 0, k = j; h < inner; h++, k += cols) {
        const x = a[ai + h]!;
        p += x * b[k]!;
        q += x * b[k + 1]!;
        r += x * b[k + 2]!;
        s += x * b[k + 3]!;
      }
      out[oi + j] = (add ? out[oi + j]! : 0) + p;
      out[oi + j + 1] = (add ? out[oi + j + 1]! : 0) + q;
      out[oi + j + 2] = (add ? out[oi + j + 2]! : 0) + r;
      out[oi + j + 3] = (add ? out[oi + j + 3]! : 0) + s;
    }
    for (; j < cols; j++) {
      let p = 0;
      for (let h = 0, k = j; h < inner; h++, k += cols) p += a[ai + h]! * b[k]!;
      out[oi + j] = (add ? out[oi + j]! : 0) + p;
    }
  }
}

export function transpose(a: Float64Array, out: Float64Array, rows: number, cols: number) {
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) out[j * rows + i] = a[i * cols + j]!;
  }
}
