pub use std::f64::consts::{PI, SQRT_2};

pub fn abs(x: f64) -> f64 {
    x.abs()
}
pub fn sqrt(x: f64) -> f64 {
    if x < 0.0 { f64::NAN } else { x.sqrt() }
}
pub fn floor(x: f64) -> f64 {
    x.floor()
}
pub fn ceil(x: f64) -> f64 {
    x.ceil()
}

pub fn round(x: f64) -> f64 {
    if x.is_nan() {
        return quiet_nan(x);
    }
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    if (-0.5..0.0).contains(&x) {
        return -0.0;
    }
    let lower = x.floor();
    if x - lower >= 0.5 { lower + 1.0 } else { lower }
}
pub fn sign(x: f64) -> f64 {
    if x == 0.0 || x.is_nan() {
        x
    } else {
        x.signum()
    }
}
const QUIET_BIT: u64 = 1 << 51;

fn quiet_nan(x: f64) -> f64 {
    f64::from_bits(x.to_bits() | QUIET_BIT)
}

// The arm64 builtin reduces from an infinity. The first operand is already
// quiet when the second operand reaches FMIN/FMAX.
fn min_max_nan(x: f64, y: f64) -> f64 {
    if !x.is_nan() || (y.is_nan() && y.to_bits() & QUIET_BIT == 0) {
        quiet_nan(y)
    } else {
        quiet_nan(x)
    }
}
pub fn min(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        min_max_nan(x, y)
    } else if x < y || (x == y && x.is_sign_negative()) {
        x
    } else {
        y
    }
}
pub fn max(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        min_max_nan(x, y)
    } else if x > y || (x == y && x.is_sign_positive()) {
        x
    } else {
        y
    }
}
pub fn hypot(x: f64, y: f64) -> f64 {
    let (a, b) = (x.abs(), y.abs());
    if a.is_infinite() || b.is_infinite() {
        return f64::INFINITY;
    }
    let m = max(a, b);
    if m.is_nan() {
        return f64::NAN;
    }
    if m == 0.0 {
        return m;
    }
    let (a, b) = (a / m, b / m);
    (a * a + b * b).sqrt() * m
}
pub fn hypot_many(args: &[f64]) -> f64 {
    match args {
        [] => return 0.0,
        [a] => return a.abs(),
        [a, b] => return hypot(*a, *b),
        _ => {}
    }
    if args.iter().any(|a| a.is_infinite()) {
        return f64::INFINITY;
    }
    let m = args.iter().fold(0., |m, a| max(m, a.abs()));
    if m.is_nan() {
        return f64::NAN;
    }
    if m == 0. {
        return m;
    }
    let (mut sum, mut correction) = (0., 0.);
    for a in args {
        let n = a.abs() / m;
        let term = n * n - correction;
        let next = sum + term;
        correction = (next - sum) - term;
        sum = next;
    }
    sum.sqrt() * m
}
pub fn to_uint32(x: f64) -> u32 {
    let bits = x.to_bits();
    let shift = ((bits >> 52) & 2047) as i32 - 1023;
    if !(0..84).contains(&shift) {
        return 0;
    }
    let mantissa = (bits & ((1 << 52) - 1)) | (1 << 52);
    let low = if shift >= 52 {
        (mantissa as u32) << (shift - 52)
    } else {
        (mantissa >> (52 - shift)) as u32
    };
    if bits >> 63 == 0 {
        low
    } else {
        low.wrapping_neg()
    }
}
pub fn to_int32(x: f64) -> i32 {
    to_uint32(x) as i32
}
pub fn imul(x: f64, y: f64) -> i32 {
    to_int32(x).wrapping_mul(to_int32(y))
}
pub fn hash_unit(n: f64) -> f64 {
    let x = (to_uint32(n) ^ 0x9e3779b9).wrapping_mul(0x85ebca6b);
    let x = (x ^ (x >> 13)).wrapping_mul(0xc2b2ae35);
    f64::from(x ^ (x >> 16)) / 4294967296.
}
pub fn hue_noise(seed: f64, t: f64, i: f64) -> f64 {
    let u = hash_unit(seed + t * 997. + i * 131. + 3.);
    let v = hash_unit(seed * 3. + t * 1597. + i * 313. + 11.);
    let w = hash_unit(seed * 7. + t * 53. + i * 17. + 29.);
    (u + v + w) / 1.5 - 1.
}
