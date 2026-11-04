// These leaf functions accept every binary64 bit pattern and do not access Rust memory.
unsafe extern "C" {
    fn js_sin(x: f64) -> f64;
    fn js_cos(x: f64) -> f64;
    fn js_exp(x: f64) -> f64;
    fn js_log(x: f64) -> f64;
    fn js_log1p(x: f64) -> f64;
    fn js_tanh(x: f64) -> f64;
    fn js_atan2(y: f64, x: f64) -> f64;
}
pub fn sin(x: f64) -> f64 {
    unsafe { js_sin(x) }
}
pub fn cos(x: f64) -> f64 {
    unsafe { js_cos(x) }
}
pub fn exp(x: f64) -> f64 {
    unsafe { js_exp(x) }
}
pub fn log(x: f64) -> f64 {
    unsafe { js_log(x) }
}
pub fn log1p(x: f64) -> f64 {
    unsafe { js_log1p(x) }
}
pub fn tanh(x: f64) -> f64 {
    unsafe { js_tanh(x) }
}
pub fn atan2(y: f64, x: f64) -> f64 {
    unsafe { js_atan2(y, x) }
}
