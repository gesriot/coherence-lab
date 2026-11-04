pub(crate) type Vector = [f64; 2];
pub(crate) struct Observation {
    pub(crate) x: Vec<f64>,
    pub(crate) copy: Option<Vector>,
}
