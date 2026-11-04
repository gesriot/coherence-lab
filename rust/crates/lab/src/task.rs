// Frozen design parameters shared by training, perception and the evaluators.
pub(crate) mod object {
    pub(crate) const UNCERTAINTY_GROWTH: f64 = 0.03;
    pub(crate) const EMPTY_VIEWS: u32 = 3;
    pub(crate) const GHOST_DELAY: u32 = 20;
    pub(crate) const OUTDOOR_FIRST: u32 = 12;
    pub(crate) const OUTDOOR_LAST: u32 = 35;
    pub(crate) const INITIAL: [f64; 3] = [0.28, 0.28, 0.5];
    pub(crate) const KEEP_ALL: [f64; 3] = [0.001, 0.001, 0.999];
    pub(crate) const IDEAL: [f64; 3] = [0.001, 0.02, 0.1];
    pub(crate) const RADIUS_FRACTION: f64 = 0.5;
    pub(crate) const MAX_SEEN: u32 = 20;
    pub(crate) const RETROSPECTIVE: u32 = 30;
    pub(crate) const DELTA: f64 = 1e-4;
    pub(crate) const LOGIT_LIMIT: f64 = 9.;
    pub(crate) const CEILING: f64 = 0.75;
    pub(crate) const FLOOR: f64 = 0.9;
    pub(crate) const STABILITY: f64 = 0.85;
}
pub(crate) mod sensorimotor {
    pub(crate) const MIN_OFFSET: f64 = 0.1;
    pub(crate) const MAX_OFFSET: f64 = 0.2;
    pub(crate) const MIN_GAIN: f64 = 0.5;
    pub(crate) const MAX_GAIN: f64 = 2.;
    pub(crate) const MEMORY: usize = 256;
    pub(crate) const BASELINE: [f64; 2] = [1., 1.];
    pub(crate) const FIRST_SCORED: usize = 1;
    pub(crate) const SACCADE: f64 = 4.;
    pub(crate) const TIE: f64 = 0.005;
    pub(crate) const MIN_CELLS: u32 = 20;
    pub(crate) const MIN_TO_MEAN: f64 = 0.5;
    pub(crate) const MAX_BIAS: f64 = 0.2;
    pub(crate) const MIN_DIRECTION: f64 = 0.8;
}
#[cfg(test)]
mod tests {
    use super::{object as o, sensorimotor as s};
    #[test]
    fn object_protocol_keeps_the_design_parameters() {
        assert_eq!(
            (o::UNCERTAINTY_GROWTH, o::EMPTY_VIEWS, o::GHOST_DELAY),
            (0.03, 3, 20)
        );
        assert_eq!((o::OUTDOOR_FIRST, o::OUTDOOR_LAST), (12, 35));
        assert_eq!(o::INITIAL, [0.28, 0.28, 0.5]);
        assert_eq!(o::KEEP_ALL, [0.001, 0.001, 0.999]);
        assert_eq!(o::IDEAL, [0.001, 0.02, 0.1]);
        assert_eq!(
            (o::RADIUS_FRACTION, o::MAX_SEEN, o::RETROSPECTIVE),
            (0.5, 20, 30)
        );
        assert_eq!((o::DELTA, o::LOGIT_LIMIT), (1e-4, 9.));
        assert_eq!((o::CEILING, o::FLOOR, o::STABILITY), (0.75, 0.9, 0.85));
        use crate::Scenario::*;
        assert_eq!(
            (1..=7).map(crate::genesis::curriculum).collect::<Vec<_>>(),
            [
                Continuity, Vanish, Binding, Nursery, Continuity, Binding, Nursery
            ]
        );
    }
    #[test]
    fn gain_protocol_keeps_the_design_parameters() {
        assert_eq!(
            (s::MIN_OFFSET, s::MAX_OFFSET, s::MIN_GAIN, s::MAX_GAIN),
            (0.1, 0.2, 0.5, 2.)
        );
        assert_eq!(
            (s::MEMORY, s::BASELINE, s::FIRST_SCORED, s::SACCADE),
            (256, [1., 1.], 1, 4.)
        );
        assert_eq!(
            (
                s::TIE,
                s::MIN_CELLS,
                s::MIN_TO_MEAN,
                s::MAX_BIAS,
                s::MIN_DIRECTION
            ),
            (0.005, 20, 0.5, 0.2, 0.8)
        );
    }
}
