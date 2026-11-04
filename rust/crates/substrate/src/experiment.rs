use crate::{
    Error, Scale,
    element::{Element, Kind},
    input::*,
    training::Trainer,
};
use lab::{Occlusion, Scenario, advance};
#[derive(Clone, Copy, Debug)]
pub struct TrainingOptions {
    pub kind: Kind,
    pub directed: bool,
    pub motion: Option<Motion>,
    pub world: WorldSettings,
    pub weighted: bool,
    pub occlusion: Occlusion,
}
impl Default for TrainingOptions {
    fn default() -> Self {
        Self {
            kind: Kind::Plain,
            directed: true,
            motion: None,
            world: WorldSettings::default(),
            weighted: false,
            occlusion: Occlusion::WholeObject,
        }
    }
}
struct ExitWeights {
    world: lab::World,
    detector: Detector,
    grid: usize,
}
impl ExitWeights {
    fn new(sc: Scenario, seed: u32, settings: WorldSettings, grid: usize) -> Self {
        let world = world(sc, seed, settings);
        let mut detector = Detector::default();
        detector.read(&world, grid);
        Self {
            world,
            detector,
            grid,
        }
    }
    fn at(&mut self, time: usize) -> Vec<f64> {
        advance(&mut self.world);
        debug_assert_eq!(self.world.t as usize, time);
        let mut weights = vec![1.; self.grid * self.grid];
        if let Some(e) = self.detector.read(&self.world, self.grid).0 {
            debug_assert_eq!(e.time, time);
            debug_assert!(e.length > 0);
            if e.wall {
                for i in e.cells {
                    weights[i] = 50.;
                }
            }
        }
        weights
    }
}
pub struct TrainingTrace {
    pub weights: Vec<f64>,
    pub predictions: Vec<Vec<f64>>,
}
pub(crate) fn train(
    m: &mut Element,
    scale: Scale,
    run: u32,
    opt: TrainingOptions,
    mut complete: impl FnMut(usize, &Element) -> Result<(), Error>,
    mut trace: Option<&mut Vec<TrainingTrace>>,
) -> Result<(), Error> {
    scale.validate()?;
    let mut coach = Trainer::new(m, scale.memory);
    for i in 0..scale.episodes {
        let sc = [
            Scenario::Continuity,
            Scenario::Vanish,
            Scenario::Nursery,
            Scenario::Binding,
        ][i % 4];
        let seed = 20000 + 100 * run + i as u32;
        let grid = m.grid;
        let seq = movie(
            sc,
            seed,
            grid,
            scale.steps,
            opt.occlusion,
            opt.world,
            opt.motion,
        );
        let mut hint_world = (m.kind == Kind::Hinted).then(|| world(sc, seed, opt.world));
        let mut hint = |time| {
            let w = hint_world
                .as_mut()
                .expect("only the hinted model requests occupancy");
            while (w.t as usize) < time {
                advance(w);
            }
            occupancy(w, grid)
        };
        let mut weights = opt
            .weighted
            .then(|| ExitWeights::new(sc, seed, opt.world, grid));
        let mut weight = |time| {
            weights
                .as_mut()
                .expect("only exit weighting requests maps")
                .at(time)
        };
        let mut predictions = vec![];
        coach.episode(
            m,
            seq,
            (m.kind == Kind::Hinted).then_some(&mut hint),
            opt.weighted.then_some(&mut weight),
            trace.is_some().then_some(&mut predictions),
        )?;
        if let Some(traces) = trace.as_deref_mut() {
            traces.push(TrainingTrace {
                weights: m.weights.clone(),
                predictions,
            });
        }
        complete(i + 1, m)?;
    }
    Ok(())
}
pub fn training_trace(
    scale: Scale,
    run: u32,
    opt: TrainingOptions,
) -> Result<Vec<TrainingTrace>, Error> {
    scale.validate()?;
    let mut m = Element::new(
        run,
        scale.grid,
        scale.depth,
        opt.directed,
        opt.kind,
        opt.motion.is_some(),
    );
    let mut traces = vec![];
    train(&mut m, scale, run, opt, |_, _| Ok(()), Some(&mut traces))?;
    Ok(traces)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_exposed_target_cells_receive_wall_exit_weight() {
        let setting = WorldSettings {
            random: true,
            columns: Some(2),
        };
        let mut weights = ExitWeights::new(Scenario::Continuity, 72, setting, 16);
        let mut w = world(Scenario::Continuity, 72, setting);
        let mut det = Detector::default();
        det.read(&w, 16);
        let mut hits = 0;
        for time in 1..=72 {
            advance(&mut w);
            let event = det.read(&w, 16).0;
            for (i, weight) in weights.at(time).into_iter().enumerate() {
                let exposed = event
                    .as_ref()
                    .is_some_and(|e| e.wall && e.cells.contains(&i));
                assert_eq!(weight, if exposed { 50. } else { 1. });
                hits += usize::from(exposed);
            }
        }
        assert!(hits > 0);
    }
    #[test]
    fn partial_segments_repeat_only_for_equal_seeds() -> Result<(), Error> {
        let scale = Scale {
            episodes: 3,
            depth: 2,
            steps: 32,
            ..Scale::default()
        };
        let run = |s| {
            Ok::<_, Error>(
                training_trace(scale, s, TrainingOptions::default())?
                    .pop()
                    .expect("three episodes")
                    .weights,
            )
        };
        let a = run(1)?;
        assert_eq!(a, run(1)?);
        assert_ne!(a, run(2)?);
        assert_ne!(a, Element::new(1, 16, 2, true, Kind::Plain, false).weights);
        Ok(())
    }
}
