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
        let mut weights = opt.weighted.then(|| {
            let w = world(sc, seed, opt.world);
            let mut d = Detector::default();
            d.read(&w, grid);
            (w, d)
        });
        let mut weight = |time| {
            let (w, det) = weights
                .as_mut()
                .expect("only exit weighting requests target maps");
            advance(w);
            debug_assert_eq!(w.t as usize, time);
            let mut v = vec![1.; grid * grid];
            if let Some(e) = det.read(w, grid).0 {
                debug_assert_eq!(e.time, time);
                debug_assert!(e.length > 0);
                if e.wall {
                    for i in e.cells {
                        v[i] = 50.;
                    }
                }
            }
            v
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
