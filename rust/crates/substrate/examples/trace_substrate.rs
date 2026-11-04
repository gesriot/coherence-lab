use anyhow::{Context, Result, bail};
use lab::Occlusion;
use std::io::{self, Write};
use substrate::*;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [variant, depth, episodes, steps] = args.as_slice() else {
        bail!("expected variant depth episodes steps")
    };
    let scale = Scale {
        depth: depth.parse().context("invalid depth")?,
        episodes: episodes.parse().context("invalid episodes")?,
        steps: steps.parse().context("invalid steps")?,
        ..Scale::default()
    };
    let occlusion = if matches!(variant.as_str(), "plain" | "isotropic") {
        Occlusion::WholeObject
    } else {
        Occlusion::PerCell
    };
    let kind = match variant.as_str() {
        "plain" | "isotropic" | "local" => Kind::Plain,
        "hinted" => Kind::Hinted,
        "memory" | "weighted" | "executed" | "zero" | "shuffled" | "random-copy" => Kind::Memory,
        _ => bail!("unknown variant"),
    };
    let mut opt = TrainingOptions {
        occlusion,
        kind,
        directed: variant != "isotropic",
        ..TrainingOptions::default()
    };
    if variant == "weighted" {
        opt.weighted = true;
        opt.world = WorldSettings {
            random: true,
            columns: Some(2),
        };
    }
    if matches!(
        variant.as_str(),
        "executed" | "zero" | "shuffled" | "random-copy"
    ) {
        opt.motion = Some(Motion {
            gain: if variant == "shuffled" {
                Gain::Episodic
            } else {
                Gain::Fixed
            },
            copy: match variant.as_str() {
                "zero" => CopyMode::Zero,
                "shuffled" => CopyMode::Shuffled,
                _ => CopyMode::Executed,
            },
            random: variant == "random-copy",
        });
    }
    let mut out = io::BufWriter::new(io::stdout().lock());
    for episode in training_trace(scale, 1, opt).context("trace substrate training")? {
        for row in episode.predictions.iter().chain([&episode.weights]) {
            for x in row {
                out.write_all(&x.to_le_bytes())
                    .context("write binary64 trace")?;
            }
        }
    }
    out.flush().context("flush trace")
}
