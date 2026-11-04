use anyhow::{Context, Result, bail};
use lab::{Config, Episode, Models, Policy, Scenario};
use std::io::{self, Write};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let scenario: Scenario = args.next().context("missing scenario")?.parse()?;
    let policy: Policy = args.next().context("missing policy")?.parse()?;
    let seed = args
        .next()
        .context("missing seed")?
        .parse()
        .context("invalid seed")?;
    let steps = args
        .next()
        .context("missing step count")?
        .parse()
        .context("invalid step count")?;
    let mode = args.next().context("missing mode")?;
    if args.next().is_some() || !matches!(mode.as_str(), "live" | "replay" | "learned") {
        bail!("expected scenario policy seed steps live|replay|learned");
    }
    let mut episode = Episode::with_models(
        scenario,
        seed,
        policy,
        (mode == "learned").then(Models::default),
        None,
    );
    let mut cfg = Config::new(policy);
    let mut out = io::BufWriter::new(io::stdout().lock());
    for frame in 0..=steps {
        if frame > 0 {
            cfg.replay = mode == "replay" && frame > 14;
            episode.step(&cfg);
        }
        serde_json::to_writer(&mut out, &episode.snapshot(policy)).context("serialize snapshot")?;
        out.write_all(b"\n").context("write snapshot")?;
    }
    out.flush().context("flush trace")
}
