use anyhow::{Context, Result, bail};
use lab::{Condition, training_trace};
use std::io::{self, Write};
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let condition = match args.next().as_deref() {
        Some("full") => Condition::Full,
        Some("first-order") => Condition::FirstOrder,
        Some("strong-first-order") => Condition::StrongFirstOrder,
        Some("no-protocol") => Condition::NoProtocol,
        Some("oracle") => Condition::Oracle,
        Some("frozen") => Condition::Frozen,
        _ => bail!("invalid training condition"),
    };
    let seed = args
        .next()
        .context("missing seed")?
        .parse()
        .context("invalid seed")?;
    let count = args
        .next()
        .context("missing episode count")?
        .parse()
        .context("invalid count")?;
    let mut out = io::BufWriter::new(io::stdout().lock());
    for frame in training_trace(condition, seed, count) {
        serde_json::to_writer(&mut out, &frame).context("serialize training episode")?;
        out.write_all(b"\n").context("write training episode")?;
    }
    out.flush().context("flush training trace")
}
