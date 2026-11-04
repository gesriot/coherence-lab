use anyhow::{Context, Result, bail};
use jscompat as js;
use std::io::{self, Read};

fn main() -> Result<()> {
    let op = std::env::args().nth(1).context("missing operation")?;
    let mut bytes = Vec::new();
    io::stdin()
        .read_to_end(&mut bytes)
        .context("read reference vectors")?;
    let (chunks, remainder) = bytes.as_chunks::<32>();
    if !remainder.is_empty() {
        bail!("incomplete reference record");
    }
    for (i, row) in chunks.iter().enumerate() {
        let mut nums = [0.; 4];
        for (n, b) in nums.iter_mut().zip(row.as_chunks::<8>().0) {
            *n = f64::from_le_bytes(*b);
        }
        let [x, y, z, want] = nums;
        let got = match op.as_str() {
            "sin" => js::sin(x),
            "cos" => js::cos(x),
            "exp" => js::exp(x),
            "log" => js::log(x),
            "log1p" => js::log1p(x),
            "atan2" => js::atan2(x, y),
            "tanh" => js::tanh(x),
            "abs" => js::abs(x),
            "ceil" => js::ceil(x),
            "floor" => js::floor(x),
            "sqrt" => js::sqrt(x),
            "round" => js::round(x),
            "sign" => js::sign(x),
            "min" => js::min(x, y),
            "max" => js::max(x, y),
            "hypot" => js::hypot(x, y),
            "hypot3" => js::hypot_many(&[x, y, z]),
            "hypot4" => js::hypot_many(&[x, y, z, x]),
            "imul" => f64::from(js::imul(x, y)),
            "int32" => f64::from(js::to_int32(x)),
            "uint32" => f64::from(js::to_uint32(x)),
            "hashUnit" => js::hash_unit(x),
            "hueNoise" => js::hue_noise(x, y, z),
            _ => bail!("unknown operation {op}"),
        };
        if want.to_bits() != got.to_bits() {
            bail!(
                "{op} record {i}: ({x:?}, {y:?}, {z:?}) expected {:016x}, got {:016x}",
                want.to_bits(),
                got.to_bits()
            );
        }
    }
    Ok(())
}
