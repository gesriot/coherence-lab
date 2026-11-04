use std::io::{Write, stdout};
fn main() -> std::io::Result<()> {
    let mut out = std::io::BufWriter::new(stdout().lock());
    for i in 1..=18000 {
        for x in [0.9, 0.999] {
            out.write_all(&jscompat::pow_u32(x, i).to_le_bytes())?;
        }
    }
    out.flush()
}
