use anyhow::{Context, Result, bail};
use jscompat::{
    Value, console_log, console_table, inspect_number, inspect_string, number, to_fixed,
};
use std::io::{self, Read, Write};

fn main() -> Result<()> {
    let mode = std::env::args().nth(1).context("missing check mode")?;
    let mut buf = Vec::new();
    io::stdin().read_to_end(&mut buf).context("read cases")?;
    let mut out = io::BufWriter::new(io::stdout().lock());
    if mode == "points" {
        let (cases, tail) = buf.as_chunks::<8>();
        if !tail.is_empty() {
            bail!("truncated points case");
        }
        for bytes in cases {
            writeln!(
                out,
                "{}",
                jscompat::signed_tenth(f64::from_le_bytes(*bytes) * 100.)
            )
            .context("write points")?;
        }
    } else if mode == "numbers" {
        let (cases, tail) = buf.as_chunks::<9>();
        if !tail.is_empty() {
            bail!("truncated number case");
        }
        for case in cases {
            let x = f64::from_le_bytes(case[..8].try_into().context("binary64 input")?);
            writeln!(
                out,
                "{}\t{}\t{}",
                number(x),
                to_fixed(x, case[8])?,
                inspect_number(x)
            )
            .context("write formatted number")?;
        }
    } else {
        let mut input = &buf[..];
        let mut rows = Vec::new();
        while !input.is_empty() {
            let mut size = [0; 4];
            input.read_exact(&mut size).context("read string size")?;
            let (bytes, rest) = input
                .split_at_checked(u32::from_le_bytes(size) as usize)
                .context("truncated string")?;
            let s = std::str::from_utf8(bytes).context("decode string")?;
            input = rest;
            if mode == "inspect" {
                writeln!(out, "{}", inspect_string(s)).context("write inspected string")?;
            }
            rows.push(vec![
                Value::from(s),
                Value::Number(-(rows.len() as f64) / 10.),
            ]);
        }
        if mode == "table" {
            writeln!(out, "{}", console_table(&["Условие", "Доля · –"], &rows)?)
                .context("write table")?;
        } else if mode == "log" {
            for row in rows {
                writeln!(out, "{}", console_log(&row)).context("write log")?;
            }
        } else if mode != "inspect" {
            bail!("unknown mode {mode}");
        }
    }
    out.flush().context("flush output")
}
