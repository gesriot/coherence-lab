use crate::width_data::WIDTHS;
use regex::Regex;
use std::{fmt::Write, sync::LazyLock};
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FormatError {
    #[error("fraction digits must be between 0 and 100")]
    FractionDigits,
    #[error("table row {row} has more values than columns")]
    RowWidth { row: usize },
}

pub fn number(x: f64) -> String {
    ryu_js::Buffer::new().format(x).to_owned()
}
pub fn to_fixed(x: f64, digits: u8) -> Result<String, FormatError> {
    if digits > 100 {
        return Err(FormatError::FractionDigits);
    }
    if !x.is_finite() || x.abs() >= 1e21 {
        return Ok(number(x));
    }
    // ECMAScript rounds the exact integer |x| * 10^digits, choosing the larger tie.
    let bits = x.abs().to_bits();
    let exponent = ((bits >> 52) & 2047) as i32;
    let mantissa = (bits & ((1 << 52) - 1)) | (u64::from(exponent != 0) << 52);
    let shift = exponent.max(1) - 1023 - 52 + i32::from(digits);
    let mut scaled =
        num_bigint::BigUint::from(mantissa) * num_bigint::BigUint::from(5u8).pow(digits.into());
    if shift < 0 {
        scaled += num_bigint::BigUint::from(1u8) << (-shift - 1);
        scaled >>= -shift;
    } else {
        scaled <<= shift;
    }
    let mut s = scaled.to_str_radix(10);
    let places = usize::from(digits);
    if places > 0 {
        if s.len() <= places {
            s = format!("{}{}", "0".repeat(places + 1 - s.len()), s);
        }
        s.insert(s.len() - places, '.');
    }
    if x < 0.0 {
        s.insert(0, '-');
    }
    Ok(s)
}
pub fn inspect_number(x: f64) -> String {
    if x.to_bits() == (-0.0f64).to_bits() {
        "-0".into()
    } else {
        number(x)
    }
}

fn quoted(s: &[u16]) -> String {
    let single = 39u16;
    let q = if s.contains(&single) && !s.contains(&34) {
        '"'
    } else if s.contains(&single)
        && !s.contains(&96)
        && !s.windows(2).any(|a| a == [36, 123])
        && s.contains(&34)
    {
        '`'
    } else {
        '\''
    };
    let mut out = String::from(q);
    for c in char::decode_utf16(s.iter().copied()) {
        match c {
            Err(e) => {
                write!(out, "\\u{:04x}", e.unpaired_surrogate())
                    .expect("writing to a String is infallible");
            }
            Ok(c) => match c {
                '\x08' => out.push_str("\\b"),
                '\t' => out.push_str("\\t"),
                '\n' => out.push_str("\\n"),
                '\x0c' => out.push_str("\\f"),
                '\r' => out.push_str("\\r"),
                '\\' => out.push_str("\\\\"),
                c if c == q => {
                    out.push('\\');
                    out.push(c);
                }
                c if c.is_control() => {
                    write!(out, "\\x{:02X}", c as u32).expect("writing to a String is infallible");
                }
                _ => out.push(c),
            },
        }
    }
    out.push(q);
    out
}
pub fn inspect_string(s: &str) -> String {
    let utf16: Vec<_> = s.encode_utf16().collect();
    let n = utf16.len().min(10000);
    let mut out = String::new();
    if n > 76 && utf16[..n].contains(&10) {
        for (i, line) in utf16[..n].split_inclusive(|&c| c == 10).enumerate() {
            if i > 0 {
                out.push_str(" +\n  ");
            }
            out.push_str(&quoted(line));
        }
    } else {
        out = quoted(&utf16[..n]);
    }
    if utf16.len() > n {
        let rest = utf16.len() - n;
        write!(
            out,
            "... {rest} more character{}",
            if rest == 1 { "" } else { "s" }
        )
        .expect("writing to a String is infallible");
    }
    out
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Value {
    Number(f64),
    Text(String),
    Missing,
}
impl From<String> for Value {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}
impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Self::Text(s.into())
    }
}
impl From<f64> for Value {
    fn from(n: f64) -> Self {
        Self::Number(n)
    }
}

pub fn console_log(args: &[Value]) -> String {
    args.iter()
        .map(|v| match v {
            Value::Text(s) => s.clone(),
            Value::Number(x) => inspect_number(*x),
            Value::Missing => "undefined".into(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

// ANSI pattern from Node internal/util/inspect (ansi-regex, MIT, Sindre Sorhus).
static ANSI: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\x1b\x{9b}][\[\]()#;?]*(?:(?:(?:;[-a-zA-Z0-9/#&.:=?%@~_]+)*|[a-zA-Z0-9]+(?:;[-a-zA-Z0-9/#&.:=?%@~_]*)*)?(?:\x07|\x1b\\|\x{9c})|(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-PR-TZcf-nq-uy=><~])").expect("constant ANSI pattern is valid")
});
pub fn string_width(s: &str) -> usize {
    let s = ANSI.replace_all(s, "");
    let first = s.bytes().position(|b| b >= 127).unwrap_or(s.len());
    let ascii = s.as_bytes()[..first].iter().filter(|&&b| b >= 32).count();
    ascii
        + s[first..]
            .nfc()
            .map(|c| {
                let end = WIDTHS.partition_point(|&(start, _)| start <= c as u32);
                WIDTHS[end - 1].1
            })
            .sum::<usize>()
}

pub fn console_table(headers: &[&str], rows: &[Vec<Value>]) -> Result<String, FormatError> {
    let header: Vec<String> = std::iter::once("(index)".to_owned())
        .chain(headers.iter().map(|s| s.to_string()))
        .collect();
    let mut widths: Vec<_> = header.iter().map(|s| string_width(s)).collect();
    let mut body = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        if row.len() > headers.len() {
            return Err(FormatError::RowWidth { row: i });
        }
        let mut cells = vec![i.to_string()];
        for value in row {
            cells.push(match value {
                Value::Number(x) => inspect_number(*x),
                Value::Text(s) => quoted_table(s),
                Value::Missing => String::new(),
            });
        }
        cells.resize(header.len(), String::new());
        for (width, cell) in widths.iter_mut().zip(&cells) {
            *width = (*width).max(string_width(cell));
        }
        body.push(cells);
    }
    let border = |left, mid: &str, right| {
        format!(
            "{left}{}{right}",
            widths
                .iter()
                .map(|w| "─".repeat(w + 2))
                .collect::<Vec<_>>()
                .join(mid)
        )
    };
    let row = |cells: &[String]| {
        format!(
            "│ {} │",
            cells
                .iter()
                .zip(&widths)
                .map(|(s, w)| format!("{s}{}", " ".repeat(w - string_width(s))))
                .collect::<Vec<_>>()
                .join(" │ ")
        )
    };
    let mut lines = vec![border('┌', "┬", '┐'), row(&header), border('├', "┼", '┤')];
    lines.extend(body.iter().map(|cells| row(cells)));
    lines.push(border('└', "┴", '┘'));
    Ok(lines.join("\n"))
}
fn quoted_table(s: &str) -> String {
    // console.table uses compact inspection with an infinite break length.
    let utf16: Vec<_> = s.encode_utf16().collect();
    if utf16.len() <= 10000 {
        return quoted(&utf16);
    }
    let rest = utf16.len() - 10000;
    format!(
        "{}... {rest} more character{}",
        quoted(&utf16[..10000]),
        if rest == 1 { "" } else { "s" }
    )
}

// Intl.NumberFormat rounds the shortest decimal, not the exact binary fraction.
// Russian grouping uses NBSP and suppresses the sign of a rounded zero.
pub fn signed_tenth(x: f64) -> String {
    if x.is_nan() {
        return "не число".into();
    }
    if x.is_infinite() {
        return if x.is_sign_negative() { "-∞" } else { "+∞" }.into();
    }
    let shortest = number(x.abs());
    let (mantissa, exponent) = shortest
        .split_once('e')
        .map_or((shortest.as_str(), 0), |(m, e)| {
            (
                m,
                e.parse::<i32>().expect("number emits a decimal exponent"),
            )
        });
    let decimals = mantissa.split_once('.').map_or(0, |(_, f)| f.len() as i32);
    let digits = mantissa.replace('.', "");
    let mut n = num_bigint::BigUint::parse_bytes(digits.as_bytes(), 10)
        .expect("finite number has decimal digits");
    let shift = exponent + 1 - decimals;
    let ten = num_bigint::BigUint::from(10u8);
    if shift < 0 {
        let divisor = ten.pow((-shift) as u32);
        n = (n + &divisor / 2u8) / divisor;
    } else {
        n *= ten.pow(shift as u32);
    }
    if n == num_bigint::BigUint::from(0u8) {
        return "0".into();
    }
    let mut digits = n.to_str_radix(10);
    let fraction = digits.pop().expect("nonzero integer has digits");
    if digits.is_empty() {
        digits.push('0');
    }
    let mut result = String::from(if x < 0. { "-" } else { "+" });
    for (i, c) in digits.chars().enumerate() {
        if i != 0 && (digits.len() - i).is_multiple_of(3) {
            result.push(' ');
        }
        result.push(c);
    }
    if fraction != '0' {
        result.push(',');
        result.push(fraction);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decimal_ties_and_signed_zero() -> Result<(), FormatError> {
        assert_eq!(to_fixed(-2.5, 0)?, "-3");
        assert_eq!(to_fixed(-0.0, 2)?, "0.00");
        assert_eq!(inspect_number(-0.0), "-0");
        assert_eq!(
            to_fixed(1.3510625679457612e-17, 40)?,
            "0.0000000000000000135106256794576116487314"
        );
        assert!(to_fixed(1., 101).is_err());
        Ok(())
    }
    #[test]
    fn terminal_width_uses_icu_on_the_normalized_suffix() {
        assert_eq!(string_width("\x1b[31mЖ·–\x1b[0m"), 3);
        assert_eq!(string_width("е\u{308}東京"), 5);
        assert_eq!(string_width("👩‍🔬"), 4);
    }
}
