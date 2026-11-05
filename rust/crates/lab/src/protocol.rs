use crate::{BenchRow, Policy, Proportion, Verdict};
use jscompat::to_fixed;

pub fn decimal(value: impl Into<Option<f64>>, digits: u8) -> String {
    let Some(value) = value.into() else {
        return "–".into();
    };
    to_fixed(value, digits)
        .expect("bench fields use at most three fractional digits")
        .replace('.', ",")
}
pub fn format_rate(p: Proportion) -> String {
    format!(
        "{}/{} · {}% [{}–{}%]",
        p.k,
        p.n,
        decimal(100. * p.rate, 1),
        decimal(100. * p.low, 1),
        decimal(100. * p.high, 1)
    )
}

pub const AGE_ASSUMPTION: &str = "Возраст трека – число шагов с момента рождения гипотезы. В этом эксперименте он считается полем первого порядка.";
pub const SHARED_DETECTION: &str = "Оценщик обнаружил повтор в обоих условиях. Различение повтора по этой ошибке – свойство прогноза первого порядка.";
pub fn criterion(configurations: u32) -> String {
    format!(
        "Удержание A проверяется за 72 шага, различение пары – за 200. На {} конфигурациях полный контур должен опередить каждый из трёх контролей («Первый порядок», «Только запись», «Лишняя память») более чем на 20 п. п. хотя бы по одной задаче. Тот же порог применяется отдельно к сравнению с «Новизной» для проверки альтернативного объяснения. Оценщик обнаруживает повтор по ошибке прогноза за шаги 17–40 при AUC ≥ 0,9.",
        configurations
    )
}
pub fn causal_label(verdict: Verdict) -> &'static str {
    match verdict {
        Verdict::Supported => "подтверждена на этих конфигурациях",
        Verdict::Mixed => "смешанный результат",
        Verdict::Unsupported => "не подтверждена",
    }
}
pub fn exploration_note(rows: &[BenchRow]) -> &'static str {
    let full = rows.iter().find(|r| r.id == Policy::Full);
    let first = rows.iter().find(|r| r.id == Policy::FirstOrder);
    if let (Some(a), Some(b)) = (full, first)
        && a.explore_rate.max(b.explore_rate) > 1.5 * a.explore_rate.min(b.explore_rate)
    {
        "Доли шагов обхода у полного контура и первого порядка различаются более чем в полтора раза."
    } else {
        ""
    }
}
pub fn points(delta: f64) -> String {
    jscompat::signed_tenth(delta * 100.)
}
