use super::decimal as d;
use anyhow::{Context, Result};
use jscompat::{Value, console_table, max, round};
use lab::*;
use std::fmt::Write;

pub(crate) fn table(out: &mut String, headers: &[&str], rows: Vec<Vec<Value>>) -> Result<()> {
    writeln!(
        out,
        "{}",
        console_table(headers, &rows).context("format experiment table")?
    )?;
    Ok(())
}
pub(crate) fn interval(e: Estimate, digits: u8) -> String {
    format!(
        "{} [{}; {}]",
        d(e.mean, digits),
        d(e.low, digits),
        d(e.high, digits)
    )
}
fn status(v: &[f64]) -> &'static str {
    if stable(v) {
        "устойчив"
    } else {
        "неустойчив"
    }
}
pub(crate) fn design(args: &[String]) -> Result<String> {
    let mut ids = Vec::new();
    for arg in args {
        let id = arg.parse()?;
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    if ids.is_empty() {
        ids.extend(CONTROLS);
    }
    let report = genesis_design(&ids);
    let mut out =
        String::from("Проверка выполнимости · прогоны 701–710 · оценочные зёрна 3001–3020\n");
    if ids.contains(&Control::Oracle) {
        writeln!(
            out,
            "Шаги обучения оракула: движение {}, существование {}.",
            d(report.rates.motion, 3),
            d(report.rates.existence, 3)
        )?;
    }
    for (id, curve) in &report.curves {
        writeln!(out, "{}", id.label())?;
        let rows = curve
            .iter()
            .map(|p| {
                vec![
                    f64::from(p.step).into(),
                    d(p.accuracy, 4).into(),
                    d(p.retention, 3).into(),
                    d(p.ghost_free, 3).into(),
                    p.existence.map(|v| d(v, 3)).join(" / ").into(),
                    p.weights.map(|x| d(x, 3)).join(" ").into(),
                ]
            })
            .collect();
        table(
            &mut out,
            &[
                "Эпизод",
                "Точность",
                "Удержание",
                "Без призрака",
                "λ_hidden / λ_open / m",
                "W по строкам",
            ],
            rows,
        )?;
    }
    if let Some(split) = report
        .curves
        .iter()
        .find(|(id, _)| *id == Control::Oracle)
        .and_then(|(_, v)| v.last()?.split)
    {
        for (label, p) in [
            "Движение оракула + идеальные вероятности",
            "Идеальное движение + вероятности оракула",
        ]
        .iter()
        .zip(split)
        {
            writeln!(
                out,
                "{label} · точность {} · удержание {} · без призрака {}",
                d(p[0], 4),
                d(p[1], 3),
                d(p[2], 3)
            )?;
        }
    }
    for (r, v) in &report.stability {
        writeln!(
            out,
            "Оракул · η существования {} · {} · {}",
            d(*r, 3),
            v.iter().map(|&x| d(x, 4)).collect::<Vec<_>>().join(" "),
            status(v)
        )?;
    }
    if CONTROLS.iter().all(|id| ids.contains(id)) {
        let [f, k, i, o] = CONTROLS.map(|id| {
            report
                .curves
                .iter()
                .find(|item| item.0 == id)
                .expect("all controls present")
                .1
                .iter()
                .map(|p| p.accuracy)
                .collect::<Vec<_>>()
        });
        let extra = report
            .stability
            .iter()
            .map(|v| v.1.clone())
            .collect::<Vec<_>>();
        writeln!(
            out,
            "{}",
            if permits(&f, &k, &i, &o, &extra) {
                "Постановка готова."
            } else {
                "Постановка не готова."
            }
        )?;
        if o[1] >= 0.9 {
            writeln!(
                out,
                "Оракул достиг порога уже после четырёх эпизодов. Возможно раннее насыщение."
            )?;
        }
    }
    Ok(out)
}
pub(crate) fn tune() -> Result<String> {
    let mut out =
        String::from("Подбор на первом порядке · прогоны 521–530 · оценочные зёрна 2041–2060\n");
    let range = GENESIS_TUNING;
    let mut best = -1.;
    let mut chosen = String::new();
    let units = ((STAGES.len() - 1) as u32 * 2 * EVAL_COUNT) as f64;
    for motion in MOTION_RATES {
        for existence in EXISTENCE_RATES {
            let rates = Rates::new(motion, existence);
            let successes = range.seeds().fold(0., |sum, seed| {
                sum + round(area(&trial(Condition::FirstOrder, seed, range.eval, rates)) * units)
            });
            let pair = format!(
                "η движения {}, η существования {}",
                d(motion, 3),
                d(existence, 3)
            );
            writeln!(
                out,
                "{pair} · площадь {}",
                d(successes / (f64::from(range.runs) * units), 6)
            )?;
            if successes > best {
                best = successes;
                chosen = pair;
            }
        }
    }
    writeln!(out, "Выбрано: {chosen}")?;
    Ok(out)
}

type Runs = Vec<Vec<Point>>;
fn accuracy(runs: &Runs, i: usize) -> Estimate {
    score_interval(&runs.iter().map(|v| v[i].accuracy).collect::<Vec<_>>(), 40.)
}

pub(crate) fn comparison() -> Result<String> {
    let range = GENESIS_COMPARISON;
    let mut experiments: Vec<(&str, Runs)> = CONTROLS
        .iter()
        .map(|&id| (id.label(), control_runs(id, ORACLE_RATES, range)))
        .collect();
    experiments.push((
        "Оракул (шаги участников)",
        control_runs(Control::Oracle, SELECTED_RATES, range),
    ));
    let scores: Vec<Vec<_>> = experiments
        .iter()
        .map(|(_, v)| (0..STAGES.len()).map(|i| accuracy(v, i).mean).collect())
        .collect();
    let ready = permits(&scores[0], &scores[1], &scores[2], &scores[3], &scores[4..]);
    let mut out = String::from("Проверки выполнимости\n");
    for (i, (label, _)) in experiments[..4].iter().enumerate() {
        let (v, text) = if i < 2 {
            (
                scores[i].iter().copied().fold(f64::NEG_INFINITY, max),
                "≤ 0,75 во всех точках",
            )
        } else {
            (scores[i][STAGES.len() - 1], "≥ 0,9 в последней точке")
        };
        writeln!(out, "{label} · {} · порог {text}", d(v, 3))?;
    }
    for i in [3, 4] {
        let (label, runs) = &experiments[i];
        writeln!(
            out,
            "{label} · {} · {}",
            scores[i]
                .iter()
                .map(|&v| d(v, 4))
                .collect::<Vec<_>>()
                .join(" "),
            status(&scores[i])
        )?;
        let below = runs
            .iter()
            .filter(|r| r[STAGES.len() - 1].accuracy < 0.9)
            .count();
        writeln!(
            out,
            "{label} · прогонов с итоговой точностью ниже 0,9: {below} из {}",
            range.runs
        )?;
    }
    if ready {
        for id in PARTICIPANTS {
            experiments.push((
                id.label(),
                range
                    .seeds()
                    .map(|seed| trial(id, seed, range.eval, SELECTED_RATES))
                    .collect(),
            ));
        }
        writeln!(
            out,
            "{}",
            if accuracy(&experiments[5].1, 1).mean >= 0.9
                && accuracy(&experiments[6].1, 1).mean >= 0.9
            {
                "Обучение слишком быстрое: задача не различает условия"
            } else {
                "Насыщения в точке 4 нет."
            }
        )?;
    } else {
        writeln!(
            out,
            "Сравнение не интерпретируется: проверка выполнимости не пройдена."
        )?;
    }
    writeln!(
        out,
        "Прогоны 61–90. Оценочные зёрна 1041–1060. Интервалы 95% по прогонам."
    )?;
    writeln!(
        out,
        "Шаги обучения оракула: {} / {}. Участники: {} / {}.",
        d(ORACLE_RATES.motion, 3),
        d(ORACLE_RATES.existence, 3),
        d(SELECTED_RATES.motion, 3),
        d(SELECTED_RATES.existence, 3)
    )?;
    let mut rows = vec![];
    for (name, runs) in &experiments {
        for (i, &stage) in STAGES.iter().enumerate() {
            let avg =
                |f: fn(Point) -> f64| Value::from(d(average(runs.iter().map(|r| f(r[i]))), 3));
            rows.push(vec![
                (*name).into(),
                f64::from(stage).into(),
                interval(accuracy(runs, i), 3).into(),
                avg(|p| p.retention),
                avg(|p| p.ghost_free),
                f64::from(runs.iter().map(|r| r[i].untracked).sum::<u32>()).into(),
                avg(|p| p.motion_error),
                avg(|p| p.hidden),
                avg(|p| p.open),
                avg(|p| p.miss),
            ]);
        }
    }
    table(
        &mut out,
        &[
            "Условие",
            "Эпизоды",
            "Точность [95%]",
            "Удержание A",
            "Без призрака",
            "Без трека",
            "Ошибка движения",
            "λ_hidden",
            "λ_open",
            "m",
        ],
        rows,
    )?;
    if ready {
        let full = &experiments[5].1;
        let mut effects = Vec::new();
        for (label, runs) in &experiments[6..] {
            let delta: Vec<_> = full
                .iter()
                .zip(runs)
                .map(|(f, r)| area(f) - area(r))
                .collect();
            let e = score_interval(&delta, 200.);
            writeln!(
                out,
                "Разность площадей · полный контур минус «{label}» · {}",
                interval(e, 3)
            )?;
            effects.push(effect(e));
        }
        writeln!(
            out,
            "Ускорение обучения контуром (Genesis): {}",
            effects[0].acceleration()
        )?;
        writeln!(out, "Вклад протокола внимания: {}", effects[1].protocol())?;
        writeln!(
            out,
            "Альтернативное объяснение правилами первого порядка: {}",
            replacement(effects[0], effects[2])
        )?;
        if effects[2] == Effect::Slower {
            writeln!(
                out,
                "Контур замедляет обучение относительно сильного первого порядка."
            )?;
        }
    }
    writeln!(
        out,
        "Обучаются модели движения и существования объекта. Заданы классификация клеток, геометрия сенсора, эфферентная копия, оценка освещения, связывание и оттенки, правила внимания и веса E."
    )?;
    Ok(out)
}
