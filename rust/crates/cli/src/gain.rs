use crate::{
    decimal as d,
    genesis::{interval, table},
};
use anyhow::{Result, bail};
use lab::*;
use std::fmt::Write;
fn fixed(out: &mut String, p: &FixedPoint) -> Result<()> {
    writeln!(
        out,
        "Неподвижная точка · среднее r* − 1 при верной оценке: {} · предел по модулю {} · информативных осей {}",
        d(p.bias, 6),
        d(p.limit, 6),
        p.informative
    )?;
    writeln!(
        out,
        "Направление при исходной оценке: {} из {} ({}%) · порог 80%",
        p.correct,
        p.directional,
        d(100. * p.direction, 2)
    )?;
    writeln!(
        out,
        "{}",
        if p.passes() {
            "Проверка неподвижной точки пройдена."
        } else {
            "Проверка неподвижной точки не пройдена."
        }
    )?;
    Ok(())
}
fn stability(out: &mut String, s: &GainSummary) -> Result<()> {
    writeln!(
        out,
        "{}",
        if stable(&s.curve.iter().map(|p| p.q).collect::<Vec<_>>()) {
            "Кривая устойчива."
        } else {
            "Кривая неустойчива."
        }
    )?;
    Ok(())
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
        ids.extend(GAIN_CONTROLS);
    }
    let mut out = String::from(
        "Проверка сенсомоторной зависимости · прогоны 801–810 · оценочные зёрна 3101–3120\n«Хранить всё» совпадает с «Без обучения»: окно считается расположенным там, куда скомандовал агент.\n",
    );
    let mut results = vec![];
    for id in ids {
        for &rate in if id.trains() {
            &GAIN_RATES[..]
        } else {
            &[0.][..]
        } {
            let s = gain_summary(&gain_runs(id, rate, GAIN_DESIGN));
            if rate != 0. {
                writeln!(out, "{} · η усиления {}", id.label(), d(rate, 3))?;
            } else {
                writeln!(out, "{}", id.label())?;
            }
            let rows = s
                .curve
                .iter()
                .map(|p| {
                    vec![
                        f64::from(p.step).into(),
                        d(p.q, 5).into(),
                        d(p.error, 5).into(),
                        d(p.gain[0], 5).into(),
                        d(p.gain[1], 5).into(),
                    ]
                })
                .collect();
            table(
                &mut out,
                &["Шаг", "q", "Ошибка калибровки", "Оценка X", "Оценка Y"],
                rows,
            )?;
            writeln!(
                out,
                "Итоговая точность {} · удержание A {} · без призрака {}",
                d(s.accuracy, 4),
                d(s.retention, 3),
                d(s.ghost_free, 3)
            )?;
            if id == GAIN_CONTROLS[2] {
                stability(&mut out, &s)?;
            }
            results.push((id, rate, s));
        }
    }
    let get = |id| results.iter().find(|r| r.0 == id).map(|r| &r.2);
    let oracles: Vec<_> = results.iter().filter(|r| r.0 == GAIN_CONTROLS[2]).collect();
    if let (Some(f), Some(i)) = (get(GAIN_CONTROLS[0]), get(GainCondition::Ideal))
        && oracles.len() == GAIN_RATES.len()
    {
        let p = fixed_point(GAIN_DESIGN);
        fixed(&mut out, &p)?;
        let ready =
            p.passes() && controls_pass(f, i, &oracles.iter().map(|r| &r.2).collect::<Vec<_>>());
        writeln!(
            out,
            "{}",
            if ready {
                "Критерий готовности пройден."
            } else {
                "Критерий готовности не пройден."
            }
        )?;
        if ready {
            let scores = oracles
                .iter()
                .map(|r| (r.1, r.2.curve[GAIN_STAGES.len() - 1].q))
                .collect::<Vec<_>>();
            writeln!(
                out,
                "Собственный шаг оракула по итоговому q: {}",
                d(design_gain_rate(&scores), 3)
            )?;
        }
        writeln!(
            out,
            "{}",
            if oracles.iter().all(|r| r.2.curve[1].q >= 0.9) {
                "Насыщение в первой точке обучения при всех шагах."
            } else {
                "Насыщения при всех шагах в первой точке обучения нет."
            }
        )?;
    }
    Ok(out)
}
pub(crate) fn tune() -> Result<String> {
    let mut out =
        String::from("Подбор на первом порядке · прогоны 541–550 · оценочные зёрна 2081–2100\n");
    let mut best = f64::NEG_INFINITY;
    let mut chosen = GAIN_RATES[0];
    for rate in GAIN_RATES {
        let runs = gain_runs(
            GainCondition::Training(Condition::FirstOrder),
            rate,
            GAIN_TUNING,
        );
        let score = average(runs.iter().map(|r| gain_area(&r.curve)));
        writeln!(out, "η усиления {} · площадь q {}", d(rate, 3), d(score, 6))?;
        if score > best {
            best = score;
            chosen = rate;
        }
    }
    writeln!(out, "Выбран шаг усиления {}.", d(chosen, 3))?;
    Ok(out)
}
fn print_curve(out: &mut String, label: &str, runs: &[GainRun], s: &GainSummary) -> Result<()> {
    writeln!(out, "{label}")?;
    let rows = s
        .curve
        .iter()
        .enumerate()
        .map(|(i, p)| {
            vec![
                f64::from(p.step).into(),
                interval(
                    bootstrap(&runs.iter().map(|r| r.curve[i].q).collect::<Vec<_>>()),
                    4,
                )
                .into(),
                d(p.error, 5).into(),
                d(p.gain[0], 5).into(),
                d(p.gain[1], 5).into(),
            ]
        })
        .collect();
    table(
        out,
        &[
            "Шаг",
            "q [95%]",
            "Ошибка калибровки",
            "Оценка X",
            "Оценка Y",
        ],
        rows,
    )?;
    let accuracy = score_interval(
        &runs
            .iter()
            .map(|r| r.secondary.accuracy)
            .collect::<Vec<_>>(),
        40.,
    );
    writeln!(
        out,
        "Итоговая точность {} · удержание A {} · без призрака {}",
        interval(accuracy, 4),
        d(s.retention, 3),
        d(s.ghost_free, 3)
    )?;
    writeln!(
        out,
        "Информативных пар на эпизод: {} · информативных выборок из памяти: {}",
        d(s.pairs, 3),
        d(s.draws, 3)
    )?;
    writeln!(out, "Саккад на эпизод: {}", d(s.saccades, 3))?;
    Ok(())
}
fn js_whitespace(c: char) -> bool {
    matches!(c,'\u{0009}'..='\u{000d}'|' '|'\u{a0}'|'\u{1680}'|'\u{2000}'..='\u{200a}'|'\u{2028}'|'\u{2029}'|'\u{202f}'|'\u{205f}'|'\u{3000}'|'\u{feff}')
}
pub(crate) fn comparison(args: &[String]) -> Result<String> {
    if args.len() > 1
        || args.first().is_some_and(|s| {
            s.replacen(',', ".", 1)
                .trim_matches(js_whitespace)
                .parse::<f64>()
                .ok()
                != Some(GAIN_SELECTED)
        })
    {
        bail!("Для сравнения зафиксирован шаг {}.", d(GAIN_SELECTED, 3));
    }
    let mut out = String::from("Прогоны 121–150 · оценочные зёрна 1081–1100\n");
    let p = fixed_point(GAIN_COMPARISON);
    fixed(&mut out, &p)?;
    let [frozen, ideal, oracle, _] = GAIN_CONTROLS;
    let controls = [
        (frozen, 0.),
        (ideal, 0.),
        (oracle, GAIN_ORACLE),
        (oracle, GAIN_SELECTED),
    ]
    .map(|(id, rate)| gain_runs(id, rate, GAIN_COMPARISON));
    let summaries = controls.each_ref().map(|r| gain_summary(r));
    for (i, label) in [
        frozen.label().to_owned(),
        ideal.label().to_owned(),
        format!("Оракул · η усиления {}", d(GAIN_ORACLE, 3)),
        format!(
            "Оракул (шаг участников) · η усиления {}",
            d(GAIN_SELECTED, 3)
        ),
    ]
    .iter()
    .enumerate()
    {
        print_curve(&mut out, label, &controls[i], &summaries[i])?;
        if i >= 2 {
            stability(&mut out, &summaries[i])?;
        }
    }
    if let Some(result) = compare_gain(summaries.each_ref(), &p, |id| {
        gain_runs(GainCondition::Training(id), GAIN_SELECTED, GAIN_COMPARISON)
    }) {
        writeln!(
            out,
            "Допуск по собственному сигналу, калибровке и итоговой точности пройден."
        )?;
        for (id, runs) in &result.participants {
            print_curve(&mut out, id.label(), runs, &gain_summary(runs))?;
        }
        writeln!(
            out,
            "{}",
            if result.saturated {
                "Обучение слишком быстрое: задача не различает условия."
            } else {
                "Насыщения в точке 10 шагов нет."
            }
        )?;
        for (id, e, _) in &result.differences {
            writeln!(
                out,
                "Разность площадей · полный контур минус «{}» · {}",
                id.label(),
                interval(*e, 4)
            )?;
        }
        let first = result.differences[0].2;
        writeln!(
            out,
            "Ускорение обучения контуром (сенсомоторная зависимость): {}",
            first.acceleration()
        )?;
        writeln!(
            out,
            "Вклад протокола внимания: {}",
            result.differences[1].2.protocol()
        )?;
        writeln!(
            out,
            "Альтернативное объяснение правилами первого порядка: {}",
            replacement(first, result.differences[2].2)
        )?;
    } else {
        writeln!(
            out,
            "Сравнение не интерпретируется: проверка выполнимости не пройдена."
        )?;
    }
    Ok(out)
}
