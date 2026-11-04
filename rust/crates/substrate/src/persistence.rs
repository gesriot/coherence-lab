use crate::{
    Error, Scale,
    element::{Element, Kind},
    evaluation::{OFFSETS, Scores, evaluate},
    experiment::{TrainingOptions, train},
    probes::mean,
    report::{averages, fmt, line, pool, seconds},
};
use lab::Occlusion;
use std::{fmt::Write, time::Instant};
#[derive(Clone, Copy, Debug)]
pub struct PersistenceSettings {
    pub scale: Scale,
    pub control_runs: usize,
    pub wide_depth: usize,
}
impl Default for PersistenceSettings {
    fn default() -> Self {
        Self {
            scale: Scale::default(),
            control_runs: 3,
            wide_depth: 28,
        }
    }
}
impl PersistenceSettings {
    fn validate(self) -> Result<(), Error> {
        self.scale.validate()?;
        if self.control_runs == 0 || self.control_runs > self.scale.runs || self.wide_depth == 0 {
            Err(Error::Dimensions)
        } else {
            Ok(())
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    Sobel,
    Isotropic,
    Memory,
    Wide,
    Hint,
}
struct Job {
    family: Family,
    run: u32,
}
struct ResultRow {
    family: Family,
    run: u32,
    learned: Scores,
    initial: Option<Scores>,
    trained: Instant,
}
fn family_name(f: Family, s: PersistenceSettings) -> String {
    match f {
        Family::Sobel => format!("Собель {}", s.scale.depth),
        Family::Isotropic => "Без направлений".into(),
        Family::Memory => format!("Память {}", s.scale.depth),
        Family::Wide => format!("Память {}", s.wide_depth),
        Family::Hint => format!("Подсказка {}", s.scale.depth),
    }
}
fn jobs(order: &[Family], s: PersistenceSettings) -> Vec<Job> {
    order
        .iter()
        .flat_map(|&family| {
            (1..=if matches!(family, Family::Hint | Family::Isotropic) {
                s.control_runs
            } else {
                s.scale.runs
            })
                .map(move |n| Job {
                    family,
                    run: n as u32,
                })
        })
        .collect()
}
fn run_job(
    job: &Job,
    s: PersistenceSettings,
    occ: Occlusion,
    progress: &(impl Fn(&str) + Sync),
) -> Result<ResultRow, Error> {
    let directed = job.family != Family::Isotropic;
    let depth = if job.family == Family::Wide {
        s.wide_depth
    } else {
        s.scale.depth
    };
    let kind = match job.family {
        Family::Memory | Family::Wide => Kind::Memory,
        Family::Hint => Kind::Hinted,
        _ => Kind::Plain,
    };
    let make = || Element::new(job.run, s.scale.grid, depth, directed, kind, false);
    let mut m = make();
    let start = Instant::now();
    let name = family_name(job.family, s);
    train(
        &mut m,
        s.scale,
        job.run,
        TrainingOptions {
            kind,
            directed,
            occlusion: occ,
            ..TrainingOptions::default()
        },
        |n, _| {
            if occ == Occlusion::PerCell && n % 250 == 0 {
                progress(&format!(
                    "{name} · {}: {n}/{} эпизодов.",
                    job.run, s.scale.episodes
                ));
            }
            Ok(())
        },
        None,
    )?;
    let trained = Instant::now();
    if occ == Occlusion::WholeObject {
        progress(&format!(
            "{}, прогон {}: обучение {} с.",
            if directed {
                "Собель"
            } else {
                "Без направлений"
            },
            job.run,
            seconds(start)
        ));
    }
    let eval = |m: &Element| {
        evaluate(
            Some(m),
            s.scale.grid,
            occ,
            s.scale.evaluation.steps,
            &s.scale.evaluation.probe_seeds(),
            &s.scale.evaluation.test_seeds(),
        )
    };
    Ok(ResultRow {
        family: job.family,
        run: job.run,
        trained,
        learned: eval(&m)?,
        initial: if matches!(job.family, Family::Hint | Family::Isotropic) {
            None
        } else {
            Some(eval(&make())?)
        },
    })
}
fn viable(main: &[&Scores], initial: &[&Scores], runs: usize) -> bool {
    main.len() == runs
        && initial.len() == runs
        && mean(main.iter().map(|s| s.changed)) <= 0.8
        && main.iter().all(|s| s.changed <= 0.85)
        && initial.iter().all(|s| s.changed >= 1.)
        && mean(main.iter().map(|s| s.probes[0][0])) >= 0.75
}
pub fn b0(s: PersistenceSettings, progress: impl Fn(&str) + Sync) -> Result<String, Error> {
    s.validate()?;
    let start = Instant::now();
    let jobs = jobs(&[Family::Sobel, Family::Isotropic], s);
    let mut out = format!(
        "B0: {} независимых обучений по {} эпизодов × {} шагов.\n",
        jobs.len(),
        s.scale.episodes,
        s.scale.steps
    );
    let rows = pool(&jobs, |j| run_job(j, s, Occlusion::WholeObject, &progress))?;
    let raw = evaluate(
        None,
        s.scale.grid,
        Occlusion::WholeObject,
        s.scale.evaluation.steps,
        &s.scale.evaluation.probe_seeds(),
        &s.scale.evaluation.test_seeds(),
    )?;
    let main: Vec<_> = rows
        .iter()
        .filter(|r| r.family == Family::Sobel)
        .map(|r| &r.learned)
        .collect();
    let init: Vec<_> = rows.iter().filter_map(|r| r.initial.as_ref()).collect();
    let iso: Vec<_> = rows
        .iter()
        .filter(|r| r.family == Family::Isotropic)
        .map(|r| &r.learned)
        .collect();
    out += "Ошибки – доли ошибки копии; «Призрак» – средняя абсолютная ошибка по каналам.\nУ копии зонд читает три канала текущего кадра.\n";
    for curve in [false, true] {
        if curve {
            writeln!(out,"\nСредняя предсказанная насыщенность в клетках выхода\nУсловие / шаг от выхода\t{}",offsets()).expect("writing to String is infallible");
        } else {
            out += "Условие\tВсе клетки\tМеняющиеся\tВыход\tПризрак\tЗнак скорости\tСкрытый x (R²)\tПеремешано (R²)\n";
        }
        let values = |s: &Scores| {
            if curve {
                s.curve.to_vec()
            } else {
                vec![
                    s.all,
                    s.changed,
                    s.exit,
                    s.ghost,
                    s.probes[0][0],
                    s.probes[1][0],
                    s.probes[1][1],
                ]
            }
        };
        for (name, group) in [
            ("Собель", &main),
            ("Без обучения", &init),
            ("Без направлений", &iso),
        ] {
            let mut matrix = vec![];
            for (i, s) in group.iter().enumerate() {
                let v = values(s);
                line(&mut out, &format!("{name} {}", i + 1), &v);
                matrix.push(v);
            }
            line(&mut out, &format!("{name} · среднее"), &averages(&matrix));
            if name == "Собель" {
                line(&mut out, "Копия", &values(&raw));
            }
        }
    }
    out += "\nПроверка B0 по неокруглённым значениям:\n";
    writeln!(
        out,
        "Меняющиеся: среднее {} (≤ 0,80); худший прогон {} (≤ 0,85).",
        fmt(mean(main.iter().map(|s| s.changed))),
        fmt(main
            .iter()
            .map(|s| s.changed)
            .fold(f64::NEG_INFINITY, jscompat::max))
    )
    .expect("writing to String is infallible");
    writeln!(
        out,
        "Без обучения: минимум {} (≥ 1,0).",
        fmt(init
            .iter()
            .map(|s| s.changed)
            .fold(f64::INFINITY, jscompat::min))
    )
    .expect("writing to String is infallible");
    writeln!(
        out,
        "Зонд знака скорости: среднее {} (≥ 0,75).",
        fmt(mean(main.iter().map(|s| s.probes[0][0])))
    )
    .expect("writing to String is infallible");
    writeln!(
        out,
        "Жизнеспособность B0: {}.\nДля метрик постоянства порог не задан.",
        if viable(&main, &init, s.scale.runs) {
            "подтверждена"
        } else {
            "не подтверждена"
        }
    )
    .expect("writing to String is infallible");
    let last = rows
        .iter()
        .map(|r| r.trained)
        .max()
        .expect("B0 has directed runs");
    writeln!(
        out,
        "Время от старта до окончания всех обучений: {} с.\nПолный запуск с оценкой: {} с.",
        jscompat::to_fixed((last - start).as_secs_f64(), 1).expect("one decimal place"),
        seconds(start)
    )
    .expect("writing to String is infallible");
    Ok(out)
}
pub(crate) fn offsets() -> String {
    OFFSETS.map(|n| n.to_string()).join("\t")
}
fn partial(main: &[&Scores], runs: usize) -> (f64, f64, f64, f64, bool) {
    let exit = mean(main.iter().map(|s| s.exit));
    let worst = main
        .iter()
        .map(|s| s.exit)
        .fold(f64::NEG_INFINITY, jscompat::max);
    let at = mean(main.iter().map(|s| s.curve[6]));
    let before = mean(main.iter().map(|s| mean(s.curve[..3].iter().copied())));
    (
        exit,
        worst,
        at,
        before,
        main.len() == runs && exit <= 0.90 && worst <= 0.95 && at >= 0.03 && at > before,
    )
}
pub fn b1(s: PersistenceSettings, progress: impl Fn(&str) + Sync) -> Result<String, Error> {
    s.validate()?;
    let start = Instant::now();
    let jobs = jobs(
        &[Family::Wide, Family::Hint, Family::Memory, Family::Sobel],
        s,
    );
    let mut out = format!(
        "B1: {} обучений × {} эпизодов × {} шагов. Закрытие по клеткам.\n",
        jobs.len(),
        s.scale.episodes,
        s.scale.steps
    );
    let rows = pool(&jobs, |j| run_job(j, s, Occlusion::PerCell, &progress))?;
    let raw = evaluate(
        None,
        s.scale.grid,
        Occlusion::PerCell,
        s.scale.evaluation.steps,
        &s.scale.evaluation.probe_seeds(),
        &s.scale.evaluation.test_seeds(),
    )?;
    writeln!(
        out,
        "Оценка: {}; обучение зондов: {}. Ошибки относительно копии; призрак – абсолютная ошибка.",
        s.scale.evaluation.test_range(),
        s.scale.evaluation.probe_range()
    )
    .expect("writing to String is infallible");
    writeln!(
        out,
        "Возврат видимости: {} событий; в {} из них A касался стены во время скрытия.",
        raw.events, raw.wall_events
    )
    .expect("writing to String is infallible");
    for table in 0..5 {
        let title = match table {
            0 => {
                "Условие\tВсе клетки\tМеняющиеся\tВыход\tПризрак\tВыход после касания стены".into()
            }
            1 => format!("Насыщенность / шаг от выхода\t{}", offsets()),
            _ => format!(
                "{}\tЗонд\tНачальный\tСырой вход\tПеремешано\tПрирост",
                [
                    "Знак скорости (доля верных)",
                    "Скрытый x (R²)",
                    "Скрытый x с шага 4 (R²)"
                ][table - 2]
            ),
        };
        writeln!(out, "\n{title}").expect("writing to String is infallible");
        let values = |s: &Scores, initial: &Scores| match table {
            0 => vec![s.all, s.changed, s.exit, s.ghost, s.wall],
            1 => s.curve.to_vec(),
            _ => {
                let k = table - 2;
                vec![
                    s.probes[k][0],
                    initial.probes[k][0],
                    raw.probes[k][0],
                    s.probes[k][1],
                    s.probes[k][0] - initial.probes[k][0],
                ]
            }
        };
        line(&mut out, "Копия / сырой вход", &values(&raw, &raw));
        for init in [false, true] {
            if init && table >= 2 {
                break;
            }
            for f in [Family::Sobel, Family::Memory, Family::Wide, Family::Hint] {
                if init && f == Family::Hint {
                    continue;
                }
                let name = format!(
                    "{}{}",
                    if init { "Начальный " } else { "" },
                    family_name(f, s)
                );
                let mut matrix = vec![];
                for r in rows.iter().filter(|r| r.family == f) {
                    let initial = r
                        .initial
                        .as_ref()
                        .or_else(|| {
                            rows.iter()
                                .find(|x| x.family == Family::Memory && x.run == r.run)?
                                .initial
                                .as_ref()
                        })
                        .expect("hinted runs have paired initial memory models");
                    let v = values(if init { initial } else { &r.learned }, initial);
                    line(&mut out, &format!("{name} · {}", r.run), &v);
                    matrix.push(v);
                }
                line(&mut out, &format!("{name} · среднее"), &averages(&matrix));
            }
        }
    }
    out += "\nПодсказка при полном скрытии\tКлетки A\tПустая стена\tВыход\n";
    let mut matrix = vec![];
    for r in rows.iter().filter(|r| r.family == Family::Hint) {
        let v = vec![r.learned.hint[0], r.learned.hint[1], r.learned.exit];
        line(&mut out, &format!("Прогон {}", r.run), &v);
        matrix.push(v);
    }
    line(&mut out, "Среднее", &averages(&matrix));
    let (exit, worst, at, before, pass) = partial(
        &rows
            .iter()
            .filter(|r| r.family == Family::Memory)
            .map(|r| &r.learned)
            .collect::<Vec<_>>(),
        s.scale.runs,
    );
    writeln!(
        out,
        "Память {}: выход {} (≤ 0,90), худший {} (≤ 0,95).",
        s.scale.depth,
        fmt(exit),
        fmt(worst)
    )
    .expect("writing to String is infallible");
    writeln!(
        out,
        "Ожидание в точке 0: {} (≥ 0,03), среднее в −6…−4: {} (требуется рост).",
        fmt(at),
        fmt(before)
    )
    .expect("writing to String is infallible");
    writeln!(
        out,
        "Критерий частичного постоянства: {}.\nВремя полного запуска: {} с.",
        if pass {
            "выполнен"
        } else {
            "не выполнен"
        },
        seconds(start)
    )
    .expect("writing to String is infallible");
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gates_use_unrounded_scores_all_runs_and_separate_bounds() {
        let test = |changed, acc, init| {
            let s = Scores {
                changed,
                probes: [[acc, 0.]; 3],
                ..Scores::default()
            };
            let i = Scores {
                changed: init,
                ..Scores::default()
            };
            viable(&[&s; 5], &[&i; 5], 5)
        };
        assert!(test(0.8, 0.75, 1.));
        assert!(!test(0.80001, 0.75, 1.));
        assert!(!test(0.7, 0.75, 0.99999));
        assert!(!test(0.7, 0.74999, 1.));
        let good = Scores {
            changed: 0.6,
            probes: [[0.75, 0.]; 3],
            ..Scores::default()
        };
        let bad = Scores {
            changed: 0.85001,
            ..good.clone()
        };
        let initial = Scores {
            changed: 1.,
            ..good.clone()
        };
        assert!(!viable(
            &[&good, &good, &good, &good, &bad],
            &[&initial; 5],
            5
        ));
        let score = |exit, at, before| {
            let mut s = Scores {
                exit,
                ..Scores::default()
            };
            s.curve[..3].fill(before);
            s.curve[6] = at;
            s
        };
        assert!(partial(&[&score(0.90, 0.03, 0.02); 5], 5).4);
        for s in [
            score(0.90001, 0.03, 0.02),
            score(0.85, 0.02999, 0.02),
            score(0.85, 0.04, 0.04),
        ] {
            assert!(!partial(&[&s; 5], 5).4);
        }
        let low = score(0.8, 0.03, 0.02);
        assert!(!partial(&[&low, &low, &low, &low, &score(0.95001, 0.03, 0.02)], 5).4);
    }
}
