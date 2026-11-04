mod gain;
mod genesis;
use anyhow::{Context, Result, bail};
use jscompat::{Value, console_log, console_table, to_fixed};
use lab::{BenchReport, Proportion, ReplayResult, Verdict, run_bench};
use std::io::{self, Write};

fn decimal(value: f64, digits: u8) -> String {
    to_fixed(value, digits)
        .expect("bench fields use at most three fractional digits")
        .replace('.', ",")
}
fn rate(p: Proportion) -> String {
    format!(
        "{}/{} · {}% [{}–{}%]",
        p.k,
        p.n,
        decimal(100. * p.rate, 1),
        decimal(100. * p.low, 1),
        decimal(100. * p.high, 1)
    )
}
fn replay_description(r: &ReplayResult) -> String {
    let detection = if r.detected {
        "обнаружен"
    } else {
        "не обнаружен"
    };
    format!(
        "{}: {detection} (AUC {}). Средняя ошибка в живом эпизоде / при повторе: {} / {}. Медиана парной разницы: {}. Пар с большей ошибкой при повторе: {}.",
        r.label,
        decimal(r.auc, 3),
        decimal(r.live_error, 3),
        decimal(r.replay_error, 3),
        decimal(r.median_difference, 3),
        rate(r.higher)
    )
}
fn render(report: BenchReport) -> Result<String> {
    let mut lines = vec![
        format!(
            "Удержание A проверяется за 72 шага, различение пары – за 200. На {} конфигурациях полный контур должен опередить каждый из трёх контролей («Первый порядок», «Только запись», «Лишняя память») более чем на 20 п. п. хотя бы по одной задаче. Тот же порог применяется отдельно к сравнению с «Новизной» для проверки альтернативного объяснения. Оценщик обнаруживает повтор по ошибке прогноза за шаги 17–40 при AUC ≥ 0,9.",
            report.configurations
        ),
        String::from(
            "Возраст трека – число шагов с момента рождения гипотезы. В этом эксперименте он считается полем первого порядка.",
        ),
        String::from("Доли успеха и 95%-е интервалы Уилсона"),
    ];
    let rows = report
        .rows
        .iter()
        .map(|r| {
            vec![
                Value::from(r.label),
                rate(r.retention).into(),
                rate(r.pair).into(),
                format!("{}%", decimal(r.explore_rate * 100., 2)).into(),
                r.self_error
                    .map_or_else(|| "–".into(), |v| decimal(v, 3))
                    .into(),
                decimal(r.action_error, 3).into(),
            ]
        })
        .collect::<Vec<_>>();
    lines.push(
        console_table(
            &[
                "Условие",
                "Удержание A",
                "Пара различена",
                "Доля шагов обхода",
                "Ошибка самомодели",
                "Ошибка действия",
            ],
            &rows,
        )
        .context("format benchmark table")?,
    );
    let verdict = match report.causal {
        Verdict::Supported => "подтверждена на этих конфигурациях",
        Verdict::Mixed => "смешанный результат",
        Verdict::Unsupported => "не подтверждена",
        _ => bail!("unsupported causal verdict"),
    };
    lines.push(console_log(&[
        "Причинная роль модели второго порядка:".into(),
        verdict.into(),
    ]));
    let alternative = if report.alternative_excluded {
        "исключено"
    } else {
        "не исключено"
    };
    lines.push(console_log(&[
        "Альтернативное объяснение правилом первого порядка:".into(),
        alternative.into(),
    ]));
    lines.extend(report.saturation);
    if report.unequal_exploration {
        lines.push("Доли шагов обхода у полного контура и первого порядка различаются более чем в полтора раза.".into());
    }
    lines.push("Обнаружение повтора по собственному прогнозу".into());
    lines.extend(report.replay.iter().map(replay_description));
    if report.replay.iter().all(|r| r.detected) {
        lines.push("Оценщик обнаружил повтор в обоих условиях. Различение повтора по этой ошибке – свойство прогноза первого порядка.".into());
    }
    Ok(lines.join("\n"))
}
fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1).map(|arg| {
        arg.into_string()
            .ok()
            .context("argument is not valid UTF-8")
    });
    let command = args.next().transpose()?.context("missing subcommand")?;
    let args: Vec<_> = args.collect::<Result<_>>()?;
    let output = match command.as_str() {
        "bench" if args.is_empty() => {
            format!("{}\n", render(run_bench()).context("render benchmark")?)
        }
        "substrate-b1b" => {
            substrate::b1b(Default::default(), |m| eprintln!("{m}")).context("run B1b")?
        }
        "substrate-b3" | "substrate-b3b" => substrate::motion_experiment(
            substrate::MotionExperiment {
                random_path: command == "substrate-b3b",
                ..Default::default()
            },
            |m| eprintln!("{m}"),
        )
        .context("run motion experiment")?,
        "substrate-b0" => {
            substrate::b0(Default::default(), |m| eprintln!("{m}")).context("run B0")?
        }
        "substrate-b1" => {
            substrate::b1(Default::default(), |m| eprintln!("{m}")).context("run B1")?
        }
        "genesis-feasibility" => genesis::design(&args)?,
        "genesis-tune" => genesis::tune()?,
        "genesis" => genesis::comparison()?,
        "sensorimotor-feasibility" => gain::design(&args)?,
        "sensorimotor-tune" => gain::tune()?,
        "sensorimotor" => gain::comparison(&args)?,
        _ => bail!("unknown command or arguments: {command}"),
    };
    io::stdout()
        .lock()
        .write_all(output.as_bytes())
        .context("write command output")
}
