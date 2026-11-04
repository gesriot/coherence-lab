use crate::{
    Error, Scale,
    element::{Element, Kind},
    evaluation::{Average, EVAL_STEPS, Predictor, Ratio, held_out, probe_seeds},
    experiment::{TrainingOptions, train},
    input::{CopyMode, Gain, Motion, MovingScene, Observation, Vector, in_wall},
    probes::{Ridge, mean, r2, shuffled},
    report::{averages, fmt, line, pool, seconds},
};
use jscompat::{floor, max};
use lab::{Occlusion, Scenario, World, advance, contains};
use std::{fmt::Write, time::Instant};
const COPIES: [CopyMode; 3] = [CopyMode::Executed, CopyMode::Zero, CopyMode::Shuffled];
fn copy_name(c: CopyMode) -> &'static str {
    match c {
        CopyMode::Executed => "Копия",
        CopyMode::Zero => "Без копии",
        CopyMode::Shuffled => "Перемешанная",
    }
}
fn gain_name(g: Gain) -> &'static str {
    match g {
        Gain::Fixed => "Постоянный",
        Gain::Episodic => "Свой в эпизоде",
    }
}
fn transition(
    totals: &mut [Ratio; 5],
    t: u32,
    cmd: Vector,
    p: &[f64],
    current: &[f64],
    next: &[f64],
) {
    let (mut sum, mut copy) = (0., 0.);
    for ((p, x), y) in p.iter().zip(current).zip(next) {
        sum += (p - y) * (p - y);
        copy += (x - y) * (x - y);
    }
    totals[0].add(sum, copy);
    if cmd == [0.; 2] {
        totals[2].add(sum, copy);
    } else {
        totals[1].add(sum, copy);
        if t < 24 {
            totals[3].add(sum, copy);
        }
        if t >= 48 {
            totals[4].add(sum, copy);
        }
    }
}
fn target(w: &World, gaze: Vector, prev: Vector, grid: usize) -> Option<(usize, [f64; 3])> {
    let a = &w.objects[0];
    if w.t == 0 || a.vanish_at.is_some_and(|at| w.t >= at) {
        return None;
    }
    let half = grid as f64 / 2.;
    let size = 50. / grid as f64;
    let c = floor((a.x - gaze[0]) / size + half);
    let r = floor((a.y - gaze[1]) / size + half);
    if c < 0. || c >= grid as f64 || r < 0. || r >= grid as f64 {
        return None;
    }
    let x = gaze[0] + (c + 0.5 - half) * size;
    let y = gaze[1] + (r + 0.5 - half) * size;
    if in_wall(w, x, y) || !contains(a, x, y) {
        return None;
    }
    let own = gaze[0] - prev[1];
    let world = a.x - prev[0];
    Some((r as usize * grid + c as usize, [own, world, world - own]))
}
fn patch(i: usize, prev: &Observation, current: &Observation, grid: usize) -> Vec<f64> {
    let mut values = vec![0.; 56];
    let mut offset = 0;
    for o in [prev, current] {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let row = (i / grid) as isize + dy;
                let col = (i % grid) as isize + dx;
                if (0..grid as isize).contains(&row) && (0..grid as isize).contains(&col) {
                    let j = 3 * (row as usize * grid + col as usize);
                    values[offset..offset + 3].copy_from_slice(&o.x[j..j + 3]);
                }
                offset += 3;
            }
        }
    }
    values[54..].copy_from_slice(&prev.copy.expect("moving frames include a command channel"));
    values
}
type Group = (Scenario, u32);
struct Context {
    groups: Vec<(Group, Average)>,
    fallback: f64,
}
impl Context {
    fn fit(groups: &[Group], labels: &[f64]) -> Result<Self, Error> {
        if groups.len() != labels.len() || labels.is_empty() {
            return Err(Error::NoExamples);
        }
        let mut sums: Vec<(Group, Average)> = vec![];
        for (&key, &y) in groups.iter().zip(labels) {
            if let Some((_, s)) = sums.iter_mut().find(|x| x.0 == key) {
                s.add(y);
            } else {
                sums.push((key, Average { sum: y, n: 1 }));
            }
        }
        Ok(Self {
            groups: sums,
            fallback: mean(labels.iter().copied()),
        })
    }
    fn predict(&self, key: Group) -> f64 {
        self.groups
            .iter()
            .find(|x| x.0 == key)
            .map_or(self.fallback, |x| x.1.value())
    }
}
#[derive(Default)]
struct Samples {
    rows: [Vec<Vec<f64>>; 3],
    labels: [Vec<f64>; 3],
    groups: Vec<Group>,
}
struct Previous {
    obs: Observation,
    cmd: Vector,
    position: Vector,
}
fn collect(
    m: &Element,
    initial: &Element,
    settings: Motion,
    seeds: &[u32],
    mut totals: Option<&mut [Ratio; 5]>,
) -> Result<Samples, Error> {
    let mut data = Samples::default();
    for &seed in seeds {
        for sc in [Scenario::Continuity, Scenario::Vanish] {
            let mut scene = MovingScene::new(sc, seed, settings, m.grid);
            let (mut trained, mut fresh) = (Predictor::new(m), Predictor::new(initial));
            let mut previous: Option<Previous> = None;
            for t in 0..=EVAL_STEPS {
                let (g, cmd, obs) = scene.read();
                if let Some(p) = &previous
                    && let Some(v) = totals.as_deref_mut()
                {
                    transition(
                        v,
                        scene.world.t - 1,
                        p.cmd,
                        trained.prediction(),
                        &p.obs.x,
                        &obs.x,
                    );
                }
                let a = trained.step(&obs.x, obs.copy)?;
                let b = fresh.step(&obs.x, obs.copy)?;
                if let Some(p) = &previous
                    && let Some((cell, labels)) = target(&scene.world, g, p.position, m.grid)
                {
                    let i = cell * m.k;
                    data.rows[0].push(a.state[i..i + m.k].to_vec());
                    data.rows[1].push(b.state[i..i + m.k].to_vec());
                    data.rows[2].push(patch(cell, &p.obs, &obs, m.grid));
                    data.groups.push((sc, scene.world.t / 8));
                    for (v, y) in data.labels.iter_mut().zip(labels) {
                        v.push(y);
                    }
                }
                previous = Some(Previous {
                    obs,
                    cmd,
                    position: [scene.world.objects[0].x, g[0]],
                });
                if t < EVAL_STEPS {
                    advance(&mut scene.world);
                }
            }
        }
    }
    Ok(data)
}
#[derive(Clone, Copy, Default)]
struct Probe {
    learned: f64,
    initial: f64,
    raw: f64,
    shuffled: f64,
    context: f64,
}
impl Probe {
    fn gain(self) -> f64 {
        self.learned - max(max(self.initial, self.raw), self.context)
    }
}
struct MotionScore {
    ratios: [f64; 5],
    probes: [Probe; 3],
    train: usize,
    test: usize,
}
fn evaluate(
    m: &Element,
    initial: &Element,
    settings: Motion,
    training: &[u32],
    test: &[u32],
) -> Result<MotionScore, Error> {
    if training.iter().any(|s| test.contains(s)) {
        return Err(Error::SeedOverlap);
    }
    let tr = collect(m, initial, settings, training, None)?;
    let mut totals = [Ratio::default(); 5];
    let te = collect(m, initial, settings, test, Some(&mut totals))?;
    let ridges = tr
        .rows
        .iter()
        .map(|x| Ridge::new(x))
        .collect::<Result<Vec<_>, _>>()?;
    let mut probes = [Probe::default(); 3];
    for (i, p) in probes.iter_mut().enumerate() {
        let y = &tr.labels[i];
        let truth = &te.labels[i];
        let score = |source: usize, labels: &[f64]| {
            let fit = ridges[source].fit(labels);
            r2(
                truth,
                &te.rows[source]
                    .iter()
                    .map(|r| fit.predict(r))
                    .collect::<Vec<_>>(),
            )
        };
        *p = Probe {
            learned: score(0, y),
            initial: score(1, y),
            raw: score(2, y),
            shuffled: score(0, &shuffled(y)),
            context: if settings.random {
                let ctx = Context::fit(&tr.groups, y)?;
                r2(
                    truth,
                    &te.groups
                        .iter()
                        .map(|&g| ctx.predict(g))
                        .collect::<Vec<_>>(),
                )
            } else {
                0.
            },
        };
    }
    Ok(MotionScore {
        ratios: totals.map(Ratio::value),
        probes,
        train: tr.groups.len(),
        test: te.groups.len(),
    })
}
fn command_gate(
    executed: &[f64],
    zero: &[f64],
    shuffled: &[f64],
    n: usize,
) -> (f64, f64, usize, bool) {
    let avg = |v: &[f64]| mean(v.iter().copied());
    let (a, b, c) = (avg(executed), avg(zero), avg(shuffled));
    let wins = executed
        .iter()
        .zip(zero)
        .zip(shuffled)
        .filter(|((x, y), z)| x < y && x < z)
        .count();
    (
        a / b,
        a / c,
        wins,
        [executed.len(), zero.len(), shuffled.len()] == [n; 3]
            && a <= 0.85 * b
            && a <= 0.85 * c
            && wins == n,
    )
}
fn representation(probes: &[Probe], n: usize) -> (f64, usize, bool) {
    let gains: Vec<_> = probes.iter().map(|p| p.gain()).collect();
    let wins = gains.iter().filter(|&&v| v > 0.).count();
    let avg = mean(gains);
    (avg, wins, probes.len() == n && avg >= 0.10 && wins == n)
}
struct Run {
    motion: Motion,
    run: u32,
    s: MotionScore,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct MotionExperiment {
    pub scale: Scale,
    pub random_path: bool,
}
pub fn motion_experiment(
    s: MotionExperiment,
    progress: impl Fn(&str) + Sync,
) -> Result<String, Error> {
    s.scale.validate()?;
    let start = Instant::now();
    let gains = if s.random_path {
        &[Gain::Fixed][..]
    } else {
        &[Gain::Fixed, Gain::Episodic][..]
    };
    let mut jobs = vec![];
    for run in 1..=s.scale.runs {
        for &gain in gains {
            for copy in COPIES {
                jobs.push((
                    run as u32,
                    Motion {
                        gain,
                        copy,
                        random: s.random_path,
                    },
                ));
            }
        }
    }
    let rows = pool(&jobs, |&(run, mo)| {
        let make = || Element::new(run, s.scale.grid, s.scale.depth, true, Kind::Memory, true);
        let mut m = make();
        train(
            &mut m,
            s.scale,
            run,
            TrainingOptions {
                kind: Kind::Memory,
                motion: Some(mo),
                occlusion: Occlusion::PerCell,
                ..Default::default()
            },
            |n, _| {
                if n % 500 == 0 {
                    progress(&format!(
                        "{} · {} · {run}: {n}/{} эпизодов.",
                        gain_name(mo.gain),
                        copy_name(mo.copy),
                        s.scale.episodes
                    ));
                }
                Ok(())
            },
            None,
        )?;
        Ok(Run {
            motion: mo,
            run,
            s: evaluate(&m, &make(), mo, &probe_seeds(), &held_out())?,
        })
    })?;
    let mut out = format!(
        "{} · {} обучений по {} эпизодов. Подвижное окно {} × {}.\n",
        if s.random_path {
            "B3b · случайные пути"
        } else {
            "B3"
        },
        jobs.len(),
        s.scale.episodes,
        s.scale.grid,
        s.scale.grid
    );
    out += "Проверка на зёрнах 3001–3030, обучение зондов на 4001–4030. Ошибки относительно копии кадра.\n";
    let group = |gain, copy| {
        rows.iter()
            .filter(|r| r.motion.gain == gain && r.motion.copy == copy)
            .collect::<Vec<_>>()
    };
    for table in 0..4 {
        let title = if table == 0 {
            "Коэффициент · условие\tПрогон\tВсе кадры\tПосле сдвига\tБез сдвига\tРанние\tПоздние"
                .into()
        } else {
            format!(
                "{} (R²)\tПрогон\tОбученный\tНачальный\tСырой вход\t{}",
                [
                    "Собственный сдвиг",
                    "Смещение A в мире",
                    "Смещение A в окне"
                ][table - 1],
                if s.random_path {
                    "Только контекст\tПеремешанные метки\tПрирост над лучшей базой"
                } else {
                    "Перемешанные метки\tПрирост к начальному\tПрирост к сырому"
                }
            )
        };
        writeln!(out, "\n{title}").expect("write to string");
        for &g in gains {
            for c in COPIES {
                let name = format!("{} · {}", gain_name(g), copy_name(c));
                let mut matrix = vec![];
                for r in group(g, c) {
                    let v = if table == 0 {
                        r.s.ratios.to_vec()
                    } else {
                        let p = r.s.probes[table - 1];
                        if s.random_path {
                            vec![p.learned, p.initial, p.raw, p.context, p.shuffled, p.gain()]
                        } else {
                            vec![
                                p.learned,
                                p.initial,
                                p.raw,
                                p.shuffled,
                                p.learned - p.initial,
                                p.learned - p.raw,
                            ]
                        }
                    };
                    line(&mut out, &format!("{name}\t{}", r.run), &v);
                    matrix.push(v);
                }
                line(&mut out, &format!("{name}\tсреднее"), &averages(&matrix));
            }
        }
    }
    for &g in gains {
        let first = &group(g, CopyMode::Executed)[0].s;
        let [a, b, c] = COPIES.map(|c| {
            group(g, c)
                .iter()
                .map(|r| r.s.ratios[1])
                .collect::<Vec<_>>()
        });
        let (vs_zero, vs_shuffle, wins, pass) = command_gate(&a, &b, &c, s.scale.runs);
        writeln!(
            out,
            "\n{} · примеров зондов: обучение {}, проверка {}.",
            gain_name(g),
            first.train,
            first.test
        )
        .expect("write to string");
        writeln!(out,"Копия / без копии {}, копия / перемешанная {} (порог ≤ 0,85); ниже обоих контролей в {wins} из {} прогонов.",fmt(vs_zero),fmt(vs_shuffle),s.scale.runs).expect("write to string");
        writeln!(
            out,
            "Критерий «копия команды используется» {}.",
            if pass {
                "выполнен"
            } else {
                "не выполнен"
            }
        )
        .expect("write to string");
    }
    if s.random_path {
        out += "\nДвижение объекта в мире\tСредний прирост\tПрогонов выше трёх баз\tКритерий\n";
        for c in COPIES {
            let (avg, wins, pass) = representation(
                &group(Gain::Fixed, c)
                    .iter()
                    .map(|r| r.s.probes[1])
                    .collect::<Vec<_>>(),
                s.scale.runs,
            );
            writeln!(
                out,
                "{}\t{}\t{wins}/{}\t{}",
                copy_name(c),
                fmt(avg),
                s.scale.runs,
                if pass {
                    "выполнен"
                } else {
                    "не выполнен"
                }
            )
            .expect("write to string");
        }
        out += "\nПрогон\tРазность R² смещения A в мире: копия минус без копии\n";
        let mut differences = vec![];
        for (a, b) in group(Gain::Fixed, CopyMode::Executed)
            .into_iter()
            .zip(group(Gain::Fixed, CopyMode::Zero))
        {
            let d = a.s.probes[1].learned - b.s.probes[1].learned;
            differences.push(d);
            writeln!(out, "{}\t{}", a.run, fmt(d)).expect("write to string");
        }
        writeln!(out, "Среднее\t{}", fmt(mean(differences))).expect("write to string");
    }
    writeln!(out, "Время полного запуска: {} с.", seconds(start)).expect("write to string");
    Ok(out)
}

#[cfg(test)]
mod tests;
