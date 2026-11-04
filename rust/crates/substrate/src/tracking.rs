use crate::{
    Error, Scale,
    element::{Element, Kind},
    evaluation::{Average, OFFSETS, Predictor, Ratio, near, saturation},
    experiment::{TrainingOptions, train},
    input::{Detector, Event, WorldSettings, center, frame, world},
    persistence::offsets,
    probes::mean,
    report::{averages, fmt, line, pool, seconds},
};
use jscompat::{abs, max};
use lab::{Occlusion, Scenario, advance};
use std::{fmt::Write, time::Instant};

struct Episode {
    frames: Vec<Vec<f64>>,
    events: Vec<Event>,
    ghost: Vec<Vec<usize>>,
}
fn episodes(setting: WorldSettings, grid: usize, steps: usize, seeds: &[u32]) -> Vec<Episode> {
    let mut all = Vec::new();
    for &seed in seeds {
        for sc in [Scenario::Continuity, Scenario::Vanish] {
            let mut w = world(sc, seed, setting);
            let mut det = Detector::default();
            let mut last = None;
            let mut ep = Episode {
                frames: vec![],
                events: vec![],
                ghost: vec![],
            };
            for t in 0..=steps {
                let (e, visible) = det.read(&w, grid);
                if let Some(e) = e {
                    ep.events.push(e);
                }
                ep.frames.push(frame(&w, grid, Occlusion::PerCell, None));
                let a = &w.objects[0];
                if !visible.is_empty() {
                    last = Some(a.clone());
                }
                let cells = if let Some(last) = &last
                    && a.vanish_at.is_some_and(|at| w.t > at && w.t - at <= 10)
                {
                    (0..grid * grid)
                        .filter(|&i| {
                            let [x, y] = center(i, grid);
                            near(last, x, y)
                        })
                        .collect()
                } else {
                    vec![]
                };
                ep.ghost.push(cells);
                if t < steps {
                    advance(&mut w);
                }
            }
            all.push(ep);
        }
    }
    all
}
#[derive(Clone, Copy, Debug)]
struct Span {
    count: usize,
    min: usize,
    max: usize,
}
impl Default for Span {
    fn default() -> Self {
        Self {
            count: 0,
            min: usize::MAX,
            max: 0,
        }
    }
}
impl Span {
    fn add(&mut self, length: usize) {
        self.count += 1;
        self.min = self.min.min(length);
        self.max = self.max.max(length);
    }
    fn text(self) -> String {
        format!(
            "{} ({}–{})",
            self.count,
            if self.count == 0 {
                "Infinity".into()
            } else {
                self.min.to_string()
            },
            self.max
        )
    }
}
#[derive(Default)]
struct Totals {
    all: Ratio,
    changed: Ratio,
    event: [Ratio; 2],
    span: [Span; 2],
    ghost: Average,
    other: Average,
    after: Average,
    curve: [Average; 9],
}
fn other_rows(cells: &[usize], grid: usize) -> Vec<usize> {
    let mut columns = vec![];
    for i in cells {
        let col = i % grid;
        if !columns.contains(&col) {
            columns.push(col);
        }
    }
    (0..grid)
        .filter(|r| cells.iter().all(|i| r.abs_diff(i / grid) >= 3))
        .flat_map(|r| columns.iter().map(move |c| r * grid + c))
        .collect()
}
impl Totals {
    fn score(&mut self, ep: &Episode, pred: &[Vec<f64>], grid: usize) {
        for (t, p) in pred.iter().enumerate().skip(1) {
            let x = &ep.frames[t];
            let prev = &ep.frames[t - 1];
            for i in (0..x.len()).step_by(3) {
                let (mut sq, mut copy, mut changed) = (0., 0., false);
                for j in i..i + 3 {
                    let e = p[j] - x[j];
                    let delta = prev[j] - x[j];
                    sq += e * e;
                    copy += delta * delta;
                    changed |= abs(delta) > 0.05;
                }
                self.all.add(sq, copy);
                if changed {
                    self.changed.add(sq, copy);
                }
            }
            for &i in &ep.ghost[t] {
                for j in 3 * i..3 * i + 3 {
                    self.ghost.add(abs(p[j] - x[j]));
                }
            }
        }
        for e in &ep.events {
            let group = usize::from(!e.wall);
            self.span[group].add(e.length);
            let t = e.time;
            for &i in &e.cells {
                let start = 3 * i;
                for (d, &x) in ep.frames[t][start..start + 3].iter().enumerate() {
                    let delta = pred[t][start + d] - x;
                    let cp = ep.frames[t - 1][start + d] - x;
                    self.event[group].add(delta * delta, cp * cp);
                }
            }
            if !e.wall {
                continue;
            }
            if t >= 7 {
                for (j, &offset) in OFFSETS.iter().enumerate() {
                    if let Some(p) = pred.get((t as i32 + offset) as usize) {
                        for &i in &e.cells {
                            self.curve[j].add(saturation(p, i));
                        }
                    }
                }
            }
            let others = other_rows(&e.cells, grid);
            for p in &pred[t.saturating_sub(6).max(1)..=t] {
                for &i in &others {
                    self.other.add(saturation(p, i));
                }
            }
            for (p, x) in pred.iter().zip(&ep.frames).skip(t + 8).take(8) {
                for &i in &e.cells {
                    if saturation(x, i) <= 0.05 {
                        self.after.add(saturation(p, i));
                    }
                }
            }
        }
    }
}
#[derive(Clone, Debug, Default)]
struct TrackingScore {
    all: f64,
    changed: f64,
    wall: f64,
    gap: f64,
    ghost: f64,
    other: f64,
    after: f64,
    span: [Span; 2],
    curve: [f64; 9],
}
fn evaluate(m: &Element, episodes: &[Episode]) -> Result<TrackingScore, Error> {
    let mut totals = Totals::default();
    for e in episodes {
        let mut p = Predictor::new(m);
        let mut pred = vec![vec![]];
        for x in &e.frames[..e.frames.len() - 1] {
            pred.push(p.step(x, None)?.prediction.clone());
        }
        totals.score(e, &pred, m.grid);
    }
    Ok(TrackingScore {
        all: totals.all.value(),
        changed: totals.changed.value(),
        wall: totals.event[0].value(),
        gap: totals.event[1].value(),
        ghost: totals.ghost.value(),
        other: totals.other.value(),
        after: totals.after.value(),
        span: totals.span,
        curve: totals.curve.map(Average::value),
    })
}
#[derive(Clone, Copy, Debug)]
pub struct TrackingSettings {
    pub scale: Scale,
    pub wide_runs: usize,
    pub wide_weighted_episodes: usize,
    pub checkpoint: usize,
}
impl Default for TrackingSettings {
    fn default() -> Self {
        Self {
            scale: Scale::default(),
            wide_runs: 3,
            wide_weighted_episodes: 6000,
            checkpoint: 500,
        }
    }
}
#[derive(Clone, Copy)]
struct Job {
    columns: u8,
    weighted: bool,
    run: u32,
    episodes: usize,
}
struct Row {
    job: Job,
    score: TrackingScore,
    formation: Vec<(usize, f64, f64)>,
}
fn label(weighted: bool) -> &'static str {
    if weighted { "Вес ×50" } else { "База" }
}
fn criterion(base: &[f64], control: &[&TrackingScore], n: usize) -> ([f64; 5], bool) {
    let exit = mean(control.iter().map(|s| s.wall));
    let worst = control.iter().map(|s| s.wall).fold(f64::NEG_INFINITY, max);
    let other = mean(control.iter().map(|s| s.other));
    let after = mean(control.iter().map(|s| s.after));
    let baseline = mean(base.iter().copied());
    (
        [exit, worst, other, after, baseline],
        base.len() == n
            && control.len() == n
            && exit <= 0.5
            && worst <= 0.65
            && other <= 0.03
            && after <= 0.08
            && baseline >= 0.9,
    )
}
pub fn b1b(s: TrackingSettings, progress: impl Fn(&str) + Sync) -> Result<String, Error> {
    s.scale.validate()?;
    if s.wide_runs == 0 || s.wide_weighted_episodes == 0 || s.checkpoint == 0 {
        return Err(Error::Dimensions);
    }
    let start = Instant::now();
    let mut jobs = vec![];
    for columns in [1, 2, 4] {
        for weighted in [false, true] {
            for n in 1..=if columns == 4 {
                s.wide_runs
            } else {
                s.scale.runs
            } {
                jobs.push(Job {
                    columns,
                    weighted,
                    run: n as u32,
                    episodes: if columns == 4 && weighted {
                        s.wide_weighted_episodes
                    } else {
                        s.scale.episodes
                    },
                });
            }
        }
    }
    jobs.sort_by_key(|j| std::cmp::Reverse(j.episodes));
    let mut rows = pool(&jobs, |&j| {
        let ws = WorldSettings {
            columns: Some(j.columns),
            random: true,
        };
        let held = episodes(
            ws,
            s.scale.grid,
            s.scale.evaluation.steps,
            &s.scale.evaluation.test_seeds(),
        );
        let mut formation = vec![];
        let mut last = None;
        let mut m = Element::new(
            j.run,
            s.scale.grid,
            s.scale.depth,
            true,
            Kind::Memory,
            false,
        );
        train(
            &mut m,
            Scale {
                episodes: j.episodes,
                ..s.scale
            },
            j.run,
            TrainingOptions {
                occlusion: Occlusion::PerCell,
                kind: Kind::Memory,
                world: ws,
                weighted: j.weighted,
                ..Default::default()
            },
            |n, m| {
                if n % s.checkpoint == 0 || n == j.episodes {
                    let v = evaluate(m, &held)?;
                    formation.push((n, v.wall, v.all));
                    last = Some(v);
                    progress(&format!(
                        "Стена {} · {} · {}: {n}/{} эпизодов.",
                        j.columns,
                        label(j.weighted),
                        j.run,
                        j.episodes
                    ));
                }
                Ok(())
            },
            None,
        )?;
        Ok(Row {
            job: j,
            score: last.expect("final episode always measures tracking"),
            formation,
        })
    })?;
    rows.sort_by_key(|r| (r.job.columns, r.job.weighted, r.job.run));
    let mut groups = vec![];
    for w in [1, 2, 4] {
        for weighted in [false, true] {
            let matched: Vec<_> = rows
                .iter()
                .filter(|r| r.job.columns == w && r.job.weighted == weighted)
                .collect();
            let prefix = format!("{w} · {} · ", label(weighted));
            for &r in &matched {
                groups.push((format!("{prefix}{}", r.job.run), vec![r]));
            }
            groups.push((prefix + "среднее", matched));
        }
    }
    let mut out = format!(
        "B1b · {} обучений. Случайные пути, память {}, закрытие по клеткам.\n",
        jobs.len(),
        s.scale.depth
    );
    writeln!(
        out,
        "Проверочные зёрна {}. Ошибки относительно копии, призрак – абсолютная ошибка.",
        s.scale.evaluation.test_range()
    )
    .expect("writing to String is infallible");
    out += "Стена · условие · прогон\tВсе\tМеняющиеся\tПризрак\tВыход\tПропуск\tДругие строки\tПосле ухода\tВыходы (кадры)\tПропуски (кадры)\n";
    for (label, g) in &groups {
        let mean = averages(
            &g.iter()
                .map(|r| {
                    let s = &r.score;
                    vec![s.all, s.changed, s.ghost, s.wall, s.gap, s.other, s.after]
                })
                .collect::<Vec<_>>(),
        );
        let [wall, gap] = g[0].score.span;
        writeln!(
            out,
            "{label}\t{}\t{}\t{}",
            mean.into_iter().map(fmt).collect::<Vec<_>>().join("\t"),
            wall.text(),
            gap.text()
        )
        .expect("writing to String is infallible");
    }
    writeln!(out, "\nОжидание / шаг от выхода\t{}", offsets())
        .expect("writing to String is infallible");
    for (label, g) in &groups {
        line(
            &mut out,
            label,
            &averages(&g.iter().map(|r| r.score.curve.to_vec()).collect::<Vec<_>>()),
        );
    }
    out += "\nФормирование\tЭпизоды\tВыход\tВсе клетки\n";
    for (label, g) in &groups {
        for (i, p) in g[0].formation.iter().enumerate() {
            line(
                &mut out,
                &format!("{label}\t{}", p.0),
                &averages(
                    &g.iter()
                        .map(|r| vec![r.formation[i].1, r.formation[i].2])
                        .collect::<Vec<_>>(),
                ),
            );
        }
    }
    for w in [1, 2] {
        let base: Vec<_> = rows
            .iter()
            .filter(|r| r.job.columns == w && !r.job.weighted)
            .map(|r| r.score.wall)
            .collect();
        let ctrl: Vec<_> = rows
            .iter()
            .filter(|r| r.job.columns == w && r.job.weighted)
            .map(|r| &r.score)
            .collect();
        let (v, pass) = criterion(&base, &ctrl, s.scale.runs);
        let [exit, worst, other, after, base] = v.map(fmt);
        writeln!(out,"Стена {w} · выход ×50 {exit} (≤ 0,50), худший {worst} (≤ 0,65), другие строки {other} (≤ 0,03), после ухода {after} (≤ 0,08), база {base} (≥ 0,90).").expect("writing to String is infallible");
        writeln!(
            out,
            "Стена {w} · критерий «слежение выполнимо» {}.",
            if pass {
                "выполнен"
            } else {
                "не выполнен"
            }
        )
        .expect("writing to String is infallible");
    }
    writeln!(
        out,
        "Стена 4 · описательное условие без порога.\nВремя полного запуска: {} с.",
        seconds(start)
    )
    .expect("writing to String is infallible");
    Ok(out)
}

#[cfg(test)]
mod tests;
