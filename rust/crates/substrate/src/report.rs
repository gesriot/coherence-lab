use crate::{Error, probes::mean};
use std::{
    fmt::Write,
    sync::atomic::{AtomicUsize, Ordering},
    thread,
    time::Instant,
};
pub(crate) fn fmt(v: f64) -> String {
    jscompat::to_fixed(v, 4)
        .expect("report precision is four fractional digits")
        .replace('.', ",")
}
pub(crate) fn seconds(start: Instant) -> String {
    jscompat::to_fixed(start.elapsed().as_secs_f64(), 1)
        .expect("timing precision is one fractional digit")
}
pub(crate) fn line(out: &mut String, label: &str, values: &[f64]) {
    writeln!(
        out,
        "{label}\t{}",
        values
            .iter()
            .map(|&v| fmt(v))
            .collect::<Vec<_>>()
            .join("\t")
    )
    .expect("writing to string is infallible");
}
pub(crate) fn averages(rows: &[Vec<f64>]) -> Vec<f64> {
    (0..rows[0].len())
        .map(|i| mean(rows.iter().map(|x| x[i])))
        .collect()
}
pub(crate) fn pool<J: Sync, R: Send>(
    jobs: &[J],
    work: impl Fn(&J) -> Result<R, Error> + Sync,
) -> Result<Vec<R>, Error> {
    let next = AtomicUsize::new(0);
    let count = thread::available_parallelism()
        .map_or(1, usize::from)
        .min(8)
        .min(jobs.len());
    thread::scope(|scope| {
        let mut workers = vec![];
        for _ in 0..count {
            workers.push(scope.spawn(|| {
                let mut done = vec![];
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(index) else { break };
                    done.push((index, work(job)?));
                }
                Ok::<_, Error>(done)
            }));
        }
        let mut completed = vec![];
        for handle in workers {
            completed.extend(handle.join().map_err(|_| Error::Worker)??);
        }
        completed.sort_by_key(|row| row.0);
        Ok(completed.into_iter().map(|(_, r)| r).collect())
    })
}
