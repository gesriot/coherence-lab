import { availableParallelism } from "node:os";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { createElement } from "./element.ts";
import { exitWeightHook } from "./exit-events.ts";
import { evaluateTracking, trackingEpisodes, type TrackingScores } from "./tracking-evaluation.ts";
import { condition, reportTracking, type Job, type Point, type Result } from "./tracking-report.ts";
import { train } from "./training.ts";

type Message = { point: Point } | { result: Result };

function work(job: Job) {
  const model = createElement(job.run, true, 16, 16, "memory");
  const world = { columns: job.columns, path: "random" } as const;
  const heldOut = trackingEpisodes(world), formation: Point[] = [];
  let scores: TrackingScores | undefined;
  train(model, job.run, "per-cell", undefined, (done) => {
    if (done % 500 !== 0) return;
    scores = evaluateTracking(model, heldOut);
    const point = { episode: done, wall: scores.wall, all: scores.all };
    formation.push(point);
    parentPort!.postMessage({ point } satisfies Message);
  }, {
    world, episodes: job.episodes, weighting: job.weighted ? "wall-exits" : "uniform",
    weightHook: job.weighted ? exitWeightHook : undefined,
  });
  if (!scores || formation.at(-1)?.episode !== job.episodes) throw new Error("Нет конечной оценки");
  parentPort!.postMessage({ result: { ...job, scores, formation } } satisfies Message);
}

function launch(job: Job): Promise<Result> {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL(import.meta.url), { workerData: job });
    let result: Result | undefined;
    w.on("message", (m: Message) => {
      if ("result" in m) result = m.result;
      else console.error(`Стена ${job.columns} · ${condition(job.weighted)} · ${job.run}: ${m.point.episode}/${job.episodes} эпизодов.`);
    });
    w.on("error", reject);
    w.on("exit", (code) => {
      if (!code && result) resolve(result);
      else reject(new Error(`Незавершённый прогон ${job.run}, код выхода ${code}`));
    });
  });
}

async function main() {
  const start = performance.now(), jobs: Job[] = [], results: Result[] = [];
  for (const columns of [1, 2, 4] as const) {
    for (const weighted of [false, true]) {
      for (let run = 1; run <= (columns === 4 ? 3 : 5); run++) {
        jobs.push({ columns, weighted, run, episodes: columns === 4 && weighted ? 6000 : 2000 });
      }
    }
  }
  jobs.sort((a, b) => b.episodes - a.episodes);
  console.log(`B1b · ${jobs.length} обучений. Случайные пути, память 16, закрытие по клеткам.`);
  await Promise.all(Array.from({ length: Math.min(8, availableParallelism()) }, async () => {
    for (let job = jobs.shift(); job; job = jobs.shift()) results.push(await launch(job));
  }));
  reportTracking(results);
  console.log(`Время полного запуска: ${((performance.now() - start) / 1000).toFixed(1)} с.`);
}

if (isMainThread) await main();
else work(workerData as Job);
