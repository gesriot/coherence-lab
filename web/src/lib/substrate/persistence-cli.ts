import { availableParallelism } from "node:os";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { createElement } from "./element.ts";
import { occupancyHook } from "./occupancy.ts";
import { evaluatePersistence } from "./persistence-probes.ts";
import { reportPersistence, variants, type Job, type Result, type Variant } from "./persistence-report.ts";
import { EPISODES, STEPS, train } from "./training.ts";

type Message = { done: number } | { result: Result };

function work(job: Job) {
  const { kind, depth } = variants[job.variant];
  const model = createElement(job.run, true, 16, depth, kind);
  train(model, job.run, "per-cell", kind === "hinted" ? occupancyHook : undefined, (done) => {
    if (done % 250 === 0) parentPort!.postMessage({ done } satisfies Message);
  });
  const learned = evaluatePersistence(model);
  const initial = kind === "hinted" ? undefined : evaluatePersistence(createElement(job.run, true, 16, depth, kind));
  parentPort!.postMessage({ result: { ...job, learned, initial } } satisfies Message);
}

function launch(job: Job): Promise<Result> {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL(import.meta.url), { workerData: job });
    let result: Result | undefined;
    w.on("message", (m: Message) => {
      if ("result" in m) result = m.result;
      else console.error(`${variants[job.variant].name} · ${job.run}: ${m.done}/${EPISODES} эпизодов.`);
    });
    w.on("error", reject);
    w.on("exit", (code) => {
      if (code === 0 && result) resolve(result);
      else reject(new Error(`Незавершённый прогон ${variants[job.variant].name} · ${job.run}, код выхода ${code}`));
    });
  });
}

async function main() {
  const start = performance.now(), results: Result[] = [];
  const order: Variant[] = ["wide", "hinted", "memory", "plain"];
  const jobs = order.flatMap((variant) =>
    Array.from({ length: variants[variant].runs }, (_, i) => ({ variant, run: i + 1 })));
  console.log(`B1: ${jobs.length} обучений × ${EPISODES} эпизодов × ${STEPS} шагов. Закрытие по клеткам.`);
  const pool = Array.from({ length: Math.min(availableParallelism(), 8) }, async () => {
    for (let job = jobs.shift(); job; job = jobs.shift()) results.push(await launch(job));
  });
  const raw = evaluatePersistence(null);
  await Promise.all(pool);
  reportPersistence(raw, results);
  console.log(`Время полного запуска: ${((performance.now() - start) / 1000).toFixed(1)} с.`);
}

if (isMainThread) await main();
else work(workerData as Job);
