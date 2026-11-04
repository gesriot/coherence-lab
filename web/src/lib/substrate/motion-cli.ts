import { availableParallelism } from "node:os";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { createElement } from "./element.ts";
import { evaluateMotion, evaluateMotionWithContext } from "./motion-evaluation.ts";
import { copies, copyNames, gains, gainNames, reportMotion, type Job, type Result } from "./motion-report.ts";
import { EPISODES, train } from "./training.ts";

type Message = { done: number } | { result: Result };

function work(job: Job) {
  const make = () => createElement(job.run, true, 16, 16, "memory", true);
  const model = make();
  train(model, job.run, "per-cell", undefined, (done) => {
    if (done % 500 === 0) parentPort!.postMessage({ done } satisfies Message);
  }, { motion: job });
  const evaluate = job.path === "random" ? evaluateMotionWithContext : evaluateMotion;
  parentPort!.postMessage({ result: { ...job, ...evaluate(model, make(), job) } } satisfies Message);
}

function launch(job: Job): Promise<Result> {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL(import.meta.url), { workerData: job });
    let result: Result | undefined;
    w.on("message", (m: Message) => {
      if ("result" in m) result = m.result;
      else console.error(`${gainNames[job.gain]} · ${copyNames[job.copy]} · ${job.run}: ${m.done}/${EPISODES} эпизодов.`);
    });
    w.on("error", reject);
    w.on("exit", (code) => {
      if (code === 0 && result) resolve(result);
      else reject(new Error(`Незавершённый прогон ${job.run}, код выхода ${code}`));
    });
  });
}

async function main() {
  const start = performance.now(), jobs: Job[] = [], results: Result[] = [];
  const randomPath = process.argv.includes("--random-path");
  for (let run = 1; run <= 5; run++) {
    for (const gain of gains) {
      if (randomPath && gain !== "fixed") continue;
      for (const copy of copies) jobs.push({ gain, copy, run, path: randomPath ? "random" : "scenario" });
    }
  }
  console.log(`${randomPath ? "B3b · случайные пути" : "B3"} · ${jobs.length} обучений по ${EPISODES} эпизодов. Подвижное окно 16 × 16.`);
  await Promise.all(Array.from({ length: Math.min(availableParallelism(), 8) }, async () => {
    for (let job = jobs.shift(); job; job = jobs.shift()) results.push(await launch(job));
  }));
  reportMotion(results, randomPath);
  console.log(`Время полного запуска: ${((performance.now() - start) / 1000).toFixed(1)} с.`);
}

if (isMainThread) await main();
else work(workerData as Job);
