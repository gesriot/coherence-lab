import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { createElement } from "./element.ts";
import { evaluate } from "./probes.ts";
import { report, type RunResult } from "./report.ts";
import { EPISODES, STEPS, train } from "./training.ts";

interface Job { run: number; directed: boolean }
type Message = { trained: number } | { result: RunResult };

function worker(job: Job) {
  const m = createElement(job.run, job.directed);
  const start = performance.now();
  train(m, job.run);
  const seconds = (performance.now() - start) / 1000;
  parentPort!.postMessage({ trained: seconds } satisfies Message);
  const learned = evaluate(m);
  const initial = job.directed ? evaluate(createElement(job.run)) : undefined;
  parentPort!.postMessage({ result: { ...job, seconds, learned, initial } } satisfies Message);
}

function launch(job: Job, trained: (seconds: number) => void): Promise<RunResult> {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL(import.meta.url), { workerData: job });
    let result: RunResult | undefined;
    w.on("message", (m: Message) => {
      if ("trained" in m) trained(m.trained);
      else result = m.result;
    });
    w.on("error", reject);
    w.on("exit", (code) => {
      if (code === 0 && result) resolve(result);
      else reject(new Error(`Незавершённый прогон ${job.run}, код выхода ${code}`));
    });
  });
}

async function main() {
  const start = performance.now();
  const jobs = [true, false].flatMap((directed) =>
    Array.from({ length: directed ? 5 : 3 }, (_, i) => ({ run: i + 1, directed })));
  console.log(`B0: ${jobs.length} независимых обучений по ${EPISODES} эпизодов × ${STEPS} шагов.`);
  let lastTraining = start;
  const pending = Promise.all(jobs.map((job) => launch(job, (seconds) => {
    lastTraining = performance.now();
    console.error(`${job.directed ? "Собель" : "Без направлений"}, прогон ${job.run}: обучение ${seconds.toFixed(1)} с.`);
  })));
  const copy = evaluate(null);
  report(copy, await pending);
  console.log(`Время от старта до окончания всех обучений: ${((lastTraining - start) / 1000).toFixed(1)} с.`);
  console.log(`Полный запуск с оценкой: ${((performance.now() - start) / 1000).toFixed(1)} с.`);
}

if (isMainThread) await main();
else worker(workerData as Job);
