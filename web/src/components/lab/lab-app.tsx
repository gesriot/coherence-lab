import { useEffect, useRef, useState } from "react";
import { Pause, Play, RotateCcw, StepForward } from "lucide-react";
import { createEpisode, defaultConfig, snapshot, stepEpisode, type Episode } from "@/lib/lab/agent";
import { criterion, runBench, type BenchReport } from "@/lib/lab/bench";
import type { Ablation, AttentionReason, ScenarioId, SourceKind, UncertaintyReason, ViewState } from "@/lib/lab/types";
import { Stage } from "@/components/lab/stage";

const SCENARIOS: { id: ScenarioId; label: string; hint: string }[] = [
  {
    id: "continuity",
    label: "Непрерывность",
    hint: "Диск A уходит в стену. Полный контур хранит его номер и смотрит на прогноз. Взгляд за яркостью – нет.",
  },
  {
    id: "binding",
    label: "Связывание",
    hint: "На шаге 16 цвета A и B меняются. Номер гипотезы должен остаться у тела, оттенок – обновиться.",
  },
  {
    id: "conflict",
    label: "Конфликт",
    hint: "Два близких оттенка в стороне от большого квадрата. Модель внимания уходит их проверять.",
  },
  {
    id: "nursery",
    label: "Свободная сцена",
    hint: "Три тела и невысокая стена. Удобно крутить шум, фиксацию взгляда и повтор записи.",
  },
];

const ABLATIONS: { id: Ablation; label: string }[] = [
  { id: "full", label: "Полный контур" },
  { id: "first-order", label: "Первый порядок" },
  { id: "record-only", label: "Только запись" },
  { id: "no-schema", label: "Без модели" },
  { id: "dummy-memory", label: "Лишняя память" },
];

const SOURCE: Record<SourceKind, string> = {
  observed: "наблюдение",
  predicted: "прогноз",
  imagined: "воображение",
  mixed: "смесь",
};

const REASON: Record<UncertaintyReason, string> = {
  none: "ясно",
  noise: "шум",
  occlusion: "окклюзия",
  conflict: "конфликт",
  absence: "пусто, хотя ждали",
  "no-data": "вне окна",
};

const ATTENTION: Record<AttentionReason, string> = {
  conflict: "развести конфликт",
  verify: "проверить прогноз",
  uncertainty: "снять неопределённость",
  explore: "обход",
  salience: "за яркостью",
  pursue: "сопровождение",
  pinned: "взгляд зафиксирован",
  idle: "ожидание",
};

function pct(value: number | null): string {
  if (value == null) return "–";
  return `${Math.round(value * 100)}%`;
}

function fresh(scenario: ScenarioId, seed: number, ablation: Ablation): { episode: Episode; view: ViewState } {
  const episode = createEpisode(scenario, seed);
  return { episode, view: snapshot(episode, ablation) };
}

export function LabApp() {
  const [scenario, setScenario] = useState<ScenarioId>("continuity");
  const [seed, setSeed] = useState(4);
  const [ablation, setAblation] = useState<Ablation>("full");
  const [pinGaze, setPinGaze] = useState(false);
  const [pin, setPin] = useState({ x: 50, y: 50 });
  const [inject, setInject] = useState(false);
  const [replay, setReplay] = useState(false);
  const [extraNoise, setExtraNoise] = useState(0);
  const [showTruth, setShowTruth] = useState(true);
  const [showHypotheses, setShowHypotheses] = useState(true);
  const [running, setRunning] = useState(true);
  const [speed, setSpeed] = useState<1 | 2 | 4>(1);
  const [bench, setBench] = useState<BenchReport | null>(null);
  const [view, setView] = useState<ViewState>(() => fresh("continuity", 4, "full").view);
  const episodeRef = useRef<Episode | null>(null);
  if (!episodeRef.current) episodeRef.current = createEpisode("continuity", 4);

  const controls = useRef({ ablation, pinGaze, pin, inject, replay, extraNoise });
  controls.current = { ablation, pinGaze, pin, inject, replay, extraNoise };

  useEffect(() => {
    const id = window.setTimeout(() => setBench(runBench()), 60);
    return () => window.clearTimeout(id);
  }, []);

  useEffect(() => {
    if (!running) return;
    const delay = speed === 1 ? 460 : speed === 2 ? 230 : 110;
    const id = window.setInterval(() => {
      const episode = episodeRef.current;
      if (!episode) return;
      const current = controls.current;
      stepEpisode(episode, {
        ...defaultConfig(),
        ablation: current.ablation,
        pinGaze: current.pinGaze,
        pinX: current.pin.x,
        pinY: current.pin.y,
        injectConfidence: current.inject,
        replay: current.replay,
        extraNoise: current.extraNoise,
      });
      setView(snapshot(episode, current.ablation));
    }, delay);
    return () => window.clearInterval(id);
  }, [running, speed]);

  function reset(nextScenario = scenario, nextSeed = seed, nextAblation = ablation) {
    const episode = createEpisode(nextScenario, nextSeed);
    episodeRef.current = episode;
    setView(snapshot(episode, nextAblation));
  }

  function stepOnce() {
    const episode = episodeRef.current;
    if (!episode) return;
    const current = controls.current;
    stepEpisode(episode, {
      ...defaultConfig(),
      ablation: current.ablation,
      pinGaze: current.pinGaze,
      pinX: current.pin.x,
      pinY: current.pin.y,
      injectConfidence: current.inject,
      replay: current.replay,
      extraNoise: current.extraNoise,
    });
    setView(snapshot(episode, current.ablation));
  }

  const target = view.evaluator.objects[0];
  const retainedLabel =
    view.evaluator.retained == null ? "ещё рано" : view.evaluator.retained ? "удержана" : "потеряна";
  const scenarioMeta = SCENARIOS.find((item) => item.id === scenario)!;
  const status = view.coupling === "broken"
    ? "Связь действие–вход разорвана"
    : pinGaze
      ? "Взгляд зафиксирован"
      : inject
        ? "Уверенность подменена"
        : ablation === "full"
          ? "Контур замкнут"
          : ablation === "record-only"
            ? "Модель не влияет"
            : ablation === "dummy-memory"
              ? "Память без модели"
              : "Модель снята";

  return (
    <main className="mx-auto w-full max-w-6xl px-4 pb-20 pt-5 sm:px-6">
      <header className="mb-5 flex flex-col gap-4 border-b border-line pb-5 sm:flex-row sm:items-end sm:justify-between">
        <div className="max-w-xl">
          <p className="font-mono text-xs tracking-widest text-accent uppercase">Лаборатория, не личность</p>
          <h1 className="mt-1 font-display text-4xl text-fg sm:text-5xl">Дирижёр</h1>
          <p className="mt-2 max-w-lg text-sm leading-relaxed text-muted">
            Минимальный контур в духе Баха: сцена собирается из конкурирующих гипотез, а модель собственного
            восприятия выбирает, куда смотреть. Это не доказательство квалиа.
          </p>
        </div>
        <div className="flex flex-col items-start gap-2 sm:items-end">
          <p className="rounded-full border border-line bg-surface px-3 py-2 font-mono text-xs text-fg">{status}</p>
          <p className="font-mono text-xs text-muted tabular-nums">
            шаг {view.step} · итерации {view.iterationsUsed}/{view.iterationBudget} · {view.attention.agreement}
          </p>
        </div>
      </header>

      <ol className="mb-5 grid grid-cols-2 gap-2 sm:grid-cols-5">
        {["Измерения", "Гипотезы", "Сцена", "Модель внимания", "Взгляд"].map((label, index) => (
          <li key={label} className="lab-card flex items-center gap-2 px-3 py-2">
            <span className="font-mono text-xs text-accent">{index + 1}</span>
            <span className="text-sm text-fg">{label}</span>
          </li>
        ))}
      </ol>
      <p className="mb-5 text-sm leading-relaxed text-muted">
        Воспринимающий – не отдельная функция и не это окно. Это вся петля: измерения уточняют гипотезы, гипотезы
        входят в ограниченную сцену, сцена обновляет модель внимания, модель двигает взгляд, взгляд меняет следующие
        измерения.
      </p>

      <div className="grid gap-4 lg:grid-cols-5">
        <section className="lab-card overflow-hidden lg:col-span-3">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-3">
            <div>
              <h2 className="font-display text-2xl">Мир оценщика</h2>
              <p className="text-xs text-muted">Агент этих подписей не получает. Клик фиксирует его взгляд.</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button type="button" className="pill-on" onClick={() => setRunning((value) => !value)}>
                {running ? <Pause className="size-4" /> : <Play className="size-4" />}
                {running ? "Пауза" : "Пуск"}
              </button>
              <button
                type="button"
                className="pill"
                onClick={() => {
                  setRunning(false);
                  stepOnce();
                }}
              >
                <StepForward className="size-4" />
                Шаг
              </button>
              <button type="button" className="pill" onClick={() => reset()}>
                <RotateCcw className="size-4" />
                Сначала
              </button>
              {([1, 2, 4] as const).map((value) => (
                <button
                  key={value}
                  type="button"
                  className={speed === value ? "pill-on" : "pill"}
                  onClick={() => setSpeed(value)}
                >
                  {value}×
                </button>
              ))}
              <button
                type="button"
                className={showTruth ? "pill-on" : "pill"}
                onClick={() => setShowTruth((value) => !value)}
              >
                Истина
              </button>
              <button
                type="button"
                className={showHypotheses ? "pill-on" : "pill"}
                onClick={() => setShowHypotheses((value) => !value)}
              >
                Гипотезы
              </button>
            </div>
          </div>
          <div className="relative aspect-square w-full bg-bg">
            <Stage
              view={view}
              showTruth={showTruth}
              showHypotheses={showHypotheses}
              onPick={(x, y) => {
                setPin({ x, y });
                setPinGaze(true);
              }}
            />
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 border-t border-line px-4 py-3 text-xs text-muted">
            <span>квадрат – сенсорное окно</span>
            <span>пунктир – гипотеза</span>
            <span>кольцо – следующий взгляд</span>
            <span>штрих – стена</span>
          </div>
        </section>

        <div className="flex flex-col gap-4 lg:col-span-2">
          <section className="lab-card p-4">
            <div className="mb-3 flex items-baseline justify-between gap-3">
              <h2 className="font-display text-2xl">Пузырь настоящего</h2>
              <p className="font-mono text-xs text-muted">{view.tracks.filter((track) => track.inWorkspace).length}/3 в окне</p>
            </div>
            {view.tracks.length === 0 ? (
              <p className="text-sm text-muted">Сцена пуста. Контур ещё не связал измерения в сущности.</p>
            ) : (
              <ul className="flex flex-col gap-2">
                {view.tracks.map((track) => (
                  <li key={track.id} className="rounded-lg border border-line bg-raised px-3 py-2">
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span
                          className="inline-block size-3 rounded-full"
                          style={{ background: `hsl(${track.hue} 70% 52%)` }}
                          aria-hidden
                        />
                        <span className="font-mono text-sm text-fg">T{track.id}</span>
                        <span className="text-xs text-muted">{SOURCE[track.source]}</span>
                      </div>
                      <span className="text-xs text-muted">{track.inWorkspace ? "в сцене" : "периферия"}</span>
                    </div>
                    <p className="mt-1 text-xs text-muted">
                      {REASON[track.reason]}
                      {track.conflict ? " · конфликт связывания" : ""}
                    </p>
                    <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-bg">
                      <div
                        className="h-full bg-accent"
                        style={{ width: `${Math.round((1 - track.uncertainty) * 100)}%` }}
                      />
                    </div>
                    <p className="mt-1 font-mono text-xs text-muted tabular-nums">
                      увер. {track.confidence.toFixed(2)} · неозн. {track.uncertainty.toFixed(2)} · свидет.{" "}
                      {track.evidence.toFixed(2)}
                    </p>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="lab-card p-4">
            <h2 className="font-display text-2xl">Модель внимания</h2>
            {view.attention.active ? (
              <div className="mt-2 space-y-2 text-sm">
                <p className="text-fg">{view.attention.note}</p>
                <p className="text-muted">
                  Намерение: {ATTENTION[view.attention.reason]}
                  {view.attention.focusId != null ? ` · фокус T${view.attention.focusId}` : ""}
                  {view.attention.influencesAction ? "" : " · на действие не влияет"}
                </p>
                <p className="font-mono text-xs text-muted tabular-nums">
                  ошибка самомодели {view.attention.selfError == null ? "–" : view.attention.selfError.toFixed(2)}
                  {" · "}память {view.memoryFrames || 0} кадров
                </p>
              </div>
            ) : (
              <p className="mt-2 text-sm text-muted">
                {view.attention.note} {view.memoryFrames > 0 ? `Кадров в памяти: ${view.memoryFrames}.` : ""}
              </p>
            )}
            <div className="mt-4">
              <h3 className="mb-2 text-xs tracking-wide text-muted uppercase">Измерение, не сцена</h3>
              <div
                className="grid w-36 grid-cols-12 gap-px"
                aria-label="Сырые клетки сенсорного окна"
              >
                {view.cells.map((cell, index) => (
                  <div
                    key={index}
                    className="aspect-square"
                    style={{ background: `hsl(${cell.hue} ${cell.saturation}% ${cell.lightness}%)` }}
                  />
                ))}
              </div>
            </div>
          </section>
        </div>
      </div>

      <section className="lab-card mt-4 p-4">
        <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="font-display text-2xl">Когерентность</h2>
            <p className="max-w-xl text-sm text-muted">
              Ниже – не «больше сознания». Это ошибка согласования. Тишина при живых измерениях штрафуется.
            </p>
          </div>
          <p className="font-mono text-sm text-accent tabular-nums">E {view.energy.total.toFixed(2)}</p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <EnergyRow label="Сенсорика" value={view.energy.sensory} />
          <EnergyRow label="Связывание" value={view.energy.binding} />
          <EnergyRow label="Время" value={view.energy.temporal} />
          <EnergyRow label="Действие" value={view.energy.action} />
          <EnergyRow label="Самомодель" value={view.energy.self} />
          <EnergyRow label="Ресурс" value={view.energy.regularization} />
        </div>
        <figure className="mt-4">
          <figcaption className="text-xs text-muted">Итерации согласования · E от исходной сцены до остановки</figcaption>
          <svg className="mt-1 h-12 w-44" viewBox="0 0 160 42" role="img"
            aria-label={view.energyTrace.map((value) => value.toFixed(3)).join(" → ")}>
            <polyline fill="none" stroke="currentColor" strokeWidth="2" className="text-accent"
              points={view.energyTrace.map((value, i) =>
                `${2 + 156 * i / Math.max(1, view.energyTrace.length - 1)},${40 - 38 * value}`).join(" ")} />
          </svg>
        </figure>
        <div className="mt-4 flex h-10 items-end gap-1" aria-hidden>
          {view.history.map((point) => (
            <div
              key={point.step}
              className="w-1.5 flex-1 bg-accent/80"
              style={{ height: `${Math.max(8, Math.round(point.total * 100))}%` }}
            />
          ))}
        </div>
        <p className="mt-3 font-mono text-xs text-muted tabular-nums">
          свет мира {view.evaluator.lightTrue.toFixed(1)}° · оценка {view.evaluator.lightEstimated.toFixed(1)}° · шум{" "}
          {view.noise.toFixed(0)}°
          {view.swapped ? " · цвета тел обменены" : ""}
          {view.energy.overconfident ? " · штраф за самоуверенность" : ""}
        </p>
      </section>

      <section className="mt-4 grid gap-4 lg:grid-cols-2">
        <div className="lab-card p-4">
          <h2 className="font-display text-2xl">Эпизод</h2>
          <p className="mt-1 text-sm text-muted">{scenarioMeta.hint}</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {SCENARIOS.map((item) => (
              <button
                key={item.id}
                type="button"
                className={item.id === scenario ? "pill-on" : "pill"}
                onClick={() => {
                  setScenario(item.id);
                  setReplay(false);
                  setPinGaze(false);
                  setInject(false);
                  reset(item.id, seed, ablation);
                }}
              >
                {item.label}
              </button>
            ))}
          </div>
          <label className="mt-4 flex items-center gap-3 text-sm text-muted">
            Семя
            <input
              type="number"
              value={seed}
              min={1}
              max={999}
              className="min-h-11 w-24 rounded-full border border-line bg-bg px-3 font-mono text-fg"
              onChange={(event) => {
                const next = Number(event.target.value);
                if (!Number.isFinite(next)) return;
                setSeed(next);
                reset(scenario, next, ablation);
              }}
            />
          </label>
          <div className="mt-4 rounded-lg border border-line bg-bg px-3 py-3">
            <p className="text-xs tracking-wide text-muted uppercase">Только оценщик</p>
            <p className="mt-1 text-sm text-fg">
              Идентичность A: <span className="text-accent">{retainedLabel}</span>
              {view.evaluator.initialTrackId != null ? ` · стартовая гипотеза T${view.evaluator.initialTrackId}` : ""}
            </p>
            <ul className="mt-2 space-y-1 font-mono text-xs text-muted">
              {view.evaluator.objects.map((object) => (
                <li key={object.id}>
                  {object.id} {object.occluded ? "скрыт" : "виден"} · гипотеза{" "}
                  {object.nearestTrack == null ? "нет" : `T${object.nearestTrack}`}
                  {object.trackHue != null ? ` · оттенок ${Math.round(object.trackHue)}` : ""}
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="lab-card p-4">
          <h2 className="font-display text-2xl">Вмешательства</h2>
          <p className="mt-1 text-sm text-muted">Меняют причину, а не подпись. Можно переключать посреди эпизода.</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {ABLATIONS.map((item) => (
              <button
                key={item.id}
                type="button"
                className={ablation === item.id ? "pill-on" : "pill"}
                onClick={() => setAblation(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <button type="button" className={pinGaze ? "pill-on" : "pill"} onClick={() => setPinGaze((value) => !value)}>
              Фиксация взгляда
            </button>
            <button type="button" className={inject ? "pill-on" : "pill"} onClick={() => setInject((value) => !value)}>
              Подменить уверенность
            </button>
            <button type="button" className={replay ? "pill-on" : "pill"} onClick={() => setReplay((value) => !value)}>
              Повтор записи
            </button>
          </div>
          <label className="mt-4 block text-sm text-muted">
            Дополнительный шум оттенка: {extraNoise}°
            <input
              type="range"
              min={0}
              max={48}
              step={1}
              value={extraNoise}
              className="mt-2 w-full accent-accent"
              onChange={(event) => setExtraNoise(Number(event.target.value))}
            />
          </label>
          <p className="mt-3 text-sm leading-relaxed text-fg">{target ? scenarioMeta.hint : ""}</p>
        </div>
      </section>

      <section className="lab-card mt-4 p-4">
        <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
          <div>
            <h2 className="font-display text-2xl">Протокол</h2>
            <p className="max-w-2xl text-sm text-muted">
              {bench ? bench.summary : "Идёт сравнение конфигураций…"}
            </p>
          </div>
          <p className="text-sm text-muted">{criterion}</p>
          <p className="text-sm text-muted">Версия A без обучения: проверяет функцию дирижёра (CTC), но не гипотезу Genesis.</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[36rem] text-left text-sm">
            <thead className="text-xs tracking-wide text-muted uppercase">
              <tr>
                <th className="py-2 pr-3 font-medium">Условие</th>
                <th className="py-2 pr-3 font-medium">Идентичность</th>
                <th className="py-2 pr-3 font-medium">Пара найдена</th>
                <th className="py-2 pr-3 font-medium">Ошибка самомодели</th>
                <th className="py-2 font-medium">Ошибка действия</th>
              </tr>
            </thead>
            <tbody>
              {(bench?.rows ?? []).map((row) => (
                <tr key={row.id} className="border-t border-line">
                  <td className="py-2 pr-3">
                    <div className="text-fg">{row.label}</div>
                    <div className="text-xs text-muted">{row.detail}</div>
                  </td>
                  <td className="py-2 pr-3 font-mono tabular-nums">{pct(row.retention)}</td>
                  <td className="py-2 pr-3 font-mono tabular-nums">{pct(row.pair)}</td>
                  <td className="py-2 pr-3 font-mono tabular-nums">
                    {row.selfError == null ? "–" : row.selfError.toFixed(2)}
                  </td>
                  <td className="py-2 font-mono tabular-nums">
                    {row.actionError == null ? "–" : row.actionError.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <dl className="mt-4 grid gap-2 text-sm sm:grid-cols-3">
          <div className="rounded-lg border border-line px-3 py-2">
            <dt className="text-xs text-muted">Механизм</dt>
            <dd>реализован</dd>
          </div>
          <div className="rounded-lg border border-line px-3 py-2">
            <dt className="text-xs text-muted">Причинная роль</dt>
            <dd>
              {bench == null ? "считается" : { supported: "подтверждена на этих конфигурациях", mixed: "смешанный результат", unsupported: "не подтверждена" }[bench.causal]}
            </dd>
          </div>
          <div className="rounded-lg border border-line px-3 py-2">
            <dt className="text-xs text-muted">Квалиа</dt>
            <dd>строки PASS нет</dd>
          </div>
        </dl>
      </section>
    </main>
  );
}

function EnergyRow({ label, value }: { label: string; value: number | null }) {
  const shown = value == null ? 0 : Math.max(0, Math.min(1, value));
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs text-muted">
        <span>{label}</span>
        <span className="font-mono tabular-nums">{value == null ? "снята" : value.toFixed(2)}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-bg">
        <div className="h-full bg-accent" style={{ width: value == null ? "0%" : `${Math.round(shown * 100)}%` }} />
      </div>
    </div>
  );
}
