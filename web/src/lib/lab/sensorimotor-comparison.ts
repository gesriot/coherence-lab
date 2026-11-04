import { bootstrap, effect, replacementVerdict } from "./genesis.ts";
import { genesisTask } from "./genesis-task.ts";
import { gainControlsPass } from "./sensorimotor-feasibility.ts";
import { gainArea, summarizeGainRuns, type GainRuns } from "./sensorimotor.ts";

const alternatives = ["first-order", "no-protocol", "strong-first-order"] as const;
type Participant = "full" | typeof alternatives[number];
export type ComparisonControls = Record<"frozen" | "ideal" | "oracle" | "participantOracle", GainRuns>;

export function compareGainConditions(controls: ComparisonControls, measure: (id: Participant) => GainRuns) {
  if (!gainControlsPass(summarizeGainRuns(controls.frozen), summarizeGainRuns(controls.ideal),
    [summarizeGainRuns(controls.oracle), summarizeGainRuns(controls.participantOracle)])) return null;

  const full = measure("full");
  const participants = new Map<Participant, GainRuns>([["full", full]]);
  const differences = alternatives.map((id) => {
    const runs = measure(id);
    participants.set(id, runs);
    const estimate = bootstrap(full.map((run, i) => gainArea(run.curve) - gainArea(runs[i]!.curve)));
    return { id, estimate, effect: effect(estimate) };
  });
  const firstOrder = differences[0]!.effect, protocol = differences[1]!.effect;
  const saturated = [full, participants.get("first-order")!].every((runs) => summarizeGainRuns(runs).curve[1]!.q >= genesisTask.controlFloor);
  const acceleration = { faster: "подтверждено на этих конфигурациях", none: "не подтверждено", slower: "контур замедляет обучение" };
  const attention = { faster: "подтверждён на этих конфигурациях", none: "не подтверждён", slower: "протокол замедляет обучение" };
  return {
    participants, differences, saturated,
    verdicts: [
      `Ускорение обучения контуром (сенсомоторная зависимость): ${acceleration[firstOrder]}`,
      `Вклад протокола внимания: ${attention[protocol]}`,
      `Альтернативное объяснение правилами первого порядка: ${replacementVerdict(firstOrder, differences[2]!.effect)}`,
    ],
  };
}
