/** Task selected using only the four controls on design runs 701–710. */
export const genesisTask = {
  trainingSteps: 80,
  checkpoints: [0, 4, 8, 16, 32, 64],
  curriculum: ["continuity", "vanish", "binding", "nursery", "continuity", "binding", "nursery"],
  uncertaintyGrowth: 0.03,
  emptyViews: 3,
  ghostDelay: 20,
  outdoorFirst: 12,
  outdoorLast: 35,
  keepAllHazard: 0.001,
  idealHazards: { occlusion: 0.005, absence: 0.5, "no-data": 0.03 },
  oracleRates: { motion: 0.03, permanence: 0.003 },
  controlCeiling: 0.75,
  controlFloor: 0.9,
} as const;
