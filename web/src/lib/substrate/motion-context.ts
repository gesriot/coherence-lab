export function contextGroup(scenario: "continuity" | "vanish", time: number) {
  return scenario + "/" + Math.floor(time / 8);
}

// Only training labels enter the group and global means; inference needs just a group key.
export function fitContext(groups: readonly string[], labels: readonly number[]) {
  if (groups.length !== labels.length || labels.length === 0) {
    throw new Error("Для базы контекста нужны обучающие примеры с метками");
  }
  const sums = new Map<string, { sum: number; count: number }>();
  let total = 0;
  for (let i = 0; i < labels.length; i++) {
    const group = groups[i]!, y = labels[i]!;
    const entry = sums.get(group);
    if (entry) {
      entry.sum += y;
      entry.count++;
    } else sums.set(group, { sum: y, count: 1 });
    total += y;
  }
  const fallback = total / labels.length;
  return (group: string) => {
    const entry = sums.get(group);
    return entry ? entry.sum / entry.count : fallback;
  };
}
