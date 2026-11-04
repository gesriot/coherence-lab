import { createHash } from 'node:crypto';
import { referenceModel } from '../../web/src/lib/lab/feasibility.ts';
import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { createEpisode, defaultConfig, snapshot, stepEpisode } from '../../web/src/lib/lab/agent.ts';
import { freshModels } from '../../web/src/lib/lab/learning.ts';

const cwd = fileURLToPath(new URL('../', import.meta.url));
if (process.versions.node !== '26.3.1') throw Error('Requires Node 26.3.1');
const build = spawnSync('cargo', ['build', '--release', '--example', 'trace'], { cwd, stdio: 'inherit' });
if (build.status !== 0) process.exit(1);
const dir = process.argv[2] ?? mkdtempSync(join(tmpdir(), 'lab-traces-'));
mkdirSync(dir, { recursive: true });
console.log(dir);

function difference(a, b, path) {
  if (a === b) return;
  if (a && b && typeof a === 'object' && typeof b === 'object' && Array.isArray(a) === Array.isArray(b)) {
    for (const key of new Set([...Object.keys(a), ...Object.keys(b)])) {
      const diff = difference(a[key], b[key], path + '.' + key);
      if (diff) return diff;
    }
    return;
  }
  return path + ': reference=' + JSON.stringify(a) + ' rust=' + JSON.stringify(b);
}

let total = 0, runs = 0;
// JSON.stringify uses JS insertion order (including optional track fields).
// Take that order from the reference and every value from the Rust frame.
function ordered(frame, reference) {
  if (Array.isArray(reference)) return reference.map((v,i) => ordered(frame[i],v));
  if (reference && typeof reference === 'object') return Object.fromEntries(
    Object.keys(reference).map(k => [k,ordered(frame[k],reference[k])]));
  return frame;
}

function check(scenario, policy, seed, steps, mode, digest) {
  const name = [scenario, policy, seed, mode].join('_');
  const ep = createEpisode(scenario, seed, policy, mode === 'ideal' ? referenceModel('ideal') : mode === 'learned' ? freshModels() : undefined);
  const cfg = { ...defaultConfig(), ablation: policy };
  const reference = [];
  for (let i = 0; i <= steps; i++) {
    if (i) {
      cfg.replay = mode === 'replay' && i > 14;
      stepEpisode(ep, cfg);
    }
    reference.push(JSON.stringify(snapshot(ep, policy)));
  }
  writeFileSync(join(dir, name + '.ts.ndjson'), reference.join('\n') + '\n');
  const p = spawnSync('target/release/examples/trace', [scenario, policy, String(seed), String(steps), mode], { cwd, maxBuffer: 32 << 20 });
  if (p.status !== 0) throw p.error ?? Error(p.stderr.toString());
  writeFileSync(join(dir, name + '.rs.ndjson'), p.stdout);
  const actual = p.stdout.toString().trimEnd().split('\n');
  if (actual.length !== reference.length) throw Error(name + ': frame count ' + actual.length);
  for (let i = 0; i < actual.length; i++) {
    const r = JSON.parse(actual[i]), t = JSON.parse(reference[i]);
    const diff = difference(t, r, '$');
    if (diff) throw Error(name + ' frame ' + i + ' ' + diff);
    if (digest) digest.update(JSON.stringify(ordered(r,t)));
  }
  total += actual.length;
  runs++;
}

for (const seed of [1, 2, 17]) {
  for (const policy of ['full','first-order','novelty','record-only','no-schema','dummy-memory']) {
    check('continuity', policy, seed, 72, 'live');
    check('conflict', policy, seed, 200, 'live');
    if (['full','first-order'].includes(policy)) check('continuity', policy, seed, 40, 'replay');
  }
  check('vanish', 'full', seed, 40, 'learned');
}
const digest = createHash('sha256');
for (const scenario of ['continuity', 'vanish']) check(scenario, 'first-order', 3101, 80, 'ideal', digest);
const sha = digest.digest('hex');
if (sha !== '39cd01bdff23f6aebf63328aac7617808cc5cdc6875627e373bf067cd3f31eb5') throw Error('bodyless trajectory: ' + sha);
console.log(runs + ' episodes, ' + total + ' complete snapshots match');
