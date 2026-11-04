import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const rust = fileURLToPath(new URL('../', import.meta.url));
const web = fileURLToPath(new URL('../../web/', import.meta.url));
const root = fileURLToPath(new URL('../../', import.meta.url));
if (process.version !== 'v26.3.1') throw Error('Requires Node 26.3.1');

const cases = [
  ['b0', 'cli'],
  ['b1', 'persistence-cli'],
  ['b1b', 'tracking-cli'],
  ['b3', 'motion-cli'],
  ['b3b', 'motion-cli', '--random-path'],
];
const requested = process.argv.slice(2);
for (const id of requested) {
  if (!cases.some(([name]) => name === id)) throw Error('unknown case ' + id);
}

function run(command, args, cwd, progress = false) {
  const start = performance.now();
  const result = spawnSync(command, args, {
    cwd, maxBuffer: 64 << 20, stdio: ['ignore', 'pipe', progress ? 'inherit' : 'pipe'],
  });
  if (result.status !== 0) {
    throw result.error ?? Error(command + ' exited ' + (result.signal ?? result.status) + ': ' + (result.stderr ?? ''));
  }
  return { bytes: result.stdout, seconds: (performance.now() - start) / 1000 };
}

const git = (...args) => run('git', args, root).bytes.toString().trim();
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
function cleanTree() {
  if (git('status', '--porcelain', '--untracked-files=all', '--', 'web/src/lib')) throw Error('dirty reference tree');
  return git('rev-parse', 'HEAD:web/src/lib');
}
const tree = cleanTree();
const cacheRoot = process.env.SUBSTRATE_REFERENCE_CACHE ?? join(tmpdir(), 'coherence-substrate-cache');
const cache = resolve(cacheRoot, tree);
const where = relative(root, cache);
if (!isAbsolute(where) && where.split(sep)[0] !== '..') throw Error('reference cache must be outside the repository');
mkdirSync(cache, {recursive: true});

function reference(name, script, args) {
  const path = join(cache, name + '.json');
  if (cleanTree() !== tree) throw Error('reference tree changed');
  if (existsSync(path)) {
    const record = JSON.parse(readFileSync(path, 'utf8'));
    const bytes = Buffer.from(record.stdout, 'base64');
    if (record.tree !== tree || record.node !== process.version || record.name !== name ||
        record.script !== script || JSON.stringify(record.args) !== JSON.stringify(args) ||
        sha(bytes) !== record.sha256 || !Number.isFinite(record.seconds) || record.seconds < 0) {
      throw Error('invalid reference cache: ' + path);
    }
    return {bytes, seconds: record.seconds};
  }

  const result = run(process.execPath, ['src/lib/substrate/' + script + '.ts', ...args], web, true);
  if (cleanTree() !== tree) throw Error('reference tree changed during run');
  const tmp = path + '.' + process.pid + '.tmp';
  writeFileSync(tmp, JSON.stringify({
    tree, node: process.version, name, script, args, seconds: result.seconds,
    sha256: sha(result.bytes), stdout: result.bytes.toString('base64'),
  }));
  renameSync(tmp, path);
  return result;
}

function comparable(name, bytes) {
  // Match only the trailing timing lines. Keep all preceding bytes, including their final LF.
  const patterns = name === 'b0' ? [
    /^Время от старта до окончания всех обучений: [0-9]+\.[0-9] с\.$/u,
    /^Полный запуск с оценкой: [0-9]+\.[0-9] с\.$/u,
  ] : [/^Время полного запуска: [0-9]+\.[0-9] с\.$/u];
  let end = bytes.length;
  for (const pattern of patterns.toReversed()) {
    if (bytes[end - 1] !== 10) throw Error(name + ' missing final newline');
    const start = bytes.lastIndexOf(10, end - 2) + 1;
    if (!pattern.test(bytes.subarray(start, end - 1).toString())) throw Error(name + ' unexpected timing line');
    end = start;
  }
  return bytes.subarray(0, end);
}

run('cargo', ['build', '--release', '-p', 'cli'], rust, true);
console.log('web/src/lib ' + tree);
for (const [name, script, ...args] of cases) {
  if (requested.length && !requested.includes(name)) continue;
  const ts = reference(name, script, args);
  const rs = run(join(rust, 'target/release/cli'), ['substrate-' + name], rust, true);
  const a = comparable(name, ts.bytes), b = comparable(name, rs.bytes);
  if (!a.equals(b)) {
    let i = 0;
    while (i < Math.min(a.length, b.length) && a[i] === b[i]) i++;
    throw Error(name + ' differs at byte ' + i + '\nNode ' + a.subarray(i, i + 120) + '\nRust ' + b.subarray(i, i + 120));
  }
  console.log('substrate-' + name + ' | ' + a.length + ' B | ' + sha(a) +
    ' | Node ' + ts.seconds.toFixed(3) + ' s | Rust ' + rs.seconds.toFixed(3) + ' s');
}
