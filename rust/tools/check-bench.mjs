import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const rust = fileURLToPath(new URL('../', import.meta.url));
const web = fileURLToPath(new URL('../../web/', import.meta.url));
if (process.versions.node !== '26.3.1') throw Error('Requires Node 26.3.1');
function run(cwd, command, args) {
  const start = performance.now();
  const result = spawnSync(command, args, { cwd });
  if (result.status !== 0) throw result.error ?? Error(result.stderr.toString());
  return { stdout: result.stdout, seconds: (performance.now() - start) / 1000 };
}

run(rust, 'cargo', ['build', '--release', '-p', 'cli']);
const reference = run(web, 'node', ['src/lib/lab/bench-cli.ts']);
const port = run(rust, 'cargo', ['run', '--release', '-p', 'cli', '--', 'bench']);
if (!port.stdout.equals(reference.stdout)) {
  let at = 0;
  while (at < Math.min(port.stdout.length, reference.stdout.length) &&
         port.stdout[at] === reference.stdout[at]) at++;
  throw Error('stdout differs at byte ' + at +
    '\nTypeScript: ' + JSON.stringify(reference.stdout.subarray(at, at + 80).toString()) +
    '\nRust: ' + JSON.stringify(port.stdout.subarray(at, at + 80).toString()));
}
console.log(reference.stdout.length + ' identical bytes, SHA-256 ' +
  createHash('sha256').update(reference.stdout).digest('hex'));
console.log('Node ' + reference.seconds.toFixed(3) + ' s, Rust ' + port.seconds.toFixed(3) + ' s');
