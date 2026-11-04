import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const rust = fileURLToPath(new URL('../', import.meta.url));
const web = fileURLToPath(new URL('../../web/', import.meta.url));
if (process.version !== 'v26.3.1') throw Error('Requires Node 26.3.1');
const scripts = [
  ['bench', 'bench-cli'],
  ['genesis-feasibility', 'feasibility-cli', ['ideal', 'frozen', 'ideal'], ['ideal', 'full']],
  ['genesis-tune', 'genesis-tune'],
  ['genesis', 'genesis-cli'],
  ['sensorimotor-feasibility', 'sensorimotor-feasibility-cli', ['ideal', 'frozen', 'ideal'], ['ideal', 'constructor']],
  ['sensorimotor-tune', 'sensorimotor-tune'],
  ['sensorimotor', 'sensorimotor-cli', ['0,300'], ['0.03']],
];
function run(cwd, command, args) {
  const start = performance.now();
  const result = spawnSync(command, args, {cwd});
  if (result.error) throw result.error;
  if (result.signal) throw Error(command + ' killed by ' + result.signal);
  return {...result, seconds: ((performance.now() - start) / 1000).toFixed(3)};
}
// The reference throws at the CLI boundary. Its source excerpt, stack and
// runtime version belong to Node; the exception message is the CLI contract.
function message(stderr) {
  const match = /^Error: (.+)$/m.exec(stderr.toString());
  if (!match) throw Error('missing exception message: ' + stderr);
  return match[1];
}
const built = run(rust, 'cargo', ['build','--release','-p','cli']);
if (built.status !== 0) throw Error(built.stderr.toString());
for (const [name, script, valid, invalid] of scripts) {
  for (const args of [[], valid, invalid].filter(Boolean)) {
    const id = [name, ...args].join(' ');
    const ts = run(web, process.execPath, ['src/lib/lab/' + script + '.ts', ...args]);
    const rs = run(rust, 'cargo', ['run','--release','-p','cli','--',name,...args]);
    if (ts.status !== rs.status || (args === invalid ? ts.status === 0 : ts.status !== 0)) {
      throw Error(id + ' exit codes Node=' + ts.status + ' Rust=' + rs.status + '\n' + ts.stderr + '\n' + rs.stderr);
    }
    if (!ts.stdout.equals(rs.stdout)) {
      let at = 0;
      while (at < Math.min(ts.stdout.length,rs.stdout.length) && ts.stdout[at] === rs.stdout[at]) at++;
      throw Error(id + ' stdout byte ' + at + '\nNode ' + ts.stdout.subarray(at,at+100) + '\nRust ' + rs.stdout.subarray(at,at+100));
    }
    const digest = createHash('sha256').update(ts.stdout).digest('hex');
    if (args === invalid && message(ts.stderr) !== message(rs.stderr)) throw Error(id + ' error message differs');
    console.log(id + ' | ' + ts.stdout.length + ' B | ' + digest + ' | Node ' + ts.seconds + ' s | Rust ' + rs.seconds +
      ' s | exit ' + ts.status + (args === invalid ? ' | ' + message(ts.stderr) : ''));
  }
}
