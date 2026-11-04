import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { trainingCurve, evaluate } from '../../web/src/lib/lab/genesis.ts';
import { conditions } from '../../web/src/lib/lab/genesis-training.ts';
import { Learner, Replay } from '../../web/src/lib/lab/experience.ts';

const cwd = fileURLToPath(new URL('../', import.meta.url));
if (process.version !== 'v26.3.1') throw Error('Requires Node 26.3.1');
const build = spawnSync('cargo', ['build','--release','--example','training_trace'], {cwd, stdio:'inherit'});
if (build.status !== 0) process.exit(1);
const dir = process.argv[2] ?? mkdtempSync(join(tmpdir(), 'training-traces-'));
mkdirSync(dir, {recursive:true});
const sample = Replay.prototype.sample;
const update = Learner.prototype.update;
let draws = [], current;
Replay.prototype.sample = function(...args) {
  const value = sample.apply(this,args);
  current.push(value ?? null);
  return value;
};
Learner.prototype.update = function() {
  current = [];
  update.call(this);
  const [movement,gap] = current;
  draws.push({movement,gap});
};

function compare(a,b,path){
  if(a===b)return;
  if (a && b && typeof a === 'object' && typeof b === 'object' && Array.isArray(a) === Array.isArray(b)) {
    for(const key of new Set([...Object.keys(a),...Object.keys(b)])) compare(a[key],b[key],path+'.'+key);
    return;
  }
  throw Error(path+' reference='+JSON.stringify(a)+' rust='+JSON.stringify(b));
}
let episodes=0, samples=0;
for (const seed of [61,701]) for (const id of Object.keys(conditions)) {
  draws = [];
  const frames = trainingCurve(id,seed,{motion:0.03,existence:0.1},[0,1,2,3,4,5,6,7,8],(m,episode)=>{
    const out = {episode,weights:structuredClone(m.weights),logits:[m.logits.hidden,m.logits.open,m.logits.miss],draws,
      point:evaluate(m,conditions[id].gaze,3001,2)};
    draws=[];return out;
  });
  const p = spawnSync('target/release/examples/training_trace',[id,String(seed),'8'],{cwd,maxBuffer:64<<20});
  if(p.status!==0) throw p.error ?? Error(p.stderr.toString());
  const name=join(dir,id+'_'+seed);
  writeFileSync(name+'.ts.ndjson',frames.map(x=>JSON.stringify(x)).join('\n')+'\n');
  writeFileSync(name+'.rs.ndjson',p.stdout);
  const actual=p.stdout.toString().trimEnd().split('\n').map(s=>JSON.parse(s));
  for(const [i,f] of frames.entries()){compare(f,actual[i],id+' seed '+seed+' episode '+i);samples+=f.draws.length;}
  if(actual.length!==frames.length) throw Error(name+' episode count');
  episodes+=frames.length;
}
console.log(episodes+' training snapshots, '+samples+' step selections match in '+dir);
