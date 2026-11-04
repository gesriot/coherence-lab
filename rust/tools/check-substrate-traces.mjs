import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createElement, blankState, forward, stepBuffers } from '../../web/src/lib/substrate/element.ts';
import { frames, movingFrames } from '../../web/src/lib/substrate/input.ts';
import { occupancyHook } from '../../web/src/lib/substrate/occupancy.ts';
import { exitWeightHook } from '../../web/src/lib/substrate/exit-events.ts';
import { trainer } from '../../web/src/lib/substrate/training.ts';

const cwd = fileURLToPath(new URL('../', import.meta.url));
const build = spawnSync('cargo',['build','--release','--example','trace_substrate'],{cwd,stdio:'inherit'});
if(build.status !== 0) process.exit(1);
const powers=spawnSync('cargo',['run','--quiet','--release','--example','adam_powers'],{cwd});
if(powers.status!==0) throw powers.error ?? Error(powers.stderr.toString());
for(let i=1;i<=18000;i++) for(const [j,x] of [0.9,0.999].entries())
  if(x**i!==powers.stdout.readDoubleLE(16*(i-1)+8*j)) throw Error('Adam power step '+i);
for(const [variant,depth] of [['plain',16],['isotropic',16],['local',16],['memory',16],['memory',28],['hinted',16],
  ['weighted',16],['executed',16],['zero',16],['shuffled',16],['random-copy',16]]) {
  const kind = ['plain','isotropic','local'].includes(variant) ? 'plain' : variant === 'hinted' ? 'hinted' : 'memory';
  const moving = ['executed','zero','shuffled','random-copy'].includes(variant);
  const model = createElement(1,variant !== 'isotropic',16,depth,kind,moving);
  const run = spawnSync('target/release/examples/trace_substrate',[variant,String(depth),'3','80'],{cwd,maxBuffer:32<<20});
  if(run.status !== 0) throw run.error ?? Error(run.stderr.toString());
  const train = trainer(model);
  let offset=0;
  function compare(a,episode,step,field) {
    const bytes = Buffer.from(a.buffer,a.byteOffset,a.byteLength);
    if(!bytes.equals(run.stdout.subarray(offset,offset+bytes.length))) {
      for(let i=0;i<a.length;i++) if(bytes.readBigUInt64LE(i*8)!==run.stdout.readBigUInt64LE(offset+i*8))
        throw Error(variant+' depth '+depth+' episode '+episode+' step '+step+' '+field+'['+i+']: Node '+a[i]+' Rust '+run.stdout.readDoubleLE(offset+i*8));
    }
    offset+=bytes.length;
  }
  for(const [i,sc] of ['continuity','vanish','nursery'].entries()){
    const seed=20100+i;
    const world=variant==='weighted'?{path:'random',columns:2}:{};
    const sequence=moving?movingFrames(sc,seed,80,{gain:variant==='shuffled'?'episodic':'fixed',copy:variant==='random-copy'?'executed':variant,path:variant==='random-copy'?'random':'scenario'}):
      frames(sc,seed,80,['plain','isotropic'].includes(variant)?'whole-object':'per-cell',world);
    function* observed() {
      let state=blankState(model),t=0;
      for(const item of sequence){
        yield item;
        if(t<80){
          const out=stepBuffers(model);
          forward(model,state,item.frame??item,out,item.copy);
          compare(out.prediction,i+1,t,'prediction');
          state=out.state;
        }
        t++;
      }
    }
    train(observed(),kind==='hinted'?occupancyHook(sc,seed,world):undefined,variant==='weighted'?exitWeightHook(sc,seed,world):undefined);
    compare(model.weights,i+1,80,'weights');
  }
  if(offset!==run.stdout.length) throw Error('trailing trace bytes');
  console.log(variant,depth,'3 episodes × 80 predictions and all weights match');
}
