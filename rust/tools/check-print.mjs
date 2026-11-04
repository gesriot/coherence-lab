import assert from 'node:assert/strict';
import { Console } from 'node:console';
import { Writable } from 'node:stream';
import { spawnSync } from 'node:child_process';
import { inspect } from 'node:util';
import { fileURLToPath } from 'node:url';

const cwd=fileURLToPath(new URL('../',import.meta.url));
if(process.versions.node!=='26.3.1') throw Error('Requires Node 26.3.1');
const build=spawnSync('cargo',['build','--release','--example','check_print'],{cwd,stdio:'inherit'});
if(build.status!==0) process.exit(1);
const run=(mode,input)=>{
  const p=spawnSync('target/release/examples/check_print',[mode],{input,cwd,maxBuffer:64<<20});
  if(p.status!==0) throw Error(p.error?.message??p.stderr.toString());return p.stdout.toString();
};
let state=71357; const word=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return state>>>0;};
const bits=Buffer.alloc(8);
const edges=[-0,0,NaN,-Infinity,Infinity,Number.MIN_VALUE,Number.MAX_VALUE,-2.5,2.5,1e-6,1e-7,1e20,1e21,1.005,1000000000000000128];
const cases=Buffer.alloc(9*(100000+101*edges.length)); const expected=[];
for(let i=0;i<cases.length/9;i++){
  bits.writeUInt32LE(word());bits.writeUInt32LE(word(),4);
  const x=i<100000?(i%2?bits.readDoubleLE():(word()/2**32-0.5)*1000):edges[(i-100000)%edges.length];
  const digits=i<100000?word()%101:Math.floor((i-100000)/edges.length);
  cases.writeDoubleLE(x,9*i);cases[9*i+8]=digits;
  expected.push([String(x),x.toFixed(digits),inspect(x)].join('\t'));
}
const actual=run('numbers',cases).trimEnd().split('\n');
assert.equal(actual.length,expected.length);
for(let i=0;i<actual.length;i++)assert.equal(actual[i],expected[i],`number ${i}`);
const suites=[['', 'Жёлтый', "д'Артаньян", 'И "он" и \'я\'', '\'\"`\\', '12/200 · 6,0% [0,0–7,6%]',
  'е\u0308', '東京', '👩‍🔬','\x01\x85\n\t', '🦊'.repeat(6),'\x1b[31mцвет\x1b[0m'],
  Array.from({length:100},(_,i)=>'аб'.repeat(i%12)+String.fromCodePoint(32+word()%1100))];
for(const texts of suites){
  const buf=Buffer.concat(texts.flatMap(s=>{const b=Buffer.from(s),n=Buffer.alloc(4);n.writeUInt32LE(b.length);return[n,b]}));
  let output=''; const c=new Console(new Writable({write(x,e,cb){output+=x.toString();cb();}}));
  c.table(texts.map((s,i)=>({Условие:s,'Доля · –':-i/10}))); assert.equal(run('table',buf),output);
  output='';texts.forEach((s,i)=>c.log(s,-i/10));assert.equal(run('log',buf),output);
  assert.equal(run('inspect',buf),texts.map(s=>inspect(s)+'\n').join(''));
}
console.log(`${expected.length} number/toFixed/inspect cases; ${suites.length} tables, log and string inspection match`);
