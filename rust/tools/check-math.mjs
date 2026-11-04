import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { hashUnit, hueNoise } from '../../web/src/lib/lab/math.ts';

const cwd = fileURLToPath(new URL('../', import.meta.url));
if (process.versions.node !== '26.3.1') throw Error('Requires Node 26.3.1');
const build = spawnSync('cargo', ['build', '--release', '-p', 'jscompat', '--example', 'check_math'], {cwd, stdio:'inherit'});
if (build.status !== 0) process.exit(1);
let state = 0x31415926;
function word() { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return state >>> 0; }
const bits = Buffer.alloc(8);
function randomDouble() {
  bits.writeUInt32LE(word()); bits.writeUInt32LE(word(), 4);
  const x = bits.readDoubleLE();
  return Number.isFinite(x) ? x : NaN;
}
const edge = [0, -0, Number.MIN_VALUE, -Number.MIN_VALUE, 2**-1022, -(2**-1022),
  Number.MAX_VALUE, -Number.MAX_VALUE, -Infinity, Infinity, NaN, -1, 1, 0.5, -0.5,
  0.49999999999999994, -0.5000000000000001, 2**52, 2**53, 2**32+1, -(2**32+1),
  709.782712893384, -745.1332191019411, Math.PI/2, -Math.PI, 2**20];
const ops = {sin:Math.sin, cos:Math.cos, exp:Math.exp, log:Math.log, log1p:Math.log1p,
  tanh:Math.tanh, atan2:Math.atan2, abs:Math.abs, ceil:Math.ceil, floor:Math.floor, round:Math.round,
  sqrt:Math.sqrt, sign:Math.sign, min:(x,y)=>Math.min(x,y), max:(x,y)=>Math.max(x,y),
  hypot:(x,y)=>Math.hypot(x,y), hypot3:(x,y,z)=>Math.hypot(x,y,z), hypot4:(x,y,z)=>Math.hypot(x,y,z,x),
  int32:x=>x|0,uint32:x=>x>>>0,imul:Math.imul,hashUnit,hueNoise};
let failed = false;
for (const [op, fn] of Object.entries(ops)) {
  const randomCount = 200_000;
  const buf = Buffer.alloc((randomCount + edge.length**2)*32);
  let offset = 0;
  function add(x,y,z) {
    const expected = fn(x,y,z);
    buf.writeDoubleLE(x,offset); buf.writeDoubleLE(y,offset+8); buf.writeDoubleLE(z,offset+16);
    buf.writeDoubleLE(expected,offset+24); offset+=32;
  }
  for(let i=0;i<randomCount;i++) {
    function draw() {
      const u=word()/2**32;
      if (i < 100_000) {
        if (op === 'log') return u || Number.MIN_VALUE;
        if (op === 'log1p') return -u;
        if (['exp','tanh'].includes(op)) return (u-0.5)*24;
        if (['int32','uint32','imul','hashUnit','hueNoise'].includes(op)) return Math.floor(u*2**32);
        return (u-0.5)*720;
      }
      switch(i%5){
        case 0:return (u-0.5)*2;
        case 1:return (u-0.5)*1500;
        case 2:return u*360;
        case 3:return randomDouble();
        default:return (u-0.5)*2**45;
      }
    }
    add(draw(),draw(),draw());
  }
  for(const [i,x] of edge.entries()) for(const [j,y] of edge.entries()) add(x,y,edge[(i+j)%edge.length]);
  const checked = spawnSync('target/release/examples/check_math',[op],{cwd,input:buf,stdio:['pipe','inherit','inherit']});
  if(checked.status !== 0) failed=true;
  else console.log(op, offset/32);
}
if(failed) process.exitCode=1;
