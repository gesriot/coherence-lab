// Generate ICU column-width runs once, not at application startup.
import { writeFileSync } from 'node:fs';
if (process.versions.node !== '26.3.1') throw Error('Requires Node 26.3.1');
const width = process.binding('icu').getStringWidth;
const lines = ['// Node 26.3.1 ICU character widths. Regenerate with tools/width-data.mjs.',
  'pub(crate) const WIDTHS: &[(u32, usize)] = &['];
let last = -1;
for (let code = 0; code <= 0x10ffff; code++) {
  const n = width(String.fromCodePoint(code));
  if (n !== last) lines.push(`    (0x${code.toString(16)}, ${n}),`);
  last = n;
}
writeFileSync(new URL('../crates/jscompat/src/width_data.rs', import.meta.url), lines.join('\n')+'\n];\n');
