import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';
const sandbox = {window: {}};
vm.runInNewContext(readFileSync(new URL('../static/comparison.js', import.meta.url), 'utf8'), sandbox);
const summarize = sandbox.window.PrismRAWComparison;

test('quality and latency can favor different methods', () => {
  const rows = summarize({processing_ms:{classical:10,hybrid:20}, metrics:{classical:{psnr:30,delta_e_00:2},hybrid:{psnr:32,delta_e_00:3}}});
  assert.equal(rows[0].methods.join(','),'hybrid');
  assert.equal(rows[1].methods.join(','),'classical');
  assert.equal(rows[2].methods.join(','),'classical');
});
test('missing references have no invented winners', () => {
  const rows = summarize({processing_ms:{classical:10},metrics:{}});
  assert.equal(rows[0].methods.length,0); assert.equal(rows[1].value,null);
  assert.equal(rows[2].methods[0],'classical');
});
test('ties and perfect matches remain explicit', () => {
  const rows = summarize({processing_ms:{classical:10,hybrid:10},metrics:{classical:{psnr:null,delta_e_00:0},hybrid:{psnr:null,delta_e_00:0}}});
  assert.equal(rows[0].value,Infinity); assert.equal(rows[0].methods.length,2);
  assert.equal(rows[2].methods.length,2);
});
