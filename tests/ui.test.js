'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const app = require('../public/app.js');

// Fixtures are strictly local to tests and never emitted into public/.
const referenceTime = Date.parse('2026-11-08T12:00:00Z');
function fixture() {
  const raw_draws=Array.from({length:20},(_,i)=>({round:1230+i,date:new Date(Date.parse('2002-12-07T12:00:00Z')+(1229+i)*7*86400000).toISOString().slice(0,10),numbers:[4,15,23,27,34,35],bonus:1}));
  return {
    schema_version:1, status:'current', updated_at:'2026-11-08T10:00:00Z',
    source:{ name:'동행복권', url:'https://www.dhlottery.co.kr/', api_url:'https://www.dhlottery.co.kr/lt645/selectPstLt645InfoNew.do', latest_round:1249, latest_draw_date:'2026-11-07', verified_at:'2026-11-08T10:00:00Z' },
    model:{ id:'ema-frequency-v1', name:'EMA·출현 빈도 통계', timesfm_used:false },
    target_round:1250, predicted_sum:{point:138, low:126, high:150},
    recommended_odd_even:'홀 4 : 짝 2', recommended_high_low:'저 2 : 고 4', hot_zones:'21~30 · 31~40',
    history_rounds:raw_draws.map(d=>d.round), history_sums:new Array(20).fill(138), raw_draws,sample_count:20,
    ball_weights:Object.fromEntries(Array.from({length:45},(_,i)=>[String(i+1),raw_draws[0].numbers.includes(i+1)?8.5:1])), message:'테스트 전용 데이터'
  };
}
function freshFixture() {
  const data = fixture();
  const first=Date.parse('2002-12-07T12:00:00Z');
  const round=Math.floor((Date.now()-first)/(7*86400000))+1;
  data.source.latest_round=round;
  data.target_round=round+1;
  data.source.latest_draw_date=new Date(first+(round-1)*7*86400000).toISOString().slice(0,10);
  data.raw_draws.forEach((draw,i)=>{draw.round=round-19+i;draw.date=new Date(first+(draw.round-1)*7*86400000).toISOString().slice(0,10);});
  data.history_rounds=data.raw_draws.map(d=>d.round);
  data.updated_at=data.source.verified_at=new Date().toISOString();
  return data;
}
class FakeElement {
  constructor(tag='div') { this.tag=tag;this.children=[];this.listeners={};this.attributes={};this.dataset={};this.textContent='';this.disabled=false;this.hidden=false; }
  append(...children) {this.children.push(...children);}
  replaceChildren(...children) {this.children=children;}
  setAttribute(name,value) {this.attributes[name]=value;}
  addEventListener(name,fn) {this.listeners[name]=fn;}
  click() {if(!this.disabled)this.listeners.click?.();}
  allText() {return this.textContent+this.children.map(c=>c.allText()).join(' ');}
}
function fakeDocument({svgFailure=false}={}) {
  const html=fs.readFileSync(path.join(__dirname,'../public/index.html'),'utf8');
  const nodes=Object.fromEntries([...html.matchAll(/\bid="([^"]+)"/g)].map(([,id])=>[id,new FakeElement()]));
  const listeners={};
  return {
    nodes,
    listeners,
    addEventListener(name,fn) {listeners[name]=fn;},
    removeEventListener(name) {delete listeners[name];},
    getElementById(id) {assert.ok(nodes[id],`Missing actual HTML element ${id}`);return nodes[id];},
    createElement(tag) {return new FakeElement(tag);},
    createElementNS(ns,tag) {if(svgFailure)throw new Error('SVG unavailable');return new FakeElement(tag);}
  };
}
function memoryStorage(initial='[]') {
  let value=initial;
  return {getItem:()=>value,setItem:(key,next)=>{value=next;},removeItem:()=>{value=null;}};
}
const response = data => async()=>({ok:true,json:async()=>data});

test('accepts explicit supported schema and rejects legacy embedded model data',()=>{
  assert.equal(app.validateForecast(fixture(),referenceTime).target_round,1250);
  assert.throws(()=>app.validateForecast({target_round:1249,model_name:'TimesFM'},referenceTime));
});

test('rejects malformed provenance, dates, history, range and weights',()=>{
  const changes=[
    d=>d.source.url='https://www.dhlottery.co.kr.evil.example/',
    d=>d.source.url='javascript:alert(1)',
    d=>d.source.verified_at=null,
    d=>d.source.latest_draw_date='2026-02-31',
    d=>d.target_round=1248,
    d=>d.history_rounds[0]=1228,
    d=>d.history_sums[0]=256,
    d=>d.history_sums=[],
    d=>d.predicted_sum.low=151,
    d=>d.predicted_sum.point=NaN,
    d=>d.ball_weights['1']=0,
    d=>delete d.ball_weights['45'],
    d=>d.ball_weights['46']=1,
    d=>d.model.timesfm_used=true,
    d=>delete d.raw_draws,
    d=>d.raw_draws[0].bonus=4,
    d=>d.raw_draws[0].numbers=[1,1,2,3,4,5],
    d=>d.history_sums[0]=130,
    d=>d.ball_weights['2']=2,
    d=>d.source.verified_at='2026-11-08T11:00:00Z',
    d=>d.source.verified_at='2026-11-07T11:59:59Z',
    d=>d.source.latest_draw_date='2026-11-06',
    d=>d.recommended_odd_even='홀 3 : 짝 3'
  ];
  changes.forEach(change=>{const data=fixture();change(data);assert.throws(()=>app.validateForecast(data,referenceTime));});
});

test('a static current file expires exactly at next Saturday 21:00 KST',()=>{
  const data=fixture();
  assert.equal(app.effectiveStatus(data,Date.parse('2026-11-14T11:59:59Z')),'current');
  assert.equal(app.effectiveStatus(data,Date.parse('2026-11-14T12:00:00Z')),'stale');
  data.status='stale';assert.equal(app.effectiveStatus(data,referenceTime),'stale');
  data.status='unavailable';assert.equal(app.effectiveStatus(data,referenceTime),'unavailable');
});

test('generated games respect uniqueness, fixed numbers and declared sum constraints',()=>{
  let seed=83421;
  const rng=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
  for(let i=0;i<200;i++) {
    const game=app.sampleGame(fixture(),[4,23,41],rng);
    assert.ok(app.validNumbers(game.numbers));
    assert.ok([4,23,41].every(n=>game.numbers.includes(n)));
    assert.ok(game.sum>=126&&game.sum<=150);
  }
});

test('impossible fixed conditions and exhausted attempts return an error, never an out-of-range game',()=>{
  const data=fixture();data.predicted_sum={low:21,point:25,high:30};
  assert.throws(()=>app.sampleGame(data,[43,44,45]),/동시에 만족/);
  assert.throws(()=>app.sampleGame(fixture(),[],()=>0,3),/시도 횟수/);
  assert.throws(()=>app.sampleGame(fixture(),[],()=>1),/난수/);
  assert.throws(()=>app.sampleGame(fixture(),[4,4]),/고정수/);
});

test('corrupt and malicious stored entries are not rendered or overwritten',()=>{
  const storage=memoryStorage('not-json');
  assert.ok(app.readSaved(storage).error);
  assert.equal(storage.getItem(),'not-json');
  const valid={round:1249,numbers:[1,2,3,4,5,6],sum:21};
  const injected={...valid,round:'<img src=x onerror=alert(1)>'};
  const mixed=memoryStorage(JSON.stringify([valid,injected]));
  assert.deepEqual(app.readSaved(mixed).games,[valid]);
  assert.ok(app.readSaved(mixed).error);
  assert.equal(JSON.parse(mixed.getItem()).length,2);
});

test('HTTP failure, network error, invalid JSON and legacy JSON all fail closed without generated values',async()=>{
  const fetchers=[
    async()=>({ok:false,status:404}),
    async()=>{throw new Error('Network failure');},
    async()=>({ok:true,json:async()=>{throw new SyntaxError('Bad JSON');}}),
    response({target_round:1249,model_name:'TimesFM'})
  ];
  for(const fetcher of fetchers) {
    const doc=fakeDocument();const instance=app.start(doc,fetcher,()=>memoryStorage());await instance.ready;
    assert.equal(doc.nodes['status-panel'].dataset.state,'unavailable');
    assert.equal(doc.nodes['btn-generate'].disabled,true);
    assert.equal(doc.nodes['top-weights-container'].children.length,0);
    instance.generate();assert.equal(doc.nodes['generated-results'].children.length,0);
    assert.match(doc.nodes['chart-container'].allText(),/확인된 추첨 데이터가 없습니다/);
  }
});

test('unavailable output never renders values even if it contains legacy number fields',async()=>{
  const data=freshFixture();data.status='unavailable';
  const doc=fakeDocument();const instance=app.start(doc,response(data),()=>memoryStorage());await instance.ready;
  assert.equal(doc.nodes['status-panel'].dataset.state,'unavailable');
  assert.equal(doc.nodes['source-round'].textContent,'—');
  assert.equal(doc.nodes['top-weights-container'].children.length,0);
});

test('stale data retains history while generation is disabled',async()=>{
  const data=freshFixture();data.status='stale';
  const doc=fakeDocument();const instance=app.start(doc,response(data),()=>memoryStorage());await instance.ready;
  assert.equal(doc.nodes['status-panel'].dataset.state,'stale');
  assert.equal(doc.nodes['btn-generate'].disabled,true);
  assert.equal(doc.nodes['history-table'].children.length,20);
  assert.ok(doc.nodes['top-weights-container'].children.every(b=>b.disabled));
  instance.generate();assert.equal(doc.nodes['generated-results'].children.length,0);
});

test('SVG failure and blocked browser storage do not prevent a valid current generator',async()=>{
  const data=freshFixture();
  const doc=fakeDocument({svgFailure:true});
  const instance=app.start(doc,response(data),()=>{throw new Error('Storage blocked');});await instance.ready;
  assert.equal(doc.nodes['status-panel'].dataset.state,'current');
  assert.equal(doc.nodes['btn-generate'].disabled,false);
  assert.match(doc.nodes['storage-message'].textContent,/사용할 수 없습니다/);
  assert.match(doc.nodes['chart-container'].allText(),/데이터 표/);
  assert.equal(doc.nodes['history-table'].children.length,20);
  instance.generate();assert.equal(doc.nodes['generated-results'].children.length,5);
});

test('saving a new game leaves malformed storage intact and tells the user',async()=>{
  const doc=fakeDocument();const storage=memoryStorage('BROKEN');
  const instance=app.start(doc,response(freshFixture()),()=>storage);await instance.ready;
  instance.generate();
  doc.nodes['generated-results'].children[0].children[1].children[1].click();
  assert.equal(storage.getItem(),'BROKEN');
  assert.match(doc.nodes['storage-message'].textContent,/초기화/);
});

test('generation and saving use server-derived round, sum and numbers',async()=>{
  const doc=fakeDocument();const storage=memoryStorage();
  const data=freshFixture();
  const instance=app.start(doc,response(data),()=>storage);await instance.ready;
  instance.generate();doc.nodes['generated-results'].children[0].children[1].children[1].click();
  const saved=JSON.parse(storage.getItem());
  assert.equal(saved[0].round,data.source.latest_round+1);
  assert.ok(app.validNumbers(saved[0].numbers));
  assert.equal(saved[0].sum,saved[0].numbers.reduce((a,b)=>a+b,0));
  assert.equal(doc.nodes['saved-games-list'].children.length,1);
});

test('page has no external script dependency or old fabricated claims',()=>{
  const html=fs.readFileSync(path.join(__dirname,'../public/index.html'),'utf8');
  assert.doesNotMatch(html,/<script[^>]+src=["']https?:/);
  assert.doesNotMatch(html,/제1249회|128\s*~\s*152|TimesFM Engine|FALLBACK_DATA|신뢰구간 80%|광고 영역/);
  assert.match(html,/TimesFM 모델은 사용하지 않습니다/);
});

test('an open tab expires on status refresh and clears old generated games',async()=>{
  const data=fixture();const doc=fakeDocument();let now=Date.parse('2026-11-14T11:59:59Z');
  const instance=app.start(doc,response(data),()=>memoryStorage(),{now:()=>now});await instance.ready;
  instance.generate();assert.equal(doc.nodes['generated-results'].children.length,5);
  now=Date.parse('2026-11-14T12:00:00Z');instance.refreshStatus();
  assert.equal(doc.nodes['status-panel'].dataset.state,'stale');
  assert.equal(doc.nodes['btn-generate'].disabled,true);
  assert.equal(doc.nodes['generated-results'].children.length,0);
  assert.match(doc.nodes['generator-message'].textContent,/새 추첨 회차/);
  instance.destroy();
});

test('a save click after weekly rollover expires the page instead of silently storing a past-round game',async()=>{
  const data=fixture();const doc=fakeDocument();const storage=memoryStorage();let now=Date.parse('2026-11-14T11:59:59Z');
  const instance=app.start(doc,response(data),()=>storage,{now:()=>now});await instance.ready;
  instance.generate();const saveButton=doc.nodes['generated-results'].children[0].children[1].children[1];
  now=Date.parse('2026-11-14T12:00:00Z');saveButton.click();
  assert.equal(storage.getItem(),'[]');
  assert.equal(doc.nodes['status-panel'].dataset.state,'stale');
  assert.match(doc.nodes['storage-message'].textContent,/저장하지 않았습니다/);
  instance.destroy();
});

test('rollover is wired to the minute timer and visibility event with clean teardown',async()=>{
  const doc=fakeDocument();let now=Date.parse('2026-11-14T11:59:59Z');let tick;let cleared=false;
  const timer={unref(){}};
  const instance=app.start(doc,response(fixture()),()=>memoryStorage(),{
    now:()=>now,
    setInterval(fn,delay){assert.equal(delay,60000);tick=fn;return timer;},
    clearInterval(value){assert.equal(value,timer);cleared=true;}
  });
  await instance.ready;
  assert.equal(typeof doc.listeners.visibilitychange,'function');
  instance.generate();
  now=Date.parse('2026-11-14T12:00:00Z');tick();
  assert.equal(doc.nodes['status-panel'].dataset.state,'stale');
  assert.equal(doc.nodes['generated-results'].children.length,0);
  instance.destroy();assert.equal(cleared,true);assert.equal(doc.listeners.visibilitychange,undefined);
});
