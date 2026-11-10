'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {execFileSync}=require('node:child_process');
const core=require('../public/app.js');
const ui=require('../public/timesfm.js');
const ROOT=path.join(__dirname,'..');
const NOW=Date.parse('2026-11-10T15:00:00Z');

// All synthetic numbers and the fake model remain in test-process memory.
// This fixture checks the producer/consumer contract; it is not evidence of inference.
const fixture=JSON.parse(execFileSync('python',['-c',String.raw`
import copy,json,random
from datetime import datetime,timezone
from unittest.mock import patch
from scripts import fetch_and_predict as official
from scripts import timesfm_forecast as pipeline
now=datetime(2026,11,10,15,tzinfo=timezone.utc)
last=official.latest_expected_round(now)
rng=random.Random(77531)
draws=[]
for r in range(last-167,last+1):
    nums=sorted(rng.sample(range(1,46),6))
    bonus=rng.choice([n for n in range(1,46) if n not in nums])
    draws.append({'round':r,'date':official.round_draw_date(r).isoformat(),'numbers':nums,'bonus':bonus})
reference=official.build_report(draws[-64:],now)
history={'schema_version':1,'status':'current','source':copy.deepcopy(reference['source']),'updated_at':official.utc_timestamp(now),'count':len(draws),'raw_draws':draws}
class FakeRunner:
    def __init__(self):
        self.metadata=pipeline.desired_model_metadata(64)
        self.metadata.update(checkpoint_verified=True,runtime_versions={'timesfm':'2.0.2'})
    def predict(self,contexts):
        self.metadata['timesfm_used']=True
        return [sum(v)/len(v)+0.1*v[-1] for v in contexts]
class FrozenDateTime(datetime):
    @classmethod
    def now(cls,tz=None):
        return now if tz is not None else now.replace(tzinfo=None)
with patch.object(pipeline,'datetime',FrozenDateTime):
    report=pipeline.analyze(history,FakeRunner(),context=64,holdout=104,bootstrap=100)
report['generated_at']=official.utc_timestamp(now)
for key in ('by_iso_year','by_iso_week','by_iso_year_week'):
    report['statistics'].pop(key)
report['statistics']['target_week'].pop('by_iso_year')
report['statistics']['detail_source']='./data/draw-history.json'
print(json.dumps({'report':report,'reference':reference,'history':history},ensure_ascii=True))
`],{cwd:ROOT,encoding:'utf8',maxBuffer:8*1024*1024}));
const copy=()=>structuredClone(fixture);
class Element {
  constructor(tag='div'){this.tag=tag;this.children=[];this.listeners={};this.attributes={};this.dataset={};this.textContent='';this.value='';this.hidden=false;}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  addEventListener(name,fn){this.listeners[name]=fn;}
  removeEventListener(name){delete this.listeners[name];}
  removeAttribute(name){delete this.attributes[name];if(name==='href')delete this.href;}
  allText(){return this.textContent+this.children.map(c=>c.allText()).join(' ');}
}
function documentStub(){
  const html=fs.readFileSync(path.join(ROOT,'public/index.html'),'utf8');
  const nodes=Object.fromEntries([...html.matchAll(/\bid="([^"]+)"/g)].map(([,id])=>[id,new Element()]));
  const listeners={};
  return {nodes,listeners,getElementById(id){assert.ok(nodes[id],`Missing HTML id ${id}`);return nodes[id];},createElement(tag){return new Element(tag);},addEventListener(name,fn){listeners[name]=fn;},removeEventListener(name){delete listeners[name];}};
}
function fetcherFor(data,overrides={}){
  return async url=>overrides[url]?overrides[url]():({ok:true,status:200,json:async()=>url.includes('timesfm.json')?data.report:url.includes('draw-history.json')?data.history:data.reference});
}
async function start(data=copy(),overrides={},options={}){
  const doc=documentStub(),app=ui.start(doc,fetcherFor(data,overrides),core,{now:()=>NOW,...options});await app.ready;return {doc,app};
}

test('trimmed Python report and independent official history satisfy the browser contract',()=>{
  const data=copy();assert.equal(ui.validateReport(data.report,data.reference,data.history,core,NOW),data.report);
  assert.equal(data.report.validation.evaluated_count,104);
  assert.equal(data.report.recommendations.methods.combined.games.length,5);
});

test('invalid provenance, model claims, source rounds, history and recommendations are rejected',()=>{
  const mutations=[
    d=>d.report.model.timesfm_used=false,
    d=>d.report.model.checkpoint_verified=false,
    d=>d.report.model.revision='not-a-revision',
    d=>d.report.source.latest_round--,
    d=>d.report.source.verified_at='2026-11-11T00:00:00Z',
    d=>d.report.prediction.rank_scores['1']=2,
    d=>d.report.prediction.raw_scores['4']=NaN,
    d=>d.report.recommendations.methods.combined.games[0]=[1,1,2,3,4,5],
    d=>d.report.recommendations.methods.combined.games.pop(),
    d=>d.report.recommendations.methods.combined.games[1]=d.report.recommendations.methods.combined.games[0],
    d=>d.report.recommendations.methods.combined.weights[0]+=.1,
    d=>d.history.raw_draws[0].bonus=d.history.raw_draws[0].numbers[0],
    d=>d.history.raw_draws[0].date='2020-01-01',
    d=>d.history.source.api_url='https://invalid.example/data',
    d=>d.history.status='stale',
    d=>d.report.statistics.overall.numbers[0].main_rate=.99,
    d=>d.report.statistics.overall.numbers[0].second_qualifying_rate=.99,
    d=>d.report.statistics.historical_rule.derived_combinations_are_independent_draws=true,
    d=>d.report.statistics.historical_rule.first_prize_weight=.5,
    d=>d.report.statistics.target_week.sample_count++,
    d=>d.report.statistics.combination_repeat_summary.first_prize.maximum_count++,
    d=>d.report.statistics.detail_source='https://untrusted.example/data'
  ];
  mutations.forEach(mutate=>{const data=copy();mutate(data);assert.throws(()=>ui.validateReport(data.report,data.reference,data.history,core,NOW));});
});

test('retrospective metrics, actual results, prize ranks and paired deltas must agree with the ledger',()=>{
  const mutations=[
    d=>d.report.validation.evaluated_count=103,
    d=>d.report.validation.rows[0].context_end_round++,
    d=>d.report.validation.rows[0].actual_bonus=46,
    d=>d.report.validation.rows[0].methods.combined.games[0].matches=6,
    d=>d.report.validation.metrics.combined.prize_rank_counts['1']++,
    d=>d.report.validation.metrics.uniform.avg_matches_per_game+=.1,
    d=>d.report.validation.paired_vs_uniform.combined.mean_delta+=.1,
    d=>d.report.validation.paired_vs_uniform.combined.ci95=[1,-1],
    d=>d.report.validation.bootstrap.resampling_unit='individual_game'
  ];
  mutations.forEach(mutate=>{const data=copy();mutate(data);assert.throws(()=>ui.validateReport(data.report,data.reference,data.history,core,NOW));});
});

test('current valid data renders five six-number games and full 104-by-five comparison',async()=>{
  const {doc,app}=await start();
  assert.equal(doc.nodes['timesfm-panel'].dataset.state,'current');
  assert.equal(doc.nodes['timesfm-results'].hidden,false);
  assert.equal(doc.nodes['timesfm-games'].children.length,5);
  assert.ok(doc.nodes['timesfm-games'].children.every(row=>row.children[0].children.length===7));
  assert.equal(doc.nodes['timesfm-metrics'].children.length,4);
  assert.equal(doc.nodes['timesfm-comparison-table'].children.length,3);
  assert.match(doc.nodes['timesfm-evaluation-window'].textContent,/104회차 × 방식별 5게임 = 각 520게임/);
  assert.equal(doc.nodes['timesfm-frequency-table'].children.length,45);
  assert.match(doc.nodes['timesfm-library'].textContent,/timesfm 2.0.2/);
  app.destroy();
});

test('missing files, HTTP errors, malformed JSON and stale official reference never render recommendations',async()=>{
  const cases=[
    {overrides:{'./data/timesfm.json':()=>({ok:false,status:404})}},
    {overrides:{'./data/draw-history.json':()=>({ok:false,status:500})}},
    {overrides:{'./data/timesfm.json':()=>({ok:true,json:async()=>{throw new Error('bad JSON');}})}},
    {mutate:d=>d.reference.status='stale'},
    {mutate:d=>d.history.status='stale'},
    {mutate:d=>d.report.source.latest_round--}
  ];
  for(const item of cases){const data=copy();item.mutate?.(data);const {doc,app}=await start(data,item.overrides);assert.equal(doc.nodes['timesfm-results'].hidden,true);assert.equal(doc.nodes['timesfm-games'].children.length,0);assert.notEqual(doc.nodes['timesfm-panel'].dataset.state,'current');app.destroy();}
});

test('a failed partial run with real model calls still exposes no model values',async()=>{
  const data=copy();data.report.status='failed';data.report.prediction=null;data.report.recommendations=null;data.report.statistics=null;data.report.model.timesfm_used=true;
  const {doc,app}=await start(data);assert.equal(doc.nodes['timesfm-panel'].dataset.state,'failed');assert.equal(doc.nodes['timesfm-results'].hidden,true);assert.equal(doc.nodes['timesfm-games'].children.length,0);assert.equal(doc.nodes['timesfm-model-name'].textContent,'');app.destroy();
});

test('same-week/year selectors use raw official counts with separate main and bonus denominators',async()=>{
  const data=copy();const {doc,app}=await start(data);const select=doc.nodes['timesfm-stat-scope'];
  select.value='overall';select.listeners.change();
  assert.match(doc.nodes['timesfm-stat-denominator'].textContent,/주·보너스 168회/);
  const row=doc.nodes['timesfm-frequency-table'].children[0];
  const expected=data.history.raw_draws.filter(d=>d.numbers.includes(1)).length;
  const bonus=data.history.raw_draws.filter(d=>d.bonus===1).length;
  assert.equal(row.children[1].textContent,expected);assert.equal(row.children[3].textContent,bonus);
  assert.equal(row.children[2].textContent,(100*expected/168).toFixed(2)+'%');
  assert.equal(row.children[5].textContent,5*expected+6*bonus);
  assert.equal(row.children[6].textContent,(100*(5*expected+6*bonus)/(6*168)).toFixed(2)+'%');
  const yearOption=select.children.find(o=>o.value.startsWith('year:'));select.value=yearOption.value;select.listeners.change();
  assert.equal(doc.nodes['timesfm-frequency-table'].children.length,45);
  doc.nodes['timesfm-history-week'].listeners.change();assert.equal(doc.nodes['timesfm-week-combinations'].children.length,8);
  app.destroy();
});

test('ISO week/year uses the draw calendar correctly at year boundaries',()=>{
  assert.deepEqual(ui.isoWeek('2021-01-02'),{year:2020,week:53});
  assert.deepEqual(ui.isoWeek('2021-01-09'),{year:2021,week:1});
});

test('exact-combination singletons do not become a misleading top-ranked preview',async()=>{
  const {doc,app}=await start();assert.match(doc.nodes['timesfm-repeat-note'].textContent,/모두 1회씩 등장/);assert.equal(doc.nodes['timesfm-repeat-preview'].children.length,0);app.destroy();
});

test('open-tab weekly rollover removes recommendation values and model metadata',async()=>{
  let now=NOW;const {doc,app}=await start(copy(),{},{now:()=>now});assert.equal(doc.nodes['timesfm-games'].children.length,5);
  now=Date.parse('2026-11-14T12:00:00Z');app.refreshStatus();assert.equal(doc.nodes['timesfm-panel'].dataset.state,'stale');assert.equal(doc.nodes['timesfm-games'].children.length,0);assert.equal(doc.nodes['timesfm-results'].hidden,true);assert.equal(doc.nodes['timesfm-model-name'].textContent,'');app.destroy();
});

test('no-inference fallback and old sum contract cannot masquerade as number recommendations',()=>{
  const data=copy();delete data.report.prediction.raw_scores;data.report.forecast={point:138};assert.throws(()=>ui.validateReport(data.report,data.reference,data.history,core,NOW));
  const html=fs.readFileSync(path.join(ROOT,'public/index.html'),'utf8');assert.doesNotMatch(html,/합계 예측 실험|timesfm-point|MAE|RMSE/);assert.match(html,/TimesFM 번호 순위 50%/);assert.match(html,/실제 2등 당첨자가 구매한 조합 데이터가 아닙니다/);
  assert.ok(html.indexOf('id="timesfm-panel"')<html.indexOf('aria-labelledby="summary-heading"'));
});

test('replay presents original inference time separately from statistics recalculation time',async()=>{
  const data=copy();
  data.report.audit_replay={source_sha256:'a'.repeat(64),model_score_ledger_sha256:'b'.repeat(64),original_generated_at:data.report.generated_at,raw_model_predictions_unchanged:true,inference_rerun:false,origin_count:105,series_per_origin:45};
  data.report.model.completed_forecast_calls=105;data.report.generated_at=new Date(NOW+30000).toISOString();
  const {doc,app}=await start(data,{},{now:()=>NOW+30000});
  assert.equal(doc.nodes['timesfm-panel'].dataset.state,'current');
  assert.equal(doc.nodes['timesfm-recalculated-row'].hidden,false);
  assert.notEqual(doc.nodes['timesfm-generated-at'].textContent,doc.nodes['timesfm-recalculated-at'].textContent);
  assert.match(doc.nodes['timesfm-replay-note'].textContent,/그대로 재사용/);
  for(const mutate of [d=>d.report.audit_replay.original_generated_at='2026-11-10T16:00:00Z',d=>d.report.audit_replay.inference_rerun=true,d=>d.report.audit_replay.raw_model_predictions_unchanged=false,d=>d.report.audit_replay.source_sha256='bad',d=>d.report.model.completed_forecast_calls=1]){
    const altered=structuredClone(data);mutate(altered);assert.throws(()=>ui.validateReport(altered.report,altered.reference,altered.history,core,NOW+30000));
  }
  app.destroy();
});
