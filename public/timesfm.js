'use strict';

const TimesFMExperiments = (() => {
  const METHODS = ['timesfm', 'statistics', 'combined', 'uniform'];
  const LABELS = {timesfm:'TimesFM만',statistics:'통계만',combined:'TimesFM + 통계',uniform:'균등 무작위'};
  const MODEL_ID = 'google/timesfm-2.5-200m-pytorch';
  const REVISION = '1d952420fba87f3c6dee4f240de0f1a0fbc790e3';
  const CHECKPOINT = '2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8';
  const FIRST_DRAW = Date.parse('2002-12-07T12:00:00Z');
  const WEEK = 7 * 86400000;
  const finite = n => typeof n === 'number' && Number.isFinite(n);
  const object = o => o !== null && typeof o === 'object' && !Array.isArray(o);
  const integer = (n,min=0,max=10000) => Number.isSafeInteger(n) && n >= min && n <= max;
  const close = (a,b) => finite(a) && finite(b) && Math.abs(a-b) <= 1e-8;
  const same = (a,b) => JSON.stringify(a) === JSON.stringify(b);
  const requireValue = (condition,message='실험 데이터의 형식이나 근거가 일치하지 않습니다.') => {if(!condition)throw new Error(message);};
  const stamp = value => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}T.*(?:Z|[+-]\d\d:\d\d)$/.test(value) && Number.isFinite(Date.parse(value));
  const drawDate = round => new Date(FIRST_DRAW+(round-1)*WEEK).toISOString().slice(0,10);
  const expectedRound = now => Math.max(0,Math.floor((now-FIRST_DRAW)/WEEK)+1);
  const numbers = values => Array.isArray(values) && values.length===6 && values.every((n,i)=>integer(n,1,45)&&(!i||n>values[i-1]));

  function isoWeek(date) {
    const d=new Date(date+'T00:00:00Z');
    d.setUTCDate(d.getUTCDate()+4-(d.getUTCDay()||7));
    const year=d.getUTCFullYear();
    return {year,week:Math.ceil((((d-Date.UTC(year,0,1))/86400000)+1)/7)};
  }
  function vector(map) {
    requireValue(object(map)&&Object.keys(map).length===45);
    const values=Array.from({length:45},(_,i)=>map[String(i+1)]);
    requireValue(values.every(finite));return values;
  }
  function rankScores(values) {
    const sorted=values.map((value,index)=>({value,index})).sort((a,b)=>a.value-b.value);
    const ranks=new Array(45);let start=0;
    while(start<45){let end=start+1;while(end<45&&sorted[end].value===sorted[start].value)end++;const rank=(start+end-1)/2/44;for(let i=start;i<end;i++)ranks[sorted[i].index]=rank;start=end;}
    return ranks;
  }
  function checkGames(games) {
    requireValue(Array.isArray(games)&&games.length===5&&games.every(numbers)&&new Set(games.map(g=>g.join(','))).size===5,'6개의 서로 다른 번호로 된 5게임을 확인할 수 없습니다.');
  }
  function secondCombinations(main,bonus) {
    return main.map((_,i)=>main.filter((n,j)=>j!==i).concat(bonus).sort((a,b)=>a-b)).sort(compareNumbers);
  }
  function compareNumbers(a,b) {for(let i=0;i<6;i++){if(a[i]!==b[i])return a[i]-b[i];}return 0;}
  function groupCounts(draws) {
    const main=new Array(45).fill(0),bonus=new Array(45).fill(0);
    draws.forEach(d=>{d.numbers.forEach(n=>main[n-1]++);bonus[d.bonus-1]++;});return {main,bonus};
  }
  function checkGroup(group,draws) {
    requireValue(object(group)&&group.sample_count===draws.length&&group.rate_denominator==='draws in this group'&&group.second_qualifying_rate_denominator==='six derived combinations per actual draw'&&Array.isArray(group.numbers)&&group.numbers.length===45);
    const counts=groupCounts(draws),n=draws.length;
    group.numbers.forEach((item,i)=>{
      requireValue(object(item)&&item.number===i+1&&item.main_hits===counts.main[i]&&item.bonus_hits===counts.bonus[i]);
      requireValue(n ? close(item.main_rate,counts.main[i]/n)&&close(item.bonus_rate,counts.bonus[i]/n) : item.main_rate===null&&item.bonus_rate===null);
      requireValue(close(item.smoothed_main_rate,(counts.main[i]+20*6/45)/(n+20)));
      const secondHits=5*counts.main[i]+6*counts.bonus[i],smoothedSecond=(secondHits/6+20*6/45)/(n+20);
      requireValue(item.second_qualifying_hits===secondHits&&(n?close(item.second_qualifying_rate,secondHits/(6*n)):item.second_qualifying_rate===null)&&close(item.smoothed_second_qualifying_rate,smoothedSecond)&&close(item.historical_selection_score,.75*item.smoothed_main_rate+.25*smoothedSecond));
    });
  }
  function checkRepeat(summary,draws,second=false) {
    const counts=new Map();
    draws.forEach(d=>(second?secondCombinations(d.numbers,d.bonus):[d.numbers]).forEach(ns=>{const key=ns.join(',');counts.set(key,(counts.get(key)||0)+1);}));
    const sorted=[...counts].map(([key,count])=>({numbers:key.split(',').map(Number),count})).sort((a,b)=>b.count-a.count||compareNumbers(a.numbers,b.numbers));
    const preview=sorted.slice(0,10),cutoff=preview.length?preview.at(-1).count:null;
    requireValue(object(summary)&&summary.observations===draws.length*(second?6:1)&&summary.unique_combinations===counts.size&&summary.repeated_combinations===sorted.filter(x=>x.count>1).length&&summary.maximum_count===(sorted[0]?.count||0));
    requireValue(summary.preview_limit===10&&same(summary.preview,preview)&&summary.preview_cutoff_count===cutoff&&summary.total_tied_at_cutoff===sorted.filter(x=>x.count===cutoff).length);
  }
  function makeGroup(draws) {
    const counts=groupCounts(draws),n=draws.length;
    return {sample_count:n,numbers:Array.from({length:45},(_,i)=>({number:i+1,main_hits:counts.main[i],bonus_hits:counts.bonus[i],main_rate:n?counts.main[i]/n:null,bonus_rate:n?counts.bonus[i]/n:null,second_qualifying_hits:5*counts.main[i]+6*counts.bonus[i],second_qualifying_rate:n?(5*counts.main[i]+6*counts.bonus[i])/(6*n):null}))};
  }
  function detailGroups(draws) {
    const years=new Map(),weeks=new Map();
    draws.forEach(draw=>{const iso=isoWeek(draw.date);if(!years.has(iso.year))years.set(iso.year,[]);if(!weeks.has(iso.week))weeks.set(iso.week,[]);years.get(iso.year).push(draw);weeks.get(iso.week).push(draw);});
    return {
      by_iso_year:[...years].map(([iso_year,group])=>({iso_year,...makeGroup(group)})),
      by_iso_week:[...weeks].sort((a,b)=>a[0]-b[0]).map(([iso_week,group])=>({iso_week,...makeGroup(group)})),
      by_iso_year_week:draws.map(d=>{const iso=isoWeek(d.date);return {iso_year:iso.year,iso_week:iso.week,rounds:[d.round],first_prize_combinations:[{numbers:d.numbers,count:1}],second_prize_qualifying_combinations:secondCombinations(d.numbers,d.bonus).map(numbers=>({numbers,count:1}))};})
    };
  }
  function validateHistory(history,reference,now) {
    requireValue(object(history)&&history.schema_version===1&&history.status==='current'&&object(history.source)&&Array.isArray(history.raw_draws)&&integer(history.count,32,10000)&&history.count===history.raw_draws.length,'연도·주차 통계의 공식 원본 이력을 확인할 수 없습니다.');
    const source=history.source;
    requireValue(source.name==='동행복권'&&source.url==='https://www.dhlottery.co.kr/'&&source.api_url===reference.source.api_url&&source.latest_round===reference.source.latest_round&&source.latest_draw_date===reference.source.latest_draw_date,'연도·주차 자료와 최신 공식 추첨 회차가 다릅니다.');
    requireValue(stamp(source.verified_at)&&stamp(history.updated_at)&&Date.parse(source.verified_at)<=Date.parse(history.updated_at)&&Date.parse(history.updated_at)<=now+60000&&expectedRound(Date.parse(source.verified_at))===source.latest_round);
    const first=history.raw_draws[0]?.round;
    requireValue(integer(first,1)&&first+history.count-1===source.latest_round);
    history.raw_draws.forEach((draw,i)=>requireValue(object(draw)&&draw.round===first+i&&draw.date===drawDate(draw.round)&&numbers(draw.numbers)&&integer(draw.bonus,1,45)&&!draw.numbers.includes(draw.bonus)));
    return history.raw_draws;
  }
  function validateStatistics(stats,input,target,reference,history,now) {
    requireValue(object(stats)&&stats.sample_count===input.count&&stats.first_round===input.first_round&&stats.last_round===input.last_round&&stats.first_date===drawDate(input.first_round)&&stats.last_date===input.last_draw_date&&stats.target_date===target);
    const targetIso=isoWeek(target);
    requireValue(stats.target_iso_year===targetIso.year&&stats.target_iso_week===targetIso.week&&stats.smoothing?.prior_strength_draws===20&&close(stats.smoothing.main_rate_prior,6/45));
    const rule=stats.historical_rule;
    requireValue(object(rule)&&rule.id==='first-second-inclusion-v1'&&rule.first_prize_weight===.75&&rule.second_prize_qualifying_weight===.25&&rule.second_combinations_per_draw===6&&rule.prior_strength_actual_draws===20&&rule.derived_combinations_are_independent_draws===false&&rule.weights_selected_using_holdout_metrics===false,'1등·2등 가능 조합 통계의 고정 결합 규칙을 확인할 수 없습니다.');
    requireValue(stats.detail_source==='./data/draw-history.json');
    const draws=validateHistory(history,reference,now).filter(d=>d.round>=input.first_round&&d.round<=input.last_round);
    requireValue(draws.length===input.count&&draws[0].round===input.first_round&&draws.at(-1).round===input.last_round);
    for(const previous of reference.raw_draws){if(previous.round<input.first_round)continue;const actual=draws[previous.round-input.first_round];requireValue(actual&&same(actual.numbers,previous.numbers)&&actual.bonus===previous.bonus,'통계에 포함된 추첨 결과가 현재 공식 자료와 다릅니다.');}
    checkGroup(stats.overall,draws);
    const seasonal=draws.filter(d=>isoWeek(d.date).week===targetIso.week);
    checkGroup(stats.target_week,seasonal);
    requireValue(object(stats.combination_repeat_summary));
    checkRepeat(stats.combination_repeat_summary.first_prize,draws);
    checkRepeat(stats.combination_repeat_summary.second_prize_qualifying,draws,true);
    checkRepeat(stats.target_week.first_prize,seasonal);
    checkRepeat(stats.target_week.second_prize_qualifying,seasonal,true);
    return draws;
  }
  function checkValidation(validation,report,draws) {
    const context=report.prediction.context_length;
    requireValue(object(validation)&&['complete','insufficient','not_requested'].includes(validation.status)&&validation.protocol==='rolling-origin-one-step-number-occurrence'&&validation.context_length===context&&validation.games_per_method===5&&integer(validation.seed,0,2**32-1)&&integer(validation.requested_holdout,0,512));
    requireValue(report.recommendations.seed===validation.seed+report.prediction.target_round);
    if(validation.status!=='complete'){
      requireValue(validation.evaluated_count===0&&same(validation.rows,[])&&same(validation.metrics,{})&&same(validation.paired_vs_uniform,{}));
      requireValue(validation.status==='not_requested'?validation.requested_holdout===0:validation.requested_holdout>0&&draws.length<context+validation.requested_holdout);
      return;
    }
    const count=validation.evaluated_count;
    requireValue(integer(count,1,512)&&count===validation.requested_holdout&&Array.isArray(validation.rows)&&validation.rows.length===count&&draws.length>=context+count);
    requireValue(validation.first_target_round===report.source.latest_round-count+1&&validation.last_target_round===report.source.latest_round);
    const totals=Object.fromEntries(METHODS.map(m=>[m,{matches:0,best:0,ranks:{'1':0,'2':0,'3':0,'4':0,'5':0,none:0}}]));
    validation.rows.forEach((row,i)=>{
      const target=validation.first_target_round+i,actual=draws[target-report.input.first_round];
      requireValue(object(row)&&row.target_round===target&&row.target_date===drawDate(target)&&row.context_end_round===target-1&&row.context_start_round===target-context&&row.history_count===target-report.input.first_round&&row.seed===validation.seed+target&&same(row.actual_numbers,actual.numbers)&&row.actual_bonus===actual.bonus);
      vector(row.raw_scores);requireValue(object(row.methods));
      METHODS.forEach(method=>{
        const results=row.methods[method];requireValue(object(results)&&Array.isArray(results.games));checkGames(results.games.map(g=>g.numbers));
        let total=0,best=0;
        results.games.forEach(game=>{
          const matches=game.numbers.filter(n=>actual.numbers.includes(n)).length,bonus=game.numbers.includes(actual.bonus);
          const rank=matches===6?1:matches===5&&bonus?2:matches===5?3:matches===4?4:matches===3?5:null;
          requireValue(game.matches===matches&&game.bonus_match===bonus&&game.rank===rank);
          total+=matches;best=Math.max(best,matches);totals[method].ranks[rank===null?'none':String(rank)]++;
        });
        requireValue(close(results.mean_matches,total/5)&&results.best_matches===best);
        totals[method].matches+=total;totals[method].best+=best;
      });
    });
    requireValue(object(validation.metrics)&&object(validation.paired_vs_uniform));
    METHODS.forEach(method=>{
      const metric=validation.metrics[method],computed=totals[method];
      requireValue(object(metric)&&metric.games===count*5&&close(metric.avg_matches_per_game,computed.matches/(count*5))&&close(metric.avg_best_of_five,computed.best/count)&&object(metric.prize_rank_counts));
      Object.keys(computed.ranks).forEach(rank=>requireValue(metric.prize_rank_counts[rank]===computed.ranks[rank]));
      if(method!=='uniform'){
        const paired=validation.paired_vs_uniform[method],delta=(computed.matches-totals.uniform.matches)/(count*5);
        requireValue(object(paired)&&close(paired.mean_delta,delta)&&paired.rounds===count&&paired.unit==='mean_matches_per_game_by_round'&&Array.isArray(paired.ci95)&&paired.ci95.length===2&&paired.ci95.every(n=>finite(n)&&n>=-6&&n<=6)&&paired.ci95[0]<=paired.ci95[1]);
      }
    });
    const bootstrap=validation.bootstrap;
    requireValue(object(bootstrap)&&integer(bootstrap.resamples,100,10000)&&bootstrap.seed===validation.seed&&bootstrap.confidence_level===.95&&bootstrap.method==='paired-round-percentile'&&bootstrap.resampling_unit==='draw_round');
  }
  function validateReport(report,reference,history,core,now=Date.now()) {
    requireValue(object(report)&&report.schema_version===1&&['success','failed'].includes(report.status));
    requireValue(stamp(report.generated_at)&&Date.parse(report.generated_at)<=now+60000);
    if(report.status==='failed'){requireValue(report.prediction===null&&report.recommendations===null);return report;}
    requireValue(core&&typeof core.validateForecast==='function','공식 자료 검증 기능을 불러오지 못했습니다.');
    core.validateForecast(reference,now);
    requireValue(reference.status==='current'&&core.effectiveStatus(reference,now)==='current','공식 추첨 자료가 최신 상태가 아닙니다.');
    const source=report.source,model=report.model,input=report.input,prediction=report.prediction;
    requireValue(object(source)&&source.name==='동행복권'&&source.url==='https://www.dhlottery.co.kr/'&&source.api_url===reference.source.api_url&&source.latest_round===reference.source.latest_round&&source.latest_draw_date===reference.source.latest_draw_date,'실험 결과와 현재 공식 자료의 마지막 회차가 다릅니다.');
    requireValue(stamp(source.verified_at)&&Date.parse(source.verified_at)<=Date.parse(report.generated_at)&&expectedRound(Date.parse(source.verified_at))===source.latest_round);
    requireValue(object(model)&&model.id===MODEL_ID&&model.name==='TimesFM 2.5 200M (PyTorch)'&&model.package==='timesfm'&&model.package_version==='2.0.2'&&model.revision===REVISION&&model.checkpoint_sha256===CHECKPOINT&&model.checkpoint_verified===true&&model.timesfm_used===true&&model.series==='weekly_main_number_occurrence_0_or_1'&&model.point_statistic==='median_0.5_quantile'&&model.runtime_versions?.timesfm==='2.0.2','실제 TimesFM 모델의 실행 근거를 확인할 수 없습니다.');
    requireValue(object(input)&&integer(input.count,32,10000)&&integer(input.first_round,1)&&input.last_round===source.latest_round&&input.first_round+input.count-1===input.last_round&&input.last_draw_date===source.latest_draw_date&&/^[a-f0-9]{64}$/.test(input.draws_sha256));
    requireValue(object(prediction)&&integer(prediction.context_length,32,512)&&prediction.context_length%32===0&&prediction.context_length<=input.count&&prediction.context_end_round===source.latest_round&&prediction.context_start_round===source.latest_round-prediction.context_length+1&&prediction.target_round===source.latest_round+1&&prediction.target_date===drawDate(prediction.target_round)&&prediction.horizon===1&&prediction.score_kind==='relative_rank_not_probability');
    requireValue(model.forecast_config?.max_context===prediction.context_length&&model.forecast_config?.infer_is_positive===false);
    const raw=vector(prediction.raw_scores),ranks=rankScores(raw),declaredRanks=vector(prediction.rank_scores);
    requireValue(ranks.every((rank,i)=>close(rank,declaredRanks[i])));
    const draws=validateStatistics(report.statistics,input,prediction.target_date,reference,history,now);
    const overall=rankScores(report.statistics.overall.numbers.map(n=>n.historical_selection_score)),seasonal=rankScores(report.statistics.target_week.numbers.map(n=>n.historical_selection_score));
    const recommendations=report.recommendations;
    requireValue(object(recommendations)&&recommendations.games_per_method===5&&integer(recommendations.seed,0,2**32+10000)&&object(recommendations.methods));
    METHODS.forEach(method=>{
      const result=recommendations.methods[method];requireValue(object(result));checkGames(result.games);
      requireValue(Array.isArray(result.weights)&&result.weights.length===45);
      result.weights.forEach((weight,i)=>{const wanted=method==='uniform'?1:Math.max(.01,method==='timesfm'?ranks[i]:method==='statistics'?.6*overall[i]+.4*seasonal[i]:.5*ranks[i]+.3*overall[i]+.2*seasonal[i]);requireValue(close(weight,wanted),'조합의 가중치가 공개된 결합 규칙과 다릅니다.');});
    });
    requireValue(report.error===null);checkValidation(report.validation,report,draws);
    if(report.audit_replay!==undefined){
      const audit=report.audit_replay,origins=1+(report.validation.status==='complete'?report.validation.evaluated_count:0);
      requireValue(object(audit)&&/^[a-f0-9]{64}$/.test(audit.source_sha256)&&/^[a-f0-9]{64}$/.test(audit.model_score_ledger_sha256)&&stamp(audit.original_generated_at)&&Date.parse(source.verified_at)<=Date.parse(audit.original_generated_at)&&Date.parse(audit.original_generated_at)<=Date.parse(report.generated_at)&&audit.raw_model_predictions_unchanged===true&&audit.inference_rerun===false&&audit.origin_count===origins&&audit.series_per_origin===45&&model.completed_forecast_calls===origins,'모델 원시 결과 재사용 기록을 확인할 수 없습니다.');
    }
    return report;
  }

  function start(doc=document,fetcher=fetch,core=typeof LottoFM!=='undefined'?LottoFM:null,runtime={}) {
    const now=runtime.now||Date.now,$=id=>doc.getElementById(id);
    let report=null,reference=null,history=null,statsView=null,current=false;
    const setText=(id,text)=>{$(id).textContent=text;};
    const element=(tag,className,text)=>{const e=doc.createElement(tag);if(className)e.className=className;if(text!==undefined)e.textContent=text;return e;};
    const time=value=>new Date(value).toLocaleString('ko-KR',{timeZone:'Asia/Seoul',hour12:false})+' KST';
    const color=n=>n<=10?'ball-1':n<=20?'ball-10':n<=30?'ball-20':n<=40?'ball-30':'ball-40';
    const signed=n=>(n>0?'+':'')+n.toFixed(3);
    function hide(state,message){
      current=false;$('timesfm-panel').dataset.state=state;$('timesfm-results').hidden=true;
      setText('timesfm-status',state==='stale'?'최신 실험 결과 대기':state==='failed'?'모델 실행 실패':'실험 결과 확인 불가');
      setText('timesfm-message',message);
      for(const id of ['timesfm-games','timesfm-other-games','timesfm-frequency-table','timesfm-week-combinations','timesfm-repeat-preview','timesfm-metrics','timesfm-comparison-table'])$(id).replaceChildren();
      for(const id of ['timesfm-target','timesfm-model-name','timesfm-source-round','timesfm-source-verified','timesfm-generated-at','timesfm-recalculated-at','timesfm-replay-note','timesfm-revision','timesfm-library','timesfm-comparison','timesfm-bootstrap-note','timesfm-evaluation-window','timesfm-seasonal-note','timesfm-repeat-note','timesfm-seasonal-repeat-note'])setText(id,'');
      $('timesfm-recalculated-row').hidden=true;$('timesfm-replay-note').hidden=true;
      $('timesfm-model-link').removeAttribute('href');setText('timesfm-model-link','');
    }
    function games(container,items){
      container.replaceChildren();items.forEach((game,i)=>{
        const row=element('div','game-row'),balls=element('div','balls');balls.append(element('span','game-label',`${'ABCDE'[i]} 조합`));
        game.forEach(n=>balls.append(element('span',`ball ${color(n)}`,n)));row.append(balls);container.append(row);
      });
    }
    function frequencies(){
      if(!current)return;
      const value=$('timesfm-stat-scope').value,stats=statsView;
      let group=stats.overall;
      if(value==='target')group=stats.target_week;
      else if(value.startsWith('year:'))group=stats.by_iso_year.find(g=>g.iso_year===Number(value.slice(5)));
      else if(value.startsWith('week:'))group=stats.by_iso_week.find(g=>g.iso_week===Number(value.slice(5)));
      if(!group)return;
      setText('timesfm-stat-denominator',group.sample_count?`분모: 주·보너스 ${group.sample_count}회 / 2등 가능 조합 ${6*group.sample_count}개`:'관측 표본 없음 · 비율 계산 불가');
      $('timesfm-frequency-table').replaceChildren();
      group.numbers.forEach(n=>{const row=element('tr');[n.number,n.main_hits,n.main_rate===null?'—':(100*n.main_rate).toFixed(2)+'%',n.bonus_hits,n.bonus_rate===null?'—':(100*n.bonus_rate).toFixed(2)+'%',n.second_qualifying_hits,n.second_qualifying_rate===null?'—':(100*n.second_qualifying_rate).toFixed(2)+'%'].forEach(value=>row.append(element('td','metric-value',value)));$('timesfm-frequency-table').append(row);});
    }
    function historyWeek(){
      if(!current)return;const group=statsView.by_iso_year_week[Number($('timesfm-history-week').value)];if(!group)return;
      const container=$('timesfm-week-combinations');container.replaceChildren();container.append(element('p','muted',`${group.rounds.join(', ')}회 · ${group.iso_year}년 ${group.iso_week}주차`));
      group.first_prize_combinations.forEach(item=>container.append(element('p','',`1등 번호: ${item.numbers.join(' · ')} (${item.count}회)`)));
      group.second_prize_qualifying_combinations.forEach((item,i)=>container.append(element('p','note',`2등 가능 조합 ${i+1}: ${item.numbers.join(' · ')} (${item.count}회)`)));
    }
    function repeatText(summary,label){
      if(!summary.observations)return `${label}: 표본이 없습니다.`;
      if(summary.maximum_count===1)return `${label}: ${summary.observations}개 관측 조합이 모두 1회씩 등장했습니다. 반복 우위가 있는 조합은 없습니다.`;
      return `${label}: ${summary.observations}개 관측 중 서로 다른 조합 ${summary.unique_combinations}개, 반복된 조합 ${summary.repeated_combinations}개, 최대 ${summary.maximum_count}회입니다.`;
    }
    function renderStats(){
      const stats=statsView,select=$('timesfm-stat-scope');select.replaceChildren();
      const option=(value,label)=>{const o=element('option','',label);o.value=value;select.append(o);};
      option('overall',`전체 입력 기간 (${stats.sample_count}회)`);option('target',`대상과 같은 ${stats.target_iso_week}주차 (${stats.target_week.sample_count}회)`);
      [...stats.by_iso_year].reverse().forEach(g=>option(`year:${g.iso_year}`,`${g.iso_year}년 (${g.sample_count}회)`));
      stats.by_iso_week.forEach(g=>option(`week:${g.iso_week}`,`${g.iso_week}주차 · 모든 입력 연도 (${g.sample_count}회)`));select.value='target';
      setText('timesfm-seasonal-note',`대상: ISO ${stats.target_iso_year}년 ${stats.target_iso_week}주차. 같은 주차의 과거 표본은 ${stats.target_week.sample_count}회입니다. ${stats.target_week.sample_count?'작은 표본의 관측 차이를 예측 우위로 해석하지 않습니다.':'같은 주차 관측이 없어 계절 순위는 모두 동률로 처리됩니다.'}`);
      const weekSelect=$('timesfm-history-week');weekSelect.replaceChildren();
      for(let i=stats.by_iso_year_week.length-1;i>=0;i--){const g=stats.by_iso_year_week[i],o=element('option','',`${g.iso_year}년 ${g.iso_week}주차 · ${g.rounds[0]}회`);o.value=String(i);weekSelect.append(o);}weekSelect.value=String(stats.by_iso_year_week.length-1);
      setText('timesfm-repeat-note',repeatText(stats.combination_repeat_summary.first_prize,'전체 기간 1등 번호 조합')+' '+repeatText(stats.combination_repeat_summary.second_prize_qualifying,'전체 기간 2등 가능 조합'));
      setText('timesfm-seasonal-repeat-note',repeatText(stats.target_week.first_prize,'같은 주차 1등 번호 조합')+' '+repeatText(stats.target_week.second_prize_qualifying,'같은 주차 2등 가능 조합'));
      $('timesfm-repeat-preview').replaceChildren();
      const summary=stats.combination_repeat_summary.first_prize;
      if(summary.maximum_count>1){$('timesfm-repeat-preview').append(element('p','note',`최대 ${summary.preview_limit}개 표시. 끝 순위와 ${summary.total_tied_at_cutoff}개 조합이 동률일 수 있으며, 동률은 번호 오름차순으로만 정렬합니다.`));summary.preview.forEach(item=>$('timesfm-repeat-preview').append(element('p','',`${item.numbers.join(' · ')}: ${item.count}회`)));}
      frequencies();historyWeek();
    }
    function renderValidation(){
      const validation=report.validation,complete=validation.status==='complete';$('timesfm-validation-results').hidden=!complete;
      if(!complete){setText('timesfm-evaluation-window',validation.status==='not_requested'?'과거 순차 검증을 아직 실행하지 않았습니다. 아래 조합의 성능 우위를 확인한 상태가 아닙니다.':`예정된 ${validation.requested_holdout}회 전체 검증에 필요한 자료가 부족합니다. 일부 검증을 전체 결과처럼 표시하지 않습니다.`);return;}
      setText('timesfm-evaluation-window',`${validation.first_target_round}~${validation.last_target_round}회 · ${validation.evaluated_count}회차 × 방식별 5게임 = 각 ${validation.evaluated_count*5}게임. 모델 입력 길이는 대상 직전 ${validation.context_length}회입니다.`);
      $('timesfm-metrics').replaceChildren();$('timesfm-comparison-table').replaceChildren();
      ['combined','timesfm','statistics','uniform'].forEach(method=>{
        const metric=validation.metrics[method],row=element('tr');[LABELS[method],metric.avg_matches_per_game.toFixed(3),metric.avg_best_of_five.toFixed(3),...['1','2','3','4','5','none'].map(r=>metric.prize_rank_counts[r])].forEach(value=>row.append(element('td','metric-value',value)));$('timesfm-metrics').append(row);
        if(method!=='uniform'){const pair=validation.paired_vs_uniform[method],r=element('tr');[LABELS[method],signed(pair.mean_delta),`${signed(pair.ci95[0])} ~ ${signed(pair.ci95[1])}`].forEach(value=>r.append(element('td','metric-value',value)));$('timesfm-comparison-table').append(r);}
      });
      const combined=validation.paired_vs_uniform.combined;
      let conclusion=combined.mean_delta<=0?'결합 방식의 평균 주 번호 일치는 이 검증에서 균등 무작위보다 높지 않았습니다.':combined.ci95[0]<=0?'결합 방식의 평균 일치는 더 높았지만 차이의 95% 구간에 0이 포함되어, 이 검증으로 우위를 확인할 수 없습니다.':'이 검증 구간에서는 결합 방식의 평균 일치 차이가 양수였고 95% 구간도 0보다 높았습니다. 다른 기간과 미래 회차의 우위를 뜻하지 않습니다.';
      if(combined.ci95[1]<0)conclusion+=' 차이의 95% 구간도 음수였습니다.';
      setText('timesfm-comparison',conclusion);
      setText('timesfm-bootstrap-note',`같은 회차끼리 비교한 차이를 ${validation.bootstrap.resamples.toLocaleString('ko-KR')}번 회차 단위로 재표집한 95% 구간입니다. 한 회차의 5게임을 서로 독립인 5개 추첨으로 취급하지 않습니다. 이 구간은 당첨 확률 구간이 아닙니다.`);
    }
    function render(){
      statsView={...report.statistics,...detailGroups(history.raw_draws.filter(d=>d.round>=report.input.first_round&&d.round<=report.input.last_round))};
      current=true;$('timesfm-panel').dataset.state='current';$('timesfm-results').hidden=false;
      setText('timesfm-status','실제 모델 추론 확인');setText('timesfm-message','공식 입력 자료, 모델 리비전, 6개 번호 조합과 검증 기록의 일치를 확인했습니다. 실험 결과를 당첨 보장으로 해석하지 마세요.');
      setText('timesfm-target',`${report.prediction.target_round}회 · ${report.prediction.target_date} 대상 조합 5게임`);games($('timesfm-games'),report.recommendations.methods.combined.games);
      $('timesfm-other-games').replaceChildren();['timesfm','statistics','uniform'].forEach(method=>{const section=element('section');section.append(element('h3','',LABELS[method]));const list=element('div','results');games(list,report.recommendations.methods[method].games);section.append(list);$('timesfm-other-games').append(section);});
      setText('timesfm-model-name',report.model.name);setText('timesfm-source-round',`${report.input.first_round}~${report.input.last_round}회 (${report.input.count}회) · 마지막 추첨 ${report.source.latest_draw_date}`);
      setText('timesfm-source-verified',time(report.source.verified_at));setText('timesfm-generated-at',time(report.audit_replay?.original_generated_at||report.generated_at));setText('timesfm-revision',report.model.revision);setText('timesfm-library',`${report.model.package} ${report.model.package_version}`);
      $('timesfm-recalculated-row').hidden=!report.audit_replay;$('timesfm-replay-note').hidden=!report.audit_replay;
      if(report.audit_replay){setText('timesfm-recalculated-at',time(report.generated_at));setText('timesfm-replay-note','기존 실제 모델 추론 결과를 그대로 재사용하고 통계 결합 규칙과 과거 비교 결과만 재계산했습니다. 모델을 다시 실행한 결과는 아닙니다.');}
      $('timesfm-model-link').href=`https://huggingface.co/${MODEL_ID}/tree/${REVISION}`;setText('timesfm-model-link',MODEL_ID);
      renderStats();renderValidation();
    }
    function refreshStatus(){if(current&&(!core||core.effectiveStatus(reference,now())!=='current'||report.source.latest_round!==expectedRound(now())))hide('stale','새 추첨 회차가 되었거나 공식 자료가 만료되어 모델 번호 조합을 숨겼습니다. 새로고침 후 최신 실험 상태를 확인해 주세요.');}
    $('timesfm-stat-scope').addEventListener('change',frequencies);$('timesfm-history-week').addEventListener('change',historyWeek);
    const ready=(async()=>{
      const controller=typeof AbortController==='function'?new AbortController():null,timeout=controller?setTimeout(()=>controller.abort(),15000):null;
      try{
        const responses=await Promise.all(['./data/timesfm.json','./data/forecast.json','./data/draw-history.json'].map(url=>fetcher(url,{cache:'no-store',...(controller?{signal:controller.signal}:{})})));
        if(!responses[0].ok){hide('unavailable',responses[0].status===404?'실제 TimesFM 실행 결과가 아직 제공되지 않았습니다. 모델 번호 조합을 표시하지 않습니다.':'실험 결과를 불러오지 못했습니다. 확인되지 않은 모델 번호 조합을 표시하지 않습니다.');return;}
        report=await responses[0].json();
        if(report?.schema_version===1&&report?.status==='failed'){validateReport(report,null,null,core,now());hide('failed','TimesFM 실행을 완료하지 못했습니다. 통계 결과나 임의 번호를 모델 결과로 대신 표시하지 않습니다.');return;}
        if(!responses[1].ok)throw new Error('현재 공식 추첨 자료를 확인할 수 없습니다.');reference=await responses[1].json();
        if(reference?.status==='stale'||reference?.status==='unavailable'){hide('stale','공식 추첨 자료가 최신 상태가 아니어서 모델 번호 조합을 숨겼습니다.');return;}
        if(!responses[2].ok)throw new Error('연도·주차 통계의 공식 원본 이력을 불러오지 못했습니다.');history=await responses[2].json();
        validateReport(report,reference,history,core,now());render();refreshStatus();
      }catch(error){hide('unavailable',error.name==='AbortError'?'실험 자료 확인 시간이 초과되었습니다. 잠시 후 새로고침해 주세요.':`${error.message} 모델 번호 조합을 표시하지 않습니다.`);}
      finally{if(timeout)clearTimeout(timeout);}
    })();
    const timer=(runtime.setInterval||setInterval)(refreshStatus,60000);if(timer.unref)timer.unref();
    doc.addEventListener?.('visibilitychange',refreshStatus);doc.defaultView?.addEventListener('focus',refreshStatus);
    const destroy=()=>{(runtime.clearInterval||clearInterval)(timer);doc.removeEventListener?.('visibilitychange',refreshStatus);doc.defaultView?.removeEventListener('focus',refreshStatus);};
    return {ready,refreshStatus,destroy};
  }
  return {validateReport,rankScores,isoWeek,start};
})();
if(typeof module!=='undefined'&&module.exports)module.exports=TimesFMExperiments;
if(typeof document!=='undefined')TimesFMExperiments.start();
