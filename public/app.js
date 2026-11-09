'use strict';

// Accept only versioned pipeline output with explicit provenance. No draw fallback.
const LottoFM = (() => {
  const STORAGE_KEY = 'lottofm_saved';
  const DAY = 86400000;
  const FIRST_DRAW = Date.parse('2002-12-07T12:00:00Z');
  const SOURCE_API = 'https://www.dhlottery.co.kr/lt645/selectPstLt645InfoNew.do';
  const expectedRound = now => Math.max(0, Math.floor((now - FIRST_DRAW) / (7 * DAY)) + 1);
  const roundDate = round => new Date(FIRST_DRAW + (round - 1) * 7 * DAY).toISOString().slice(0,10);
  const isInt = value => Number.isInteger(value);
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  const isoTime = value => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}T.*(?:Z|[+-]\d\d:\d\d)$/.test(value) && Number.isFinite(Date.parse(value));
  const realDate = value => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value) && Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value;

  function officialUrl(value) {
    try { const url = new URL(value); return url.protocol === 'https:' && ['dhlottery.co.kr', 'www.dhlottery.co.kr'].includes(url.hostname) && !url.username && !url.password; }
    catch { return false; }
  }

  function validateForecast(data, now = Date.now()) {
    if (!record(data) || data.schema_version !== 1 || !['current', 'stale', 'unavailable'].includes(data.status)) throw new Error('지원하지 않는 데이터 형식입니다.');
    if (!record(data.source) || data.source.name !== '동행복권' || !officialUrl(data.source.url) || data.source.api_url !== SOURCE_API) throw new Error('공식 데이터 출처를 확인할 수 없습니다.');
    if (!record(data.model) || data.model.id !== 'ema-frequency-v1' || data.model.timesfm_used !== false || data.model.name !== 'EMA·출현 빈도 통계') throw new Error('계산 방식 정보를 확인할 수 없습니다.');
    // One minute permits modest client clock drift, never a future draw.
    if (!isoTime(data.updated_at) || Date.parse(data.updated_at) > now + 60000) throw new Error('데이터 갱신 시각이 올바르지 않습니다.');
    if (data.message !== undefined && (typeof data.message !== 'string' || data.message.length > 1000)) throw new Error('설명 데이터가 올바르지 않습니다.');
    if (data.status === 'unavailable') return data;
    const source = data.source;
    if (!isInt(source.latest_round) || source.latest_round < 1 || source.latest_round > expectedRound(now) || !realDate(source.latest_draw_date) || source.latest_draw_date !== roundDate(source.latest_round) || !isoTime(source.verified_at) || Date.parse(source.verified_at) > Date.parse(data.updated_at) || source.latest_round !== expectedRound(Date.parse(source.verified_at))) throw new Error('추첨 회차 또는 출처 확인 정보가 올바르지 않습니다.');
    if (data.target_round !== source.latest_round + 1) throw new Error('대상 회차와 원본 회차가 일치하지 않습니다.');
    if (!Array.isArray(data.history_rounds) || !Array.isArray(data.history_sums) || !data.history_rounds.length || data.history_rounds.length > 1000 || data.history_rounds.length !== data.history_sums.length || data.history_rounds.at(-1) !== source.latest_round) throw new Error('추첨 이력의 길이 또는 마지막 회차가 올바르지 않습니다.');
    if (!data.history_rounds.every((n, i, arr) => isInt(n) && n > 0 && (!i || n === arr[i - 1] + 1)) || !data.history_sums.every(n => isInt(n) && n >= 21 && n <= 255)) throw new Error('추첨 이력의 값이 올바르지 않습니다.');
    const sum = data.predicted_sum;
    if (!record(sum) || !isInt(sum.low) || !isInt(sum.high) || !finite(sum.point) || sum.low < 21 || sum.high > 255 || sum.low > sum.point || sum.point > sum.high) throw new Error('합계 범위가 올바르지 않습니다.');
    const weights = data.ball_weights;
    if (!record(weights) || Object.keys(weights).length !== 45 || !Array.from({ length:45 }, (_, i) => weights[String(i + 1)]).every(n => finite(n) && n > 0 && n <= 1000000)) throw new Error('번호별 가중치가 올바르지 않습니다.');
    const draws = data.raw_draws;
    if (!Array.isArray(draws) || draws.length < 20 || draws.length > 512 || data.sample_count !== draws.length) throw new Error('확인된 원본 추첨 이력이 없습니다.');
    const frequencies = new Array(46).fill(0);
    const oddCounts = new Array(7).fill(0);
    const highCounts = new Array(7).fill(0);
    const totals = [];
    draws.forEach((draw,i) => {
      if (!record(draw) || !isInt(draw.round) || draw.round < 1 || draw.round > source.latest_round || (i && draw.round !== draws[i-1].round + 1) || draw.date !== roundDate(draw.round) || !validNumbers(draw.numbers) || !isInt(draw.bonus) || draw.bonus < 1 || draw.bonus > 45 || draw.numbers.includes(draw.bonus)) throw new Error('원본 추첨 이력의 회차·날짜·번호가 올바르지 않습니다.');
      totals.push(draw.numbers.reduce((a,b)=>a+b,0));
      draw.numbers.forEach(n=>frequencies[n]++);
      oddCounts[draw.numbers.filter(n=>n%2).length]++;
      highCounts[draw.numbers.filter(n=>n>=23).length]++;
    });
    if (draws.at(-1).round !== source.latest_round || draws.at(-1).date !== source.latest_draw_date || JSON.stringify(data.history_rounds) !== JSON.stringify(draws.slice(-20).map(d=>d.round)) || JSON.stringify(data.history_sums) !== JSON.stringify(totals.slice(-20))) throw new Error('표시된 이력과 원본 추첨 이력이 일치하지 않습니다.');
    let ema = totals[0];
    totals.slice(1).forEach(value=>{ema=.25*value+.75*ema;});
    const center = .65*ema + .35*totals.reduce((a,b)=>a+b,0)/totals.length;
    if (Math.abs(sum.point-center) > .050001 || sum.low !== Math.max(21,Math.floor(sum.point-12)) || sum.high !== Math.min(255,Math.ceil(sum.point+12))) throw new Error('합계 통계와 원본 추첨 이력이 일치하지 않습니다.');
    const expectedFrequency = draws.length * 6 / 45;
    // Accept only the rounding tolerance of the Python producer's 4-digit weights.
    if (Array.from({length:45},(_,i)=>i+1).some(n=>Math.abs(weights[n]-(1+frequencies[n]/expectedFrequency)) > .000050001)) throw new Error('번호별 가중치와 원본 출현 빈도가 일치하지 않습니다.');
    const oddMode = oddCounts.indexOf(Math.max(...oddCounts));
    const highMode = highCounts.indexOf(Math.max(...highCounts));
    // Normalize raw occurrence counts with integer scores, avoiding rounded-weight ties.
    const zones = [[1,10],[11,20],[21,30],[31,40],[41,45]].map(([lo,hi])=>({name:`${lo}~${hi}`,score:Array.from({length:hi-lo+1},(_,i)=>frequencies[lo+i]).reduce((a,b)=>a+b,0)*(10/(hi-lo+1))})).sort((a,b)=>b.score-a.score);
    if (data.recommended_odd_even !== `홀 ${oddMode} : 짝 ${6-oddMode}` || data.recommended_high_low !== `저 ${6-highMode} : 고 ${highMode}` || data.hot_zones !== zones.slice(0,2).map(z=>z.name).join(' · ')) throw new Error('요약 통계와 원본 추첨 이력이 일치하지 않습니다.');
    for (const key of ['recommended_odd_even', 'recommended_high_low', 'hot_zones']) if (data[key] !== undefined && (typeof data[key] !== 'string' || data[key].length > 1000)) throw new Error('설명 데이터가 올바르지 않습니다.');
    return data;
  }

  function effectiveStatus(data, now = Date.now()) {
    if (data.status === 'unavailable') return 'unavailable';
    // Saturday's next draw is due one week later, by 21:00 KST (12:00 UTC).
    // A static "current" file must expire even if scheduled updates stop running.
    return data.status === 'stale' || data.source.latest_round !== expectedRound(now) ? 'stale' : 'current';
  }

  function validNumbers(numbers) {
    return Array.isArray(numbers) && numbers.length === 6 && new Set(numbers).size === 6 && numbers.every(n => isInt(n) && n >= 1 && n <= 45);
  }

  function readSaved(storage) {
    try {
      const parsed = JSON.parse(storage.getItem(STORAGE_KEY) || '[]');
      if (!Array.isArray(parsed)) throw new Error('Invalid saved list');
      const valid = parsed.filter(item => record(item) && isInt(item.round) && item.round > 0 && validNumbers(item.numbers) && item.sum === item.numbers.reduce((a,b) => a+b, 0));
      return { games:valid.slice(0,15), error:valid.length !== parsed.length ? '일부 저장 항목의 형식이 잘못되어 표시하지 않았습니다. 원본 저장 내용은 유지됩니다.' : '' };
    } catch { return { games:[], error:'브라우저 저장 내용을 읽을 수 없습니다. 원본 저장 내용은 유지됩니다.' }; }
  }

  function sampleGame(data, fixed = [], random = Math.random, maxAttempts = 1000) {
    if (!Array.isArray(fixed) || fixed.length > 3 || new Set(fixed).size !== fixed.length || !fixed.every(n => isInt(n) && n >= 1 && n <= 45)) throw new Error('고정수는 서로 다른 1~45 번호를 최대 3개 선택해 주세요.');
    const remaining = Array.from({ length:45 }, (_, i) => i + 1).filter(n => !fixed.includes(n));
    const fixedSum = fixed.reduce((a,b) => a+b, 0);
    const needed = 6 - fixed.length;
    const minSum = fixedSum + remaining.slice(0,needed).reduce((a,b) => a+b, 0);
    const maxSum = fixedSum + remaining.slice(-needed).reduce((a,b) => a+b, 0);
    if (minSum > data.predicted_sum.high || maxSum < data.predicted_sum.low) throw new Error('선택한 고정수와 합계 범위를 동시에 만족할 수 없습니다. 고정수를 바꿔 주세요.');
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const numbers = [...fixed];
      while (numbers.length < 6) {
        const candidates = remaining.filter(n => !numbers.includes(n));
        const total = candidates.reduce((acc,n) => acc + data.ball_weights[n], 0);
        const draw = random();
        if (!finite(draw) || draw < 0 || draw >= 1) throw new Error('난수 생성에 실패했습니다. 다시 시도해 주세요.');
        let threshold = draw * total;
        let selected = candidates.at(-1);
        for (const n of candidates) { threshold -= data.ball_weights[n]; if (threshold < 0) { selected = n; break; } }
        numbers.push(selected);
      }
      numbers.sort((a,b) => a-b);
      const sum = numbers.reduce((a,b) => a+b, 0);
      if (sum >= data.predicted_sum.low && sum <= data.predicted_sum.high) return { numbers, sum };
    }
    throw new Error('시도 횟수 안에 조건을 만족하는 조합을 찾지 못했습니다. 다시 생성하거나 고정수를 바꿔 주세요.');
  }

  function start(doc = document, fetcher = fetch, storageFactory = () => localStorage, runtime = {}) {
    let currentData = null;
    let state = 'loading';
    const fixed = new Set();
    const now = runtime.now || Date.now;
    const $ = id => doc.getElementById(id);
    const setText = (id, text) => { $(id).textContent = text; };
    const element = (tag, className, text) => { const e = doc.createElement(tag); if (className) e.className = className; if (text !== undefined) e.textContent = text; return e; };
    const formattedTime = value => new Date(value).toLocaleString('ko-KR', { timeZone:'Asia/Seoul', hour12:false }) + ' KST';
    const colorClass = n => n <= 10 ? 'ball-1' : n <= 20 ? 'ball-10' : n <= 30 ? 'ball-20' : n <= 40 ? 'ball-30' : 'ball-40';

    function savedState() {
      try { return readSaved(storageFactory()); }
      catch { return { games:[], error:'이 브라우저에서는 번호 보관함을 사용할 수 없습니다.' }; }
    }
    function renderSaved() {
      const saved = savedState();
      $('saved-games-list').replaceChildren();
      setText('storage-message', saved.error);
      if (!saved.games.length) $('saved-games-list').append(element('p','muted', '저장된 번호가 없습니다. 생성한 조합의 보관함 저장 버튼을 이용하세요.'));
      for (const item of saved.games) $('saved-games-list').append(gameRow(item, `${item.round}회`, false));
    }
    function saveGame(game) {
      if (!currentData || effectiveStatus(currentData,now()) !== 'current') { refreshStatus(); setText('storage-message','새 추첨 회차가 되어 이전 생성 결과를 저장하지 않았습니다. 새로고침 후 최신 상태를 확인해 주세요.'); return; }
      const saved = savedState();
      // Saving must not overwrite a corrupt existing store.
      if (saved.error) { setText('storage-message', saved.error + ' 새 번호를 저장하려면 전체 삭제로 보관함을 초기화해 주세요.'); return; }
      try {
        storageFactory().setItem(STORAGE_KEY, JSON.stringify([{ round:currentData.target_round, numbers:game.numbers, sum:game.sum, savedAt:new Date().toISOString() }, ...saved.games].slice(0,15)));
        renderSaved();
        setText('storage-message', '이 브라우저에 조합을 저장했습니다.');
      } catch { setText('storage-message', '브라우저 저장 공간에 기록하지 못했습니다. 번호를 별도로 보관해 주세요.'); }
    }
    function gameRow(game, label, canSave) {
      const row = element('div','game-row');
      const balls = element('div','balls');
      balls.append(element('span','game-label',label));
      for (const n of game.numbers) balls.append(element('span',`ball ${colorClass(n)}`,n));
      const actions = element('div','game-actions');
      const odds = game.numbers.filter(n => n % 2).length;
      actions.append(element('span','note',`합계 ${game.sum} · 홀짝 ${odds}:${6-odds}`));
      if (canSave) { const button = element('button','secondary','보관함 저장'); button.type='button'; button.addEventListener('click', () => saveGame(game)); actions.append(button); }
      row.append(balls, actions);
      return row;
    }
    function renderWeights() {
      $('top-weights-container').replaceChildren();
      if (currentData && state !== 'unavailable') {
        const entries = Object.entries(currentData.ball_weights).map(([n,w]) => ({ n:Number(n), w })).sort((a,b) => b.w-a.w || a.n-b.n).slice(0,10);
        for (const {n,w} of entries) {
          const button = element('button','weight-button', `${n}번 · ${w.toFixed(2)}`);
          button.type = 'button'; button.disabled = state !== 'current'; button.setAttribute('aria-pressed', String(fixed.has(n)));
          button.addEventListener('click', () => {
            if (state !== 'current') return;
            if (fixed.has(n)) fixed.delete(n);
            else if (fixed.size < 3) fixed.add(n);
            else { setText('generator-message','고정수는 최대 3개까지 선택할 수 있습니다.'); return; }
            setText('generator-message','고정수 조건을 바꿨습니다. 5게임 생성을 눌러 새 조합을 만드세요.');
            $('generated-results').replaceChildren(); renderWeights();
          });
          $('top-weights-container').append(button);
        }
      }
      setText('fixed-count', fixed.size);
      setText('fixed-list-text', fixed.size ? [...fixed].sort((a,b)=>a-b).join(', ') + '번' : '없음');
      $('btn-clear-fixed').disabled = state !== 'current' || fixed.size === 0;
    }
    function renderChart() {
      $('chart-container').replaceChildren();
      $('history-table').replaceChildren();
      $('history-details').hidden = !currentData || state === 'unavailable';
      if (!currentData || state === 'unavailable') { $('chart-container').append(element('p','muted','표시할 수 있는 확인된 추첨 데이터가 없습니다.')); return; }
      const rounds = currentData.history_rounds;
      const sums = currentData.history_sums;
      setText('chart-description', `동행복권에서 확인한 ${rounds.length}회차의 당첨번호 6개 합계입니다. 보너스 번호는 제외합니다.`);
      for (let i=0;i<rounds.length;i++) { const row=element('tr'); row.append(element('td','',`${rounds[i]}회`), element('td','',sums[i])); $('history-table').append(row); }
      // Native SVG removes the previous chart CDN dependency. The table is always available.
      try {
        const ns = 'http://www.w3.org/2000/svg';
        const svg = doc.createElementNS(ns, 'svg'); svg.setAttribute('viewBox','0 0 700 250'); svg.setAttribute('role','img'); svg.setAttribute('aria-label', `${rounds[0]}회부터 ${rounds.at(-1)}회까지 당첨번호 합계. 상세 값은 아래 표에서 확인할 수 있습니다.`);
        const min = Math.max(0, Math.floor((Math.min(...sums)-10)/20)*20); const max = Math.ceil((Math.max(...sums)+10)/20)*20;
        const x = i => rounds.length === 1 ? 350 : 48 + i * 624/(rounds.length-1); const y = sum => 210-(sum-min)*180/(max-min);
        const svgPart = (tag,attrs,text) => { const part=doc.createElementNS(ns,tag); for (const [key,value] of Object.entries(attrs)) part.setAttribute(key,String(value)); if(text!==undefined) part.textContent=text; svg.append(part); return part; };
        for (let value=min;value<=max;value+=20) { const yy=y(value); svgPart('line',{ x1:48,y1:yy,x2:672,y2:yy,stroke:'#263247' }); svgPart('text',{ x:36,y:yy+4,'text-anchor':'end',fill:'#a7b3c7','font-size':12 },value); }
        svgPart('polyline',{ points:sums.map((s,i)=>`${x(i)},${y(s)}`).join(' '), fill:'none',stroke:'#818cf8','stroke-width':2.5 });
        sums.forEach((s,i) => svgPart('circle',{ cx:x(i),cy:y(s),r:3.5,fill:'#a5b4fc' }));
        svgPart('text',{ x:48,y:237,fill:'#a7b3c7','font-size':12 },`${rounds[0]}회`);
        svgPart('text',{ x:672,y:237,fill:'#a7b3c7','font-size':12,'text-anchor':'end' },`${rounds.at(-1)}회`);
        $('chart-container').append(svg);
      } catch { $('chart-container').append(element('p','muted','차트를 표시하지 못했습니다. 아래 데이터 표를 이용해 주세요.')); }
    }
    function unavailable(message) {
      state='unavailable'; currentData=null;
      $('status-panel').dataset.state=state;
      setText('data-status','데이터 확인 불가');
      setText('status-message',message);
      setText('target-round-badge','통계·번호 생성 대기');
      setText('source-status','추첨 데이터 미확인');
      for (const id of ['source-round','source-verified','updated-at-text','pred-sum-range','pred-odd-even','pred-high-low','pred-hot-zone']) setText(id,'—');
      setText('pred-sum-point','통계 확인 대기');
      setText('chart-description','출처를 확인한 추첨 결과가 있으면 차트를 표시합니다.');
      $('source-link').href='https://www.dhlottery.co.kr/';
      fixed.clear();
      $('generated-results').replaceChildren();
      setText('model-type-badge','EMA·출현 빈도 통계 · TimesFM 미사용');
      setText('generator-description','최신 공식 추첨 데이터를 확인할 때까지 가중치 번호 생성을 중단합니다.');
      $('btn-generate').disabled=true;
      renderWeights(); renderChart();
    }
    function renderData() {
      state=effectiveStatus(currentData,now());
      if (state === 'unavailable') { const data=currentData; unavailable(data.message || '공식 추첨 결과를 확인하지 못했습니다.'); setText('updated-at-text',formattedTime(data.updated_at)); return; }
      $('status-panel').dataset.state=state;
      setText('data-status', state === 'current' ? '공식 추첨 데이터 확인 완료' : '이전 데이터 · 최신 갱신 필요');
      setText('status-message', state === 'current' ? (currentData.message || '표시된 회차까지 확인한 결과입니다.') : '마지막으로 확인된 통계를 표시합니다. 최신 결과 확인 전까지 번호 생성은 중단됩니다.');
      setText('source-status',state === 'current' ? '출처 확인 완료' : '이전 확인 결과');
      setText('source-round',`${currentData.source.latest_round}회 · ${currentData.source.latest_draw_date}`);
      setText('source-verified', formattedTime(currentData.source.verified_at));
      setText('updated-at-text',formattedTime(currentData.updated_at));
      $('source-link').href=currentData.source.url;
      setText('target-round-badge',state === 'current' ? `${currentData.target_round}회 조합 생성` : '갱신 대기 · 이전 통계');
      setText('model-type-badge',`${currentData.model.name} · TimesFM 미사용`);
      setText('pred-sum-range',`${currentData.predicted_sum.low} ~ ${currentData.predicted_sum.high}`);
      setText('pred-sum-point',`EMA·평균 혼합값 ${currentData.predicted_sum.point} · 신뢰구간 아님`);
      setText('pred-odd-even',currentData.recommended_odd_even || '제공되지 않음');
      setText('pred-high-low',currentData.recommended_high_low || '제공되지 않음');
      setText('pred-hot-zone',currentData.hot_zones || '제공되지 않음');
      setText('generator-description',state === 'current' ? `합계 ${currentData.predicted_sum.low}~${currentData.predicted_sum.high}와 선택한 고정수를 만족하는 가중 무작위 조합입니다. 당첨 확률 향상을 의미하지 않습니다.` : '과거 가중치를 표시합니다. 최신 데이터가 확인되면 생성을 다시 사용할 수 있습니다.');
      $('btn-generate').disabled=state !== 'current';
      renderWeights(); renderChart();
    }
    function generate() {
      $('generated-results').replaceChildren();
      if (!currentData || effectiveStatus(currentData,now()) !== 'current') { refreshStatus(); setText('generator-message','최신 추첨 데이터를 확인한 뒤 번호를 생성할 수 있습니다.'); return; }
      try {
        const games = Array.from({length:5},()=>sampleGame(currentData,[...fixed]));
        games.forEach((game,i)=>$('generated-results').append(gameRow(game,`${'ABCDE'[i]} 조합`,true)));
        setText('generator-message','5게임을 생성했습니다. 게임 간 같은 조합이 나올 수 있습니다.');
      } catch(error) { setText('generator-message',error.message); }
    }
    function refreshStatus() {
      if (currentData && state === 'current' && effectiveStatus(currentData,now()) !== 'current') {
        fixed.clear(); $('generated-results').replaceChildren(); renderData();
        setText('generator-message','새 추첨 회차가 되어 이전 생성 결과를 지웠습니다. 새로고침 후 최신 상태를 확인해 주세요.');
      }
    }
    $('btn-generate').addEventListener('click',generate);
    $('btn-clear-fixed').addEventListener('click',()=>{fixed.clear();$('generated-results').replaceChildren();setText('generator-message','고정수를 초기화했습니다.');renderWeights();});
    $('btn-clear-saved').addEventListener('click',()=>{try {storageFactory().removeItem(STORAGE_KEY);renderSaved();} catch {setText('storage-message','브라우저 저장 내용을 삭제하지 못했습니다.');}});
    renderSaved();
    const ready = (async()=>{
      const controller = typeof AbortController === 'function' ? new AbortController() : null;
      const timeout = controller ? setTimeout(()=>controller.abort(),15000) : null;
      try {
        const response = await fetcher('./data/forecast.json', {cache:'no-store', ...(controller ? {signal:controller.signal} : {})});
        if (!response.ok) throw new Error(`통계 파일을 불러오지 못했습니다 (HTTP ${response.status}).`);
        currentData = validateForecast(await response.json(),now());
        renderData();
      } catch(error) { unavailable(error.name === 'AbortError' ? '데이터 요청 시간이 초과되었습니다. 잠시 후 새로고침해 주세요.' : `${error.message} 확인되지 않은 번호나 통계는 표시하지 않습니다.`); }
      finally {if(timeout) clearTimeout(timeout);}
    })();
    const timer = (runtime.setInterval || setInterval)(refreshStatus,60000);
    if (timer.unref) timer.unref();
    doc.addEventListener?.('visibilitychange',refreshStatus);
    doc.defaultView?.addEventListener('focus',refreshStatus);
    const destroy = () => {(runtime.clearInterval || clearInterval)(timer);doc.removeEventListener?.('visibilitychange',refreshStatus);doc.defaultView?.removeEventListener('focus',refreshStatus);};
    return {ready,generate,refreshStatus,destroy};
  }
  return {validateForecast,effectiveStatus,validNumbers,readSaved,sampleGame,start};
})();
if (typeof module !== 'undefined' && module.exports) module.exports=LottoFM;
if (typeof document !== 'undefined') LottoFM.start();
