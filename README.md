# LottoFM

동행복권 로또 6/45 추첨 결과를 요약하고 가중 무작위 번호 조합을 생성하는 정적 사이트입니다. **현재 구현은 TimesFM을 사용하지 않습니다.** 모델 표시는 `EMA·출현 빈도 통계`이며, 당첨 확률을 높인다는 검증 결과가 없습니다.

## 데이터와 계산

- 출처: [동행복권 추첨 결과](https://www.dhlottery.co.kr/lt645/result)
- 공식 페이지가 사용하는 공개 JSON 경로: `/lt645/selectPstLt645InfoNew.do`
- 최초 요청: `srchDir=center&srchLtEpsd=<회차>`, 이전 페이지: `srchDir=older&srchCursorLtEpsd=<최저 회차>`
- 최근 64회차의 회차·추첨일·서로 다른 1~45 번호 6개·보너스 번호·연속성을 검증합니다. 추첨 완료 예상 시각은 토요일 21:00 KST입니다.
- 합계 중심값은 EMA(alpha=0.25) 65%와 표본 평균 35%를 합한 값입니다. 중심값 ±12는 임의의 생성 필터이며 신뢰구간이 아닙니다. 번호 가중치는 `1 + 관측빈도 / (표본수 × 6 / 45)`입니다.
- 홀짝·고저는 과거 표본의 최빈 비율, 구간 순위는 번호당 평균 출현 빈도입니다. 구간의 동률은 낮은 구간부터 정렬합니다. 다음 회차에 유리하다는 의미가 없습니다.
- `forecast.json`에는 원본 추첨 번호, 출처, 검증 시각, 갱신 상태와 계산 방식이 함께 저장됩니다.

## 실패와 보존

공식 응답을 검증하지 못하면 수집기는 종료 코드 1을 반환합니다. 검증 가능한 기존 자료만 `stale`로 남기며, 기존 자료가 없거나 구버전·손상 자료이면 `unavailable`로 전환합니다. 가짜 추첨 데이터를 생성하는 경로는 없습니다. 프론트엔드도 스키마와 원본 추첨을 검증하고, 오래된 상태나 확인 불가 상태에서는 새 번호 생성을 중단합니다.

번호 보관함은 해당 브라우저에 최대 15개를 저장합니다. 자동 당첨 판정은 구현되어 있지 않습니다. 손상된 저장 내용을 새 번호 저장으로 덮어쓰지 않습니다.

## 로컬 실행

Python 3.11 이상, Node.js 22 이상을 사용합니다. 외부 패키지를 설치할 필요가 없습니다.

```sh
python -m unittest discover -s tests -p 'test_*.py' -v
node --test tests/*.test.js
python scripts/fetch_and_predict.py
python -m http.server 8765 --bind 127.0.0.1 --directory public
```

`http://127.0.0.1:8765`를 열어 확인합니다. `file://`로 직접 열면 JSON 요청이 차단될 수 있습니다. `--output <경로>`로 검증용 JSON을 별도로 저장할 수 있습니다. 테스트의 합성 fixture는 임시 디렉터리에서만 사용하며 공개 데이터로 저장하지 않습니다.

## 자동 실행과 Cloudflare Pages

`Validate Lotto Site`는 push/PR 시 오프라인 회귀 테스트와 하위 모듈 없이 소스를 체크아웃할 수 있는지 검사합니다. `Weekly Lotto Statistics Update`는 토요일 21:10 KST, 수동 실행 또는 main의 수집기·해당 workflow 변경 시 공식 데이터를 갱신합니다. 생성 JSON만 변경하는 커밋은 수집기를 다시 실행하지 않습니다. 수집 실패 시에도 `stale/unavailable` 상태를 커밋한 뒤 실행을 실패로 보고합니다. 갱신 workflow의 기존 `contents: write` 권한은 유지했고 추가 권한은 요구하지 않습니다.

Cloudflare Pages는 정적 파일을 제공하도록 기존 프로젝트의 출력 디렉터리를 `public`로 설정합니다. 빌드는 필요하지 않습니다. GitHub에 연결된 기존 프로젝트의 실제 배포 성공 상태와 배포 URL은 별도로 확인해야 합니다. 이 저장소에는 Cloudflare 자격 증명이나 새 계정·권한 생성 절차가 포함되어 있지 않습니다.

## Git 복구 내역

기존 루트 커밋 `0ab9bcc63017ba73992df7e304fdc72285ce92e2`는 `scripts`를 gitlink(mode 160000)로 가리켰고 `.gitmodules`가 없었습니다. 실제 중첩 원본 커밋은 `4c5dfdcd9b65a669c8a248c909ae8f618a633206`입니다. 원본 폴더와 두 저장소의 기록을 별도로 보존한 뒤 `scripts/fetch_and_predict.py`를 일반 파일(mode 100644)로 추적하도록 바꿨습니다.

기존 게시 JSON의 최근 20회 합계, 번호 가중치 45개와 중심값은 원본 코드의 합성 데이터 fallback 결과와 일치했습니다. 기존 JSON을 검증된 캐시로 승격하지 않고, 공식 API에서 새로 수집한 자료로 교체했습니다.
