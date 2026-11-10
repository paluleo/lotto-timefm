# LottoFM

동행복권 공식 추첨 이력과 **실제 Google TimesFM 2.5 CPU 추론**으로 1~45 중 서로 다른 6개 번호의 후보 조합을 만드는 정적 사이트입니다. 모델·전체 이력·같은 ISO 주차의 통계를 결합한 5게임을 제시합니다. 모델 점수와 추천 가중치는 당첨 확률이 아니며, 과거 평가가 미래 당첨이나 수익을 보장하지 않습니다.

## 공식 데이터와 연도·주차 통계

- 출처는 [동행복권 추첨 결과](https://www.dhlottery.co.kr/lt645/result)이며, 공식 페이지의 `/lt645/selectPstLt645InfoNew.do` 응답만 수집합니다. 기본 전체 수집은 `center` 요청으로 최신 회차를 확인하고 `older` 요청으로 1회차까지 내려갑니다.
- `scripts/collect_history.py`는 회차·추첨일·서로 다른 본번호 6개·보너스 번호·전체 연속성을 검증합니다. 2026-11-10에 확인한 전체 이력은 **1~1,249회차**입니다. 최신 추첨 완료 예상 시각은 토요일 21:00 KST입니다.
- `public/data/draw-history.json`에 원본 번호와 출처·검증 범위를 보존합니다. 전체 수집은 페이지 사이에 0.15초를 기다리고 일시적 통신 오류에 2초·4초 간격으로 최대 3회 요청합니다. 실패 시 기존 이력 파일을 보존하고 실패를 반환합니다.
- `--recent-report public/data/forecast.json`은 공식 확인 후 30분 이내인 최신 **64회차** 자료로 기존 전체 이력을 갱신합니다. 출처·시각·요약과 1회차부터의 전체 연속성을 검증하고, 보너스를 포함해 겹치는 모든 회차가 같으며 중복 구간이 **32회 이상**일 때만 합칩니다. 자료 충돌이나 중복 부족은 실패로 처리합니다. 기존 전체 파일이 없거나 유효하지 않으면 공식 전체 수집을 시도합니다.
- 증분 갱신의 `source.verified_at`은 최근 64회 자료를 확인한 시각입니다. `source.history_verification`은 전체를 조회했던 `full_history_verified_at`, 이번에 재대조한 회차 범위와 이전 전체 번호 자료의 SHA-256을 보존합니다. 과거 전체 회차를 다시 조회했다고 표시하지 않습니다. 전체 재확인이 필요하면 `--recent-report` 없이 실행합니다.
- 연도·주차는 공식 토요일 추첨일의 **ISO-8601 연도와 주차**입니다. 전체, ISO 연도별, ISO 주차별, 연도·주차별 표본 수와 번호 출현 횟수를 계산합니다. 연말·연초의 ISO 연도는 달력 연도와 다를 수 있습니다.
- 1등 조합은 실제 본번호 6개입니다. 2등 조건 조합은 본번호 중 하나를 보너스로 바꾼 **회차당 6개의 파생 조합**입니다. 실제 구매된 당첨 복권이나 당첨자 목록이 아니며, 독립적인 6회 추첨으로 세지 않습니다.
- 정확히 같은 조합의 반복도 집계합니다. 동률 조합의 표시 순서는 번호 오름차순이며 예측 우선순위가 아닙니다. 같은 ISO 주차의 표본이 적다는 점도 함께 표시합니다.

## 실제 TimesFM 번호 추천

`scripts/timesfm_forecast.py`는 각 번호가 본번호에 포함됐는지를 주별 `0/1`로 바꾸어 **45개 시계열**을 구성합니다. 각 시계열의 직전 64회로 다음 한 회를 추론합니다. 번호 합계 예측으로 여섯 번호를 대신하는 방식이 아닙니다.

모델은 [`google/timesfm-2.5-200m-pytorch`](https://huggingface.co/google/timesfm-2.5-200m-pytorch), Python 패키지는 `timesfm==2.0.2`입니다. 모델 버전과 패키지 버전은 다른 식별자입니다. 가중치는 revision `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`와 SHA-256으로 고정·검증합니다. Apache-2.0 안내는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 있습니다.

공식 API의 0.5 분위수(중앙값)를 번호별 **상대 순위 점수**로 사용합니다. 출현 확률로 보정된 값이 아닙니다. 음수 신호를 임의로 0으로 잘라 순위 동률을 만들지 않으며, 동률에는 평균 순위를 부여합니다. CPU에서 `torch_compile=False`, 문맥 64, 내부 최대 예측 길이 128, 실제 예측 길이 1, 배치 4를 사용합니다.

조합 생성 규칙은 평가 결과에 맞춰 튜닝하지 않는 고정 실험 규칙입니다.

| 방식 | 번호 선택 가중치 |
| --- | --- |
| TimesFM | 모델 상대 순위 |
| 통계 | 전체 이력 순위 60% + 같은 ISO 주차 순위 40% |
| 결합 | 모델 순위 50% + 전체 이력 순위 30% + 같은 ISO 주차 순위 20% |
| 균등 무작위 | 45개 번호의 가중치가 동일 |

전체 이력과 같은 주차의 통계 점수는 **1등 본번호 포함률 75% + 2등 조건 조합 포함률 25%**를 사용합니다. 각 포함률에 실제 추첨 20회에 해당하는 균등 사전값(`6/45`)을 적용합니다. 2등 조건의 번호 포함 횟수는 `5 × 본번호 출현 횟수 + 6 × 보너스 출현 횟수`이며, 이를 6으로 나누어 실제 회차 단위로 환산한 뒤 사전값을 더합니다. 이 점수도 당첨 확률이 아닙니다.

균등 사전값은 같은 그룹 안에서 번호의 상대 순위를 바꾸지 않습니다. 추천은 상대 순위를 사용하므로, 표본 수가 적다고 같은 주차 통계의 고정 반영 비중(20%)이 자동으로 줄어들지는 않습니다. 이는 작은 표본의 불확실성을 줄였다는 의미가 아닙니다.

최소 가중치 0.01을 적용하고 고정 seed의 비복원 가중 추출로 게임당 서로 다른 6개 번호를 뽑습니다. 같은 방식 안에서 중복 게임은 제외합니다. `public/data/timesfm.json`에는 실제 실행 여부, 모델 revision·체크섬·라이브러리 버전, 사용 이력, 상대 점수, 후보 조합과 평가 결과가 포함됩니다.

## 시간 순서를 지킨 평가

기본 평가는 최근 **104회차**, 네 방식 각각 **회차당 5게임**입니다. 각 대상 회차 직전 64회만 모델에 입력하고 연도·주차 통계도 대상 회차 이전 자료로 다시 계산합니다. 대상 회차의 당첨번호와 보너스는 조합 생성이 끝난 뒤 채점에만 사용합니다. 방식별로 같은 회차 seed를 사용합니다.

게임당 평균 일치 개수, 회차별 5게임 중 최대 일치 개수, 1~5등 조건 충족 횟수를 보고합니다. 균등 무작위와의 평균 일치 개수 차이를 같은 회차끼리 비교하고, **회차 단위 bootstrap 2,000회**로 95% 구간을 계산합니다. 이는 당첨 확률 구간이 아닙니다. 같은 회차의 5게임을 독립적인 추첨으로 취급하지 않습니다.

모델 사전학습 데이터에 이 공개 이력이 포함됐는지는 확인되지 않았습니다. 과거 평가를 미래 성능의 증명으로 해석하지 않습니다. 결과 수치는 실행 산출물과 화면에 표시하며, 이 문서는 특정 방식의 우위를 주장하지 않습니다.

### 초기 검증 기록 — 2026-11-10

[실행 산출물](public/data/timesfm.json)의 고정 평가 구간은 **1,146~1,249회차**, 총 104회이며 각 방식당 5게임씩 520게임입니다. 게임당 본번호 평균 일치 개수는 결합 방식 **0.746153846개**, 균등 무작위 **0.776923077개**였습니다. 결합−균등 차이는 **−0.030769231개**, 회차 단위 bootstrap 95% 구간은 **[−0.115384615, 0.053846154]**로 0을 포함했습니다. **이번 평가에서 결합 방식의 우위는 확인되지 않았습니다.** 네 방식 모두 1등·2등 조건 충족은 각각 0게임이었습니다. 이는 실제 복권 구매나 수익 기록이 아닌 과거 자료의 모의 평가입니다.

관찰된 실제 추론 산출물의 SHA-256을 별도로 확인한 경우에만 `--replay-model-report`와 `--replay-sha256`으로 같은 원본 자료·문맥·45개 점수를 재사용하여 조합 규칙을 재평가할 수 있습니다. 이 결과의 `audit_replay`에는 원본 해시와 실제 추론 시각, 추론을 다시 실행하지 않았다는 사실이 남습니다. `generated_at`은 새 결과 생성 시각이며, 실제 추론 시각은 `audit_replay.original_generated_at`입니다. 기본 주간 workflow는 이 재사용 경로 대신 실제 모델을 실행합니다.

## 실패 처리와 기존 통계 기능

TimesFM 가중치 다운로드·체크섬 검증·라이브러리 로드·추론·평가 중 실패하면 실패 상태만 기록합니다. **EMA나 임의 값으로 모델 결과를 대체하지 않습니다.** 화면에서도 출처·시각·원본 이력·추천 규칙·평가 요약을 검증하고 확인 불가하거나 새 추첨 회차가 된 모델 조합을 숨깁니다.

기존 `scripts/fetch_and_predict.py`와 `public/data/forecast.json`은 **EMA·출현 빈도 통계 기능**으로 분리해 유지합니다. 합계 중심값은 EMA(alpha=0.25) 65%와 표본 평균 35%이며, 중심값 ±12는 임의의 생성 필터로서 신뢰구간이 아닙니다. 이 경로는 TimesFM을 사용하지 않는다고 명시합니다. 공식 자료 실패 시 검증 가능한 기존 자료만 `stale`로 남기고 없으면 `unavailable`로 전환합니다. 가짜 추첨 데이터를 만드는 경로는 없습니다.

기존 브라우저 번호 보관함은 최대 15개를 저장하며 자동 당첨 판정은 구현되어 있지 않습니다. 손상된 저장 내용을 새 번호 저장으로 덮어쓰지 않습니다.

## 로컬 CPU 환경

실제 모델 환경은 **Python 3.14**입니다. 고정 런타임은 `timesfm 2.0.2`, `torch 2.15.0+cpu`, `numpy 2.5.4`, `huggingface_hub 2.4.0`, `safetensors 0.8.0`입니다. [requirements-timesfm.txt](requirements-timesfm.txt)에 의존성을 명시합니다. CPU PyTorch를 먼저 설치해야 CUDA 패키지 설치를 피할 수 있습니다.

저장소 루트에서 실행하는 Windows PowerShell 예시입니다. 가상환경과 모델 캐시는 저장소 밖에 둡니다.

```powershell
python -m venv ..\timesfm-runtime
..\timesfm-runtime\Scripts\python.exe -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
..\timesfm-runtime\Scripts\python.exe -m pip install -r requirements-timesfm.txt
..\timesfm-runtime\Scripts\python.exe scripts/fetch_and_predict.py
..\timesfm-runtime\Scripts\python.exe scripts/collect_history.py --recent-report public/data/forecast.json
..\timesfm-runtime\Scripts\python.exe scripts/timesfm_forecast.py --input public/data/draw-history.json --output ../timesfm-full.json --context 64 --holdout 104 --threads 4 --batch-size 4 --cache-dir ../.model-cache
..\timesfm-runtime\Scripts\python.exe scripts/publish_timesfm_report.py --input ../timesfm-full.json
```

Linux에서는 `python3.14 -m venv ../timesfm-runtime`으로 생성하고 실행 경로를 `../timesfm-runtime/bin/python`으로 바꾸면 됩니다. 고정 CPU PyTorch의 Python 3.14 Windows x64 및 Linux x86_64 wheel이 공식 인덱스에 있습니다. 최초 모델 가중치 다운로드는 약 925MB이며 전체 104회 평가에는 로컬 CPU에서 약 8분이 걸렸습니다. 설치·다운로드 시간은 별도이고 CPU와 네트워크에 따라 달라집니다. GPU나 유료 추론 API는 사용하지 않습니다.

현재 회차의 실제 45개 시계열 추론만 확인하려면 `--holdout 0`을 사용합니다. 이때 과거 평가 완료로 표시되지 않습니다. 가중치가 있는 환경에서는 `--local-files-only`로 다운로드 없이 실행할 수 있습니다. 가중치·가상환경·전체 감사 산출물은 정적 사이트에 올리지 않습니다.

오프라인 회귀 테스트와 로컬 화면 확인:

```sh
python -m unittest discover -s tests -p 'test_*.py' -v
node --test tests/*.test.js
python -m http.server 8765 --bind 127.0.0.1 --directory public
```

Node.js 22 이상을 사용합니다. `http://127.0.0.1:8765`로 접속하며 `file://`로 직접 열면 JSON 요청이 차단될 수 있습니다. 테스트의 합성 fixture는 임시 파일이나 메모리에서만 사용하고 공개 데이터로 저장하지 않습니다.

## 자동 갱신과 배포

`Weekly Lotto Statistics Update`는 **매주 토요일 21:10 KST**, 수동 실행 또는 main의 수집·모델 스크립트/의존성/workflow 변경 시 실행합니다. 공식 자료 수집, CPU 런타임 준비, 실제 TimesFM 추론과 104회 평가를 마친 뒤 JSON을 커밋합니다. 생성 JSON만 변경하는 커밋은 수집기를 다시 실행하지 않습니다. 수집이나 추론 실패 시 명시적인 실패 상태를 남기며 실행도 실패로 보고합니다. 전체 모델 감사 결과는 GitHub Actions의 `timesfm-number-audit` artifact로 보관합니다.

매주 최근 64회 수집을 먼저 수행하며, 실패하면 5초 간격으로 최대 3회 시도합니다. 이어서 위의 증분 검증으로 전체 이력을 갱신합니다. 전체 이력 갱신 단계는 최대 5분이며, 최근 구간 확인과 이전 전체 확인 시각을 구분해 게시합니다. 이 짧은 재시도를 모두 실패하면 해당 실행은 실패로 끝나고, 별도의 시간별·일별 재게시 감시는 수행하지 않습니다.

기존 GitHub Actions CPU runner와 Cloudflare 연결을 사용합니다. 연결과 기존 권한이 정상이라면 모델용 계정 연결·GPU 결제·유료 API 설정은 필요 없습니다. 최초 설치·다운로드가 있으므로 정해진 시각에 결과가 즉시 나타나는 것은 아닙니다. workflow는 최대 45분을 허용합니다.

Cloudflare Workers 프로젝트 `lotto-timefm`은 `public`의 정적 파일만 제공합니다. 모델 추론은 GitHub Actions 또는 로컬 CPU에서 수행합니다. 루트 `wrangler.jsonc`의 `assets.directory`는 `./public`이며, 기존 Workers Builds는 저장소 루트, 빈 Build command, Deploy command `npx wrangler deploy`를 사용합니다. 계정 토큰·가중치·추론 런타임은 배포 디렉터리에 포함하지 않습니다.

## 원본과 Git 복구 기록

기존 루트 커밋 `0ab9bcc63017ba73992df7e304fdc72285ce92e2`는 `scripts`를 gitlink(mode 160000)로 가리켰고 `.gitmodules`가 없었습니다. 실제 중첩 원본 커밋은 `4c5dfdcd9b65a669c8a248c909ae8f618a633206`입니다. 원본 폴더와 두 저장소의 기록을 별도로 보존한 뒤 Python 소스를 일반 파일(mode 100644)로 추적하도록 복구했습니다.

원본에는 TimesFM 1.0 로드를 시도하는 코드가 있었으나 모든 오류를 EMA로 바꾸는 경로가 있었습니다. 기존 게시 JSON의 최근 20회 합계·번호 가중치 45개·중심값은 원본의 합성 추첨 데이터와 EMA 계산으로 재현됐습니다. 이 JSON을 검증된 캐시로 승격하지 않고 공식 자료로 교체했습니다. 이후 실제 TimesFM 2.5 번호별 추론과 실행 증거를 별도 경로로 추가했습니다.
