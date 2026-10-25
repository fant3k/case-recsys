# Воспроизведение

Порядок воспроизведения экспериментов отчёта [REPORT.md](REPORT.md). Исходная схема: Mac готовит данные и передаёт проект, Windows с RTX 3060 12 ГБ считает. Подойдёт и одна машина с CUDA: все команды запускаются из корня репозитория, пути в них относительные.

## 0. Без GPU: таблицы и графики из сохранённых результатов

Небольшие JSON всех запусков, на которые опирается отчёт, лежат в `research/results/` (21 октября) и `research/results/runs/<запуск>/` (22–25 октября, включая финальный тест и новую когорту). Графики строятся только из них:

```bash
python -m pip install matplotlib numpy
python scripts/make_report_figures.py
```

`artifacts/` в Git не входит: там checkpoints и полные логи на машине, где шли вычисления.

## 1. Окружение

- **Python 3.11** (на Windows проверено 3.11.9), **PyTorch с CUDA** (проверено 2.14.1+cu126) — ставится отдельно с pytorch.org под свою видеокарту.
- Остальное: `pip install -r requirements-windows.txt` (numpy 2.2.6, polars 2.0.0, transformers 4.57.1, lightgbm 4.7.0, scipy, scikit-learn).
- Для подготовки данных на Mac/Linux без GPU достаточно `pip install -r requirements-research.txt`.

### Готовые веса

Чтобы пропустить обучение, веса опубликованы релизом `weights-v1` (около 580 МБ): `python3 scripts/fetch_weights.py` скачивает их, проверяет SHA256 по `research/weights-manifest.json` и кладёт по путям `artifacts/...`, которые ожидает код. Хэши совпадают с записанными в `locked-manifest.json` финального теста и новой когорты. С ними можно запускать пробный прогон `python -m recsys.final_test --window selection --out artifacts/final-dryrun-selection-20261023-02` (шаг 6 раздела «Финальный тест») и остальные шаги оценки без обучения; обучение с нуля в этом случае не нужно.

## 2. Данные

```bash
python3 research/selection/fetch_data.py yambda
python3 research/selection/audit_yambda.py
```

Первая команда скачивает Yambda-50M `sequential/50m/listens.parquet` и проверяет закреплённые SHA256. Вторая выбирает 1000 пользователей с seed 42 и сохраняет их полные истории в `data/processed/yambda-1000-users-full-histories.parquet` (4 862 977 прослушиваний). Временные границы, каталог и метрики заданы в `research/protocol/case2-v1.json` и в коде не меняются.

Передача на Windows с проверкой SHA256 каждого файла нужна только для схемы Mac → Windows (нужен SSH-доступ по ключу; адрес задаётся переменной окружения `DEBUG_WINDOWS_HOST`, например `user@host`). На одной машине с CUDA этот шаг пропускается: `recsys.pilot` и `recsys.train_sft` сами считают SHA256 исходников, если файла `transfer-manifest.json` нет.

```bash
python3 scripts/windows.py sync
```

## 3. Тесты

```bash
python -m unittest discover -s tests
```

123 тестов. Среди них: цель видит только строго более ранние timestamp-группы, признаки реранкера не меняются от будущих событий, градиенты RL совпадают с аналитическими, перевзвешивание топ-K сохраняет отношения вероятностей внутри хвоста, решающие правила финального теста, защита от повторного открытия финального окна и новой когорты, кодирование предполагаемого отклика и стандартный NDCG пятёрки при многошаговой генерации.

Скрипты `scripts/*.ps1` написаны под машину автора (путь `E:\Projects\Debug`, задачи планировщика Windows); на другой машине их нужно править, а команды из таблиц ниже относительны и запускаются из корня репозитория напрямую.

Шаги ниже запускать **в указанном порядке**: каждый читает checkpoints предыдущих по фиксированным путям (колонка «Нужно»). Имена выходных папок совпадают с использованными в отчёте; скрипт отказывается писать в существующую папку. Время — фактическое на RTX 3060. На Windows длинные шаги запускались через планировщик задач, чтобы переживать разрыв SSH: `scripts/start-*-windows.ps1`; скрипт отказывается стартовать, если задача или выходная папка уже существуют.

## 4. Первая цепочка (раздел 4: RL v1 и confirmation)

Исторические запуски 21 октября. Нужны раньше основной цепочки: модель награды v1 (`reward-audit-20261021-01/history-last.pt`) служит базой сравнения в аудите RM v2, а GPT-2 первой цепочки — строкой «исходная GPT-2» финального теста.

```bash
python -m recsys.pilot --out artifacts/calibration-20261021-01 --sft-steps 4000 --rm-steps 5000 --rl-steps 200 --batch-size 16 --reward-users 64 --max-seconds 300
python -m recsys.train_sft --out artifacts/sft-20261021-01 --steps 8000 --batch-size 128 --eval-every 1000 --max-seconds 1200 --patience 3 --min-steps 4000
python -m recsys.hybrid
python -m recsys.day_objective
python -m recsys.reward_audit
python -m recsys.rl_comparison
```

`recsys.confirmation` — однократная оценка окна confirmation; окно уже использовано, повторный запуск воспроизводит число, но не является новой проверкой. Описания и результаты — `docs/*-RESULTS.md` и `research/results/*-2026-10-21.json`.

## 5. Основная цепочка (разделы 2–3, 5–6 отчёта)

| # | Команда | Нужно | Время | Проверка на selection |
|---|---|---|---:|---|
| 1 | `python -m recsys.full_sft --out artifacts/pps-20261022-01 --seconds-per-arm 600 --eval-every-seconds 150 --arms full:0.1:0.01,full+pps:0.1:0.01,full+pps+fb:0.1:0.01,full+pps+fb+pos:0.1:0.01` | данные | ≈45 мин | `full+pps+fb` last: NDCG@10 0,1473 — опорная политика |
| 2 | `python -m recsys.full_sft --out artifacts/fullsft-reg-20261022-01 --seconds-per-arm 600 --eval-every-seconds 150 --arms full:0.1:0.01,full:0.3:0.01,full:0.3:0.1,full:0.5:0.1` | данные | ≈45 мин | `full-d0.1-wd0.01` best: 0,1310 — GPT-2 для признаков реранкера |
| 3 | `python -m recsys.full_sft --out artifacts/early-gpt-20261022-01 --seconds-per-arm 600 --eval-every-seconds 300 --arms full:0.1:0.01 --target-end 25032400` | данные | ≈10 мин | ранняя GPT-2 только на целях до первого обучающего дня реранкера |
| 4 | `python -m recsys.reward_v2 --out artifacts/reward-v2-20261022-01 --steps 4000 --ensemble 3` | шаг 1, раздел 4 (RM v1) | ≈17 мин | ансамбль MAE 0,2447, PairAccuracy 0,555 |
| 5 | `python -m recsys.reranker --out artifacts/reranker-20261022-02` | шаги 2, 3 | ≈1 мин | с признаками GPT-2: 0,2349 |
| 6 | `python -m recsys.rl_v2 --out artifacts/rl-v2-20261023-01` | шаги 1, 4 | ≈55 мин | 36 политик; RL секунды β=0,3: 0,1464 |
| 7 | `python -m recsys.rl_distill --out artifacts/rl-v3-20261023-01` | шаги 1, 4 | ≈23 мин | оптимум топ-50: 0,1585; 10 обученных вариантов |
| 8 | долгосрочные награды — три команды ниже | шаг 1 | ≈55 мин | раздел 6 |

Долгосрочные награды (шаг 8), по порядку:

```bash
python -m recsys.reward_v2 --out artifacts/reward-v2-continuation-20261023-01 --steps 4000 --ensemble 3 --target continuation
python -m recsys.rl_v2 --out artifacts/rl-v2-continuation-20261023-01 --rm-dir artifacts/reward-v2-continuation-20261023-01 --rewards completion,completion_lcb --offline-arms= --tag continuation/
python -m recsys.reward_v2 --out artifacts/reward-v2-quantile-20261023-01 --steps 4000 --ensemble 3 --target quantile
```

На Windows эту цепочку запускает `scripts/run-rl-v2-pipeline.ps1`. Её последний шаг, RL на квантильной награде, прервался сбоем драйвера; он выполнен отдельно 25 октября: `python -m recsys.rl_v2 --out artifacts/rl-v2-quantile-20261025-01 --rm-dir artifacts/reward-v2-quantile-20261023-01 --rewards completion --betas 0.03,0.1,0.3,1.0 --offline-arms sft,exp:quantile:0.1,exp:quantile:0.3,exp:quantile:1.0 --tag quantile/` (`scripts/run-rl-quantile-20261025.ps1`, ≈30 мин; отчёт, раздел 6).

## 6. Финальный тест

Нужны шаги 1, 2, 4, 5, 6 основной цепочки и GPT-2 первой цепочки. Сначала пробный прогон на уже использованном окне selection. Он не открывает финальное окно и должен воспроизвести известные значения: опорная 0,1473, A 0,1583, B 0,1464, частота 0,1805, память 0,1958, реранкер 0,2349.

```bash
python -m recsys.final_test --window selection --out artifacts/final-dryrun-selection-20261023-02
```

Затем финальное окно, один раз:

```bash
python -m recsys.final_test --window final --open-final-test
```

До загрузки событий скрипт записывает в `locked-manifest.json` SHA256 документа кандидатов, протокола, кода и всех checkpoints и атомарно создаёт маркер `artifacts/FINAL-TEST-OPENED.json`; при наличии маркера повторный запуск отказывает. В `final_test.py` маркер создаётся **до** расчёта хэшей: при отсутствующем файле запуск упал бы, уже заблокировав окно. Этот риск не сработал; в коде новой когорты порядок исправлен, а `final_test.py` после запуска не менялся, чтобы его хэш совпадал с манифестом. Поэтому перед `--open-final-test` обязателен пробный прогон на selection: он читает те же checkpoints и падает, если какого-то файла нет. Вывод — только в `artifacts/final-test-20261023-01`, около 35 с.

В этом исследовании финальное окно уже открыто 23.10.2026. Результат и манифест — `research/results/runs/final-test-20261023-01/`. Запуск на новой копии воспроизводит этот результат, но не является новой независимой проверкой.

## 7. Проверки после финального теста (разделы 3, 5, 7, 9 отчёта)

Исследовательские расчёты на окне selection и однократная проверка на новых пользователях.

| # | Команда | Нужно | Время | Результат |
|---|---|---|---:|---|
| 9 | `python -m recsys.multistep artifacts/multistep-20261025-01` | шаги 1, 4, 6 | ≈4 мин | генерация на 5 шагов (раздел 7); опорная, жадно, симулятор: прогноз судьи 0,681 на шаге 5 |
| 10 | `python -m recsys.review_checks artifacts/review-checks-20261025-01` | шаги 1, 4, 8 (квантильная RM) | ≈1 мин | перевзвешивание топ-50 моделью награды 0,1583; пятёрка без повторов 0,147 / 0,162 |
| 11 | `python -m recsys.pairacc_recheck artifacts/pairacc-recheck-20261025-01` | шаги 1, 4 | ≈1 мин | PairAccuracy в контексте аудита и запроса, по сессиям |
| 12 | `python -m recsys.exploratory --experiment e1 --out artifacts/exploratory-e1-20261023-01` | шаги 1, 4 | ≈4 мин | голова с попарными признаками: NDCG 0,1612 |
| 13 | `python -m recsys.exploratory --experiment e1 --seed-offset k --out artifacts/exploratory-e1-seedk-20261025-01` для k = 1…4 (`scripts/run-e1-seeds.ps1`) | шаги 1, 4 | ≈15 мин | разброс по seed: 0,1612 ± 0,0007 |
| 14 | `python -m recsys.exploratory --experiment e1 --arms sft_next,sft_positive --out artifacts/exploratory-e1-sft-20261025-01` | шаги 1, 4 | ≈4 мин | контроль без модели награды: 0,1348 / 0,1398 |
| 15 | `python -m recsys.exploratory --experiment e2 --out artifacts/exploratory-e2-20261025-01` | шаги 1, 4 | ≈9 мин | RL на пятёрках: отдача по симулятору 3,515 → 3,537 |
| 16 | `python research/selection/select_new_cohort.py`, затем `python -m recsys.new_cohort --cohort original --window selection` (пробный) и `python -m recsys.new_cohort --cohort new --window final --open-new-cohort` (один раз) | шаги 1, 2, 4, 5, 6, 12 | ≈1 мин | новая когорта: голова E1 0,1287, A 0,1231, опорная 0,1140 |
| 17 | `python -m recsys.metric_ceiling` | данные | секунды, без GPU | потолок NDCG@10 0,90–0,92, Recall@10 0,49–0,55 |

Скрипт `scripts/run-review-fixes-20261025.ps1` запускает шаги 9, 10, 11, 14 и 15 одной задачей. Более ранние версии шагов 9, 10 и 15 (`multistep-20261023-01`, `review-checks-20261023-02`, `exploratory-e2-20261023-01/-02`) сохранены в `research/results/runs/`; они давали те же выводы, а изменения кода между версиями описаны в `docs/REVIEW-CHECKS-RESULTS.md`.

Ранние диагностические запуски 21 октября (`recsys.diagnostics`, `recsys.actr`, `recsys.feature_ablation`, `recsys.multitarget`, `recsys.repeat_gate`) не нужны последующим шагам. Их сохранённые результаты лежат в `research/results/` (`diagnostics-`, `actr-`, `features-`, `multitarget-`, `repeat-gate-2026-10-21.json`), а модули запускаются без обязательных аргументов и пишут в `artifacts/<имя запуска>`; отчёт их подробно не разбирает.

### Дополнительные опыты 25 октября (второй проход)

Исследовательские расчёты на selection, кроме N3. Планы с правилами чтения, написанные до запусков: `docs/PRECISION-PLAN.md`, `docs/REPLAY-PLAN.md`, `docs/JOINT-PLAN.md`, `docs/BSEEDS-PLAN.md`, `docs/NEW-COHORT-N3-PLAN.md`, `docs/SFTKL-PLAN.md`.

| # | Команда | Нужно | Время | Результат |
|---|---|---|---:|---|
| 18 | `python -m recsys.rl_distill --out artifacts/precision-fp32-20261025-01 --arms reverse:pool,forward:pool --grad-diagnostic 4 --precision fp32 --cache-fp32` и то же с `--precision fp16scaled --out artifacts/precision-fp16scaled-20261025-01` без `--cache-fp32` (`scripts/run-precision-20261025.ps1`) | шаги 1, 4 | ≈31 мин | доли оптимума те же, что в FP16: 0,434 / 0,420 («дослушивание») |
| 19 | `python -m recsys.rl_v2 --out artifacts/rl-v2-quantile-20261025-01 --rm-dir artifacts/reward-v2-quantile-20261023-01 --rewards completion --betas 0.03,0.1,0.3,1.0 --offline-arms sft,exp:quantile:0.1,exp:quantile:0.3,exp:quantile:1.0 --tag quantile/` | шаги 1, 8 | ≈30 мин | RL берёт 91–96% аналитического прироста награды |
| 20 | `python -m recsys.exploratory --experiment e1 --arms all,joint --seed-offset k --out artifacts/joint-seedk-20261025-01` для k = 0, 1, 2 (`scripts/run-joint-20261025.ps1`) | шаги 1, 4 | ≈8 мин на seed | совместное обучение: NDCG 0,1596 / 0,1603 / 0,1575 |
| 21 | `python -m recsys.rl_v2 --out artifacts/rl-v2-bseedk-20261025-01 --rewards seconds --betas 0.3 --offline-arms= --seed-offset k` для k = 1, 2, затем `python -m recsys.replay_outcome --out artifacts/replay-bseeds-selection-20261025-01 --extra-policy B_seed1=… --extra-policy B_seed2=…` (`scripts/run-bseeds-20261025.ps1`) | шаги 1, 4, 6 | ≈25 мин | разброс кандидата B по seed |
| 22 | `python3 research/selection/fetch_data.py yambda_likes`, затем `python -m recsys.replay_outcome --likes data/raw/yambda-sequential-50m-likes.parquet` | шаги 1, 4, 6 | ≈1 мин | реально прослушанные секунды и лайки рекомендованных треков |
| 23 | `python -m recsys.new_cohort --cohort original --window confirmation` (пробный), затем `python -m recsys.new_cohort --cohort new --window confirmation --open-new-cohort` (один раз) | шаги 1, 2, 4, 5, 6, 12, 16 | ≈1 мин | N3: A +0,0142, голова +0,0170 к опорной |
| 24 | `python -m recsys.multiplicity` (по сохранённым строкам, без GPU) | шаги 6, 16, 23 | секунды | все шесть подтверждающих гипотез вместе: нижние границы выше нуля |
| 25 | `python -m recsys.exploratory --experiment e1 --arms all,joint,sft_kl:0.1,sft_kl:1,sft_kl:10 --seed-offset k --out artifacts/joint-rows-seedk-20261025-01` для k = 0, 1, 2 (`scripts/run-joint-rows-20261025.ps1`) | шаги 1, 4 | ≈12 мин на seed | интервал «совместное обучение − голова» и supervised-контроль с KL; план `docs/SFTKL-PLAN.md` |
| 26 | `python -m recsys.multistep_fix` (план `docs/MULTISTEP-FIX-PLAN.md`) | шаги 1, 4, 6 | ≈1 мин | пятёрка: без PPS-самоподкрепления дубликатов 0,94 против 0,97; запрет повторов +0,07 NDCG@5 |

Исходная версия `recsys/new_cohort.py` (до режима confirmation) сохранена в `research/code-snapshots/new-cohort-20261024-01/`.

`recsys.reward_audit` не разбирает аргументы командной строки: его запуск с `--help` начнёт расчёт и создаст папку результата. Запускайте его только командой из таблицы выше. Остальные модули `recsys` печатают справку по `--help`.

## 8. Воспроизводимость чисел

Seeds выборок, инициализаций и bootstrap зафиксированы в коде и записываются в `plan.json` каждого запуска. Обучение на GPU с FP16 autocast не бит-в-бит детерминировано, поэтому повторное обучение даст близкие, но не идентичные checkpoints и метрики. Оценка готовых checkpoints детерминирована: два пробных прогона финального теста совпали до шестого знака. Выдача кандидата A чувствительна к точности наград: FP16 вместо FP32 сдвигает NDCG на 0,0003 (`recsys/a_precision_check.py`).

**Версии кода.** Манифесты финального теста, новой когорты и N3 хранят SHA256 кода на момент запуска. После запусков исправлялись и дополнялись `rl_v2.py`, `rl_distill.py`, `reranker.py`, `exploratory.py`, `new_cohort.py` и `final_test.py` (порядок маркера, доля прироста J, режим confirmation, новые необязательные опции); сохранённые результаты от этого не меняются, но текущие файлы с манифестами не совпадают. Точные версии, использованные в запусках, лежат в [research/code-snapshots/](../research/code-snapshots/README.md), их SHA256 совпадают с манифестами (проверено скриптом: для каждого запуска каждый записанный в манифесте файл либо совпадает с текущим, либо есть в снимках).
