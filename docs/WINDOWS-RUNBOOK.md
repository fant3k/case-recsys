# Запуск экспериментов на Windows

Проект: `E:\Projects\Debug`. Python: `.venv\Scripts\python.exe`. Команды ниже выполняются из этой папки, обычные права администратора для обучения не нужны.

## Проверки перед обучением

```powershell
Set-Location E:\Projects\Debug
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Девять проверок охватывают время и tied-группы, OOV, знаменатель метрик, дубликаты, padding и RL-градиент/аналитический оптимум на маленькой задаче. Они проверяют корректность отдельных механизмов; не доказывают качество рекомендаций.

## Технический пилот

```powershell
.\.venv\Scripts\python.exe -m recsys.pilot --out artifacts/my-new-pilot --sft-steps 80 --rm-steps 200 --rl-steps 40 --batch-size 16 --reward-users 64 --max-seconds 900
```

Выходная папка должна быть новой. Программа отказывается перезаписывать старый опыт. Используется selection validation; confirmation и final test не оцениваются. Небольшая выборка пользователей для ожидаемого reward фиксирована seed и не зависит от будущей активности.

Пилот включает GPT-2, RM, продолженную SFT, RL completion/beta=0,03, аналитический H=1-контроль и H=5-диагностику. Это не полная исследовательская матрица. Полный основной протокол требует второй функции награды, других beta, абляции RM без истории, интервалов и повторов.

Калибровочный пример с большим числом шагов:

```powershell
.\.venv\Scripts\python.exe -m recsys.pilot --out artifacts/my-new-calibration --sft-steps 4000 --rm-steps 5000 --rl-steps 200 --batch-size 16 --reward-users 64 --max-seconds 300
```

Настройки текущей реализации: SFT AdamW lr=0,0003; RM lr=0,001; continued SFT и RL lr=0,0001, новые оптимизаторы; weight decay=0,01, gradient norm clip=1. FP16 autocast с GradScaler initial scale=1024; logits/loss/KL — FP32. RM batch в четыре раза больше policy batch. Moving baseline RL: train global clipped-response mean, затем EMA с коэффициентом 0,95. Первый короткий пилот использовал baseline=0; его исходники отдельно сохранены.

## Результаты запуска

- `status.json`, `run.log`: состояние, этап и возможная ошибка.
- `report.json`: итог, метрики, времена и ограничения интерпретации.
- `configuration.json`, `source-manifest.json`, `code-snapshot.zip`: постановка, окружение, хеши и исходники.
- `sft-checkpoint.pt`: веса, оптимизатор, scaler, RNG и каталог; восстановление предсказаний/оптимизатора проверяется самим пилотом.
- `reward-model.pt`, `continued-sft.pt`, `rl-checkpoint.pt`: остальные состояния. Полноценный CLI resume после обрыва пока не реализован, не считать наличие checkpoint гарантией автоматического продолжения.
- `*-trace.json`: loss, шаги, время и пики PyTorch CUDA allocator.
- `rankings.json`, `ranking-per-user.json`: выдача и пользовательские метрики.
- `reward-validation.json`: ошибка RM и простых ориентиров на выборке наблюдаемых in-catalog событий.
- `reward-policy-comparison.json`: expected reward, reward top-10 и KL на заранее выбранной группе.
- `h5-diagnostic.json`: гипотетическая генерация без выдуманных пользовательских откликов.

## Измерение batch size

```powershell
.\.venv\Scripts\python.exe -m recsys.benchmark --out artifacts/my-new-benchmark.json --batches 16 64 128 256
```

Это короткое измерение throughput с пятью прогревочными и десятью измеряемыми шагами на размер batch. Оно не сохраняет обученную модель и не использует heldout-качество. Замер только SFT не гарантирует, что более сложная конфигурация RL имеет такой же пик памяти.

## Управление с Mac

Из `<корень репозитория>`:

```bash
python3 scripts/windows.py sync
python3 scripts/windows.py exec scripts/pilot-windows.ps1
python3 scripts/windows.py fetch artifacts/pilot-20261021-01/report.json
```

`sync` отправляет только исходники, документацию, протокол, requirements и подготовленную выборку 32 МБ. Mac `.venv`, SSH-ключи, `.git`, сырые архивы и старые облачные логи не передаются. Архив и каждый файл сверяются SHA256. PS1-файлы содержат конкретные run IDs: для повторения укажите новое имя выхода, не удаляйте предыдущие результаты.

Текущий запуск — foreground-процесс через SSH. Работа после разрыва соединения или выключения Mac не проверена и не гарантируется. Для длительного основного обучения сначала нужен отдельный устойчивый launcher. Готовые завершённые результаты и checkpoints сохраняются на Windows.
