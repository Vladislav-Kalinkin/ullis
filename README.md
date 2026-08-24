# Ullis

Ullis — компактный тренер dense ternary Hyena language models для Apple Silicon. Основной backend — Metal; `--backend cpu` оставлен как детерминированный reference fallback.

## Контракт памяти

- Persistent model state хранится в FP16: embedding, latent ternary masters и compact implicit filters.
- Ternary projection — две packed bitplanes и row scale; FP32 не является постоянной копией весов.
- Optimizer — clipped stateless SGD без momentum/variance-state.
- Streamed MTP cross-entropy не создаёт `[batch,time,vocab]` logits/probabilities.
- Hyena convolution использует bounded overlap-save FFT, поэтому scratch зависит от filter/chunk, а не от всего context.

Metal-resident путь держит MTP heads, loss gradient, обратный activation stream, projection updates и FP16 code refresh на GPU. Текущая filter-backward reference bridge читает только компактные `O(D*order)` statistics, не активации и не logits.

## Первый прогон

```sh
cargo test
cargo run --release -- train \
  --data examples/first-train.jsonl \
  --run runs/hello \
  --steps 100 \
  --learning-rate 0.01 \
  --checkpoint-every 25
```

CLI обучает BPE на самом dataset до создания модели и сохраняет `config.json`, `tokenizer.json`, `metrics.jsonl` и lossless FP16 `checkpoint.json`.

Вывод содержит raw loss одного окна и `ema`. Соседние окна могут иметь разный loss; прогресс оценивают по EMA и одинаковой validation выборке, а не по обязательному падению каждой строки.

Продолжение обучения:

```sh
cargo run --release -- train --data examples/first-train.jsonl \
  --run runs/hello --resume runs/hello/checkpoint.json --steps 100
```

CPU запуск явно:

```sh
cargo run --release -- train --data examples/first-train.jsonl \
  --run runs/cpu-check --steps 100 --backend cpu
```

Полная справка без расширения: [USAGE](/Users/vladislavkalinkin/ullis/USAGE).

## Dataset

Каждая строка JSONL — conversation. `assistant.thinking` обязателен и остаётся отдельной размеченной частью training text. Неисполняемые пока поля `tool_calls` / `tool_call_id` сохраняют schema для будущих agent traces.

```json
{"id":"demo","messages":[{"role":"user","content":"What is 2+2?"},{"role":"assistant","thinking":"Use arithmetic.","content":"4"}]}
```

`examples/first-train.jsonl` — repeated overfit sanity corpus: он проверяет, что loss способен снижаться, но не является полезным production corpus.
