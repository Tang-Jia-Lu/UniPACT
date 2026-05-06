# annotation/

Place the LLaVA-format JSON annotation files here before running training or
testing. Paths in `1_train.sh`, `2_train.sh`, and `3_test.sh` are relative to
this directory.

## Download

All annotation files are hosted at:

  https://huggingface.co/datasets/jialucode/MIMIC_PROGNOSIS/tree/main

Download the files listed below into this `annotation/` directory.

## Required raw files (training source)

10 per-task files used as input to `scripts/make_balanced_split.py`:

```
train_deterioration.json
train_diagnose_00.json   train_diagnose_01.json   train_diagnose_02.json
train_diagnose_03.json   train_diagnose_04.json   train_diagnose_05.json
train_diagnose_06.json
train_icu.json
train_mortality.json
```

Each entry is a LLaVA conversation; the gpt-side answer must start with
`yes` or `no` for the per-class balancing logic to work.

## Generate the two balanced splits

Run from the repo root:

```bash
python scripts/make_balanced_split.py \
    --input-dir ./annotation \
    --output    ./annotation/balanced_stage1_seed42.json \
    --seed 42

python scripts/make_balanced_split.py \
    --input-dir ./annotation \
    --output    ./annotation/balanced_stage2_seed43.json \
    --seed 43
```

Stage 1 (`1_train.sh`) consumes `balanced_stage1_seed42.json`; Stage 2
(`2_train.sh`) consumes `balanced_stage2_seed43.json`. The two seeds are
independent random draws so the stages see different (but equally
class-balanced) samples.

## Eval / test file

A single `test.json` is used by `1_train.sh`, `2_train.sh`, and `3_test.sh`
(eval-during-training and final test).

```
test.json
```
