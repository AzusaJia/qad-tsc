# QAD

QAD is a self-supervised representation learning implementation for time-series
data. It uses multi-scale patches, frequency masking, a student/EMA-teacher
architecture, and a combined quantile and pointwise loss. After pretraining, the
frozen teacher features are evaluated with a Ridge classifier.

Supported datasets: `Epilepsy`, `HAR`, `PAMAP2`, `Skoda`, and `Sleep`.

## Installation

Python 3.10 or later is recommended. Install the core dependencies:

```bash
pip install torch numpy scikit-learn huggingface_hub
```

A CUDA GPU is recommended for training, but CPU execution is also supported.

## Download the datasets

Download the datasets from
[ZihanJia/datasets-for-QAD](https://huggingface.co/datasets/ZihanJia/datasets-for-QAD).
From the project root, run:

```bash
hf download ZihanJia/datasets-for-QAD \
  --repo-type dataset \
  --local-dir datasets
```

## Quick start

The default command pretrains and evaluates all five datasets:

```bash
python run.py --result-filename result.csv
```

Run a single dataset:

```bash
python run.py --dataset PAMAP2 --result-filename pamap2.csv
```

Run a comma-separated subset:

```bash
python run.py --dataset Epilepsy,HAR --result-filename epilepsy_har.csv
```

## Resume training

Use the same seed, `loss_lambda`, and result filename as the original run, then
add `--resume`:

```bash
python run.py \
  --dataset PAMAP2 \
  --epochs 10 \
  --loss-lambda 0.5 \
  --seed 42 \
  --result-filename pamap2.csv \
  --resume
```

The program resumes from the matching `last.pt` or `best.pt` checkpoint and
skips datasets that are already complete.

## Downstream-only evaluation

If a matching checkpoint already exists, skip pretraining with:

```bash
python run.py \
  --mode downstream-only \
  --dataset PAMAP2 \
  --loss-lambda 0.5 \
  --seed 42 \
  --result-filename pamap2_eval.csv
```

For a single dataset, a checkpoint can also be provided explicitly:

```bash
python run.py \
  --mode downstream-only \
  --dataset PAMAP2 \
  --checkpoint result/checkpoints/loss_lambda0.5_seed42/PAMAP2/best.pt \
  --result-filename pamap2_eval.csv
```

## Few-shot evaluation

Few-shot evaluation requires pretrained checkpoints matching the selected seed
and `loss_lambda`:

```bash
python run.py \
  --mode few-shot \
  --dataset PAMAP2 \
  --few-shot-values 1 5 10 50 100 500 \
  --loss-lambda 0.5 \
  --seed 42 \
  --device cuda:0
```

Few-shot results are written to `result/fewshot/` by default, with one CSV
file per shot count.

## Main arguments

View all arguments with:

```bash
python run.py --help
```

## Outputs

```text
result/
├── result.csv
├── result.log
├── checkpoints/
│   └── <experiment>/<dataset>/
│       ├── best.pt
│       └── last.pt
└── fewshot/
    └── *.csv
```

The standard result CSV contains `dataset`, `linear_accuracy`, and
`linear_f1_macro`. Its final row reports the mean over all evaluated datasets.
