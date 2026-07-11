# Hate-Speech Detection for Low-Resource Spanish — XLM-RoBERTa-CNN

Detecting toxic comments / hate speech in **Spanish** — a low-resource language
for this task — by combining the multilingual transformer **XLM-RoBERTa** with a
small **CNN** classification head, and studying how different fine-tuning
strategies (pure-language vs. translation-augmented data) behave on a small
dataset.

> M.Sc. coursework project (Natural Language Processing). This is a cleaned,
> reorganized implementation refactored into Python modules. The full write-up is
> in [`report/report.pdf`](report/report.pdf).

## Overview

Hate speech is a widespread problem on social media, and moderation systems need
to work across many languages. Spanish, however, has very little publicly
available labelled data for this task. This project asks two questions:

1. Does a **multilingual** model (XLM-RoBERTa-CNN) beat a **Spanish-specific**
   one (BETO-CNN) at detecting toxic comments in Spanish?
2. When fine-tuning the multilingual model on little data, is it better to use
   **pure-language** data or **translation-augmented** data — and how does that
   interact with the number of fine-tuning epochs?

## Approach

- **Backbone.** A pre-trained transformer (XLM-RoBERTa, or BETO for the Spanish
  baseline) encodes each comment. Rather than using only the last layer, all
  **13 hidden states** (embeddings + 12 encoder layers) are stacked along the
  channel dimension.
- **Head.** A 2-D convolution over the stacked layers, followed by ReLU,
  max-pooling, dropout and a linear layer with a sigmoid — a binary
  (toxic / clean) classifier trained with binary cross-entropy.
- **Fine-tuning strategies** (across Spanish, Turkish and French):
  *pure language*, *translated to Spanish*, and *translated from Spanish*,
  each for 1 and 4 epochs.

## Key results

Reported on the toxic (positive) class of the test set. See the report for the
full per-language, per-epoch tables.

| Model (Spanish) | F1 | Recall |
| --- | --- | --- |
| XLM-RoBERTa + CNN | **0.80** | **0.93** |
| BETO + CNN | 0.80 | 0.77 |

**Findings.** The multilingual model matches BETO on F1 but clearly wins on
recall (0.93 vs. 0.77) — important when the goal is to catch as many toxic
comments as possible. For fine-tuning, **translated data helps with short
fine-tuning** (1 epoch), while **pure-language data wins with longer
fine-tuning** (4 epochs); languages that are linguistically close (Spanish and
French) are affected the most, while Turkish stays roughly stable.

## Repository structure

```
.
├── src/
│   ├── model.py      # TransformerCNN (XLM-RoBERTa / BETO backbone + CNN head)
│   ├── data.py       # CSV loading and tokenization
│   └── train.py      # training / evaluation loop (CLI)
├── report/report.pdf # project report
├── requirements.txt
└── README.md
```

## Data

The data is **not included** in this repository. It comes from the
[Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification)
challenge. Each split is a CSV with two columns:

| column | type | meaning |
| --- | --- | --- |
| `content` | str | the comment text |
| `toxic` | int | `1` = toxic / hate speech, `0` = clean |

Expected files (per language): `train_es.csv`, `val_es.csv`, `test_es.csv`, and
the Turkish / French equivalents.

## Setup and usage

```bash
pip install -r requirements.txt

# Spanish only: multilingual model vs. Spanish baseline
python src/train.py --backbone xlm-roberta-base \
    --train data/train_es.csv --val data/val_es.csv --test data/test_es.csv --epochs 4
python src/train.py --backbone beto \
    --train data/train_es.csv --val data/val_es.csv --test data/test_es.csv --epochs 4

# Multilingual fine-tuning (pure vs. translated): pass several training files
python src/train.py --backbone xlm-roberta-base \
    --train data/train_es.csv data/train_turkish.csv data/train_french.csv \
    --val data/val_es.csv --test data/test_es.csv --epochs 1
```

A GPU is recommended. Training was originally run on a single CUDA device.

## Requirements

See [`requirements.txt`](requirements.txt) — PyTorch, Hugging Face Transformers
(+ SentencePiece), scikit-learn, pandas and NumPy.
