"""Data loading and tokenization.

The datasets are CSV files with two columns:

    content : str   -- the comment text
    toxic   : int   -- binary label (1 = toxic / hate speech, 0 = clean)

They come from the Jigsaw Multilingual Toxic Comment Classification challenge
(Kaggle) and are split per language (Spanish, Turkish, French). The data itself
is not distributed with this repository.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer, XLMRobertaTokenizer

from model import BETO_ID

MAX_LENGTH = 300


def get_tokenizer(backbone: str = "xlm-roberta-base"):
    """Return the tokenizer that matches the chosen backbone."""
    if backbone == "beto":
        return BertTokenizer.from_pretrained(BETO_ID)
    return XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")


def load_csvs(paths) -> pd.DataFrame:
    """Read and concatenate one or more CSV files into a single DataFrame."""
    if isinstance(paths, str):
        paths = [paths]
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def _encode(texts, tokenizer):
    input_ids, attention_masks = [], []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)


def make_loader(df: pd.DataFrame, tokenizer, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
    """Tokenize a DataFrame and wrap it in a DataLoader."""
    input_ids, attention_masks = _encode(df["content"].values, tokenizer)
    labels = torch.tensor(df["toxic"].values)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)
