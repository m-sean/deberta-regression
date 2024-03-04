from typing import Dict, List, Tuple
from torch.utils.data import Dataset
import json
import math
import pandas as pd
import torch
from torch.utils.data import DataLoader
from functools import partial

DEFAULT_LABELS = {
    "irrelevant": 0.0, 
    "relevant": 1.0,
}

class RegressionDataset(Dataset):
    def __init__(self, records: List[dict], x_key: str = "content", y_key: str = "value") -> None:
        self.x = []
        self.y = []
        for rec in records:
            self.x.append(rec[x_key])
            self.y.append(math.sqrt(rec[y_key]))

    def __getitem__(self, idx) -> Tuple[str, int]:
        return self.x[idx], self.y[idx]

    def __len__(self) -> int:
        return len(self.y)


class FineTuningDataset(Dataset):
    def __init__(self, records: List[dict], label_to_idx: Dict[str, int]) -> None:
        self.x = []
        self.y = []
        for rec in records:
            self.x.append(rec["content"])
            self.y.append(label_to_idx[rec["label"]])

    def __getitem__(self, idx) -> Tuple[str, int]:
        return self.x[idx], self.y[idx]

    def __len__(self) -> int:
        return len(self.y)

    @classmethod
    def from_jsonl(cls, file: str, label_to_idx: Dict[str, int] = DEFAULT_LABELS):
        with open(file, "r", encoding="utf-8") as src:
            records = list(json.loads(line) for line in src)
            return cls(records, label_to_idx)


class InferenceDataset(Dataset):

    def __init__(self, records: pd.DataFrame ) -> None:
        self.x = []
        for rec in records.itertuples():
            self.x.append(rec.content)

    def __getitem__(self, idx) -> Tuple[str, int]:
        return self.x[idx]

    def __len__(self) -> int:
        return len(self.x)


def _collate_fn(x, tokenizer, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encodings = tokenizer(
        x,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    x = encodings["input_ids"].to(device)
    mask = encodings["attention_mask"].to(device)
    return x, mask

def get_inference_data(file, tokenizer):
    records = pd.read_csv(file)
    dataset = InferenceDataset(records)
    cfn = partial(
        _collate_fn,
        tokenizer=tokenizer,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )
    loader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=False, 
        collate_fn=cfn,
    )
    return records, loader
