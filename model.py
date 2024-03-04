from typing import Dict
import os
import json
import torch
import torch.nn as nn
from transformers import DebertaModel

MAX_MODEL_INPUT = 512

class DebertaFineTunedModel(nn.Module):
    def __init__(self, model_name: str) -> None:
        roberta = DebertaModel.from_pretrained(model_name)
        super(DebertaFineTunedModel, self).__init__()
        self.base = roberta
        self.fc = nn.Linear(roberta.config.hidden_size, MAX_MODEL_INPUT)
        self.relu = nn.ReLU()
        self.output = nn.Linear(MAX_MODEL_INPUT, 1)
        self.sigmoid = nn.Sigmoid()
        self._config = {"model_name": model_name}

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.base(input_ids, attention_mask=attention_mask, return_dict=False)[0]
        x = self.fc(x)
        x = self.relu(x)
        x = x * torch.unsqueeze(attention_mask, 2)
        x = torch.sum(x, 1) / torch.unsqueeze(torch.sum(attention_mask, 1), 1)
        x = self.output(x)
        x = self.sigmoid(x)
        x = x.squeeze(1)
        return x

    def trainable_params(self):
        return (param for _, param in self.named_parameters())

    def _save_config(self, dir):
        with open(f"{dir}/config.json", "w") as fp:
            json.dump(self._config, fp)

    @staticmethod
    def _load_config(dir) -> Dict[str, any]:
        with open(f"{dir}/config.json", "r") as fp:
            return json.load(fp)

    def save(self, dir: str) -> None:
        if not os.path.isdir(dir):
            os.mkdir(dir)
        self._save_config(dir)
        torch.save(self.state_dict(), f=f"{dir}/model.pt")

    @classmethod
    def load(cls, dir: str):
        config = cls._load_config(dir)
        model = cls(**config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(f"{dir}/model.pt", map_location=device)
        model.load_state_dict(state_dict)
        return model
