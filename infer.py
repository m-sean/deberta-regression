import json
from model import DebertaFineTunedModel
from transformers import AutoTokenizer
import pandas as pd
from data import get_inference_data
import torch 
from tqdm import tqdm

def main():
    model = DebertaFineTunedModel.load("roberta-large-BCS")
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    tokenizer.backend_tokenizer.enable_padding(pad_id=tokenizer.pad_token_id)
    tokenizer.backend_tokenizer.enable_truncation(
        max_length=tokenizer.model_max_length
    )
    df, loader = get_inference_data(
        "/tmp/inference-data.csv",
        tokenizer,
    )
    preds = []
    conf = []
    model.to(torch.device('cuda'))
    with torch.no_grad():
        model.eval()
        for (x, mask) in tqdm(loader):
            logits = model(x, mask)
            for l in logits.squeeze():
                if l > 0.5:
                    preds.append('relevant')
                    conf.append(l)
                else:
                    preds.append('irrelevant')
                    conf.append(1-l)

    df['roberta'] = preds
    df['conf'] = conf
    df.to_csv("inference-data-annotated.csv", index=False)

if __name__ == "__main__":
    main()