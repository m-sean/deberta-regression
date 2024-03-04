import json
import logging
from functools import partial
from tqdm import tqdm
from typing import Dict, Optional, Tuple

import convem
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from data import FineTuningDataset, RegressionDataset, DEFAULT_LABELS
from model import DebertaFineTunedModel
from optimizer import OptimizerConfig
from torch.nn import BCELoss, HuberLoss, MSELoss


def batch_collate_fn(batch, tokenizer: PreTrainedTokenizerFast, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = zip(*batch)
    encodings = tokenizer(
        x,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    x = encodings["input_ids"].to(device)
    mask = encodings["attention_mask"].to(device)
    y = torch.tensor(y, dtype=torch.float, device=device)
    return x, mask, y


def _get_alphas(train: DataLoader, n_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(n_classes, device=device)
    for _, batch_labels, _ in train:
        for label in batch_labels.tolist():
            counts[label] += 1
    return counts / counts.sum()


class DebertaModelTrainer:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        label_to_idx=DEFAULT_LABELS,
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.backend_tokenizer.enable_padding(pad_id=tokenizer.pad_token_id)
        tokenizer.backend_tokenizer.enable_truncation(
            max_length=tokenizer.model_max_length
        )
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = BCELoss()
        self.history = None
        self.label_to_idx = label_to_idx
        self.optimizer = None
        self.norm_clip = None
        self.collate_fn = partial(
            batch_collate_fn,
            tokenizer=tokenizer,
            device=self.device,
        )

    def n_classes(self) -> int:
        return len(self.label_to_idx)

    def get_idx_to_label(self) -> Dict[int, str]:
        return {idx: label for label, idx in self.tag_to_idx.items()}

    def get_data_loader(self, data: str, train: bool) -> DataLoader:
        """Loads datasets from jsonl."""
        dataset = FineTuningDataset.from_jsonl(data, self.label_to_idx)
        return DataLoader(
            dataset, self.batch_size, shuffle=train, collate_fn=self.collate_fn
        )

    def train(
        self,
        train: DataLoader,
        val: DataLoader,
        loss: str,
        epochs: int,
        save: str,
        optimizer: OptimizerConfig,
        norm_clip: Optional[float] = None,
        **model_kwargs,
    ) -> DebertaFineTunedModel:
        """Performs model training with early stopping after no improvement in validation after 3 epochs.

        Args:
            train: Training set Dataloader.
            val: Validation set Dataloader.
            save: Location to save the best model from training.
            optimizer: Optimizer configuration.
            model_kwargs: Model constructor args
        """
        n_classes = self.n_classes()
        self.norm_clip = norm_clip
        model = (
            DebertaFineTunedModel(
                model_name=self.model_name, n_classes=n_classes, **model_kwargs
            )
            .to(device=self.device)
            .train()
        )
        self.optimizer = optimizer.get_optimizer(model.trainable_params())
        self.history = {"train": [], "val": []}
        batch_ct = len(train)
        best_mcc = float('-inf')
        best_epoch = -1
        for e in range(1, epochs + 1):
            with tqdm(total=batch_ct, unit="batch") as pbar:
                pbar.set_description(f"Epoch {e}")
                t_acc, t_mcc, t_loss = self._train_step(model, train, pbar)
                v_acc, v_mcc, v_loss = self._val_step(model, val, pbar)
                pbar.set_postfix_str(
                    f"t_acc={t_acc:.3f} t_mcc={t_mcc:.3f} t_loss={t_loss:.3f} "
                    f"v_acc={v_acc:.3f} v_mcc={v_mcc:.3f} v_loss={v_loss:.3f}"
                )
            self.history["train"].append((t_acc, t_mcc, t_loss))
            self.history["val"].append((v_acc, t_mcc, v_loss))
            if v_mcc > best_mcc:
                logging.info(f"Saving model to {save}/model.pt")
                best_epoch = e
                best_mcc = v_mcc
                model.save(save)
            elif e - best_epoch > 4:
                logging.info("No improvement after 3 epochs, stopping early.")
                break
        return model

    def evaluate(
        self,
        model: DebertaFineTunedModel,
        eval: DataLoader,
        expanded_metrics: bool = True,
    ) -> Tuple[float, float, float]:
        logging.info("Running model evaluation")
        batch_ct = len(eval)
        with tqdm(total=batch_ct, unit="batch") as pbar:
            pbar.set_description("Model evaluation")
            acc, mcc, loss = self._val_step(
                model, eval, pbar, expanded_metrics=expanded_metrics
            )
            loss = f" LOSS={loss:.3f}" if loss else ""
            pbar.set_postfix_str(f"ACC={acc:.3f} MCC={mcc:.3f}{loss}")
        return acc, mcc, loss

    def _train_step(
        self,
        model: DebertaFineTunedModel,
        train: DataLoader,
        pbar: tqdm,
    ) -> Tuple[float, float, float]:
        sum_acc = 0.0
        sum_loss = 0.0
        all_labels = []
        all_preds = []
        model.train()
        for i, (x, mask, labels) in enumerate(train):
            logits = model(input_ids=x, attention_mask=mask)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            if self.norm_clip:
                clip_grad_norm_(model.parameters(), self.norm_clip)
            self.optimizer.step()
            model.zero_grad()
            labels = [l == 1.0 for l in labels.tolist()]
            logits = logits.tolist()
            bcm = convem.binary_confusion_matrix(logits, labels)
            sum_acc += bcm.accuracy()
            sum_loss += loss.detach()
            n = i + 1
            avg_acc = sum_acc / n
            avg_loss = sum_loss / n
            pbar.update()
            pbar.set_postfix_str(
                f"acc={avg_acc:.3f} mcc={bcm.mcc() or float('nan'):.3f} loss={avg_loss:.3f}"
            )
            all_labels.extend(labels)
            all_preds.extend(logits)
        bcm = convem.binary_confusion_matrix(all_preds, all_labels)
        return avg_acc, bcm.mcc() or float("nan"), avg_loss

    def _val_step(
        self,
        model: DebertaFineTunedModel,
        val: DataLoader,
        pbar: tqdm,
        expanded_metrics: bool = False,
    ) -> Tuple[float, float, float]:
        all_labels = []
        all_predicitions = []
        sum_loss = 0.0
        with torch.no_grad():
            model.eval()
            for i, (x, mask, labels) in enumerate(val):
                logits = model(input_ids=x, attention_mask=mask)
                if self.loss_fn:
                    sum_loss += self.loss_fn(logits, labels)
                all_labels.extend([l == 1.0 for l in labels.tolist()])
                all_predicitions.extend(logits.tolist())
                pbar.update()
        bcm = convem.binary_confusion_matrix(all_predicitions, all_labels)
        if expanded_metrics:
            print(f"  PRE: {bcm.precision():.5f}")
            print(f"  REC: {bcm.recall():.5f}")
            print(f"  F1S: {bcm.f1():.5f}")
            print(f"  MCC: {bcm.mcc() or float('nan'): .5f}")
            print(bcm)
        return bcm.accuracy(), bcm.mcc() or float("nan"), sum_loss / (i + 1) if sum_loss else None


class DebertaRegressionModelTrainer:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.backend_tokenizer.enable_padding(pad_id=tokenizer.pad_token_id)
        tokenizer.backend_tokenizer.enable_truncation(
            max_length=tokenizer.model_max_length
        )
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = None
        self.history = None
        self.optimizer = None
        self.norm_clip = None
        self.collate_fn = partial(
            batch_collate_fn,
            tokenizer=tokenizer,
            device=self.device,
        )

    def get_data_loader(self, data: str, train: bool, x_key: str, y_key: str) -> DataLoader:
        with open(data, 'r') as fp:
            records = [json.loads(line) for line in fp]
        """Loads datasets from jsonl."""
        dataset = RegressionDataset(records, x_key, y_key)
        dataloader = DataLoader(
            dataset, 
            self.batch_size, 
            shuffle=train, 
            collate_fn=self.collate_fn
        )
        return dataloader
    

    @staticmethod
    def empty_loader():
        return DataLoader(RegressionDataset([]))


    def set_loss_fn(self, loss: str) -> None:
        if loss == 'huber':
            self.loss_fn = HuberLoss()
        else:
            self.loss_fn = MSELoss()
        
    def train(
        self,
        train: DataLoader,
        val: DataLoader,
        loss: str,
        epochs: int,
        save: str,
        optimizer: OptimizerConfig,
        norm_clip: Optional[float] = None,
    ) -> DebertaFineTunedModel:
        """Performs model training with early stopping after no improvement in validation after 3 epochs.

        Args:
            train: Training set Dataloader.
            val: Validation set Dataloader.
            save: Location to save the best model from training.
            optimizer: Optimizer configuration.
            model_kwargs: Model constructor args
        """
        self.norm_clip = norm_clip
        self.set_loss_fn(loss)
        model = DebertaFineTunedModel(model_name=self.model_name).to(device=self.device).train()
        self.optimizer = optimizer.get_optimizer(model.trainable_params())
        self.history = {"train": [], "val": []}
        batch_ct = len(train)
        best_loss = float('inf')
        best_epoch = -1
        for e in range(1, epochs + 1):
            with tqdm(total=batch_ct, unit="batch") as pbar:
                pbar.set_description(f"Epoch {e}")
                t_loss = self._train_step(model, train, pbar)
                v_loss = self._val_step(model, val, pbar)
                if v_loss:
                    postfix = f"t_loss={t_loss:.3f} v_loss={v_loss:.3f}"
                else:
                    postfix = f"loss={t_loss:.3f}"
                pbar.set_postfix_str(postfix)
            self.history["train"].append(t_loss)
            self.history["val"].append(v_loss)
            if t_loss < best_loss or e == 1:
                logging.info(f"Saving model to {save}/model.pt")
                best_epoch = e
                best_loss = t_loss
                model.save(save)
            elif e - best_epoch > 4:
                logging.info("No improvement after 3 epochs, stopping early.")
                break
        return model

    def evaluate(
        self,
        model: DebertaFineTunedModel,
        eval: DataLoader,
        expanded_metrics: bool = True,
    ) -> Tuple[float, float, float]:
        logging.info("Running model evaluation")
        batch_ct = len(eval)
        with tqdm(total=batch_ct, unit="batch") as pbar:
            pbar.set_description("Model evaluation")
            acc, mcc, loss = self._val_step(
                model, eval, pbar, expanded_metrics=expanded_metrics
            )
            loss = f" LOSS={loss:.3f}" if loss else ""
            pbar.set_postfix_str(f"ACC={acc:.3f} MCC={mcc:.3f}{loss}")
        return acc, mcc, loss

    def _train_step(
        self,
        model: DebertaFineTunedModel,
        train: DataLoader,
        pbar: tqdm,
    ) -> Tuple[float, float, float]:
        sum_loss = 0.0
        model.train()
        for i, (x, mask, labels) in enumerate(train):
            logits = model(input_ids=x, attention_mask=mask)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            if self.norm_clip:
                clip_grad_norm_(model.parameters(), self.norm_clip)
            self.optimizer.step()
            model.zero_grad()
            sum_loss += loss.detach()
            avg_loss = sum_loss / (i + 1)
            pbar.update()
            pbar.set_postfix_str(f"loss={avg_loss:.3f}")
            model.save('tmp.mdl')
        return avg_loss

    def _val_step(
        self,
        model: DebertaFineTunedModel,
        val: DataLoader,
        pbar: tqdm,
    ) -> Tuple[float, float, float]:
        # all_labels = []
        # all_predicitions = []
        sum_loss = 0.0
        with torch.no_grad():
            model.eval()
            for i, (x, mask, labels) in enumerate(val):
                logits = model(input_ids=x, attention_mask=mask)
                if self.loss_fn:
                    sum_loss += self.loss_fn(logits, labels)
                # all_labels.extend([l == 1.0 for l in labels.tolist()])
                # all_predicitions.extend(logits.tolist())
                pbar.update()
        if sum_loss:
            return sum_loss / (i + 1)
