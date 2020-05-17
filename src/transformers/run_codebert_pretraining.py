from argparse import Namespace
from typing import Any, Dict, List, Optional, Text, Tuple, cast

import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    RobertaConfig,
    RobertaForMaskedLM,
    get_linear_schedule_with_warmup,
)
from transformers.data.data_collator import DataCollatorForLanguageModeling

from .dataset_codebert import CodeBertDataset
from .tokenization_codebert import CodeBertTokenizerFast


class CodeBertLMPretraining(pl.LightningModule):
    tokenizer: CodeBertTokenizerFast
    optimizer: Optional[Optimizer]
    lr_scheduler: Optional[_LRScheduler]

    def __init__(self, hparams: Namespace) -> None:
        super().__init__()
        self.hparams = hparams
        self.tokenizer = cast(
            CodeBertTokenizerFast,
            CodeBertTokenizerFast.from_pretrained(hparams.save_dir),
        )
        self.model = self.load_model()
        self.optimizer = None
        self.lr_scheduler = None

    def load_model(self) -> RobertaForMaskedLM:
        config = RobertaConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=768,
            num_hidden_layers=2,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            max_position_embeddings=512 + 2,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        model = RobertaForMaskedLM(config)
        model.resize_token_embeddings(len(self.tokenizer))
        return model

    # pylint: disable=arguments-differ
    def forward(
        self,
        input_ids: torch.LongTensor,
        masked_lm_labels: torch.LongTensor,
        **kwargs: Any
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        self.model.train()
        outputs = self.model(
            input_ids=input_ids, masked_lm_labels=masked_lm_labels
        )
        loss, prediction_scores = outputs[:2]
        return loss, prediction_scores

    # pylint: disable=arguments-differ, unused-argument
    def training_step(
        self, batch: Dict[Text, torch.Tensor], batch_idx: int
    ) -> Dict[Text, torch.Tensor]:
        loss, _ = self.forward(**batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    # pylint: disable=too-many-arguments
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: int,
        second_order_closure=None,
    ) -> None:
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self) -> Dict[Text, Any]:
        avg_loss = getattr(self.trainer, "avg_loss", 0.0)

        tqdm_dict = {
            "loss": "{:.3f}".format(avg_loss),
            "lr": self.lr_scheduler.get_lr(),
        }
        return tqdm_dict

    def configure_optimizers(self) -> List[Optimizer]:
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4,)
        self.optimizer = optimizer
        return [optimizer]

    def train_dataloader(self) -> DataLoader:
        dataset = CodeBertDataset(
            tokenizer=self.tokenizer,
            data_path=self.hparams.dataset_path,
            max_length=512,
        )
        collator = DataCollatorForLanguageModeling(self.tokenizer)

        train_batch_size = 5
        data_loader = DataLoader(
            dataset,
            batch_size=train_batch_size,
            num_workers=0,
            shuffle=True,
            collate_fn=collator.collate_batch,
        )

        setattr(self.hparams, "n_gpu", 1)
        setattr(self.hparams, "gradient_accumulation_steps", 32)
        setattr(self.hparams, "num_train_epochs", 1000)
        setattr(self.hparams, "warmup_steps", 5000)

        t_total = (
            (
                len(data_loader.dataset)
                // (train_batch_size * max(1, self.hparams.n_gpu))
            )
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=t_total,
        )
        self.lr_scheduler = scheduler
        return data_loader
