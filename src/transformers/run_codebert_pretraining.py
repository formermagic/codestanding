from argparse import ArgumentParser, Namespace
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Union, cast

import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler
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
    lr_scheduler: Optional[LambdaLR]

    def __init__(self, hparams: Namespace) -> None:
        super().__init__()
        self.hparams = hparams
        self.tokenizer = self.load_tokenizer()
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

    def load_tokenizer(self) -> CodeBertTokenizerFast:
        tokenizer = CodeBertTokenizerFast.from_pretrained(
            self.hparams.tokenizer_path
        )
        tokenizer = cast(CodeBertTokenizerFast, tokenizer)
        # tokenizer.backend_tokenizer.add_special_tokens(["<nl>"])
        return tokenizer

    # pylint: disable=arguments-differ
    def forward(
        self,
        input_ids: torch.LongTensor,
        masked_lm_labels: torch.LongTensor,
        **kwargs: Any,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        self.model.train()
        outputs = self.model.forward(
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
        optimizer: Optimizer,
        optimizer_idx: int,
        second_order_closure: Optional[Callable] = None,
    ) -> None:
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_progress_bar_dict(self) -> Dict[Text, Union[int, str]]:
        progress_bar_dict = super().get_progress_bar_dict()
        progress_bar_dict["lr"] = "{:.8f}".format(self.lr_scheduler.get_lr()[-1])  # type: ignore
        return progress_bar_dict

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
            data_path=self.hparams.train_data_path,
            max_length=512,
        )

        sampler = RandomSampler(dataset)
        collator = DataCollatorForLanguageModeling(self.tokenizer)
        data_loader = DataLoader(
            dataset,
            batch_size=self.hparams.train_batch_size,
            num_workers=0,
            sampler=sampler,
            collate_fn=collator.collate_batch,
        )

        def training_steps(dataset_len: int) -> int:
            batch_size = self.hparams.train_batch_size
            per_gpu_samples = dataset_len // (
                batch_size * max(1, self.hparams.gpus)
            )
            per_gpu_samples //= self.hparams.accumulate_grad_batches
            return per_gpu_samples * self.hparams.max_epochs

        if getattr(self.trainer, "max_steps") is None:
            t_total = training_steps(len(data_loader.dataset))
        else:
            trainer = cast(pl.Trainer, self.trainer)
            t_total = trainer.max_steps

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=t_total,
        )
        self.lr_scheduler = scheduler
        return data_loader

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # fmt: off
        parser.add_argument("--tokenizer_path", type=str, default=None,
                            help="A path to pretrained tokenizer saved files.")
        parser.add_argument("--warmup_steps", type=int, default=None,
                            help="A number of warmup steps to make.")
        parser.add_argument("--weight_decay", type=float, default=None,
                            help="A weight_decay value for optimizer.")
        parser.add_argument("--train_data_path", type=str, default=None,
                            help="A path to the training data file.")
        parser.add_argument("--local_rank", type=int, default=-1,
                            help="local_rank for distributed training on gpus")
        parser.add_argument("--train_batch_size", type=int, default=1,
                            help="Batch size value for training setup.")
        # fmt: on
        return parser