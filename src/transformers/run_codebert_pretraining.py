import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Union, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, RobertaConfig, RobertaForMaskedLM
from transformers.data.data_collator import DataCollatorForLanguageModeling

from .dataset_codebert import CodeBertDataset
from .optimization import get_polynomial_decay_with_warmup
from .tokenization_codebert import CodeBertTokenizerFast
from .utils import get_perplexity


class ValidSaveCallback(Callback):
    def __init__(self, filepath: Text, save_top_k: int = 3) -> None:
        self.filepath = filepath
        self.save_top_k = save_top_k

    @staticmethod
    def _keep_last_files(num: int, dirname: Text) -> None:
        paths = sorted(Path(dirname).iterdir(), key=os.path.getmtime)
        for path in paths[:-num]:
            os.remove(path)

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        save_filepath = os.path.join(self.filepath, "{epoch}-{step}")
        model_checkpoint = ModelCheckpoint(
            save_filepath, save_top_k=self.save_top_k
        )
        save_filepath = model_checkpoint.format_checkpoint_name(
            epoch=trainer.current_epoch,
            metrics=dict(**trainer.callback_metrics, step=trainer.global_step),
        )

        model_checkpoint.save_function = trainer.save_checkpoint
        # pylint: disable=protected-access
        model_checkpoint._save_model(save_filepath)

        # keep last `save_top_k` files
        self._keep_last_files(num=self.save_top_k, dirname=self.filepath)


class CodeBertLMPretraining(pl.LightningModule):
    tokenizer: CodeBertTokenizerFast
    optimizer: Optional[Optimizer]
    lr_scheduler: Optional[LambdaLR]
    trainer: Optional[pl.Trainer]

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
            num_hidden_layers=self.hparams.num_hidden_layers,
            num_attention_heads=self.hparams.num_attention_heads,
            intermediate_size=3072,
            hidden_act="gelu",
            max_position_embeddings=512 + 2,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
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
        tokenizer.backend_tokenizer.add_special_tokens(["<nl>"])
        return tokenizer

    @property
    def last_learning_rate(self) -> torch.FloatTensor:
        # pylint: disable=not-callable
        if self.lr_scheduler is None:
            values = torch.tensor([float("nan")])
        else:
            values = self.lr_scheduler.get_last_lr()  # type: ignore
            values = torch.tensor(values).mean()
        return cast(torch.FloatTensor, values.float())

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

    # pylint: disable=arguments-differ, unused-argument, not-callable
    def training_step(
        self, batch: Dict[Text, torch.Tensor], batch_idx: int
    ) -> Dict[Text, Union[torch.Tensor, Dict[Text, torch.Tensor]]]:
        # prepare logging meter values
        loss, _ = self.forward(**batch)
        perplexity = get_perplexity(loss)
        learning_rate = self.last_learning_rate
        batch_size = torch.tensor([self.hparams.batch_size])
        tensorboard_logs = {
            "train_loss": loss,
            "train_ppl": perplexity,
            "train_lr": learning_rate,
            "train_bz": batch_size,
        }

        return {
            "loss": loss,
            "ppl": perplexity,
            "lr": learning_rate,
            "log": tensorboard_logs,
        }

    # pylint: disable=arguments-differ, unused-argument
    def validation_step(
        self, batch: Dict[Text, torch.Tensor], batch_idx: int
    ) -> Dict[Text, torch.Tensor]:
        # prepare loss and ppl
        loss, _ = self.forward(**batch)
        perplexity = get_perplexity(loss)
        return {"loss": loss, "ppl": perplexity}

    def validation_epoch_end(
        self, outputs: List[Dict[Text, torch.Tensor]]
    ) -> Dict[Text, Dict[Text, torch.Tensor]]:
        # prepare average loss and ppl
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_perplexity = torch.stack([x["ppl"] for x in outputs]).mean()
        tensorboard_logs = {
            "val_loss": avg_loss,
            "val_ppl": avg_perplexity,
        }

        return {"log": tensorboard_logs}

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
        progress_bar_dict["lr"] = "{}".format(self.lr_scheduler.get_last_lr()[-1])  # type: ignore
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

        optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.98),
            eps=1e-6,
            lr=self.hparams.lr,
        )
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
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            sampler=sampler,
            collate_fn=collator.collate_batch,
        )

        def training_steps(dataset_len: int) -> int:
            num_gpus = self.trainer.gpus
            if isinstance(num_gpus, list):
                num_gpus = list(num_gpus)

            batch_size = self.hparams.batch_size
            per_gpu_samples = dataset_len // (batch_size * max(1, num_gpus))
            per_gpu_samples //= self.trainer.accumulate_grad_batches
            return per_gpu_samples * self.trainer.max_epochs

        if getattr(self.trainer, "max_steps") is None:
            t_total = training_steps(len(data_loader.dataset))
        else:
            t_total = self.trainer.max_steps

        if self.optimizer is not None:
            scheduler = get_polynomial_decay_with_warmup(
                self.optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=t_total,
                power=self.hparams.power,
            )
            self.lr_scheduler = scheduler

        return data_loader

    def val_dataloader(self) -> DataLoader:
        dataset = CodeBertDataset(
            tokenizer=self.tokenizer,
            data_path=self.hparams.val_data_path,
            max_length=512,
        )

        collator = DataCollatorForLanguageModeling(self.tokenizer)
        data_loader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=collator.collate_batch,
        )

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
        parser.add_argument("--power", type=float, default=1.0,
                            help="A power of learning rate decay.")
        parser.add_argument("--train_data_path", type=str, default=None,
                            help="A path to the training data file.")
        parser.add_argument("--val_data_path", type=str, default=None,
                            help="A path to the validation data file.")
        parser.add_argument("--local_rank", type=int, default=-1,
                            help="local_rank for distributed training on gpus")
        parser.add_argument("--batch_size", type=int, default=1,
                            help="Batch size value for training setup.")
        parser.add_argument("--lr", type=float, default=0.001,
                            help="Train learning rate for optimizer.")
        parser.add_argument("--num_hidden_layers", type=int, default=6,
                            help="A number of transformer encoder hidden layers.")
        parser.add_argument("--num_attention_heads", type=int, default=12,
                            help="A number of self-attention heads.")
        parser.add_argument("--num_workers", type=int, default=0,
                            help="A number of workers for data loaders.")
        # fmt: on
        return parser


def main() -> None:
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="The WandB project name to write logs to.")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="The WandB experiment name to write logs to.")
    parser.add_argument("--wandb_id", type=str, default=None,
                        help="The WandB id to use for resuming.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="The dir to save training checkpoints.")
    parser.add_argument("--save_interval_updates", type=int, default=None,
                        help="The interval of steps between checkpoints saving.")
    # fmt: on

    parser = CodeBertLMPretraining.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    code_bert_model = CodeBertLMPretraining(hparams)
    wandb_logger = WandbLogger(
        project=hparams.wandb_project,
        name=hparams.wandb_name,
        id=hparams.wandb_id,
    )
    wandb_logger.watch(code_bert_model.model, log="gradients", log_freq=1)

    val_save = ValidSaveCallback(hparams.save_dir)
    trainer = pl.Trainer(
        gpus=hparams.gpus,
        num_nodes=hparams.num_nodes,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_steps=hparams.max_steps,
        gradient_clip_val=hparams.gradient_clip_val,
        val_check_interval=hparams.save_interval_updates,
        logger=wandb_logger,
        callbacks=[val_save],
        auto_scale_batch_size=hparams.auto_scale_batch_size,
        resume_from_checkpoint=hparams.resume_from_checkpoint,
    )

    trainer.fit(code_bert_model)


if __name__ == "__main__":
    main()
