import os
import typing
from argparse import Namespace
from collections import OrderedDict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

from fairseq import models, options
from fairseq.criterions import FairseqCriterion
from fairseq.data import (
    BacktranslationDataset,
    FairseqDataset,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    LanguagePairDataset,
    RoundRobinZipDatasets,
)
from fairseq.models import BaseFairseqModel
from fairseq.optim import FairseqOptimizer
from fairseq.sequence_generator import SequenceGenerator
from fairseq.tasks import FairseqTask, register_task
from fairseq.utils import deprecation_warning

from .masked_dictionary import MaskedDictionary
from .masked_language_pair_dataset import MaskedLanguagePairDataset
from .noisy_language_pair_dataset import NoisyLanguagePairDataset


def infer_mono_lang_pairs(steps: List[str]) -> List[str]:
    langs = [s.split("-")[0] for s in steps if len(s) > 0]
    lang_pairs = [f"{lang}-{lang}" for lang in langs]
    return lang_pairs


def infer_para_lang_pairs(steps: List[str]) -> List[str]:
    pairs = [pair for pair in set(steps) if len(pair) > 0]
    pairs = ["-".join(sorted(pair.split("-"))) for pair in pairs]
    lang_pairs = list(set(pairs))
    return lang_pairs


class DatasetKey(Enum):
    MT = ""
    MEMT = "memt: "
    MASS = "mass: "
    BT = "bt: "
    EVAL = ""

    def paired_with(self, name: str) -> str:
        return self.value + name

    def keyed_dataset(self, datasets: Dict) -> List[Tuple[str, FairseqDataset]]:
        return [
            (self.paired_with(lang_pair), dataset)
            for lang_pair, dataset in datasets.items()
        ]


@register_task("unsupervised_mass")
class UnsupervisedMASSTask(FairseqTask):
    def __init__(
        self,
        args: Namespace,
        dicts: Dict[str, MaskedDictionary],
        training: bool,
    ) -> None:
        super().__init__(args)
        self.dicts = dicts
        self.training = training
        self.backtranslators: Dict[str, Callable] = {}
        self.langs = list(dicts.keys())

        if training:
            self.lang_pairs = set(
                args.mono_lang_pairs
                + args.para_lang_pairs
                + [args.eval_lang_pair]
            )
        else:
            self.lang_pairs = [f"{args.source_lang}-{args.target_lang}"]

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        # fmt: off
        parser.add_argument("data",
                            help="column separated paths to data directories")

        parser.add_argument("--langs", default=None, metavar="LANGS",
                            help="comma-separated list of languages in tasks: en,de,fr")
        parser.add_argument("--source-langs", default=None, metavar="LANGS",
                            help="comma-separated list of source languages: en,fr")
        parser.add_argument("--target-langs", default=None, metavar="LANGS",
                            help="comma-separated list of target languages: en,fr")
        parser.add_argument("--valid-lang-pairs", default="", metavar="LANG-PAIRS",
                            help="comma-separated list of language pairs: en-en, zh-zh")

        parser.add_argument("--mass_steps", default="", metavar="LANG-PAIRS",
                            help="mass for monolingual data (en-en,zh-zh)")
        parser.add_argument("--mt_steps", default="", metavar="LANG-PAIRS",
                            help="supervised machine translation data (en-zh,zh-en)")
        parser.add_argument("--memt_steps", default="", metavar="LANG-PAIRS",
                            help="Masked encoder for machine translation")
        parser.add_argument("--bt_steps", default="", metavar="LANG-TRIPLETS",
                            help="backtranslation triplets (en-fr-en, fr-en-fr)")

        parser.add_argument("--left-pad-source", default="True", type=str, metavar="BOOL",
                            help="pad the source on the left (default: True)")
        parser.add_argument("--left-pad-target", default="False", type=str, metavar="BOOL",
                            help="pad the target on the left (default: False)")
        parser.add_argument("--max-source-positions", default=1024, type=int, metavar="N",
                            help="max number of tokens in the source sequence")
        parser.add_argument("--max-target-positions", default=1024, type=int, metavar="N",
                            help="max number of tokens in the target sequence")

        parser.add_argument("-s", "--source-lang", default=None, metavar="SRC",
                            help='source language (only needed for inference)')
        parser.add_argument("-t", "--target-lang", default=None, metavar="TARGET",
                            help="target language (only needed for inference)")

        parser.add_argument("--lazy-load", action="store_true",
                            help="load the dataset lazily")
        parser.add_argument("--raw-text", default=False, action="store_true",
                            help="load raw text dataset")

        parser.add_argument("--mask-s2s-prob", default=0.15, type=float,
                            help="probability of replacing a token with mask")
        parser.add_argument("--mask-s2s-mask-keep-rand", default="0.8,0.1,0.1", type=str,
                            help="Word prediction probability for decoder mask")
        # fmt: on

    @classmethod
    def setup_task(
        cls, args: Namespace, **kwargs: Any
    ) -> "UnsupervisedMASSTask":

        # split and prepare lang pairs
        args.langs = args.langs.split(",")
        args.source_langs = args.source_langs.split(",")
        args.target_langs = args.target_langs.split(",")
        args.valid_lang_pairs = [
            s for s in args.valid_lang_pairs.split(",") if len(s) > 0
        ]

        if getattr(args, "raw_text", False):
            deprecation_warning(
                "--raw-text is deprecated, please use --dataset-impl=raw"
            )
            args.dataset_impl = "raw"
        elif getattr(args, "lazy_load", False):
            deprecation_warning(
                "--lazy-load is deprecated, please use --dataset-impl=lazy"
            )
            args.dataset_impl = "lazy"

        paths = args.data.split(":")
        dictionary = cls.load_dictionary(os.path.join(paths[0], "dict.txt"))
        print(f"| dictionary: {len(dictionary)} types")

        return cls(args, dictionary)

    @classmethod
    def load_dictionary(cls, filename: str) -> MaskedDictionary:
        return MaskedDictionary.load(filename)

    def load_dataset(
        self, split: str, combine: bool = False, **kwargs: Any
    ) -> None:
        epoch = getattr(kwargs, "epoch", 0)
        paths = self.args.data.split(":")
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = load_indexed_dataset(
            split_path, self.dictionary, self.args.dataset_impl, combine
        )

        if dataset is None:
            raise FileNotFoundError(
                f"Dataset not found: {split} ({split_path})"
            )

        self.datasets[split] = self.build_masked_dataset(dataset)

    def build_masked_dataset(
        self, dataset: FairseqDataset
    ) -> MaskedLanguagePairDataset:
        block_dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample,
            self.dictionary.pad(),
            self.dictionary.eos(),
            self.args.sample_break_mode,
        )

        keep_rand = self.args.mask_s2s_mask_keep_rand.split(",")
        pred_probs = torch.FloatTensor([float(x) for x in keep_rand])

        max_source_positions, _ = self.max_positions()

        masked_dataset = MaskedLanguagePairDataset(
            source_dataset=block_dataset,
            source_sizes=block_dataset.sizes,
            target_dataset=None,
            target_sizes=None,
            source_dict=self.source_dictionary,
            target_dict=None,
            source_lang_id=0,
            target_lang_id=None,
            left_pad_source=False,
            left_pad_target=None,
            max_source_positions=max_source_positions,
            max_target_positions=None,
            mask_prob=0.15,
            block_size=64,
            shuffle=True,
            ratio=None,
            training=True,
            pred_probs=pred_probs,
        )

        return masked_dataset

    @property
    def source_dictionary(self) -> MaskedDictionary:
        return self.dictionary

    @property
    def target_dictionary(self) -> MaskedDictionary:
        return self.dictionary

    def max_positions(self) -> Tuple[int, int]:
        max_positions = 1024
        if hasattr(self.args, "max_positions"):
            max_positions = min(max_positions, self.args.max_positions)
        if hasattr(self.args, "max_source_positions"):
            max_positions = min(max_positions, self.args.max_source_positions)
        if hasattr(self.args, "max_target_positions"):
            max_positions = min(max_positions, self.args.max_target_positions)
        return (max_positions, max_positions)

    def build_model(self, args: Namespace) -> BaseFairseqModel:
        return models.build_model(args, self)

    def train_step(
        self,
        sample: Dict,
        model: BaseFairseqModel,
        criterion: FairseqCriterion,
        optimizer: FairseqOptimizer,
        ignore_grad: bool = False,
    ) -> Tuple[torch.Tensor, int, Dict[str, int]]:
        model.train()
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(
        self, sample: Dict, model: BaseFairseqModel, criterion: FairseqCriterion
    ) -> Tuple[torch.Tensor, int, Dict[str, int]]:
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    def inference_step(
        self,
        generator,  # TODO: Figure out the type
        models: List[BaseFairseqModel],  # TODO: Figure out the type
        sample: Dict,
        prefix_tokens: Optional[torch.Tensor] = None,
    ):
        pass
