import copy
import logging
import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Text, Union

import torch

from fairseq import metrics
from fairseq.criterions import FairseqCriterion
from fairseq.data import (
    DenoisingDataset,
    Dictionary,
    FairseqDataset,
    TokenBlockDataset,
)
from fairseq.models import FairseqModel, build_model
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_denoising import MultilingualDenoisingTask
from fairseq_contrib.data import MaskedDictionary
from fairseq_contrib.utils import WandBLogger

MetricType = Union[torch.Tensor, int]


logger = logging.getLogger(__name__)


@register_task("multilingual_denoising_wrapper")
class MultilingualDenoisingTaskWrapper(MultilingualDenoisingTask):
    def __init__(self, args: Namespace, dictionary: MaskedDictionary) -> None:
        super().__init__(args, copy.deepcopy(dictionary))
        self.dictionary = dictionary
        self.seed = args.seed
        self.langs = args.langs
        self.args = args

        self.mask_idx = dictionary.mask()

        self.logger = WandBLogger(
            project="codestanding", name=args.wandb_name, config=args
        )

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        MultilingualDenoisingTask.add_args(parser)
        # fmt: off
        parser.add_argument("--wandb-name", type=str, default=None,
                            help="A name for the experiment logged in WandB")
        # fmt: on

    def build_model(self, args) -> FairseqModel:
        model = build_model(args, self)
        self.logger.watch_model(model)
        return model

    def build_dataset_for_inference(
        self,
        src_tokens: List[torch.Tensor],
        src_lengths: List[int],
        **kwargs: Any,
    ) -> FairseqDataset:
        dataset = TokenBlockDataset(
            src_tokens,
            src_lengths,
            block_size=self.args.tokens_per_sample - 2,
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.args.sample_break_mode,
        )

        # dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())
        # dataset = AppendTokenDataset(dataset, self.source_dictionary.eos())

        mask_whole_words = (
            None  # get_whole_word_mask(self.args, self.dictionary)
        )

        dataset = DenoisingDataset(
            dataset,
            dataset.sizes,
            self.dictionary,
            self.mask_idx,
            mask_whole_words,
            shuffle=self.args.shuffle_instance,
            seed=self.seed,
            args=self.args,
            eos=None,
        )

        return dataset

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = args.data.split(":")
        assert len(paths) > 0

        data_path = paths[0]
        dictionary_path = os.path.join(data_path, "dict.txt")
        dictionary = cls.load_dictionary(dictionary_path)

        if args.langs is None:
            languages = sorted(
                [
                    name
                    for name in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, name))
                ]
            )
        else:
            languages = args.langs.split(",")

        if args.add_lang_token:
            for lang in languages:
                dictionary.add_symbol("[{}]".format(lang))

        logger.info("| dictionary: %d types", len(dictionary))
        if not hasattr(args, "shuffle_instance"):
            args.shuffle_instance = False
        return cls(args, dictionary)

    @classmethod
    def load_dictionary(cls, filename: str) -> MaskedDictionary:
        dictionary = Dictionary.load(filename)
        print(f"Loading dictionary... len={len(dictionary)}")
        return dictionary

    def reduce_metrics(
        self,
        logging_outputs: List[Dict[Text, Dict[Text, MetricType]]],
        criterion: FairseqCriterion,
    ) -> None:
        super().reduce_metrics(logging_outputs, criterion)

        # write metrics to WandB
        num_updates = metrics.get_meter(name="default", key="num_updates").val
        self.logger.log(key="train", step=num_updates)
        self.logger.log(key="valid", step=num_updates)
