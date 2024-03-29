import logging
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Text, Union

import torch

from fairseq import metrics
from fairseq.criterions import FairseqCriterion
from fairseq.data import Dictionary
from fairseq.models import FairseqModel, build_model
from fairseq.tasks import register_task
from fairseq.tasks.masked_lm import MaskedLMTask
from fairseq_contrib.utils import WandBLogger

MetricType = Union[torch.Tensor, int]


logger = logging.getLogger(__name__)


@register_task("masked_lm_wrapper")
class MaskedLMTaskWrapper(MaskedLMTask):
    def __init__(self, args: Namespace, dictionary: Dictionary) -> None:
        super().__init__(args, dictionary)

        self.logger = WandBLogger(
            project=args.wandb_project,
            exp_id=args.wandb_id,
            exp_name=args.wandb_name,
            config=args,
        )

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        MaskedLMTask.add_args(parser)
        WandBLogger.add_args(parser)

    @classmethod
    def load_dictionary(cls, filename: str) -> Dictionary:
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol("<nl>")
        return dictionary

    def build_model(self, args) -> FairseqModel:
        model = build_model(args, self)
        self.logger.watch_model(model)
        return model

    def reduce_metrics(
        self,
        logging_outputs: List[Dict[Text, Dict[Text, MetricType]]],
        criterion: FairseqCriterion,
    ) -> None:
        super().reduce_metrics(logging_outputs, criterion)

        # write metrics to WandB
        num_updates = metrics.get_meter(name="default", key="num_updates")
        self.logger.log(key="train_inner", tag="train", step=num_updates.val)
        self.logger.log(key="valid", tag="valid", step=num_updates.val)
