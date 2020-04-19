import logging
from argparse import Namespace
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

        self.logger = WandBLogger(project="codestanding", config=args)

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
        num_updates = metrics.get_meter(name="default", key="num_updates").val
        self.logger.log(key="train", step=num_updates)
        self.logger.log(key="valid", step=num_updates)
