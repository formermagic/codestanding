from argparse import ArgumentParser, Namespace
from numbers import Number
from typing import Any, Dict, Optional, Text

import wandb

from fairseq import utils
from fairseq.logging import metrics
from fairseq.logging.meters import AverageMeter
from fairseq.models import FairseqModel


class WandBLogger:
    def __init__(
        self,
        project: Text,
        exp_id: Optional[Text] = None,
        exp_name: Optional[Text] = None,
        config: Optional[Namespace] = None,
    ) -> None:
        wandb.init(project=project, id=exp_id, name=exp_name, config=config)

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        # fmt: off
        parser.add_argument("--wandb-project", type=str, default=None,
                            help="A WandB project name to use for logging")
        parser.add_argument("--wandb-name", type=str, default=None,
                            help="A name for the experiment logged in WandB")
        parser.add_argument("--wandb-id", type=str, default=None,
                            help="An id for the experiment logged in WandB")
        # fmt: on

    @staticmethod
    def _extract_stats(key: Text) -> Dict[Text, Any]:
        if key not in metrics._aggregators:
            return {}
        stats = metrics.get_smoothed_values(key)
        if "nll_loss" in stats and "ppl" not in stats:
            stats["ppl"] = utils.get_perplexity(stats["nll_loss"])
        wall = metrics.get_meter("default", "wall")
        stats["wall"] = round(wall.elapsed_time, 0)
        return stats

    def log(
        self,
        key: Text,
        stats: Optional[Dict[Text, Number]] = None,
        tag: Optional[Text] = None,
        step: Optional[Text] = None,
    ) -> None:
        if stats is None:
            stats = self._extract_stats(key)
        if step is None:
            step = stats.get("num_updates", -1)
        tag = tag or key
        for stat_key in stats.keys() - {"num_updates"}:
            log_key = f"{tag}/{stat_key}"
            if isinstance(stats[stat_key], AverageMeter):
                wandb.log({log_key: stats[stat_key].val}, step=step)
            elif isinstance(stats[stat_key], Number):
                wandb.log({log_key: stats[stat_key]}, step=step)

    def watch_model(self, model: FairseqModel) -> None:
        wandb.watch(model)
