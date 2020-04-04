from argparse import Namespace
from numbers import Number
from typing import Any, Dict, Optional, Text

import wandb

from fairseq import utils
from fairseq.logging import metrics
from fairseq.logging.meters import AverageMeter


class WandBLogger:
    def __init__(self, project: Text, config: Namespace) -> None:
        wandb.init(project=project, config=config)

    def _extract_stats(self, key: Text) -> Dict[Text, Any]:
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
        for key in stats.keys() - {"num_updates"}:
            if isinstance(stats[key], AverageMeter):
                wandb.log({key: stats[key].val}, step=step)
            elif isinstance(stats[key], Number):
                wandb.log({key: stats[key]}, step=step)
