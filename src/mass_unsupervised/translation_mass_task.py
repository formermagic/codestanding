from argparse import Namespace
from typing import Tuple

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from .masked_dictionary import MaskedDictionary


@register_task("translation_mass")
class TranslationMASSTask(TranslationTask):
    def __init__(
        self,
        args: Namespace,
        src_dict: MaskedDictionary,
        tgt_dict: MaskedDictionary,
    ) -> None:
        super().__init__(args, src_dict, tgt_dict)

    @classmethod
    def load_dictionary(cls, filename: str) -> MaskedDictionary:
        return MaskedDictionary.load(filename)

    def max_positions(self) -> Tuple[int, int]:
        return (self.args.max_source_positions, self.args.max_target_positions)
