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

        # check if source/target langs are listed in langs
        for lang in args.source_langs:
            assert lang in args.langs
        for lang in args.target_langs:
            assert lang in args.langs

        # check if valid langs are in source-target pairs
        for lang_pair in args.valid_lang_pairs:
            src, tgt = lang_pair.split("-")
            assert src in args.source_langs and tgt in args.target_langs

        # split task steps language pairs
        args.mass_steps = [s for s in args.mass_steps.split(",") if len(s) > 0]
        args.mt_steps = [s for s in args.mt_steps.split(",") if len(s) > 0]
        args.memt_steps = [s for s in args.memt_steps.split(",") if len(s) > 0]
        args.bt_steps = [s for s in args.bt_steps.split(",") if len(s) > 0]

        # infer monolingual lang pairs
        mono_lang_pairs = infer_mono_lang_pairs(args.mass_steps + args.bt_steps)
        setattr(args, "mono_lang_pairs", mono_lang_pairs)

        # check if mono lang pairs are in source-target pairs
        for lang_pair in args.mono_lang_pairs:
            src, tgt = lang_pair.split("-")
            assert src in args.source_langs and tgt in args.target_langs

        # check if backtranslation langs are in source-target-source relationship
        for lang_tuple in args.bt_steps:
            src, tgt, src_out = lang_tuple.split("-")
            assert src in args.source_langs
            assert tgt in args.target_langs
            assert src == src_out and src != tgt

        # infer paralingual lang pairs
        para_lang_pairs = infer_para_lang_pairs(args.mt_steps + args.memt_steps)
        setattr(args, "para_lang_pairs", para_lang_pairs)

        # check if para lang pairs are in source-target pairs
        for lang_pair in args.mt_steps + args.memt_steps:
            src, tgt = lang_pair.split("-")
            assert src in args.source_langs and tgt in args.target_langs

        # prepare left pad options
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # check if inference langs are listed in source/target langs
        if args.source_lang is not None:
            assert args.source_lang in args.source_langs
        if args.target_lang is not None:
            assert args.target_lang in args.target_langs

        # prepare langs lookup dicts
        lang2idx = {}
        idx2lang = {}
        for idx, lang in enumerate(args.langs):
            lang2idx[lang] = idx
            idx2lang[idx] = lang
        setattr(args, "lang2idx", lang2idx)
        setattr(args, "idx2lang", idx2lang)
        setattr(args, "n_langs", len(lang2idx))

        # prepare eval lang pairs
        if args.source_lang is not None and args.target_lang is not None:
            eval_lang_pair = f"{args.source_lang}-{args.target_lang}"
            setattr(args, "eval_lang_pair", eval_lang_pair)
            training = False
        else:
            if len(args.para_lang_pairs) > 0:
                lang_pairs = list(set(args.mt_steps + args.memt_steps))
                setattr(args, "eval_lang_pair", lang_pairs[0])
            else:
                setattr(args, "eval_lang_pair", args.mono_lang_pairs[0])
            training = True

        eval_para = len(args.para_lang_pairs) > 0
        setattr(args, "eval_para", eval_para)

        # prepare dataset_impl option
        if getattr(args, "raw_text", False):
            deprecation_warning(
                "--raw-text is deprecated, please use --dataset-impl=raw"
            )
            args.dataset_impl = "raw"

        if getattr(args, "lazy_load", False):
            deprecation_warning(
                "--lazy-load is deprecated, please use --dataset-impl=lazy"
            )
            args.dataset_impl = "lazy"

        # prepare mass task masking params
        keep_rand = args.mask_s2s_mask_keep_rand.split(",")
        pred_probs = torch.FloatTensor([float(x) for x in keep_rand])
        setattr(args, "pred_probs", pred_probs)

        # load dicts for all listed langs
        dicts = OrderedDict()
        for lang in args.langs:
            filename = os.path.join(args.data, f"dict.{lang}.txt")
            dicts[lang] = cls.load_dictionary(filename)

            if len(dicts) > 0:
                assert dicts[lang].pad() == dicts[args.langs[0]].pad()
                assert dicts[lang].eos() == dicts[args.langs[0]].eos()
                assert dicts[lang].unk() == dicts[args.langs[0]].unk()
                assert dicts[lang].mask() == dicts[args.langs[0]].mask()

            print("| [{}] dictionary: {} types".format(lang, len(dicts[lang])))

        return cls(args, dicts, training)

    @classmethod
    def load_dictionary(cls, filename: str) -> MaskedDictionary:
        return MaskedDictionary.load(filename)

    def load_dataset(
        self, split: str, combine: bool = False, **kwargs: Any
    ) -> None:
        paths = self.args.data.split(os.pathsep)
        epoch = kwargs.get("epoch", None)
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        def split_exists(split: str, lang: str) -> bool:
            filename = os.path.join(data_path, f"{split}.{lang}")
            raw_text = self.args.dataset_impl == "raw"
            if raw_text and IndexedRawTextDataset.exists(filename):
                return True
            if not raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def split_para_exists(split: str, key: str, lang: str) -> bool:
            filename = os.path.join(data_path, f"{split}.{key}.{lang}")
            raw_text = self.args.dataset_impl == "raw"
            if raw_text and IndexedRawTextDataset.exists(filename):
                return True
            if not raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(
            path: str, dictionary: MaskedDictionary
        ) -> Optional[IndexedDataset]:
            if self.args.dataset_impl == "raw":
                return IndexedRawTextDataset(path, dictionary)
            if IndexedDataset.exists(path):
                if self.args.dataset_impl == "lazy":
                    return IndexedDataset(path, fix_lua_indexing=True)
                return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        src_mono_datasets = {}
        for lang_pair in self.args.mono_lang_pairs:
            lang = lang_pair.split("-")[0]
            if split_exists(split, lang):
                prefix = os.path.join(data_path, f"{split}.{lang}")
            else:
                raise FileNotFoundError(
                    f"Not Found available {split} dataset for ({lang}) lang"
                )

            src_mono_datasets[lang_pair] = indexed_dataset(
                prefix, self.dicts[lang]
            )

            n_samples = len(src_mono_datasets[lang_pair])
            print(f"| monolingual {split}-{lang}: {n_samples} examples")

        src_para_datasets = {}
        for lang_pair in self.args.para_lang_pairs:
            src, tgt = lang_pair.split("-")
            key = "-".join(sorted([src, tgt]))
            if not split_para_exists(split, key, src):
                raise FileNotFoundError(
                    "Not Found available {}-{} para dataset for ({}) lang".format(
                        split, key, src
                    )
                )
            if not split_para_exists(split, key, tgt):
                raise FileNotFoundError(
                    "Not Found available {}-{} para dataset for ({}) lang".format(
                        split, key, tgt
                    )
                )

            prefix = os.path.join(data_path, f"{split}.{key}")
            if f"{key}.{src}" not in src_para_datasets:
                src_para_datasets[key + "." + src] = indexed_dataset(
                    prefix + "." + src, self.dicts[src]
                )
            if f"{key}.{tgt}" not in src_para_datasets:
                src_para_datasets[key + "." + tgt] = indexed_dataset(
                    prefix + "." + tgt, self.dicts[tgt]
                )

            src_len = len(src_para_datasets[key + "." + src])
            trt_len = len(src_para_datasets[key + "." + tgt])
            print(f"| bilingual {split} {src}-{tgt}.{src}: {src_len} examples")
            print(f"| bilingual {split} {src}-{tgt}.{tgt}: {trt_len} examples")

        mt_para_dataset = {}
        for lang_pair in self.args.mt_steps:
            src, tgt = lang_pair.split("-")
            key = "-".join(sorted([src, tgt]))
            src_key = key + "." + src
            tgt_key = key + "." + tgt

            src_dataset = src_para_datasets[src_key]
            tgt_dataset = src_para_datasets[tgt_key]
            mt_para_dataset[lang_pair] = LanguagePairDataset(
                src_dataset,
                src_dataset.sizes,
                self.dicts[src],
                tgt_dataset,
                tgt_dataset.sizes,
                self.dicts[tgt],
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )

        eval_para_dataset = {}
        if split != "train":
            for lang_pair in self.args.valid_lang_pairs:
                src, tgt = lang_pair.split("-")
                if src == tgt:
                    src_key = src + "-" + tgt
                    tgt_key = src + "-" + tgt
                    src_dataset = src_mono_datasets[src_key]
                    tgt_dataset = src_mono_datasets[tgt_key]
                else:
                    key = "-".join(sorted([src, tgt]))
                    src_key = key + "." + src
                    tgt_key = key + "." + tgt
                    src_dataset = src_para_datasets[src_key]
                    tgt_dataset = src_para_datasets[tgt_key]

                eval_para_dataset[lang_pair] = LanguagePairDataset(
                    src_dataset,
                    src_dataset.sizes,
                    self.dicts[src],
                    tgt_dataset,
                    tgt_dataset.sizes,
                    self.dicts[tgt],
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=self.args.max_source_positions,
                    max_target_positions=self.args.max_target_positions,
                )

        memt_para_dataset = {}
        if split == "train":
            for lang_pair in self.args.memt_steps:
                src, tgt = lang_pair.split("-")
                key = "-".join(sorted([src, tgt]))
                src_key = key + "." + src
                tgt_key = key + "." + tgt

                src_id, tgt_id = (
                    self.args.lang2idx[src],
                    self.args.lang2idx[tgt],
                )
                src_dataset = src_para_datasets[src_key]
                tgt_dataset = src_para_datasets[tgt_key]

                memt_para_dataset[lang_pair] = NoisyLanguagePairDataset(
                    src_dataset,
                    src_dataset.sizes,
                    tgt_dataset,
                    tgt_dataset.sizes,
                    self.dicts[src],
                    self.dicts[tgt],
                    src_id,
                    tgt_id,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=self.args.max_source_positions,
                    max_target_positions=self.args.max_target_positions,
                    ratio=self.args.mask_s2s_prob,
                    pred_probs=self.args.pred_probs,
                )

        mass_mono_datasets = {}
        if split == "train":
            for lang_pair in self.args.mass_steps:
                src_dataset = src_mono_datasets[lang_pair]
                lang = lang_pair.split("-")[0]
                src_id = self.args.lang2idx[lang]

                mass_mono_datasets[lang_pair] = MaskedLanguagePairDataset(
                    source_dataset=src_dataset,
                    source_sizes=src_dataset.sizes,
                    target_dataset=None,
                    target_sizes=None,
                    source_dict=self.dicts[lang],
                    target_dict=None,
                    source_lang_id=src_id,
                    target_lang_id=None,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    ratio=self.args.mask_s2s_prob,
                    pred_probs=self.args.pred_probs,
                )

        backtranslate_datasets = {}
        if len(self.args.bt_steps) > 0 and split == "train":
            for lang_pair in self.args.bt_steps:
                src, tgt, _ = lang_pair.split("-")
                if not split_exists(split, tgt):
                    raise FileNotFoundError(
                        "Dataset not found: backtranslation {} ({})".format(
                            split, tgt
                        )
                    )

                prefix = os.path.join(data_path, f"{split}.{tgt}")
                dataset = indexed_dataset(prefix, self.dicts[tgt])
                lang_pair_dataset_tgt = LanguagePairDataset(
                    dataset,
                    dataset.sizes,
                    self.dicts[tgt],
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                )
                lang_pair_dataset = LanguagePairDataset(
                    dataset,
                    dataset.sizes,
                    src_dict=self.dicts[src],
                    tgt=dataset,
                    tgt_sizes=dataset.sizes,
                    tgt_dict=self.dicts[tgt],
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                )

                backtranslate_datasets[lang_pair] = BacktranslationDataset(
                    tgt_dataset=lang_pair_dataset_tgt,
                    src_dict=self.dicts[src],
                    tgt_dict=self.dicts[tgt],
                    backtranslation_fn=self.backtranslators[lang_pair],
                    output_collater=lang_pair_dataset.collater,
                )

                print(
                    f"| backtranslate-{tgt}: {split} "
                    + f"{data_path} "
                    + f"{len(backtranslate_datasets[lang_pair])} examples"
                )

        # combine all datasets together
        datasets = OrderedDict(
            DatasetKey.MT.keyed_dataset(mt_para_dataset)
            + DatasetKey.MEMT.keyed_dataset(memt_para_dataset)
            + DatasetKey.MASS.keyed_dataset(mass_mono_datasets)
            + DatasetKey.BT.keyed_dataset(backtranslate_datasets)
            + DatasetKey.EVAL.keyed_dataset(eval_para_dataset)
        )


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
