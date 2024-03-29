import logging
import os
from argparse import Namespace
from collections import OrderedDict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch

from fairseq import metrics, models, options
from fairseq.criterions import FairseqCriterion
from fairseq.data import (
    BacktranslationDataset,
    FairseqDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    LanguagePairDataset,
    RoundRobinZipDatasets,
    TokenBlockDataset,
)
from fairseq.data.data_utils import load_indexed_dataset
from fairseq.models import BaseFairseqModel
from fairseq.optim import FairseqOptimizer
from fairseq.sequence_generator import SequenceGenerator
from fairseq.tasks import FairseqTask, register_task
from fairseq.utils import deprecation_warning
from fairseq_contrib.data import (
    MaskedDictionary,
    MaskedLanguagePairDataset,
    NoisyLanguagePairDataset,
)
from fairseq_contrib.utils.wandb_logger import WandBLogger

try:
    from typing import OrderedDict as OrderedDictType
except ImportError:
    from typing import MutableMapping as OrderedDictType

MetricType = Union[torch.Tensor, int]


def infer_mono_lang_pairs(steps: List[str]) -> List[str]:
    langs = [s.split("-")[0] for s in steps if len(s) > 0]
    lang_pairs = [f"{lang}-{lang}" for lang in langs]
    return lang_pairs


def infer_para_lang_pairs(steps: List[str]) -> List[str]:
    pairs = [pair for pair in set(steps) if len(pair) > 0]
    pairs = ["-".join(pair.split("-")) for pair in pairs]
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
        self.lang2idx = args.lang2idx
        self.idx2lang = args.idx2lang
        self.sequence_generators: Dict[str, SequenceGenerator] = {}

        if training:
            self.lang_pairs = set(args.mono_lang_pairs + args.para_lang_pairs)
        else:
            self.lang_pairs = [f"{args.source_lang}-{args.target_lang}"]

        self.logger = WandBLogger(
            project=args.wandb_project,
            exp_id=args.wandb_id,
            exp_name=args.wandb_name,
            config=args,
        )

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

        parser.add_argument("--bt-max-len-a", default=1.1, type=float, metavar="N",
                            help="generate back-translated sequences of maximum length ax + b, where x is the "
                                 "source length")
        parser.add_argument("--bt-max-len-b", default=10.0, type=float, metavar="N",
                            help="generate back-translated sequences of maximum length ax + b, where x is the "
                                 "source length")
        parser.add_argument("--bt-beam-size", default=1, type=int, metavar="N",
                            help="beam size used in beam search of online back-translation")
        # fmt: on

        # add args for WandB logger setup
        WandBLogger.add_args(parser)

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
            return load_indexed_dataset(
                path,
                dictionary,
                dataset_impl=self.args.dataset_impl,
                combine=combine,
            )

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

                print(f"\n\nMASS lang: {src_id}")

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
                    cuda=True,
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

        eval_key: Optional[str] = None
        if not self.training:
            eval_key = self.args.eval_lang_pair

        print(f"| Prepared datasets: {list(datasets.keys())}, split: {split}")

        self.datasets[split] = RoundRobinZipDatasets(datasets, eval_key)

    @property
    def source_dictionary(self) -> MaskedDictionary:
        if self.training:
            return next(iter(self.dicts.values()))
        return self.dicts[self.args.source_lang]

    @property
    def target_dictionary(self) -> MaskedDictionary:
        if self.training:
            return next(iter(self.dicts.values()))
        return self.dicts[self.args.target_lang]

    def max_positions(self) -> OrderedDictType[str, Tuple[int, int]]:
        max_positions = 1024
        if hasattr(self.args, "max_positions"):
            max_positions = min(max_positions, self.args.max_positions)
        if hasattr(self.args, "max_source_positions"):
            max_positions = min(max_positions, self.args.max_source_positions)
        if hasattr(self.args, "max_target_positions"):
            max_positions = min(max_positions, self.args.max_target_positions)

        positions = (max_positions, max_positions)

        datasets = [
            zip_dataset.datasets for split, zip_dataset in self.datasets.items()
        ]

        key_positions = OrderedDict(
            [(key, positions) for dataset in datasets for key in dataset.keys()]
        )

        return key_positions

    def build_model(self, args: Namespace) -> BaseFairseqModel:
        model = models.build_model(args, self)
        self.logger.watch_model(model)

        if len(self.args.bt_steps) > 0 and self.training:
            for lang_pair in self.args.bt_steps:
                src, tgt, _ = lang_pair.split("-")
                key = f"{tgt}-{src}"

                decoder_lang_tok_idx = self.dicts[src].eos()
                sequence_generator = SequenceGenerator(
                    models=[model],
                    tgt_dict=self.dicts[src],
                    beam_size=args.bt_beam_size,
                    max_len_a=args.bt_max_len_a,
                    max_len_b=args.bt_max_len_b,
                )

                def backtranslate_fn(
                    sample: Dict,
                    model: BaseFairseqModel = model,
                    bos_token: int = decoder_lang_tok_idx,
                    sequence_generator: SequenceGenerator = sequence_generator,
                ) -> Dict:
                    return sequence_generator.generate(
                        [model], sample, bos_token=bos_token,
                    )

                self.sequence_generators[key] = sequence_generator
                self.backtranslators[lang_pair] = backtranslate_fn

        return model

    def train_step(
        self,
        sample: Dict,
        model: BaseFairseqModel,
        criterion: FairseqCriterion,
        optimizer: FairseqOptimizer,
        ignore_grad: bool = False,
        **unused: Any,
    ) -> Tuple[torch.Tensor, int, Dict[str, int]]:
        model.train()
        agg_loss = 0.0
        agg_sample_size = 0.0
        agg_logging_output = {}

        def forward_backward(
            model: BaseFairseqModel,
            samples: Dict,
            logging_output_key: str,
            lang_pair: str,
        ) -> None:
            nonlocal agg_loss, agg_sample_size, agg_logging_output
            if samples is None or len(samples) == 0:
                return

            samples["net_input"]["lang_pair"] = lang_pair

            loss, sample_size, logging_output = criterion(model, samples)
            if ignore_grad:
                loss *= 0
            optimizer.backward(loss)

            if loss.detach().item() == 0.0:
                logging.warning(
                    "Loss turned into zero which might indicate an issue while computing one. "
                    "Make sure the masked dataset has values to compute loss for. "
                    "Check the sample used to compute the loss: "
                    "%s",
                    samples,
                )

            agg_loss += loss.detach().item()
            agg_sample_size += sample_size
            agg_logging_output[logging_output_key] = logging_output

        for lang_pair in self.args.mt_steps:
            sample_key = DatasetKey.MT.paired_with(lang_pair)
            forward_backward(model, sample[sample_key], sample_key, lang_pair)

        for lang_pair in self.args.memt_steps:
            sample_key = DatasetKey.MEMT.paired_with(lang_pair)
            forward_backward(model, sample[sample_key], sample_key, lang_pair)

        for lang_pair in self.args.mass_steps:
            sample_key = DatasetKey.MASS.paired_with(lang_pair)
            forward_backward(model, sample[sample_key], sample_key, lang_pair)

        for lang_pair in self.args.bt_steps:
            # src-tgt-src => src-tgt
            bt_lang_pair = "-".join(lang_pair.split("-")[:2])
            sample_key = DatasetKey.BT.paired_with(lang_pair)
            forward_backward(
                model, sample[sample_key], sample_key, bt_lang_pair
            )

        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(
        self,
        sample: Dict,
        model: BaseFairseqModel,
        criterion: FairseqCriterion,
        **unused: Any,
    ) -> Tuple[torch.Tensor, int, Dict[str, int]]:
        model.eval()
        with torch.no_grad():
            agg_loss = 0.0
            agg_sample_size = 0.0
            agg_logging_output = {}

            for lang_pair in self.args.valid_lang_pairs:
                sample_key = lang_pair
                if (
                    sample_key not in sample
                    or sample[sample_key] is None
                    or len(sample[sample_key]) == 0
                ):
                    continue

                sample[sample_key]["net_input"]["lang_pair"] = lang_pair

                loss, sample_size, logging_output = criterion(
                    model, sample[sample_key]
                )

                agg_loss += loss.data.item()
                agg_sample_size += sample_size
                agg_logging_output[lang_pair] = logging_output

        return agg_loss, agg_sample_size, agg_logging_output

    def inference_step(
        self,
        generator: SequenceGenerator,
        models: List[BaseFairseqModel],
        sample: Dict,
        prefix_tokens: Optional[torch.Tensor] = None,
    ) -> Dict:
        tgt = sample["target"]
        print(f"[MASSModel] target={tgt}")
        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens
            )

    # TODO: This is just an idea of how it could work
    # Also note, that you should use RobinBobin dataset
    # as it is giving the correct max_positions (w.r.t. dicts in a multilingual manner)
    # otherwise you have to set max_positions to None to make it work.
    # def build_dataset_for_inference(
    #     self, src_tokens: torch.Tensor, src_lengths, **unused: Any
    # ) -> FairseqDataset:
    #     # print(
    #     #     f"src_tokens={src_tokens}, src_lengths={src_lengths}, kwargs={unused}"
    #     # )
    #     src_dataset = InteractiveDataset(src_tokens[0], src_lengths)
    #     source_lang = self.args.source_lang
    #     target_lang = self.args.target_lang
    #     source_lang_idx = self.args.lang2idx[source_lang]
    #     target_lang_idx = self.args.lang2idx[target_lang]
    #     return MaskedLanguagePairDataset(
    #         source_dataset=src_dataset,
    #         source_sizes=src_dataset.sizes,
    #         target_dataset=None,
    #         target_sizes=None,
    #         source_dict=self.source_dictionary,
    #         target_dict=None,
    #         source_lang_id=source_lang_idx,
    #         target_lang_id=None,
    #         left_pad_source=False,
    #         left_pad_target=False,
    #         ratio=0.15,
    #         pred_probs=torch.FloatTensor([0.8, 0.1, 0.1]),
    #     )
    def build_dataset_for_inference(
        self,
        src_tokens: List[torch.Tensor],
        src_lengths: List[int],
        **unused: Any,
    ) -> FairseqDataset:
        src_dataset = TokenBlockDataset(
            src_tokens,
            src_lengths,
            block_size=None,  # ignored for "eos" break mode
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode="eos",
        )

        masked_dataset = MaskedLanguagePairDataset(
            source_dataset=src_dataset,
            source_sizes=src_dataset.sizes,
            target_dataset=None,
            target_sizes=None,
            source_dict=self.source_dictionary,
            target_dict=None,
            source_lang_id=self.lang2idx[self.args.source_lang],
            target_lang_id=None,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            ratio=self.args.mask_s2s_prob,
            pred_probs=self.args.pred_probs,
        )

        lang_pair_dataset = LanguagePairDataset(
            src_dataset, src_dataset.sizes, self.source_dictionary
        )

        datasets = OrderedDict([("mass: inference", masked_dataset)])

        return RoundRobinZipDatasets(datasets, eval_key="mass: inference")

    def reduce_metrics(
        self,
        logging_outputs: List[Dict[str, Dict[str, MetricType]]],
        criterion: FairseqCriterion,
    ) -> None:
        def sum_over_dataset(
            key: str, agg_logging_outputs: Dict[str, Dict[str, MetricType]]
        ) -> float:
            return sum(
                logging_output[key]
                for logging_output in agg_logging_outputs.values()
            )

        dataset_keys: Set[str] = {
            dataset_key
            for logging_output in logging_outputs
            for dataset_key in logging_output.keys()
        }

        agg_logging_outputs: Dict[str, List[Dict[str, MetricType]]] = {
            dataset_key: [
                logging_output.get(dataset_key, {})
                for logging_output in logging_outputs
            ]
            for dataset_key in dataset_keys
        }

        sum_logging_outputs: Dict[str, Dict[str, MetricType]] = {}
        for dataset_key, _logging_outputs in agg_logging_outputs.items():
            loss, nll_loss, ntokens, sample_size = [], [], [], []
            for log in _logging_outputs:
                loss.append(log["loss"].cpu().item())
                nll_loss.append(log["nll_loss"].cpu().item())
                ntokens.append(log["ntokens"])
                sample_size.append(log["sample_size"])

            sum_logging_outputs[dataset_key] = {
                "loss": sum(loss),
                "nll_loss": sum(nll_loss),
                "ntokens": sum(ntokens),
                "sample_size": sum(sample_size),
            }

        # flatten logging outputs
        flat_logging_outputs: Dict[str, float] = {
            "loss": sum_over_dataset("loss", sum_logging_outputs),
            "nll_loss": sum_over_dataset("nll_loss", sum_logging_outputs),
            "ntokens": sum_over_dataset("ntokens", sum_logging_outputs),
            "sample_size": sum_over_dataset("sample_size", sum_logging_outputs),
        }

        with metrics.aggregate():
            ntokens = flat_logging_outputs["ntokens"]
            metrics.log_scalar("wpb", ntokens, priority=180, round=1)
            metrics.log_speed("wps", ntokens, priority=90, round=1)

        # log metrics
        criterion.__class__.reduce_metrics([flat_logging_outputs])

        # write metrics to WandB
        num_updates = metrics.get_meter(name="default", key="num_updates")
        self.logger.log(key="train_inner", tag="train", step=num_updates.val)
        self.logger.log(key="valid", tag="valid", step=num_updates.val)
