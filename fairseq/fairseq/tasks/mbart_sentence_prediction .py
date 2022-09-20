# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import torch

import numpy as np
from fairseq import utils
from fairseq.data import (
    BaseWrapperDataset,
    FairseqDataset,
    ConcatSentencesDataset,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    RollDataset,
    SortDataset,
    StripTokenDataset,
    data_utils,
    AppendTokenDataset,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import LegacyFairseqTask, register_task


logger = logging.getLogger(__name__)

class RmFirstTokenDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self._sizes = np.array(dataset.sizes) - 1

    def __getitem__(self, idx): 
        item = self.dataset[idx] 
        # tensor.new() create new tensor with same type and device
        item = item[1:]
        return item

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        n = self.dataset.num_tokens(index)
        n -= 1
        return n

    def size(self, index):
        n = self.dataset.size(index)
        n -= 1
        return n

class RmLastTokenDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self._sizes = np.array(dataset.sizes) - 1

    def __getitem__(self, idx): 
        item = self.dataset[idx] 
        # tensor.new() create new tensor with same type and device
        item = item[:-1]
        return item

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        n = self.dataset.num_tokens(index)
        n -= 1
        return n

    def size(self, index):
        n = self.dataset.size(index)
        n -= 1
        return n



class OneTokenDataset(BaseWrapperDataset):
    def __init__(self, dataset, mask_id):
        super().__init__(dataset)
        self._sizes = np.ones_like(dataset.sizes)
        self.mask_id = mask_id
        
    def __getitem__(self, idx):
        return torch.tensor([self.mask_id])

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        return 1

    def size(self, index):
        return 1

@register_task("mbart_sentence_prediction")
class mBARTSentencePredictionTask(LegacyFairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        parser.add_argument(
            "--num-classes",
            type=int,
            default=-1,
            help="number of classes or regression targets",
        )
        parser.add_argument(
            "--init-token",
            type=int,
            default=None,
            help="add token at the beginning of each batch item",
        )
        parser.add_argument(
            "--separator-token",
            type=int,
            default=None,
            help="add separator token between inputs",
        )
        parser.add_argument("--regression-target", action="store_true", default=False)
        parser.add_argument("--no-shuffle", action="store_true", default=False)
        parser.add_argument(
            "--shorten-method",
            default="none",
            choices=["none", "truncate", "random_crop"],
            help="if not none, shorten sequences that exceed --tokens-per-sample",
        )
        parser.add_argument(
            "--shorten-data-split-list",
            default="",
            help="comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)',
        )
        parser.add_argument("--prompt_id", 
            default="",
            help="prompt_id for split raw",
        )
        parser.add_argument(
            "--add-prev-output-tokens",
            action="store_true",
            default=False,
            help="add prev_output_tokens to sample, used for encoder-decoder arch",
        )
        parser.add_argument('--langs', required=True, metavar='LANG',
                            help='comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning.')
        parser.add_argument('--prepend-bos', action='store_true',
                            help='prepend bos token to each sentence, which matches '
                                 'mBART pretraining')
        parser.add_argument('--src-language', required=True,
                            help='the langauge id ')
        parser.add_argument('--tgt-language', required=True,
                            help='the langauge id ')
        parser.add_argument('--baseline', action='store_true',
                            help='do prompt v2 with input1 and input2 data split; or prepare baseline data format: src:-sample, tgt:-entity1 entity2')

    def __init__(self, args, data_dictionary, label_dictionary):
        super().__init__(args)
        self.dictionary = data_dictionary
        self._label_dictionary = label_dictionary
        if not hasattr(args, "max_positions"):
            self._max_positions = (
                args.max_source_positions,
                args.max_target_positions,
            )
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions

        self.langs = args.langs.split(",")
        for l in self.langs:
            self.dictionary.add_symbol("[{}]".format(l))
        self.dictionary.add_symbol("<mask>")

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        # dictionary.add_symbol("<mask>")
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, "Must set --num-classes"

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "input0", "dict.txt"),
            source=True,
        )
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        label_dict = None
        if not args.regression_target:
            # load label dictionary
            label_dict = cls.load_dictionary(
                args,
                os.path.join(args.data, "label", "dict.txt"),
                source=False,
            )
            logger.info("[label] dictionary: {} types".format(len(label_dict)))
        else:
            label_dict = data_dict
        return cls(args, data_dict, label_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_path(key, split):
            return os.path.join(self.args.data, key, split)

        def make_dataset(key, dictionary):
            split_path = get_path(key, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            return dataset

        input0 = make_dataset("input0", self.source_dictionary)
        assert input0 is not None, "could not find dataset: {}".format(
            get_path("input0", split)
        )
        input1 = make_dataset("input1", self.source_dictionary)
        input2 = make_dataset("input2", self.source_dictionary)

        if self.args.init_token is not None:
            input0 = PrependTokenDataset(input0, self.args.init_token)

        # if input1 is None or self.args.baseline:
        if input1 is None: # baseline
            src_tokens = input0
        elif input2 is None: # 最初的逻辑
            if self.args.separator_token is not None:
                input1 = PrependTokenDataset(input1, self.args.separator_token)
                src_tokens = ConcatSentencesDataset(input0, input1)
        else: # prompt v2
            input1 = PrependTokenDataset(input1, self.args.separator_token)
            input2 = PrependTokenDataset(input2, self.args.separator_token)
            src_tokens = input0

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))
        # self.args.max_positions -> self.args.max_source_positions
        src_tokens = maybe_shorten_dataset(
            src_tokens,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.max_source_positions,
            self.args.seed,
        )
        
        if not self.args.baseline:
            # zct: prepend language id
            tgt_tokens = src_tokens
            end_token_1 = (
                    self.source_dictionary.index("[{}]".format(self.args.src_language))
                )
            src_tokens = AppendTokenDataset(src_tokens, end_token_1)

            end_token_2 = (
                    self.source_dictionary.index("[{}]".format(self.args.tgt_language))
                )
            tgt_tokens = AppendTokenDataset(tgt_tokens, end_token_2)
        elif input2 is None:
            tgt_tokens = maybe_shorten_dataset(
                input1,
                split,
                self.args.shorten_data_split_list,
                self.args.shorten_method,
                self.args.max_source_positions,
                self.args.seed,
            )
            end_token_1 = (
                    self.source_dictionary.index("[{}]".format(self.args.src_language))
                )
            src_tokens = AppendTokenDataset(src_tokens, end_token_1)

            end_token_2 = (
                    self.source_dictionary.index("[{}]".format(self.args.tgt_language))
                )
            tgt_tokens = AppendTokenDataset(tgt_tokens, end_token_2)
        else:
            # TBD
            end_token_1 = (
                    self.source_dictionary.index("[{}]".format(self.args.src_language))
                )
            end_token_2 = (
                self.source_dictionary.index("[{}]".format(self.args.tgt_language))
            )
            eos_token = (
                    self.source_dictionary.index("{}".format("</s>"))
                )
            mask_token = (
                    self.source_dictionary.index("{}".format("<mask>"))
                )
            input1 = maybe_shorten_dataset(input1, split, self.args.shorten_data_split_list, self.args.shorten_method,
                self.args.max_source_positions, self.args.seed,) # default append the eos token
            input2 = maybe_shorten_dataset(input2, split, self.args.shorten_data_split_list, self.args.shorten_method,
                self.args.max_source_positions, self.args.seed,)
            input1 = RmLastTokenDataset(input1) # remove the last eos token
            input2 = RmLastTokenDataset(input2) # remove the last eos token
            src_tokens = RmLastTokenDataset(src_tokens) # remove the last eos token
            # src_tokens has end token eos.
            
            # for prompt_v2_5-8
            p_link_1 = self.dictionary.encode_line('▁The ▁sentence ▁of')[:-1]
            p_link_2 = self.dictionary.encode_line('▁includes')[:-1]
            p_link_3 = self.dictionary.encode_line('▁The ▁sentence : "')[:-1]
            p_link_4 = self.dictionary.encode_line('▁"')[:-1]
            p_link_5 = self.dictionary.encode_line('▁What ▁is ▁the ▁type ▁of ▁relationship ▁between')[:-1]
            p_link_6 = self.dictionary.encode_line('▁and')
            # src_tokens: <bos> sent

            if self.args.prompt_id == 'prompt_v2_1':
                print(">> Run experiment with prompt_v2_1 ==>  src:[ <s> sent <mask> entity1 <mask> entity2 </s> <LID> ]; tgt:[ <LID> <s> sent </s> ] ")
                
                tgt_tokens = src_tokens
                
                src_tokens = AppendTokenDataset(src_tokens, mask_token)
                src_tokens = ConcatSentencesDataset(src_tokens, input1)
                src_tokens = AppendTokenDataset(src_tokens, mask_token)
                src_tokens = ConcatSentencesDataset(src_tokens, input2)
                src_tokens = AppendTokenDataset(src_tokens, eos_token)
                src_tokens = AppendTokenDataset(src_tokens, end_token_1)
                
                
                tgt_tokens = AppendTokenDataset(tgt_tokens, eos_token)
                tgt_tokens = AppendTokenDataset(tgt_tokens, end_token_2)
                tgt_tokens = PrependTokenDataset(tgt_tokens, self.args.init_token)
            elif self.args.prompt_id == 'prompt_v2_2':  
                print(">> Run experiment with prompt_v2_2 ==>  src:[ <s> sent <mask> entity1 <mask> entity2 </s> <LID> ]; tgt:[ <LID> <s> entity1 entity2 </s> ] ")
    
                
                src_tokens = AppendTokenDataset(src_tokens, mask_token)
                src_tokens = ConcatSentencesDataset(src_tokens, input1)
                src_tokens = AppendTokenDataset(src_tokens, mask_token)
                src_tokens = ConcatSentencesDataset(src_tokens, input2)
                src_tokens = AppendTokenDataset(src_tokens, eos_token)
                src_tokens = AppendTokenDataset(src_tokens, end_token_1)
                
                tgt_tokens = ConcatSentencesDataset(input1, input2) 
                tgt_tokens = AppendTokenDataset(tgt_tokens, eos_token)
                tgt_tokens = AppendTokenDataset(tgt_tokens, end_token_2)
                tgt_tokens = PrependTokenDataset(tgt_tokens, self.args.init_token)
            elif self.args.prompt_id == 'prompt_v2_3':
                print(">> Run experiment with prompt_v2_3 ==>  src:[ <s> sent <mask> entity1 <mask> entity2 </s> <LID> ]; tgt:[ <LID> <s> entity1 <mask> entity2 </s> ] ")

                src_tokens = AppendTokenDataset(src_tokens, mask_token)
                src_tokens = ConcatSentencesDataset(src_tokens, input1)
                src_tokens = AppendTokenDataset(src_tokens, mask_token)
                src_tokens = ConcatSentencesDataset(src_tokens, input2)
                src_tokens = AppendTokenDataset(src_tokens, eos_token)
                src_tokens = AppendTokenDataset(src_tokens, end_token_1)

                tgt_tokens = input1
                tgt_tokens = AppendTokenDataset(tgt_tokens, mask_token)
                tgt_tokens = ConcatSentencesDataset(tgt_tokens, input2)
                tgt_tokens = AppendTokenDataset(tgt_tokens, eos_token)
                tgt_tokens = AppendTokenDataset(tgt_tokens, end_token_2)
                tgt_tokens = PrependTokenDataset(tgt_tokens, self.args.init_token)
            elif self.args.prompt_id == 'prompt_v2_4':
                print(">> Run experiment with prompt_v2_4 ==>  src:[ <s> sent <mask> entity1 <mask> entity2 </s> <LID> ]; tgt:[ <LID> <s> sent <mask> entity1 <mask> entity2 </s> ] ")

                src_tokens = AppendTokenDataset(src_tokens, mask_token)
                src_tokens = ConcatSentencesDataset(src_tokens, input1)
                src_tokens = AppendTokenDataset(src_tokens, mask_token)
                src_tokens = ConcatSentencesDataset(src_tokens, input2)
                src_tokens = AppendTokenDataset(src_tokens, eos_token)
                tgt_tokens = src_tokens
                src_tokens = AppendTokenDataset(src_tokens, end_token_1)
                
                tgt_tokens = AppendTokenDataset(tgt_tokens, end_token_2)
            elif self.args.prompt_id == 'prompt_v2_5':
                print(">> Run experiment with prompt_v2_5 ==>  src:[ <s> sent </s> <LID> ]; tgt:[ <LID> <s> What is the type of relationship between entity1 and entity2 </s> ] ")
                tgt_tokens = src_tokens
                tgt_tokens = OneTokenDataset(tgt_tokens, p_link_5)
                for token in p_link_5[1:]:
                    tgt_tokens = AppendTokenDataset(tgt_tokens, token)
                src_tokens = RmFirstTokenDataset(src_tokens)
                src_tokens = AppendTokenDataset(src_tokens, eos_token)
                src_tokens = AppendTokenDataset(src_tokens, end_token_1)
                src_tokens = PrependTokenDataset(src_tokens, self.args.init_token)

                tgt_tokens = ConcatSentencesDataset(tgt_tokens, input1)
                tgt_tokens = AppendTokenDataset(tgt_tokens, p_link_6)
                tgt_tokens = ConcatSentencesDataset(tgt_tokens, input2)
                tgt_tokens = AppendTokenDataset(tgt_tokens, end_token_2)
                tgt_tokens = PrependTokenDataset(tgt_tokens, self.args.init_token)
                # tgt_tokens = PrependTokenDataset(tgt_tokens, self.args.init_token)
            elif self.args.prompt_id == 'prompt_v2_6':
                print(">> Run experiment with prompt_v2_6 ==>  src:[ <s> The sentence of sent include entity1 and entity2 </s> <LID> ]; tgt:[ <LID> <s> What is the type of relationship between entity1 and entity2 </s>] ")
            
                tgt_tokens = src_tokens
                tgt_tokens = OneTokenDataset(tgt_tokens, p_link_1[0]) 
                for token in p_link_1[1:]:
                    tgt_tokens = AppendTokenDataset(tgt_tokens, token)
                src_tokens = RmFirstTokenDataset(src_tokens)
                src_tokens = ConcatSentencesDataset(tgt_tokens, src_tokens)
                # src_tokens = AppendTokenDataset(src_tokens, mask_token)
                src_tokens = AppendTokenDataset(src_tokens, p_link_2)
                src_tokens = ConcatSentencesDataset(src_tokens, input1)
                src_tokens = AppendTokenDataset(src_tokens, p_link_6)
                src_tokens = ConcatSentencesDataset(src_tokens, input2)
                src_tokens = AppendTokenDataset(src_tokens, eos_token)
                src_tokens = AppendTokenDataset(src_tokens, end_token_1)
                src_tokens = PrependTokenDataset(src_tokens, self.args.init_token)
                tgt_tokens = OneTokenDataset(tgt_tokens, p_link_5[0])

                for token in p_link_5[1:]:
                    tgt_tokens = AppendTokenDataset(tgt_tokens, token)                
                tgt_tokens = ConcatSentencesDataset(tgt_tokens, input1)
                tgt_tokens = AppendTokenDataset(tgt_tokens, p_link_6)
                tgt_tokens = ConcatSentencesDataset(tgt_tokens, input2)
                tgt_tokens = AppendTokenDataset(tgt_tokens, eos_token)
                tgt_tokens = AppendTokenDataset(tgt_tokens, end_token_2)
                tgt_tokens = PrependTokenDataset(tgt_tokens, self.args.init_token)
            elif self.args.prompt_id == 'prompt_v2_7':
                print(">> Run experiment with prompt_v2_7 ==>  src:[ <s> The sentence of sent includes entity1 <mask> entity2 </s> <LID> ]; tgt:[ <LID> <s> The sentence of sent includes entity1 <mask> entity2 ] </s> ")
                
                tgt_tokens = src_tokens
                tgt_tokens = OneTokenDataset(tgt_tokens, p_link_1[0]) 
                for token in p_link_1[1:]:
                    tgt_tokens = AppendTokenDataset(tgt_tokens, token)
                src_tokens = RmFirstTokenDataset(src_tokens)
                src_tokens = ConcatSentencesDataset(tgt_tokens, src_tokens)
                src_tokens = AppendTokenDataset(src_tokens, p_link_2)
                src_tokens = ConcatSentencesDataset(src_tokens, input1)
                src_tokens = AppendTokenDataset(src_tokens, mask_token)
                src_tokens = ConcatSentencesDataset(src_tokens, input2)
                tgt_tokens = src_tokens
                src_tokens = AppendTokenDataset(src_tokens, eos_token)
                src_tokens = AppendTokenDataset(src_tokens, end_token_1)
                src_tokens = PrependTokenDataset(src_tokens, self.args.init_token)
                
                tgt_tokens = AppendTokenDataset(tgt_tokens, eos_token)
                tgt_tokens = AppendTokenDataset(tgt_tokens, end_token_2)
                tgt_tokens = PrependTokenDataset(tgt_tokens, self.args.init_token)
            elif self.args.prompt_id == 'prompt_v2_8':
                print(">> Run experiment with prompt_v2_8 ==>  src:[ <s> The sentence : 'sent' includes entity1 <mask> entity2 </s> <LID> ]; tgt:[ <LID> <s> sent </s>] ")
                
                tgt_tokens = src_tokens
                tgt_tokens = OneTokenDataset(tgt_tokens, p_link_3[0]) 
                for token in p_link_3[1:]:
                    tgt_tokens = AppendTokenDataset(tgt_tokens, token)
                src_tokens = RmFirstTokenDataset(src_tokens)
                tmp_tokens = ConcatSentencesDataset(tgt_tokens, src_tokens)
                tgt_tokens = src_tokens
                src_tokens = AppendTokenDataset(tmp_tokens, p_link_4)
                # src_tokens = AppendTokenDataset(src_tokens, mask_token)
                src_tokens = AppendTokenDataset(src_tokens, p_link_2)
                src_tokens = ConcatSentencesDataset(src_tokens, input1)
                src_tokens = AppendTokenDataset(src_tokens, mask_token)
                src_tokens = ConcatSentencesDataset(src_tokens, input2)
                src_tokens = AppendTokenDataset(src_tokens, eos_token)
                src_tokens = AppendTokenDataset(src_tokens, end_token_1)
                src_tokens = PrependTokenDataset(src_tokens, self.args.init_token)
                
                tgt_tokens = AppendTokenDataset(tgt_tokens, eos_token)
                tgt_tokens = AppendTokenDataset(tgt_tokens, end_token_2)
                tgt_tokens = PrependTokenDataset(tgt_tokens, self.args.init_token)
            elif self.args.prompt_id == 'prompt_v2_9':
                print(">> Run experiment with prompt_v2_9 ==>  src:[ <s> The sentence : 'sent' includes entity1 <mask> entity2 </s> <LID>]; tgt:[ <LID> <S> entity1 <mask> entity2 </s> ]")
                tgt_tokens = src_tokens
                tgt_tokens = OneTokenDataset(tgt_tokens, p_link_3[0]) 
                for token in p_link_3[1:]:
                    tgt_tokens = AppendTokenDataset(tgt_tokens, token)
                src_tokens = RmFirstTokenDataset(src_tokens)
                src_tokens = ConcatSentencesDataset(tgt_tokens, src_tokens)
                src_tokens = AppendTokenDataset(src_tokens, p_link_4)
                # src_tokens = AppendTokenDataset(src_tokens, mask_token)
                src_tokens = AppendTokenDataset(src_tokens, p_link_2)
                src_tokens = ConcatSentencesDataset(src_tokens, input1)
                src_tokens = AppendTokenDataset(src_tokens, mask_token)
                src_tokens = ConcatSentencesDataset(src_tokens, input2)
                src_tokens = AppendTokenDataset(src_tokens, eos_token)
                src_tokens = AppendTokenDataset(src_tokens, end_token_1)
                src_tokens = PrependTokenDataset(src_tokens, self.args.init_token)
                
                tgt_tokens = input1
                tgt_tokens = AppendTokenDataset(tgt_tokens, mask_token)
                tgt_tokens = ConcatSentencesDataset(tgt_tokens, input2)
                tgt_tokens = AppendTokenDataset(tgt_tokens, eos_token)
                tgt_tokens = AppendTokenDataset(tgt_tokens, end_token_2)
                tgt_tokens = PrependTokenDataset(tgt_tokens, self.args.init_token)
                
        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                "src_lengths": NumelDataset(src_tokens, reduce=False),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
        }

        if self.args.add_prev_output_tokens:
            prev_tokens_dataset = RightPadDataset(
                RollDataset(tgt_tokens, 1),
                pad_idx=self.dictionary.pad(),
            )
            dataset["net_input"].update(
                prev_output_tokens=prev_tokens_dataset,
            )

        if not self.args.regression_target:
            label_dataset = make_dataset("label", self.label_dictionary)
            if label_dataset is not None:
                dataset.update(
                    target=OffsetTokensDataset(
                        StripTokenDataset(
                            label_dataset,
                            id_to_strip=self.label_dictionary.eos(),
                        ),
                        offset=-self.label_dictionary.nspecial,
                    )
                )
        else:
            label_path = "{0}.label".format(get_path("label", split))
            if os.path.exists(label_path):

                def parse_regression_target(i, line):
                    values = line.split()
                    assert (
                        len(values) == self.args.num_classes
                    ), f'expected num_classes={self.args.num_classes} regression target values on line {i}, found: "{line}"'
                    return [float(x) for x in values]

                with open(label_path) as h:
                    dataset.update(
                        target=RawLabelDataset(
                            [
                                parse_regression_target(i, line.strip())
                                for i, line in enumerate(h.readlines())
                            ]
                        )
                    )

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)

        model.register_classification_head(
            getattr(args, "classification_head_name", "sentence_classification_head"),
            num_classes=self.args.num_classes,
        )

        return model

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary
