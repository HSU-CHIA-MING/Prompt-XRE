# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import encoders
from fairseq.hub_utils import GeneratorHubInterface
from omegaconf import open_dict


logger = logging.getLogger(__name__)

# For mbart sentence prediction
class BARTHubInterface(GeneratorHubInterface):
    """A simple PyTorch Hub interface to BART.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/bart
    """

    def __init__(self, cfg, task, model, src_lid , tgt_lid):
        super().__init__(cfg, task, [model])
        self.model = self.models[0]
        self.src_lid = src_lid
        self.tgt_lid = tgt_lid

    def encode(
        self, sentence: str, *addl_sentences, no_separator=True
    ) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`).

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> bart.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> bart.encode(' world').tolist()
            [0, 232, 2]
            >>> bart.encode('world').tolist()
            [0, 8331, 2]
        """
        # tokens = self.bpe.encode(sentence)
        tokens = sentence
        if len(tokens.split(" ")) > min(self.max_positions) - 2:
            tokens = " ".join(tokens.split(" ")[: min(self.max_positions) - 2])
        # bpe_sentence = "<s> " + tokens + " </s>"
        bpe_sentence = tokens + " </s>"
        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator else ""
            bpe_sentence += " " + self.bpe.encode(s) + " </s>"
        # tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
        tokens = self.src_dict.encode_line(bpe_sentence, append_eos=False, add_if_not_exist=False)
        return tokens.long()

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = tokens == self.task.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [
            self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences
        ]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def _build_sample(self, src_tokens: List[torch.LongTensor]):
        # assert torch.is_tensor(src_tokens)
        dataset = self.task.build_dataset_for_inference(
            src_tokens,
            [x.numel() for x in src_tokens],
        )
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(lambda tensor: tensor.to(self.device), sample)
        return sample

    def generate(
        self,
        tokenized_sentences: List[torch.LongTensor],
        *args,
        inference_step_args=None,
        **kwargs
    ) -> List[List[Dict[str, torch.Tensor]]]:
        inference_step_args = inference_step_args or {}
        if "prefix_tokens" in inference_step_args:
            raise NotImplementedError("prefix generation not implemented for BART")
        else:
            bsz = len(tokenized_sentences)
            inference_step_args["prefix_tokens"] = tokenized_sentences[0].new_full(
                (bsz, 1), fill_value=self.task.source_dictionary.bos()
            ).to(device=self.device)
        return super().generate(
            tokenized_sentences,
            *args,
            inference_step_args=inference_step_args,
            **kwargs
        )

    def extract_features(
        self, tokens: torch.LongTensor, tgt_tokens: torch.LongTensor, return_all_hiddens: bool = False, classification_head_name: str = None
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > min(self.model.max_positions()):
            raise ValueError(
                "tokens exceeds maximum length: {} > {}".format(
                    tokens.size(-1), self.model.max_positions()
                )
            )

        tokens.to(device=self.device)
        if not tgt_tokens == None:
            if tokens.dim() == 1:
                tgt_tokens = tgt_tokens.unsqueeze(0)
            prev_output_tokens = tgt_tokens.clone().detach()    
        else:
            prev_output_tokens = tokens.clone().detach()    
        
        sent_num = tokens.size()[0]
        tokens = torch.cat((tokens, (torch.ones((sent_num, 1)) * self.src_lid).clone().detach().to(device=self.device)), 1).long()
        prev_output_tokens = torch.cat((prev_output_tokens, (torch.ones((sent_num, 1)) * self.tgt_lid).clone().detach().to(device=self.device)), 1).long()
        tgt_tokens = prev_output_tokens.clone().detach()    

        if not tgt_tokens == None:        
            prev_output_tokens[:, 0] = prev_output_tokens.gather(
                1,
                (tgt_tokens.ne(self.task.source_dictionary.pad()).sum(dim=1) - 1).unsqueeze(-1),
            ).squeeze()
            prev_output_tokens[:, 1:] = tgt_tokens[:, :-1]
        else:
            prev_output_tokens[:, 0] = prev_output_tokens.gather(
                1,
                (tokens.ne(self.task.source_dictionary.pad()).sum(dim=1) - 1).unsqueeze(-1),
            ).squeeze()
            prev_output_tokens[:, 1:] = tokens[:, :-1]
            
        features, extra = self.model(
            src_tokens=tokens,
            src_lengths=None,
            prev_output_tokens=prev_output_tokens,
            features_only=True,
            return_all_hiddens=return_all_hiddens,
            classification_head_name=classification_head_name
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra["inner_states"]
            return [inner_state.transpose(0, 1) for inner_state in inner_states], tokens
        elif not tgt_tokens == None:
            return features, tgt_tokens, prev_output_tokens # tgt_tokens = prev_output_tokens
        else:
            return features, tokens, prev_output_tokens # just the last layer's features

    def register_classification_head(
        self, name: str, num_classes: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_classification_head(
            name, num_classes=num_classes, embedding_size=embedding_size, **kwargs
        )

    def predict_2(self, head: str, tokens: torch.LongTensor, return_logits: bool = False, target_tokens: torch.LongTensor = None):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if target_tokens.dim() == 1:
            target_tokens = target_tokens.unsqueeze(0)
        classification_head_name="sentence_classification_head"
        logits, tokens, prev_output_tokens = self.extract_features(tokens.to(device=self.device), tgt_tokens = target_tokens.to(device=self.device), classification_head_name=classification_head_name)
            
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)

    def predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False, target_tokens: torch.LongTensor = None):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if target_tokens.dim() == 1:
            target_tokens = target_tokens.unsqueeze(0)
        features, tokens, prev_output_tokens = self.extract_features(tokens.to(device=self.device), tgt_tokens = target_tokens.to(device=self.device))
        
        try:
            sentence_representation = features[
                tokens.eq(self.task.source_dictionary.eos()), :
            ].view(features.size(0), -1, features.size(-1))[:, -1, :]
        except IndexError: # src和tgt长度不一致
            try:
                index = list(range(1, prev_output_tokens.size(1)))
                index.append(0)
                prev_output_tokens = prev_output_tokens[:, index] # move left one step/token
                sentence_representation = features[
                    prev_output_tokens.eq(self.task.source_dictionary.eos()), :
                ].view(features.size(0), -1, features.size(-1))[:, -1, :]
            except IndexError: # decoder端只输入mask
                sentence_representation = features[
                    prev_output_tokens.eq(self.task.source_dictionary.index("{}".format("<mask>"))), :
                ].view(features.size(0), -1, features.size(-1))[:, -1, :]
        
        # sentence_representation = features[
        #     tokens.eq(self.task.source_dictionary.eos()), :
        # ].view(features.size(0), -1, features.size(-1))[:, -1, :]
            
        logits = self.model.classification_heads[head](sentence_representation)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)

    def fill_mask(
        self,
        masked_inputs: List[str],
        topk: int = 5,
        match_source_len: bool = True,
        **generate_kwargs
    ):
        masked_token = '<mask>'
        batch_tokens = []
        for masked_input in masked_inputs:
            assert masked_token in masked_input, \
                "please add one {} token for the input".format(masked_token)

            text_spans = masked_input.split(masked_token)
            text_spans_bpe = (' {0} '.format(masked_token)).join(
                [self.bpe.encode(text_span.rstrip()) for text_span in text_spans]
            ).strip()
            tokens = self.task.source_dictionary.encode_line(
                '<s> ' + text_spans_bpe + ' </s>',
                append_eos=False,
                add_if_not_exist=False,
            ).long()
            batch_tokens.append(tokens)

        # ensure beam size is at least as big as topk
        generate_kwargs['beam'] = max(
            topk,
            generate_kwargs.get('beam', -1),
        )
        generate_kwargs['match_source_len'] = match_source_len
        batch_hypos = self.generate(batch_tokens, **generate_kwargs)

        return [
            [(self.decode(hypo['tokens']), hypo['score']) for hypo in hypos[:topk]]
            for hypos in batch_hypos
        ]
