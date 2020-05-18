# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch DistilBERT model
    adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
    and in part from HuggingFace PyTorch version of Google AI Bert model (https://github.com/google-research/bert)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import copy
import sys
from io import open

import itertools
import numpy as np

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from .modeling_utils import PreTrainedModel, prune_linear_layer
from .configuration_distilbert import DistilBertConfig
from .file_utils import add_start_docstrings

from .modeling_distilbert import (Embeddings,
                                  TransformerBlock,
                                  DistilBertPreTrainedModel)
from .modeling_highway_bert import entropy, HighwayException

import logging
logger = logging.getLogger(__name__)


DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'distilbert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-pytorch_model.bin",
    'distilbert-base-uncased-distilled-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-distilled-squad-pytorch_model.bin"
}


class Transformer(nn.Module):
    # this is essentially DistilBertEncoder
    # lte not added yet

    def __init__(self, config):
        super(Transformer, self).__init__()
        self.num_layers = config.n_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        layer = TransformerBlock(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.n_layers)])
        self.highway = nn.ModuleList([DistilBertHighway(config) for _ in range(config.n_layers)])
        self.early_exit_entropy = [-1 for _ in range(config.num_hidden_layers)]

        self.use_lte = False

    def set_early_exit_entropy(self, x):
        print(x)
        if (type(x) is float) or (type(x) is int):
            for i in range(len(self.early_exit_entropy)):
                self.early_exit_entropy[i] = x
        else:
            self.early_exit_entropy = x

    def forward(self, x, attn_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        all_highway_exits = ()

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(x=hidden_state,
                                         attn_mask=attn_mask,
                                         head_mask=head_mask[i])
            hidden_state = layer_outputs[-1]

            if self.output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

            current_outputs = (hidden_state,)
            if self.output_hidden_states:
                current_outputs = current_outputs + (all_hidden_states,)
            if self.output_attentions:
                current_outputs = current_outputs + (all_attentions,)

            # feed it into highway
            highway_exit = self.highway[i](current_outputs)
            highway_entropy = entropy(highway_exit[0])

            if not self.training:
                highway_exit = highway_exit + (highway_entropy,)
                all_highway_exits = all_highway_exits + (highway_exit,)

                if (
                        (i+1 < self.num_layers)
                    and (highway_entropy < self.early_exit_entropy[i])
                ):
                    new_output = (highway_exit[0],) + current_outputs[1:] + \
                                 ({'highway': all_highway_exits},)
                    raise HighwayException(new_output, i+1)
            else:
                all_highway_exits = all_highway_exits + (highway_exit,)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        outputs = (hidden_state,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        outputs = outputs + ({'highway': all_highway_exits},)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class DistilBertHighway(nn.Module):

    def __init__(self, config):
        super(DistilBertHighway, self).__init__()
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        # equivalent to pooler in original bert
        # whether to load from pre-train: unknown yet

        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.non_linear = nn.ReLU()
        self.dropout = nn.Dropout(config.seq_classif_dropout)

    def forward(self, input):
        hidden_state = input[0][:, 0]
        logits = self.classifier(self.dropout(
            self.non_linear(self.pre_classifier(hidden_state))
        ))
        return (logits, ) + input[1:]


class DistilBertModel(DistilBertPreTrainedModel):

    def __init__(self, config):
        super(DistilBertModel, self).__init__(config)

        self.embeddings = Embeddings(config)   # Embeddings
        self.transformer = Transformer(config)  # Encoder
        self.encoder = self.transformer

        self.init_weights()

    def init_highway_pooler(self):
        # for interface consistency
        pass

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.transformer.layer[layer].attention.prune_heads(heads)

    def forward(self,
                input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device) # (bs, seq_length)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)   # (bs, seq_length, dim)
        tfmr_output = self.transformer(x=inputs_embeds,
                                       attn_mask=attention_mask,
                                       head_mask=head_mask)
        hidden_state = tfmr_output[0]
        output = (hidden_state, ) + tfmr_output[1:]

        return output # last-layer hidden-state, (all hidden_states), (all attentions)


class DistilBertForSequenceClassification(DistilBertPreTrainedModel):

    def __init__(self, config):
        super(DistilBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_layers = config.n_layers

        self.distilbert = DistilBertModel(config)
        self.core = self.distilbert
        self.pre_classifier = nn.Linear(config.dim, config.dim)  # essentially Model.pooler
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, head_mask=None,
                inputs_embeds=None, labels=None,
                output_layer=-1,
                train_strategy='raw',
                layer_example_counter=None,
                step_num=-1,
                ):

        exit_layer = self.num_layers
        try:
            distilbert_output = self.distilbert(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                head_mask=head_mask,
                                                inputs_embeds=inputs_embeds)
            hidden_state = distilbert_output[0]                    # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]                    # (bs, dim)
            pooled_output = self.pre_classifier(pooled_output)   # (bs, dim)
            pooled_output = nn.ReLU()(pooled_output)             # (bs, dim)
            pooled_output = self.dropout(pooled_output)         # (bs, dim)
            logits = self.classifier(pooled_output)              # (bs, dim)

            outputs = (logits,) + distilbert_output[1:]
        except HighwayException as e:
            outputs = e.message
            exit_layer = e.exit_layer
            logits = outputs[0]


        original_entropy = entropy(logits)
        highway_entropy = []
        highway_all_logits = []
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # work with highway  exits
            highway_losses = []
            for i, highway_exit in enumerate(outputs[-1]['highway']):
                highway_logits = highway_exit[0]
                highway_all_logits.append(highway_logits)

                if self.num_labels == 1:
                    highway_loss = loss_fct(highway_logits.view(-1),
                                            labels.view(-1))
                else:
                    highway_loss = loss_fct(highway_logits.view(-1, self.num_labels),
                                            labels.view(-1))
                highway_losses.append(highway_loss)

            if train_strategy == 'raw':
                outputs = ([loss],) + outputs
            elif train_strategy.startswith("limit"):
                target_layer = int(train_strategy[5:])
                if target_layer + 1 == self.num_layers:
                    outputs = ([loss],) + outputs
                else:
                    outputs = ([highway_losses[target_layer]],) + outputs
            elif train_strategy == 'only_highway':
                outputs = ([sum(highway_losses[:-1])],) + outputs
                # exclude the final highway, of course
            elif train_strategy in ['all']:
                outputs = ([sum(highway_losses[:-1]) + loss],) + outputs
                # all highways (exclude the final one), plus the original classifier
            elif train_strategy == 'alternate':
                if step_num % 2 == 0:
                    outputs = ([loss],) + outputs
                else:
                    outputs = ([sum(highway_losses[:-1]) + loss],) + outputs
                    # all highways (exclude the final one), plus the original classifier
            else:
                raise NotImplementedError("Wrong training strategy!")

        if not self.training:
            outputs = outputs + ((original_entropy, highway_entropy), exit_layer)
            if output_layer>=0:
                outputs = (outputs[0],) +\
                          (highway_all_logits[output_layer],) +\
                          outputs[2:]

        return outputs  # (loss), logits, (hidden_states), (attentions)
