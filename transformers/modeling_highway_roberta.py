from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from .modeling_roberta import RobertaEmbeddings
from .modeling_highway_bert import BertModel, BertPreTrainedModel, entropy, HighwayException
from .configuration_roberta import RobertaConfig

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    'distilroberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    'roberta-base-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    'roberta-large-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}


class RobertaModel(BertModel):

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaModel, self).__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


class RobertaForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers

        self.roberta = RobertaModel(config)
        self.core = self.roberta
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_layer=-1,
                train_strategy='raw',
                layer_example_counter=None,
                step_num=-1):

        exit_layer = self.num_layers
        try:
            outputs = self.roberta(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds)

            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        except HighwayException as e:
            outputs = e.message
            exit_layer = e.exit_layer
            logits = outputs[0]

        original_entropy = entropy(logits)
        highway_entropy = []
        highway_all_logits = []
        if labels is not None:
            if layer_example_counter is not None:
                layer_example_counter[0] += len(labels)

            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # work with highway exits
            highway_losses = []
            for i, highway_exit in enumerate(outputs[-1]["highway"]):
                highway_logits = highway_exit[0]
                highway_all_logits.append(highway_logits)
                if not self.training:
                    highway_entropy.append(highway_exit[2])

                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    highway_loss = loss_fct(highway_logits.view(-1),
                                            labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    highway_loss = loss_fct(highway_logits.view(-1, self.num_labels),
                                            labels.view(-1))
                highway_losses.append(highway_loss)


            # loss (first entry of outputs), is no longer one variable, but a list of them
            if train_strategy.endswith("-lte"):
                lte_loss_fct = MSELoss()
                uncertainties = []
                exit_pred = []
                for i in range(self.num_layers):
                    # exit prediction
                    exit_pred.append(outputs[-1]['lte'][i])  # "uncertainty"

                    # exit label
                    if self.num_labels == 1:
                        if i + 1 == self.num_layers:
                            layer_output = logits
                        else:
                            layer_output = outputs[-1]['highway'][i][0]
                        layer_uncertainty = torch.pow(
                            layer_output.squeeze() - labels,
                            2
                        )
                    else:
                        if i + 1 == self.num_layers:
                            layer_uncertainty = entropy(logits)
                        else:
                            layer_uncertainty = entropy(highway_all_logits[i])
                    uncertainties.append(layer_uncertainty)
                exit_pred = torch.stack(exit_pred)
                exit_label = torch.stack(uncertainties).detach()

                # normalize exit label
                if self.num_labels == 1:
                    norm_exit_label = 1 - torch.exp(-exit_label)
                else:
                    norm_exit_label = torch.clamp(exit_label, min=0.05, max=0.95)
                outputs = (lte_loss_fct(exit_pred, norm_exit_label),) + outputs
            elif train_strategy == 'raw':
                outputs = (loss,) + outputs
            elif train_strategy.startswith("limit"):
                target_layer = int(train_strategy[5:])
                if target_layer+1 == self.num_layers:
                    outputs = (loss,) + outputs
                else:
                    outputs = (highway_losses[target_layer],) + outputs
            elif train_strategy == 'only_highway':
                outputs = (sum(highway_losses[:-1]),) + outputs
                # exclude the final highway, of course
            elif train_strategy in ['all']:
                outputs = (sum(highway_losses[:-1]) + loss,) + outputs
                # all highways (exclude the final one), plus the original classifier
            elif train_strategy == 'all_alternate':
                if step_num % 2 == 0:
                    outputs = (loss,) + outputs
                else:
                    outputs = (sum(highway_losses[:-1]) + loss,) + outputs
                    # all highways (exclude the final one), plus the original classifier
            elif train_strategy == 'self_distil':
                # the following input_logits are before softmax
                # final layer logits: logits
                # logits from layer[i]: outputs[-1][i][0]
                temperature = 1.0
                softmax_fct = nn.Softmax(dim=1)
                teacher_softmax = softmax_fct(logits.detach()) / temperature
                distil_losses = []
                for i in range(self.num_layers - 1):
                    student_softmax = softmax_fct(outputs[-1][i][0]) / temperature
                    distil_losses.append(
                        - temperature ** 2 * torch.sum(
                            teacher_softmax * torch.log(student_softmax))
                    )
                outputs = (sum(highway_losses[:-1]) + loss + sum(distil_losses),) \
                          + outputs
            else:
                raise NotImplementedError("Wrong training strategy!")

        if not self.training:
            outputs = outputs + ((original_entropy, highway_entropy), exit_layer)
            if output_layer >= 0:
                outputs = (outputs[0],) + \
                          (highway_all_logits[output_layer],) + \
                          outputs[2:]  ## use the highway of the last layer

        return outputs  # (loss), logits, (hidden_states), (attentions), entropy
