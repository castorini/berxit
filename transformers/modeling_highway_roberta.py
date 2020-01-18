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
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
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
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_layer=-1,
                train_strategy='raw',
                layer_example_counter=None):

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

        if not self.training:
            original_entropy = entropy(logits)
            highway_entropy = []
            highway_logits_all = []
        if labels is not None:
            if layer_example_counter is not None:
                layer_example_counter[0] += len(labels)

            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(
                    reduction='none' if train_strategy=='cascade' else 'mean')
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # work with highway exits
            highway_losses = []
            each_layer_wrong = []
            for i, highway_exit in enumerate(outputs[-1]):
                highway_logits = highway_exit[0]
                if train_strategy=='cascade':
                    if i<self.num_layers-1:
                        wrong_this_layer = torch.argmax(highway_logits, dim=1) != labels
                    else:
                        wrong_this_layer = torch.argmax(logits, dim=1) != labels
                    each_layer_wrong.append(wrong_this_layer)
                    layer_example_counter[i+1] += torch.sum(wrong_this_layer)

                if not self.training:
                    highway_logits_all.append(highway_logits)
                    highway_entropy.append(highway_exit[2])
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    highway_loss = loss_fct(highway_logits.view(-1),
                                            labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss(
                        reduction='none' if train_strategy=='cascade' else 'mean')
                    highway_loss = loss_fct(highway_logits.view(-1, self.num_labels),
                                            labels.view(-1))
                    if train_strategy=='cascade':
                        if i>0:
                            highway_loss = torch.masked_select(
                                highway_loss,
                                each_layer_wrong[-1]
                            )
                        if i==self.num_layers-1:
                            loss = torch.mean(torch.masked_select(
                                loss,
                                each_layer_wrong[-1]
                            ))
                            # loss = torch.mean(loss)
                        highway_loss = torch.mean(highway_loss)
                highway_losses.append(highway_loss)


            # loss (first entry of outputs), is no longer one variable, but a list of them
            if train_strategy == 'raw':
                outputs = ([loss],) + outputs
            elif train_strategy == 'only_highway':
                outputs = ([sum(highway_losses[:-1])],) + outputs
                # exclude the final highway, of course
            elif train_strategy in ['all', 'divide']:
                outputs = ([sum(highway_losses[:-1]) + loss],) + outputs
                # all highways (exclude the final one), plus the original classifier
            elif train_strategy == 'cascade':
                # remove all nans
                potential_losses = highway_losses[:-1] + [loss]
                valid_losses = [x for x in potential_losses if not torch.isnan(x)]
                outputs = ([sum(valid_losses)],) + outputs
            elif train_strategy == 'half':
                half_highway_losses = [
                    x for i, x in enumerate(highway_losses[:-1]) if i%2==1
                ]
                outputs = ([sum(half_highway_losses) + loss],) + outputs
                # only classifiers on odd-number layers (1,3,5,7,9,...,last)
            elif train_strategy=='neigh_distil':
                # the following input_logits are before softmax
                # logits from layer[i]: outputs[-1][i][0]
                temperature = 1.0
                softmax_fct = nn.Softmax(dim=1)
                distil_losses = []
                for i in range(self.num_layers-1):
                    teacher_softmax = softmax_fct(outputs[-1][i+1][0].detach()) / temperature
                    student_softmax = softmax_fct(outputs[-1][i][0]) / temperature
                    distil_losses.append(
                        - temperature**2 * torch.sum(
                            teacher_softmax * torch.log(student_softmax))
                    )
                outputs = ([sum(highway_losses[:-1]) + loss + sum(distil_losses)],)\
                          + outputs
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
                outputs = ([sum(highway_losses[:-1]) + loss + sum(distil_losses)],) \
                          + outputs
            elif train_strategy == 'layer_wise':
                outputs = (highway_losses[:-1] + [loss],) + outputs
            else:
                raise NotImplementedError("Wrong training strategy!")

        if not self.training:
            outputs = outputs + ((original_entropy, highway_entropy), exit_layer)
            if output_layer >= 0:
                outputs = (outputs[0],) + \
                          (highway_logits_all[output_layer],) + \
                          outputs[2:]  ## use the highway of the last layer

        return outputs  # (loss), logits, (hidden_states), (attentions), entropy
