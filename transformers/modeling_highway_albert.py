import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.configuration_albert import AlbertConfig

from .modeling_albert import (AlbertEmbeddings,
                              AlbertLayerGroup,
                              AlbertPreTrainedModel,
                              load_tf_weights_in_albert,
                              ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP)
from .modeling_highway_bert import entropy, HighwayException, BertHighway



class AlbertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.num_layers = config.num_hidden_layers
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])

        self.highway = nn.ModuleList([BertHighway(config) for _ in range(config.num_hidden_layers)])

        self.early_exit_entropy = [-1 for _ in range(config.num_hidden_layers)]

        self.use_lte = False

    def set_early_exit_entropy(self, x):
        print(x)
        if (type(x) is float) or (type(x) is int):
            for i in range(len(self.early_exit_entropy)):
                self.early_exit_entropy[i] = x
        else:
            self.early_exit_entropy = x

    def init_highway_pooler(self, pooler):
        loaded_model = pooler.state_dict()
        for highway in self.highway:
            for name, param in highway.pooler.state_dict().items():
                compact_name = name.split('.')[1]
                param.copy_(loaded_model[compact_name])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_attentions = ()
        all_hidden_states = ()
        all_highway_exits = ()

        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        if self.output_hidden_states:
            # this is different from BERT! the output of embedding layer is included
            all_hidden_states = (hidden_states,)

        for i in range(self.config.num_hidden_layers):
            """
            An example given the default hyper-params.
            iterate for 12 (num_hidden_layers) times;
            albert_layer_groups[group_idx] is executed each time, but there are actually only 1 (num_hidden_groups) layer_group;
            each layer group is a group of 1 (inner_group_num) layers.
            So in total it's one layer repeated 12 times, and a more general case is like
            ABCABCABCABC, where there are 4 groups, and each group has 3 layers
            """
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
            )
            hidden_states = layer_group_output[0]

            current_outputs = (hidden_states,)
            if self.output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # the block for highway
            highway_exit = self.highway[i](current_outputs)
            # logits, pooled_output

            highway_entropy = entropy(highway_exit[0])

            if not self.training:
                highway_exit = highway_exit + (highway_entropy,)  # logits, hidden_states(?), entropy
                all_highway_exits = all_highway_exits + (highway_exit,)

                # if np.random.rand() < 0.1:  # compare against random exit
                if (
                        i+1 < self.num_layers \
                    and highway_entropy < self.early_exit_entropy[i]
                ):
                    new_output = (highway_exit[0],) + current_outputs[1:] + \
                                 ({'highway': all_highway_exits},)
                    raise HighwayException(new_output, i+1)
            else:
                all_highway_exits = all_highway_exits + (highway_exit,)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        outputs = outputs + ({"highway": all_highway_exits},)

        return outputs  # last-layer hidden state, (all hidden states), (all attentions), all highway exits


class AlbertModel(AlbertPreTrainedModel):

    config_class = AlbertConfig
    pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_albert
    base_model_prefix = "albert"

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertEncoder(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

        self.init_weights()

    def init_highway_pooler(self):
        self.encoder.init_highway_pooler(self.pooler)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):

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
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))

        outputs = (sequence_output, pooled_output) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs


class AlbertForSequenceClassification(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size

        self.albert = AlbertModel(config)
        self.core = self.albert
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_layer=-1,
        train_strategy='raw',
        layer_example_counter=None,
        step_num=-1,
    ):

        exit_layer = self.num_layers
        try:
            outputs = self.albert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        except HighwayException as e:
            outputs = e.message
            exit_layer = e.exit_layer
            logits = outputs[0]

        batch_size = logits.shape[0]
        device = logits.device


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
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # work with highway exits
            highway_losses = []
            for i, highway_exit in enumerate(outputs[-1]["highway"]):
                highway_logits = highway_exit[0]

                if not self.training:
                    highway_logits_all.append(highway_logits)
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
                # raw_highway_losses.append(raw_highway_loss)

            if train_strategy == 'raw':
                outputs = (loss,) + outputs
            elif train_strategy.startswith("limit"):
                target_layer = int(train_strategy[5:])
                if target_layer + 1 == self.num_layers:
                    outputs = (loss,) + outputs
                else:
                    outputs = (highway_losses[target_layer],) + outputs
            elif train_strategy == 'only_highway':
                outputs = (sum(highway_losses[:-1]),) + outputs
                # exclude the final highway, of course
            elif train_strategy == 'all':
                outputs = (sum(highway_losses[:-1]) + loss,) + outputs
                # all highways (exclude the final one), plus the original classifier
            elif train_strategy == 'weight-linear':
                loss_sum = loss * self.num_layers
                weight_sum = (1 + self.num_layers) * self.num_layers / 2
                for i in range(self.num_layers - 1):
                    loss_sum += highway_losses[i] * (1 + i)
                outputs = (loss_sum / weight_sum,) + outputs
            elif train_strategy == 'alternate':
                if step_num % 2 == 0:
                    outputs = (loss,) + outputs
                else:
                    outputs = (sum(highway_losses[:-1]) + loss,) + outputs
                    # all highways (exclude the final one), plus the original classifier
            elif train_strategy == 'self_distil':
                # the following input_logits are before softmax
                # final layer logits: logits
                # logits from layer[i]: outputs[-1]["highway"][i][0]
                temperature = 1.0
                softmax_fct = nn.Softmax(dim=1)
                teacher_softmax = softmax_fct(logits.detach()) / temperature
                distil_losses = []
                for i in range(self.num_layers - 1):
                    student_softmax = softmax_fct(outputs[-1]["highway"][i][0]) / temperature
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
                outputs = (outputs[0],) +\
                          (highway_logits_all[output_layer],) +\
                          outputs[2:]  ## use the highway of the last layer

        return outputs  # (loss), logits, (hidden_states), (attentions)
