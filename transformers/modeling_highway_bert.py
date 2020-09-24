import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from .modeling_bert import BertEmbeddings, BertLayer, BertPreTrainedModel


def entropy(x):
    # x: torch.Tensor, logits BEFORE softmax
    exp_x = torch.exp(x)
    A = torch.sum(exp_x, dim=1)    # sum of exp(x_i)
    B = torch.sum(x*exp_x, dim=1)  # sum of x_i * exp(x_i)
    return torch.log(A) - B/A


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.num_layers = config.num_hidden_layers
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.highway = nn.ModuleList([BertHighway(config) for _ in range(config.num_hidden_layers)])

        self.early_exit_entropy = [-1 for _ in range(config.num_hidden_layers)]

        self.use_lte = False
        self.init_lte()

    def init_lte(self):
        self.lte_th = [0.005] * self.num_layers
        self.lte_classifier = nn.Linear(self.hidden_size, 1)
        self.lte_activation = nn.Sigmoid()

    def enable_lte(self, args):
        if args.lte_th is not None:
            if ',' not in args.lte_th:
                self.lte_th = [float(args.lte_th)] * self.num_layers
            else:
                groups = args.lte_th.split(';')
                self.lte_th = []
                for g in groups:
                    val, rep = g.split(',')
                    val, rep = float(val), int(rep)
                    self.lte_th = self.lte_th + [val] * rep

        self.use_lte = True
        self.print_fname = args.plot_data_dir + args.output_dir + '/uncertainty.txt'
        print(f'lte enabled, th={self.lte_th}')

    def set_early_exit_entropy(self, x):
        if (type(x) is float) or (type(x) is int):
            for i in range(len(self.early_exit_entropy)):
                self.early_exit_entropy[i] = x
        else:
            self.early_exit_entropy = x

    def init_highway_pooler(self, pooler):
        loaded_model = pooler.state_dict()
        for highway in self.highway:
            for name, param in highway.pooler.state_dict().items():
                param.copy_(loaded_model[name])

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        all_highway_exits = ()

        lte_outputs = []

        # batch_size = hidden_states.shape[0]
        # device = hidden_states.device

        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            current_outputs = (hidden_states,)
            if self.output_hidden_states:
                current_outputs = current_outputs + (all_hidden_states,)
            if self.output_attentions:
                current_outputs = current_outputs + (all_attentions,)

            # the block for highway
            highway_exit = self.highway[i](current_outputs)
            # logits, pooled_output

            highway_entropy = entropy(highway_exit[0])

            # the block for lte
            if self.use_lte:
                lte_input = highway_exit[1]  # hidden states
                lte_output = self.lte_activation(
                    self.lte_classifier(lte_input)
                ).squeeze()
                lte_outputs.append(lte_output)

            if not self.training:
                highway_exit = highway_exit + (highway_entropy,)  # logits, hidden_states(?), entropy
                all_highway_exits = all_highway_exits + (highway_exit,)

                if (
                        (i+1 < self.num_layers)
                    and (
                            (self.use_lte and lte_output < self.lte_th[i])
                         or (not self.use_lte and highway_entropy < self.early_exit_entropy[i])
                        )
                ):
                    new_output = (highway_exit[0],) + current_outputs[1:] + \
                                 ({'highway': all_highway_exits},)
                    raise HighwayException(new_output, i+1)
            else:
                all_highway_exits = all_highway_exits + (highway_exit,)

        if self.use_lte and self.lte_th == [0.0] * self.num_layers:
            with open(self.print_fname, 'a') as fout:
                print('\t'.join(map(
                    lambda x: str(float(x)),
                    lte_outputs
                )), file=fout)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        outputs = outputs + ({"highway": all_highway_exits},)
        if self.use_lte:
            outputs[-1]["lte"] = lte_outputs

        return outputs  # last-layer hidden state, (all hidden states), (all attentions), all highway exits


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.chosen_token = 0

        # Pooler weights also needs to be loaded, especially in Highway!

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, self.chosen_token]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def init_highway_pooler(self):
        self.encoder.init_highway_pooler(self.pooler)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
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
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0

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
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions), highway exits


class HighwayException(Exception):
    def __init__(self, message, exit_layer):
        self.message = message
        self.exit_layer = exit_layer  # start from 1!


class BertHighway(nn.Module):
    r"""A module to provide a shortcut
    from
    the output of one non-final BertLayer in BertEncoder
    to
    cross-entropy computation in BertForSequenceClassification
    """
    def __init__(self, config):
        super(BertHighway, self).__init__()
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, encoder_outputs):
        # Pooler
        pooler_input = encoder_outputs[0]
        pooler_output = self.pooler(pooler_input)
        # "return" pooler_output

        # BertModel
        bmodel_output = (pooler_input, pooler_output) + encoder_outputs[1:]
        # "return" bodel_output

        # Dropout and classification
        pooled_output = bmodel_output[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, pooled_output


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size

        self.bert = BertModel(config)
        self.core = self.bert
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                output_layer=-1, train_strategy='raw',
                layer_example_counter=None, step_num=-1):

        exit_layer = self.num_layers
        try:
            outputs = self.bert(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds)
            # sequence_output, pooled_output, (hidden_states), (attentions), highway exits

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
                    loss_fct = MSELoss()
                    highway_loss = loss_fct(highway_logits.view(-1),
                                            labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    highway_loss = loss_fct(highway_logits.view(-1, self.num_labels),
                                            labels.view(-1))
                highway_losses.append(highway_loss)

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
                        if i+1 == self.num_layers:
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
            elif train_strategy == 'all-singlelayer-jlte':  # there will be better ways
                lte_loss_fct = MSELoss()
                layer_acc = []
                exit_pred = []
                for i in range(self.num_layers):
                    # uncertainty / prob to continue
                    exit_pred.append(outputs[-1]['lte'][i])

                    # label
                    if i+1 == self.num_layers:
                        layer_output = logits
                    else:
                        layer_output = outputs[-1]['highway'][i][0]
                    if self.num_labels == 1:
                        pass
                        # correctness_loss = torch.pow(
                        #     layer_output.squeeze() - labels,
                        #     2
                        # )
                    else:
                        lte_gold = torch.eq(
                            torch.argmax(layer_output, dim=1),
                            labels
                        )  # 0 for wrong/continue, 1 for right/exit
                        correctness_loss = 1 - lte_gold.float()  # 1 for continue, match exit_pred
                    layer_acc.append(correctness_loss)
                exit_pred = torch.stack(exit_pred)
                exit_label = torch.stack(layer_acc).detach()
                total_loss = loss + sum(highway_losses[:-1]) + lte_loss_fct(exit_pred, exit_label)
                outputs = (total_loss,) + outputs
            elif train_strategy.endswith('-alllayer-jlte'):
                pass
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
            elif train_strategy == 'all':
                outputs = (sum(highway_losses[:-1])+loss,) + outputs
                # all highways (exclude the final one), plus the original classifier
            elif train_strategy == 'weight-linear':
                loss_sum = loss * self.num_layers
                weight_sum = (1 + self.num_layers) * self.num_layers / 2
                for i in range(self.num_layers-1):
                    loss_sum += highway_losses[i] * (1+i)
                outputs = (loss_sum/weight_sum,) + outputs
            elif train_strategy == 'weight-sqrt':
                loss_sum = loss * math.sqrt(self.num_layers)
                weight_sum = math.sqrt(self.num_layers)
                for i in range(self.num_layers-1):
                    loss_sum += highway_losses[i] * math.sqrt(1+i)
                    weight_sum += math.sqrt(1+i)
                outputs = (loss_sum/weight_sum,) + outputs
            elif train_strategy == 'weight-sq':
                loss_sum = loss * (self.num_layers**2)
                weight_sum = self.num_layers**2
                for i in range(self.num_layers-1):
                    loss_sum += highway_losses[i] * (1+i)**2
                    weight_sum += (1+i)**2
                outputs = (loss_sum/weight_sum,) + outputs
            elif train_strategy == 'all_alternate':
                if step_num%2==0:
                    outputs = (loss,) + outputs
                else:
                    outputs = (sum(highway_losses[:-1])+loss,) + outputs
                    # all highways (exclude the final one), plus the original classifier
            elif train_strategy == 'self_distil':
                # the following input_logits are before softmax
                # final layer logits: logits
                # logits from layer[i]: outputs[-1]["highway"][i][0]
                temperature = 1.0
                softmax_fct = nn.Softmax(dim=1)
                teacher_softmax = softmax_fct(logits.detach()) / temperature
                distil_losses = []
                for i in range(self.num_layers-1):
                    student_softmax = softmax_fct(outputs[-1]["highway"][i][0]) / temperature
                    distil_losses.append(
                        - temperature**2 * torch.sum(
                            teacher_softmax * torch.log(student_softmax))
                    )
                outputs = (sum(highway_losses[:-1]) + loss + sum(distil_losses),)\
                          + outputs
            else:
                raise NotImplementedError("Wrong training strategy!")

        if not self.training:
            outputs = outputs + ((original_entropy, highway_entropy), exit_layer)
            if output_layer >= 0:
                outputs = (outputs[0],) +\
                          (highway_all_logits[output_layer],) +\
                          outputs[2:]  ## use the highway of the last layer

        return outputs  # (loss), logits, (hidden_states), (attentions), (entropies), (exit_layer)
