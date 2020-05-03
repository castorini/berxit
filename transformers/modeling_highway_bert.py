import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from .modeling_bert import BertLayer, BertLayerNorm, BertPreTrainedModel


def entropy(x):
    # x: torch.Tensor, logits BEFORE softmax
    exp_x = torch.exp(x)
    A = torch.sum(exp_x, dim=1)    # sum of exp(x_i)
    B = torch.sum(x*exp_x, dim=1)  # sum of x_i * exp(x_i)
    return torch.log(A) - B/A


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


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

        self.use_Qmodule = False
        self.init_Qmodule()

    def init_Qmodule(self):
        # hyperparameters for balancing loss
        self.alpha = 0.01
        self.beta = 0.6
        self.gamma = 1.0

        # Qmodule itself
        self.Qmodule_size = 10
        self.Qmodule_classifier = nn.Linear(self.Qmodule_size + 4, 2)
        # +4: 3 for logits, 1 for entropy (0 padding for regression tasks)
        self.Qmodule_activation = nn.Tanh()

        self.pool_1d = torch.nn.AdaptiveAvgPool1d(self.Qmodule_size)

    def enable_Qmodule(self, args):
        if args.alpha is not None:
            self.alpha = args.alpha
        if args.beta is not None:
            self.beta = args.beta
        if args.gamma is not None:
            self.gamma = args.gamma

        self.num_labels = 2
        if args.task_name in ['sts-b']:
            self.num_labels = 1
        elif args.task_name in ['mnli']:
            self.num_labels = 3

        self.use_Qmodule = True
        print(f'Qmodule initialized')

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

        Qmodule_outputs = []
        Qmodule_classifier_outputs = []

        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        if self.use_Qmodule:
            if self.num_labels == 2:
                zeros = torch.tensor(
                    [[0.0] for _ in range(batch_size)]).to(device)
            elif self.num_labels == 1:
                zeros = torch.tensor(
                    [[0.0, 0.0, 0.0] for _ in range(batch_size)]).to(device)

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
            with torch.autograd.profiler.record_function('highway'):
                highway_exit = self.highway[i](current_outputs)
                # logits, pooled_output

            highway_entropy = entropy(highway_exit[0])

            # the block for Qmodule
            if self.use_Qmodule:
                with torch.autograd.profiler.record_function('qmodule'):
                    pooling_out = self.pool_1d(highway_exit[1].unsqueeze(0)).squeeze(0)
                    if self.num_labels==3:
                        Qmodule_classifier_input = torch.cat([
                            pooling_out,
                            highway_exit[0],
                            highway_entropy.unsqueeze(1)
                        ], dim=1)
                    elif self.num_labels==2:
                        # add extra zero-vector for shape consistence
                        Qmodule_classifier_input = torch.cat([
                            pooling_out,
                            highway_exit[0],
                            highway_entropy.unsqueeze(1),
                            zeros
                        ], dim=1)
                    elif self.num_labels==1:
                        # add extra zero-vector for shape consistence
                        Qmodule_classifier_input = torch.cat([
                            pooling_out,
                            highway_exit[0],
                            zeros
                        ], dim=1)
                    Qmodule_classifier_output = self.Qmodule_activation(
                        self.Qmodule_classifier(Qmodule_classifier_input)
                    )
                    Qmodule_classifier_outputs.append(Qmodule_classifier_output)

            if not self.training:
                with torch.autograd.profiler.record_function('earlyexit'):
                    highway_exit = highway_exit + (highway_entropy,)  # logits, hidden_states(?), entropy
                    all_highway_exits = all_highway_exits + (highway_exit,)

                    # if np.random.rand() < 0.1:  # compare against random exit
                    if (
                            (i+1 < self.num_layers)
                        and (
                                (self.use_Qmodule and torch.argmax(Qmodule_classifier_output)==1)
                             or (not self.use_Qmodule and highway_entropy < self.early_exit_entropy[i])
                            )
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
        if self.use_Qmodule:
            outputs[-1]["qmodule"] = (Qmodule_outputs, Qmodule_classifier_outputs)

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

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
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
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
        """ Forward pass on the Model.

        The model can behave as an encoder (with only self-attention) as well
        as a decoder, in which case a layer of cross-attention is added between
        the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
        Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

        To behave as an decoder the model needs to be initialized with the
        `is_decoder` argument of the configuration set to `True`; an
        `encoder_hidden_states` is expected as an input to the forward pass.

        .. _`Attention is all you need`:
            https://arxiv.org/abs/1706.03762

        """
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
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
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

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
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
                loss_fct = CrossEntropyLoss(reduction='mean')
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
                    loss_fct = CrossEntropyLoss(reduction='mean')
                    highway_loss = loss_fct(highway_logits.view(-1, self.num_labels),
                                            labels.view(-1))
                highway_losses.append(highway_loss)
                # raw_highway_losses.append(raw_highway_loss)

            # loss (first entry of outputs), is no longer one variable, but a list of them

            if train_strategy.endswith("-Qvlstm"):
                Qmodule_loss = 0
                ongoing = torch.ones([batch_size, 1]).to(device)
                for i in range(self.num_layers-1):
                    if self.num_labels==1:
                        correctness_loss = torch.pow(
                            outputs[-1]['highway'][i][0].squeeze() - labels,
                            2
                        )
                    else:
                        Qmodule_gold = torch.eq(
                            torch.argmax(outputs[-1]['highway'][i][0], dim=1),
                            labels
                        ).long()  # 0 for wrong/continue, 1 for right/exit
                        correctness_loss = 0.99 - Qmodule_gold.float()*0.98
                        # soft labels: 1->0.01, 0->0.99
                    Q_this = outputs[-1]['qmodule'][1][i]  # Q_i
                    a_0_reward = torch.tensor([-self.bert.encoder.alpha]).to(device)  # reward for continue
                    r_this = torch.stack([
                        a_0_reward.repeat(batch_size),
                        - self.bert.encoder.beta * correctness_loss
                    ], dim=1)
                    # r_this = r_this * ongoing  # only ongoing samples have reward
                    ongoing = ongoing * torch.eq(torch.argmax(Q_this, dim=1), 0).unsqueeze(1)
                    if i == self.num_layers-2:
                        Qmodule_loss += torch.mean(
                            (r_this - Q_this) ** 2
                        )
                    else:
                        Q_next = outputs[-1]['qmodule'][1][i+1]  # Q_{i+1}
                        Q_next = torch.max(Q_next, dim=1)[0].repeat(2, 1).t()
                        Qmodule_loss += torch.mean(
                            (r_this + self.bert.encoder.gamma*Q_next - Q_this) ** 2
                        )
                #     breakpoint_flag = step_num==300
                #     if breakpoint_flag:
                #         print(i)
                #         print(Q_this)
                #         print(r_this)
                # if breakpoint_flag:
                #     breakpoint()
                outputs = ([Qmodule_loss],) + outputs
            elif train_strategy == 'raw':
                outputs = ([loss],) + outputs
            elif train_strategy.startswith("limit"):
                target_layer = int(train_strategy[5:])
                if target_layer+1 == self.num_layers:
                    outputs = ([loss],) + outputs
                else:
                    outputs = ([highway_losses[target_layer]],) + outputs
            elif train_strategy=='only_highway':
                outputs = ([sum(highway_losses[:-1])],) + outputs
                # exclude the final highway, of course
            elif train_strategy in ['all']:
                outputs = ([sum(highway_losses[:-1])+loss],) + outputs
                # all highways (exclude the final one), plus the original classifier
            elif train_strategy == 'alternate':
                if step_num%2==0:
                    outputs = ([loss],) + outputs
                else:
                    outputs = ([sum(highway_losses[:-1])+loss],) + outputs
                    # all highways (exclude the final one), plus the original classifier
            elif train_strategy=='self_distil':
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
                outputs = ([sum(highway_losses[:-1]) + loss + sum(distil_losses)],)\
                          + outputs
            elif train_strategy=='layer_wise':
                outputs = (highway_losses[:-1]+[loss],) + outputs
            else:
                raise NotImplementedError("Wrong training strategy!")

        if not self.training:
            outputs = outputs + ((original_entropy, highway_entropy), exit_layer)
            if output_layer >= 0:
                outputs = (outputs[0],) +\
                          (highway_logits_all[output_layer],) +\
                          outputs[2:]  ## use the highway of the last layer

        return outputs  # (loss), logits, (hidden_states), (attentions), (entropies), (exit_layer)
