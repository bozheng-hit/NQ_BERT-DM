import enum
import math
import copy
import json
import torch
import torch.nn as nn


class EdgeType(enum.IntEnum):
    TOKEN_TO_TOKEN = 0
    TOKEN_TO_SENTENCE = 1
    TOKEN_TO_PARAGRAPH = 2
    TOKEN_TO_DOCUMENT = 3

    SENTENCE_TO_TOKEN = 4
    SENTENCE_TO_SENTENCE = 5
    SENTENCE_TO_PARAGRAPH = 6
    SENTENCE_TO_DOCUMENT = 7

    PARAGRAPH_TO_TOKEN = 8
    PARAGRAPH_TO_SENTENCE = 9
    PARAGRAPH_TO_PARAGRAPH = 10
    PARAGRAPH_TO_DOCUMENT = 11

    DOCUMENT_TO_TOKEN = 12
    DOCUMENT_TO_SENTENCE = 13
    DOCUMENT_TO_PARAGRAPH = 14
    DOCUMENT_TO_DOCUMENT = 15


class NodeType(enum.IntEnum):
    TOKEN = 0
    SENTENCE = 1
    PARAGRAPH = 2
    DOCUMENT = 3


class NodePosition(enum.IntEnum):
    MAX_PARAGRAPH = 64
    MAX_SENTENCE = 128
    MAX_TOKEN = 512


class EdgePosition(enum.auto):
    NUM_TOKEN_TO_TOKEN = 0
    NUM_TOKEN_TO_SENTENCE = 64
    NUM_TOKEN_TO_PARAGRAPH = 256
    NUM_TOKEN_TO_DOCUMENT = 512

    NUM_SENTENCE_TO_TOKEN = 64
    NUM_SENTENCE_TO_SENTENCE = 0
    NUM_SENTENCE_TO_PARAGRAPH = 32
    NUM_SENTENCE_TO_DOCUMENT = 64

    NUM_PARAGRAPH_TO_TOKEN = 256
    NUM_PARAGRAPH_TO_SENTENCE = 32
    NUM_PARAGRAPH_TO_PARAGRAPH = 0
    NUM_PARAGRAPH_TO_DOCUMENT = 32

    NUM_DOCUMENT_TO_TOKEN = 512
    NUM_DOCUMENT_TO_SENTENCE = 64
    NUM_DOCUMENT_TO_PARAGRAPH = 32
    NUM_DOCUMENT_TO_DOCUMENT = 0

    max_edge_types = [NUM_TOKEN_TO_TOKEN, NUM_TOKEN_TO_SENTENCE, NUM_TOKEN_TO_PARAGRAPH, NUM_TOKEN_TO_DOCUMENT,
                      NUM_SENTENCE_TO_TOKEN, NUM_SENTENCE_TO_SENTENCE, NUM_SENTENCE_TO_PARAGRAPH,
                      NUM_SENTENCE_TO_DOCUMENT, NUM_PARAGRAPH_TO_TOKEN, NUM_PARAGRAPH_TO_SENTENCE,
                      NUM_PARAGRAPH_TO_PARAGRAPH, NUM_PARAGRAPH_TO_DOCUMENT, NUM_DOCUMENT_TO_TOKEN,
                      NUM_DOCUMENT_TO_SENTENCE, NUM_DOCUMENT_TO_PARAGRAPH, NUM_DOCUMENT_TO_DOCUMENT]


EdgePosition.edge_type_start = [0]
for edge_type in range(1, 16):
    EdgePosition.edge_type_start.append(
        EdgePosition.edge_type_start[edge_type - 1] + EdgePosition.max_edge_types[edge_type - 1])


def get_edge_position(edge_type, edge_idx):
    return EdgePosition.edge_type_start[edge_type] + min(edge_idx, EdgePosition.max_edge_types[edge_type] - 1)


class Graph(object):
    def __init__(self):
        self.edges_src = []
        self.edges_tgt = []
        self.edges_type = []
        self.edges_pos = []
        self.st_mask = [0] * (NodePosition.MAX_TOKEN + NodePosition.MAX_SENTENCE + NodePosition.MAX_PARAGRAPH + 1)
        self.st_index = [-1] * (NodePosition.MAX_TOKEN + NodePosition.MAX_SENTENCE + NodePosition.MAX_PARAGRAPH + 1)

    def add_node(self, idx, index=-1):
        self.st_mask[idx] = 1
        self.st_index[idx] = index

    def add_edge(self, src, tgt, edge_type=-1, edge_pos=-1):
        if src < 0 or tgt < 0:
            return

        assert self.st_mask[src] > 0 and self.st_mask[tgt] > 0
        self.edges_src.append(src)
        self.edges_tgt.append(tgt)
        self.edges_type.append(edge_type)
        self.edges_pos.append(edge_pos)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Config(object):

    def __init__(self, config_json_file):
        with open(config_json_file, "r", encoding='utf-8') as reader:
            json_config = json.loads(reader.read())
        for key, value in json_config.items():
            self.__dict__[key] = value

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class GraphAttention(nn.Module):
    def __init__(self, config, max_seq_len, max_relative_position):
        super(GraphAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.max_seq_len = max_seq_len
        self.max_relative_position = max_relative_position
        self.use_relative_embeddings = config.use_relative_embeddings

        self.relative_key_embeddings = nn.Embedding(max_relative_position * 2 + 1, self.attention_head_size)

        self.relative_value_embeddings = nn.Embedding(max_relative_position * 2 + 1, self.attention_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.indices = []
        for i in range(self.max_seq_len):
            self.indices.append([])
            for j in range(self.max_seq_len):
                position = min(max(0, j - i + max_relative_position), max_relative_position * 2)
                self.indices[-1].append(position)

        self.indices = nn.Parameter(torch.LongTensor(self.indices), requires_grad=False)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, st_mask):
        batch_size = hidden_states.size(0)

        attention_mask = st_mask.unsqueeze(1).unsqueeze(2)

        attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # (batch_size, num_attention_heads, seq_len, attention_head_size)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch_size, num_attention_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if self.use_relative_embeddings:
            # (batch_size, num_attention_heads, seq_len, max_relative_position * 2 + 1)
            relative_attention_scores = torch.matmul(query_layer, self.relative_key_embeddings.weight.transpose(-1, -2))

            # fill the attention score matrix
            batch_indices = self.indices.unsqueeze(0).unsqueeze(1).expand(batch_size, self.num_attention_heads, -1, -1)
            attention_scores = attention_scores + torch.gather(input=relative_attention_scores, dim=3,
                                                               index=batch_indices)

            # new_scores_shape = (batch_size * self.num_attention_heads, self.max_seq_len, -1)
            # print(temp_tensor)
            # print("query", query_layer)
            # print("key_embeddings", self.relative_key_embeddings)
            # print("relative_scores", relative_attention_scores)
            # attention_scores = attention_scores.view(*new_scores_shape)  # + temp_tensor
            # print("attention_scores", attention_scores)
            # attention_scores = attention_scores.view(batch_size, self.num_attention_heads, -1, self.max_seq_len)
            # TODO is masking necessary?

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # (batch_size, num_attention_heads, seq_len, head_size)
        context_layer = torch.matmul(attention_probs, value_layer)

        if self.use_relative_embeddings:
            # (batch_size, num_attention_heads, seq_len, max_relative_position * 2 + 1)
            relative_attention_probs = torch.zeros_like(relative_attention_scores)
            relative_attention_probs.scatter_add_(dim=3, index=batch_indices, src=attention_probs)
            relative_values = torch.matmul(relative_attention_probs, self.relative_value_embeddings.weight)
            context_layer = context_layer + relative_values
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class IntegrationLayer(nn.Module):
    def __init__(self, config):
        super(IntegrationLayer, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.num_edge_types = config.num_edge_types

        self.use_relative_embeddings = config.use_relative_embeddings
        self.relative_key_embeddings = nn.Embedding(self.num_edge_types, self.attention_head_size)
        self.relative_value_embeddings = nn.Embedding(self.num_edge_types, self.attention_head_size)

    def forward(self, hidden_states, edges):
        edges_src, edges_tgt, edges_type, edges_pos = edges
        batch_size, seq_len = hidden_states.size(0), hidden_states.size(1)
        query_layer = self.query(hidden_states).view(batch_size * seq_len, self.num_attention_heads,
                                                     self.attention_head_size)
        key_layer = self.key(hidden_states).view(batch_size * seq_len, self.num_attention_heads,
                                                 self.attention_head_size)
        value_layer = self.value(hidden_states).view(batch_size * seq_len, self.num_attention_heads,
                                                     self.attention_head_size)
        # print(hidden_states)
        # (n_edges, n_heads, head_size)
        src_key_tensor = key_layer[edges_src]
        if self.use_relative_embeddings:
            src_key_tensor += self.relative_key_embeddings(edges_pos).unsqueeze(1)

        tgt_query_tensor = query_layer[edges_tgt]

        # (n_edges, n_heads)
        attention_scores = torch.exp((tgt_query_tensor * src_key_tensor).sum(-1) / math.sqrt(self.attention_head_size))

        sum_attention_scores = hidden_states.data.new(batch_size * seq_len, self.num_attention_heads).fill_(0)
        indices = edges_tgt.view(-1, 1).expand(-1, self.num_attention_heads)
        sum_attention_scores.scatter_add_(dim=0, index=indices, src=attention_scores)

        # print("before", attention_scores)
        attention_scores = attention_scores / sum_attention_scores[edges_tgt]
        # print("after", attention_scores)

        # (n_edges, n_heads, head_size) * (n_edges, n_heads, 1)
        src_value_tensor = value_layer[edges_src]
        if self.use_relative_embeddings:
            src_value_tensor += self.relative_value_embeddings(edges_pos).unsqueeze(1)

        src_value_tensor *= attention_scores.unsqueeze(-1)

        output = hidden_states.data.new(
            batch_size * seq_len, self.num_attention_heads, self.attention_head_size).fill_(0)
        indices = edges_tgt.view(-1, 1, 1).expand(-1, self.num_attention_heads, self.attention_head_size)
        output.scatter_add_(dim=0, index=indices, src=src_value_tensor)
        output = output.view(batch_size, seq_len, -1)

        # print(hidden_states.shape, output.shape)
        return torch.cat([hidden_states, output], dim=-1)


class AttentionOutputLayer(nn.Module):
    def __init__(self, config):
        super(AttentionOutputLayer, self).__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # print(hidden_states.shape, input_tensor.shape)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class GraphAttentionLayer(nn.Module):
    def __init__(self, config):
        super(GraphAttentionLayer, self).__init__()
        self.token_attention = GraphAttention(config, config.max_token_len, config.max_token_relative)
        self.sentence_attention = GraphAttention(config, config.max_sentence_len, config.max_sentence_relative)
        self.paragraph_attention = GraphAttention(config, config.max_paragraph_len, config.max_paragraph_relative)
        self.integration = IntegrationLayer(config)
        self.output = AttentionOutputLayer(config)

    def forward(self, input_tensor, st_mask, edges):
        token_output = self.token_attention(input_tensor[:, :NodePosition.MAX_TOKEN, :],
                                            st_mask[:, :NodePosition.MAX_TOKEN])
        # print("token_output", token_output)

        sent_start = NodePosition.MAX_TOKEN
        sent_end = sent_start + NodePosition.MAX_SENTENCE
        sentence_output = self.sentence_attention(input_tensor[:, sent_start:sent_end, :],
                                                  st_mask[:, sent_start:sent_end])
        # print("sentence_output", sentence_output)

        para_start = sent_end
        para_end = sent_end + NodePosition.MAX_PARAGRAPH
        paragraph_output = self.paragraph_attention(input_tensor[:, para_start:para_end, :],
                                                    st_mask[:, para_start:para_end])
        # print("paragraph_output", paragraph_output)

        self_output = torch.cat([token_output, sentence_output, paragraph_output, input_tensor[:, -1:, :]], dim=1)
        # return self_output
        # self_output = input_tensor
        graph_output = self.integration(self_output, edges)
        # graph_output = self_output
        # print("graph_output", graph_output)
        attention_output = self.output(graph_output, input_tensor)
        # print("attention_output", attention_output)
        return attention_output


class IntermediateLayer(nn.Module):
    def __init__(self, config):
        super(IntermediateLayer, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class OutputLayer(nn.Module):
    def __init__(self, config):
        super(OutputLayer, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.attention = GraphAttentionLayer(config)
        self.intermediate = IntermediateLayer(config)
        self.output = OutputLayer(config)

    def forward(self, hidden_states, st_mask, edges):
        attention_output = self.attention(hidden_states, st_mask, edges)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.initializer = Initializer(config)
        layer = EncoderLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, st_mask, edges, output_all_encoded_layers=True):
        hidden_states = self.initializer(hidden_states, st_mask, edges)

        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, st_mask, edges)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class Initializer(nn.Module):
    def __init__(self, config):
        super(Initializer, self).__init__()
        self.position_embeddings = nn.Embedding(NodePosition.MAX_SENTENCE + NodePosition.MAX_PARAGRAPH + 1,
                                                config.hidden_size)

    def forward(self, hidden_states, st_mask, edges):
        edges_src, edges_tgt, edges_type, edges_pos = edges
        graph_hidden = hidden_states.data.new(st_mask.size(0), st_mask.size(1), hidden_states.size(2)).fill_(0)
        graph_hidden[:, :NodePosition.MAX_TOKEN, :] = hidden_states

        # Add position embedding
        mask = st_mask[:, NodePosition.MAX_TOKEN:].eq(1).unsqueeze(-1)
        graph_hidden[:, NodePosition.MAX_TOKEN:, :] += self.position_embeddings.weight * mask.float()

        # print(graph_hidden)
        # Update by TOKEN_TO_SENTENCE
        indices_t2s = edges_type.eq(EdgeType.TOKEN_TO_SENTENCE).nonzero().view(-1).tolist()
        graph_hidden = self.average_pooling(graph_hidden, edges_src[indices_t2s], edges_tgt[indices_t2s])
        # print(graph_hidden)

        # Update by SENTENCE_TO_PARAGRAPH
        indices_s2p = edges_type.eq(EdgeType.SENTENCE_TO_PARAGRAPH).nonzero().view(-1).tolist()
        graph_hidden = self.average_pooling(graph_hidden, edges_src[indices_s2p], edges_tgt[indices_s2p])
        # print(graph_hidden)
        # Update by PARAGRAPH_TO_DOCUMENT
        indices_p2d = edges_type.eq(EdgeType.PARAGRAPH_TO_DOCUMENT).nonzero().view(-1).tolist()
        graph_hidden = self.average_pooling(graph_hidden, edges_src[indices_p2d], edges_tgt[indices_p2d])
        return graph_hidden

    @classmethod
    def average_pooling(cls, graph_hidden, edges_src, edges_tgt):
        batch_size, n_nodes, hidden_size = graph_hidden.size()
        graph_hidden = graph_hidden.view(batch_size * n_nodes, hidden_size)
        src_tensor = graph_hidden[edges_src]

        indices = edges_tgt.view(-1, 1).expand(-1, hidden_size)
        sum_hidden = graph_hidden.clone().fill_(0)
        sum_hidden.scatter_add_(dim=0, index=indices, src=src_tensor)

        n_edges = graph_hidden.data.new(batch_size * n_nodes).fill_(0)
        n_edges.scatter_add_(dim=0, index=edges_tgt, src=torch.ones_like(edges_tgt).float())
        # print(edges_src)
        # print(edges_tgt)
        indices = n_edges.nonzero().view(-1)
        graph_hidden[indices] += sum_hidden[indices] / n_edges[indices].unsqueeze(-1)

        return graph_hidden.view(batch_size, n_nodes, hidden_size)


class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(NodePosition.MAX_TOKEN, config.hidden_size)
        self.type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if type_ids is None:
            type_ids = torch.zeros_like(input_ids)

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        type_embeddings = self.type_embeddings(type_ids)

        embeddings = word_embeddings + position_embeddings + type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, Config):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_dope_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, config_path, model_path):
        pass


class GraphEncoder(PreTrainedModel):
    def __init__(self, config):
        super(GraphEncoder, self).__init__(config)
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        self.apply(self.init_weights)

    def forward(self, input_ids, type_ids, st_mask, edges, output_all_encoded_layers=True):
        embedding_output = self.embeddings(input_ids, type_ids)
        encoded_layers = self.encoder(embedding_output, st_mask, edges,
                                      output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers
