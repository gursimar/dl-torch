import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_weights):
        super(EmbeddingLayer, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings = embedding_weights.shape[0], embedding_dim = embedding_weights.shape[1], padding_idx = 0)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_weights))
        
    def forward(self, X):
        return self.embedding_layer(X)
    
class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        input_dim = params['embed_size']
        hidden_dim = params['hidden_size']
        rnn_type = params['rnn_type']
        num_layers = params['num_layers']
        self.params = params
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.rnn = getattr(nn, rnn_type)(self.input_dim, self.hidden_dim, num_layers)
        
    def forward(self, X, hidden_state):
        rnn_output, hidden_state = self.rnn(X, hidden_state)
        return rnn_output, hidden_state

    def initHidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()), Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()))
        else:
            return Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())            
        

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        input_dim = params['embed_size']
        hidden_dim = params['hidden_size']
        rnn_type = params['rnn_type']
        num_layers = params['num_layers']
        output_dim = params['target_vocab_size']
        self.max_seq_size = params['max_seq_size']
        self.params = params
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type)(self.input_dim, self.hidden_dim, num_layers)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, X, hidden_state, encoder_outputs, isPacked=False):
        rnn_output, hidden_state = self.rnn(X, hidden_state)
        if isPacked:
            rnn_output, temp = nn.utils.rnn.pad_packed_sequence(rnn_output)
        linear_out = self.linear(rnn_output)
        
        # these are added to make the interface consistent with attn decoder
#         1. param: encoder_outputs
#         2. return argument attn_weights
        attn_weights = torch.zeros(X.size(1), self.max_seq_size,1)
        
        return linear_out, hidden_state, attn_weights  
        
    def initHidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()), Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()))
        else:
            return Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())
        
class AttentionDecoder(nn.Module):
    def __init__(self, params):
        super(AttentionDecoder, self).__init__()
        input_dim = params['embed_size']
        hidden_dim = params['hidden_size']
        rnn_type = params['rnn_type']
        num_layers = params['num_layers']
        output_dim = params['target_vocab_size']
        self.max_seq_size = params['max_seq_size']
        self.params = params
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        self.attn = nn.Linear(self.hidden_dim * 2, 1)
        self.attn_combine = nn.Linear(self.hidden_dim*2,self.hidden_dim)
        self.rnn = getattr(nn, rnn_type)(self.input_dim, self.hidden_dim, num_layers)
        self.output = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, X, hidden_state, encoder_outputs, dropout = 0.5, isPacked=False):
        rnn_output, hidden_state = self.rnn(X, hidden_state)
        if isPacked:
            rnn_output, temp = nn.utils.rnn.pad_packed_sequence(rnn_output)
            
        #Attention
        key = hidden_state[0].transpose(0,1)
        combined_embedding = torch.cat([key.expand_as(encoder_outputs), encoder_outputs], dim=2)
        score = torch.tanh(self.attn(F.dropout(combined_embedding, dropout, training = self.training))).squeeze()
        attn_weights = F.softmax(score, dim=1)
        encoder_context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).transpose(0,1)
        final_embedding = self.attn_combine(torch.cat([encoder_context, rnn_output], dim=2))
        
        # Final predictions
        linear_out = self.output(final_embedding)
        
        return linear_out, hidden_state, attn_weights
        
    def initHidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()), Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()))
        else:
            return Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())
