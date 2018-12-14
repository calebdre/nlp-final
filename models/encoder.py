import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embed_size, hidden_size, n_layers = 1, dropout= 0, use_self_attn = False):
        super(Encoder, self).__init__()
        
        self.use_self_attn = use_self_attn
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, batch_first = True, bidirectional=True)
        if use_self_attn:
            self.w_s1 = nn.Linear(hidden_size * 2, hidden_size * 2)
            self.w_s2 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, batch_input, hidden):
        batch_size = batch_input.shape[0]
        embedded = self.embedding(batch_input)
        
        embedded = embedded.view(batch_size, 1, -1) # gru expects [batch_size x seq_len x features]
        if hidden.dim() == 2:
            hidden = hidden.repeat((2, 1, 1))

        outputs, hidden = self.gru(embedded, hidden)
        
        hidden = hidden.view(self.n_layers, 2, batch_size, self.hidden_size)
        hidden = hidden.sum(0).squeeze(0)
        
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        
        if self.use_self_attn:
            hidden = torch.cat([hidden[0], hidden[1]], dim = 1)
            hidden_attn = self.apply_self_attn(hidden, batch_size)
            return outputs, hidden, hidden_attn
        else:
            return outputs, hidden, None
    
    def apply_self_attn(self, hidden, batch_size): 
        self_attn = torch.tanh(self.w_s1(hidden))
        self_attn = self.w_s2(self_attn)
        self_attn = F.softmax(self_attn, dim = 1)
        applied = self_attn.transpose(0,1) @ hidden
        return applied
        
    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers * 2, batch_size, self.hidden_size)
    
    @property
    def output_size(self):
        return self.hidden_size