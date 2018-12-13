import torch.nn as nn
import torch.nn.functional as F
import torch

class Decoder(nn.Module):
    def __init__(self, target_vocab_size, hidden_size, attn = None, n_layers = 1, dropout = 0):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(target_vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers = n_layers, batch_first = True, dropout = dropout)
        self.out = nn.Linear(hidden_size, target_vocab_size)
        
        if attn is not None:
            self.attn = attn
            self.attn_projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, batch_input, hidden, encoder_out):
        embedded = self.embedding(batch_input)
        embedded = self.dropout(embedded)
        embedded = embedded.view(batch_input.shape[0], 1, -1) # gru expects [batch_size x seq_len x features]
        
        # 7 x 256
        inner_rep, hidden = self.gru(embedded, hidden)
        
        if hasattr(self, "attn"):
            # 7 x 256 * 2
            attn, attn_weights = self.attn(inner_rep, encoder_out)
            attn_applied = self.attn_projection(attn)
            attn_applied = torch.tanh(attn_applied)
            out = self.out(attn_applied)
            
            dim = 0 if len(out.shape) == 1 else 1
            out = F.log_softmax(out, dim = dim)
            return out, hidden, attn_weights
        else:
            output = self.out(inner_rep.squeeze())
            dim = 0 if len(output.shape) == 1 else 1
            output = F.log_softmax(output, dim = dim)
            return output, hidden
        
    @property
    def has_attention(self):
        return hasattr(self, "attn")