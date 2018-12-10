import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, hidden_size, batch_size, n_layers = 1, dropout= 0):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        
        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, batch_first = True, bidirectional=True)

    def forward(self, batch_input, hidden):
        embedded = self.embedding(batch_input)
        embedded = embedded.view(batch_input.shape[0], 1, -1) # gru expects [batch_size x seq_len x features]
        outputs, hidden = self.gru(embedded)
        
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        hidden = hidden[:self.n_layers]
        return outputs, hidden
    
    def init_hidden(self):
        return torch.zeros(self.batch_size, self.n_layers * 2, self.hidden_size)
    
    @property
    def output_size(self):
        return self.hidden_size