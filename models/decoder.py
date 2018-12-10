import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embedding_size, hidden_size, batch_size, n_layers = 1, dropout = 0):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding = nn.Embedding(target_vocab_size, embedding_size)
        
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers = n_layers, batch_first = True, dropout = dropout)
        self.out = nn.Linear(hidden_size, target_vocab_size)

    def forward(self, batch_input, hidden, encoder_out):
        embedded = self.embedding(batch_input)
        embedded = embedded.view(batch_input.shape[0], 1, -1) # gru expects [batch_size x seq_len x features]
        
        inner_rep, hidden = self.gru(embedded, hidden)
        inner_rep = inner_rep.squeeze()
        output = self.out(inner_rep)
        output = F.log_softmax(output, dim = 1)
        return output, hidden

    def init_hiden(self):
        return torch.zeros(self.batch_size, self.hidden_size)