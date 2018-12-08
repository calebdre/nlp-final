from collections import Counter

class Vocab:
    def build(self, data):
        vocab = " ".join(data).split(" ")
        
        vocab.append("SOS")
        self.sos_idx = len(vocab) - 1
        vocab.append("EOS")
        self.eos_idx = len(vocab) - 1
        
        self.data = data
        self.idx_token = vocab
        self.token_idx = dict(zip(vocab, range(0,len(vocab))))
        return self
    
    def from_idx(self, idx):
        return self.idx_token[idx]
    
    def from_idxs(self, idxs):
        return [self.from_idx(idx) for idx in idxs]
    
    def to_idx(self, token):
        return self.token_idx[token]
    
    def to_idxs(self, tokens):
        return [self.to_idx(token) for token in tokens]
    
    def get_idxs(self):
        return [self.to_idxs(d) for d in self.data.str.split(" ")]