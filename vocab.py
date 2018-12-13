from collections import Counter

class Vocab:
    unk_token = "<unk>"
    
    def build(self, data):
        all_tokens = " ".join(data).split(" ")
        
        counter = Counter(all_tokens)
        counter = Counter({k: c for k, c in counter.items() if c >= 15})
        vocab = list(counter)
        
        vocab.append("<sos>")
        self.sos_idx = len(vocab) - 1
        
        vocab.append("<eos>")
        self.eos_idx = len(vocab) - 1
        
        vocab.append("<pad>")
        self.pad_idx = len(vocab) - 1
        
        vocab.append(self.unk_token)
        self.unk_idx = len(vocab) - 1
        
        self.data = data
        self.idx_token = vocab
        self.token_idx = dict(zip(vocab, range(len(vocab))))
        
        return self
    
    @property
    def size(self):
        return len(self.idx_token)
    
    def from_idx(self, idx):
        return self.idx_token[idx]

    def from_idxs(self, idxs):
        return [self.from_idx(idx) for idx in idxs]
    
    def to_idx(self, token):
        if token in self.token_idx:
            return self.token_idx[token]
        else:
            return self.token_idx[self.unk_token]
    
    def to_idxs(self, tokens):
        return [self.to_idx(token) for token in tokens]
    
    def get_idxs(self):
        return [self.to_idxs(d) for d in self.data.str.split(" ")]