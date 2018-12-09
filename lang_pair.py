import random 
import torch
import collections

class LangPair:
    def __init__(self, lang1, sos_idx1, lang2, sos_idx2):
        self.lang1 = lang1
        self.sos1 = sos_idx1
        
        self.lang2 = lang2
        self.sos2 = sos_idx2
        
        self.data_len = len(lang1) - 1
        self.lang1_sent_lengths = list(set([len(sent) for sent in lang1]))
        self.lang2_sent_lengths = list(set([len(sent) for sent in lang2]))
    
    def get_sent(self, n):
        s1 = self.lang1[n]
        s2 = [self.sos2] + self.lang2[n]
        
        return (
            torch.tensor(s1, dtype=torch.long),
            torch.tensor(s2, dtype=torch.long)
        )
    
    def get_rand_sent(self, lengths, max_iters = 100):
        if not isinstance(lengths, collections.Iterable):
            raise "'lengths' must be a 2-element iterable"
        
        l1, l2 = lengths
        
        for i in range(max_iters):
            s1, s2 = self.get_sent(random.randint(0, self.data_len))
            if s1.shape[0] != l1 or s2.shape[0] != l2:
                continue
            
            return s1, s2
        
        return None

    def get_rand_batch(self, size = 32):
        while True:
            len1 = random.choice(self.lang1_sent_lengths)
            len2 = random.choice(self.lang2_sent_lengths)
            
            batch1 = []
            batch2 = []
            
            while len(batch1) < size:
                sents = self.get_rand_sent((len1, len2))
                if sents is not None:
                    s1, s2 = sents
                    batch1.append(s1)
                    batch2.append(s2)
            
            
            return torch.stack(batch1), torch.stack(batch2)
                