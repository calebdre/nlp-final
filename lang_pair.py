import random 
import torch

class LangPair:
    def __init__(self, lang1, eos_idx1, lang2, eos_idx2):
        self.lang1 = lang1
        self.eos1 = eos_idx1
        
        self.lang2 = lang2
        self.eos2 = eos_idx2
        
        self.data_len = len(lang1)
    
    def get_sentence(self, n):
        s1 = self.lang1[n] + [self.eos1]
        s2 = self.lang2[n] + [self.eos2]
        
        return (
            torch.tensor(s1, dtype=torch.long).view(-1, 1),
            torch.tensor(s2, dtype=torch.long).view(-1, 1)
        )
    
    def get_rand_sentence(self):
        return self.get_sentence(random.randint(0, self.data_len))