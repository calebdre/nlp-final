import random 

class LangPair:
    def __init__(self, lang1, eos_idx1, lang2, eos_idx2):
        self.lang1 = lang1
        self.eos1 = eos_idx1
        
        self.lang2 = lang2
        self.eos2 = eos_idx2
        
        self.data_len = len(lang1)
    
    def get_sentence(self, n):
        return (
            self.lang1[n] + [self.eos1],
            self.lang2[n] + [self.eos2],
        )
    
    def get_rand_sentence(self):
        return self.get_sentence(random.randint(0, self.data_len))