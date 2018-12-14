import random 
import collections

import torch
from torch.nn.utils.rnn import pad_sequence

class LangPair:
    def __init__(self, lang1_vocab, lang2_vocab, debug = False, device = torch.device("cpu")):
        self.lang1_vocab = lang1_vocab
        self.lang2_vocab = lang2_vocab
        
        self.lang1 = lang1_vocab.get_idxs()
        self.lang2 = lang2_vocab.get_idxs()
        
        self.debug = debug
        self.device = device
        
        self.data_len = len(self.lang1) - 1
        
    def get_sent(self, n):
        s1 = self.lang1[n]
        s2 = [self.lang2_vocab.sos_idx] + self.lang2[n] + [self.lang2_vocab.eos_idx]
        
        return (
            torch.tensor(s1, dtype=torch.long, device = self.device),
            torch.tensor(s2, dtype=torch.long, device = self.device)
        )
    
    def get_sents(self, ns):
        return [self.get_sent(n) for n in ns]
    
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
    
    def batchify(self, size = 32, lang1 = None, lang2 = None, shuffle = True):
        combinations = {}
        if lang1 is None or lang2 is None:
            lang1, lang2 = self.lang1, self.lang2
        for i, (l1, l2) in enumerate(zip(lang1, lang2)):
            key = "{} {}".format(len(l1), len(l2))
            
            l1 = torch.tensor(l1, dtype=torch.long, device = self.device)
            
            l2 = [self.lang2_vocab.sos_idx] + l2 + [self.lang2_vocab.eos_idx]
            l2 = torch.tensor(l2, dtype=torch.long, device = self.device)
            
            
            if key in combinations.keys():
                combinations[key].append((l1,l2))
            else:
                combinations[key] = [(l1,l2)]
        
        batches = []
        for key in combinations.keys():
            comb = combinations[key]
            
            s1s = [s[0] for s in comb]
            s2s = [s[1] for s in comb]
            
            s = list(zip(s1s, s2s))
            random.shuffle(s)
            s1s, s2s = zip(*s)
            
            s1s, s2s = torch.stack(s1s), torch.stack(s2s)
            comb_length = s1s.shape[0]
            if comb_length >= size:
                num_comb_batches = int(comb_length / size)
                for i in range(1, num_comb_batches+1):
                    batched_s1 = s1s[(i-1) * size:i * size]
                    batched_s2 = s2s[(i-1) * size:i * size]
                    
                    batches.append((batched_s1, batched_s2))
            else:
                batches.append((s1s, s2s))

        if shuffle:
            random.shuffle(batches)
            
        return batches
    
    def get_rand_batch(self, size = 32):
        sent_idxs = random.choices(range(self.data_len), k =size)
        sents = [self.get_sent(idx) for idx in sent_idxs]
        
        s1s = [s[0] for s in sents]
        s2s = [s[1] for s in sents]
        
        s1s = pad_sequence(s1s, padding_value = self.lang1_vocab.pad_idx, batch_first = True)
        s2s = pad_sequence(s2s, padding_value = self.lang2_vocab.pad_idx, batch_first = True)
        
        return s1s, s2s,

    def get_all_as_batches(self, size = 32):
        num_batches = int(self.data_len / size)
        batches = [self.get_sents(range((i - 1) * size, i * size)) for i in range(1, num_batches)]
        
        prepped_batches = []
        for batch in batches:
            s1s = [b[0] for b in batch]
            s2s = [b[1] for b in batch]
            
            s1s = pad_sequence(s1s, padding_value = self.lang1_vocab.pad_idx, batch_first = True)
            s2s = pad_sequence(s2s, padding_value = self.lang2_vocab.pad_idx, batch_first = True)
            
            # shuffle
            s = list(zip(s1s, s2s))
            random.shuffle(s)
            s1s, s2s = zip(*s)
            s1s, s2s = torch.stack(s1s), torch.stack(s2s)
            
            prepped_batches.append((s1s, s2s))
            
        random.shuffle(prepped_batches)
        return prepped_batches