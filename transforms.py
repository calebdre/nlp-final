import unicodedata
import string
import re
import random
from vocab import Vocab
from itertools import chain

def lower(data):
    return data.str.lower().str.strip()

def to_ascii(data):
    def fn(x):
        asci = ""
        for c in x:
            if unicodedata.category(c) != 'Mn':
                asci += unicodedata.normalize('NFD', c)
        return asci
    return data.apply(fn)

def normalize(data):
    def fn(x):
        x = re.sub(r"([.!?])", r" \1", x)
        x = re.sub(r"[^a-zA-Z.!?]+", r" ", x)
        return x
    
    return data.apply(fn)

def filter_length(data1, data2, n):
    data1 = data1[data1.apply(lambda x: len(x) <= n)]
    data2 = data2.loc[data1.index]
    return data1.reset_index(drop=True), data2.reset_index(drop=True)

def to_lang_vocab(data):
    return Vocab().build(data)