class Lang:
    sos_idx = 0
    eos_idx = 1
    
    def __init__(self, name):
        
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            self.sos_idx: "SOS", 
            self.eos_idx: "EOS"
        }
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1