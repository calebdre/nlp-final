from tqdm import tqdm
import torch
import sacrebleu

class Translator:
    def __init__(self, encoder, decoder, lang_pair, device = torch.device("cpu"),  max_output_length = 2000):
        self.encoder = encoder
        self.decoder = decoder
        
        self.max_output_length = max_output_length
        self.sos_idx = lang_pair.lang2_vocab.sos_idx
        self.lang_pair = lang_pair
        
        self.device = device
    
    def score_translations(self, inputs, targets):
        return sacrebleu.corpus_bleu(inputs, targets).score * 100
    
    def score_corpus(self, inputs, targets):
        translations = [self.translate(input) for input in inputs]
        score = self.score_translations(translations, targets)
        
        return score, translations
    
    def translate(self, sentence, method = "greedy"):
        if isinstance(sentence, list):
            sentence = torch.tensor(sentence).long()
            
        enc_outs, enc_hidden = self.encode(sentence)
        
        if method == "greedy":
            translation, attns = self.greedy_search(enc_outs, enc_hidden)
        elif method == "beam":
            translation, attns = self.beam_search(enc_outs, enc_hidden)
        else:
            raise "No such method '{}'".forat(method)
        
        translation = translation[1:]
        translation = " ".join(self.lang_pair.lang2_vocab.from_idxs(translation))
        sentence = " ".join(self.lang_pair.lang1_vocab.from_idxs(sentence))
        
        return sentence, translation, attns
    
    def encode(self, sentence):
        with torch.no_grad():
            hidden = self.encoder.init_hidden(1).to(self.device)
            outs = torch.zeros(1, sentence.shape[0], self.encoder.hidden_size, device = self.device)
            
            for i in tqdm(range(len(sentence)), desc = "Encoding Input", unit = "token"):
                out, hidden = self.encoder(sentence[i].view(1,1), hidden)
                outs[0, i] = out[0, 0]
            
            return outs, hidden[:self.encoder.n_layers]
        
    def beam_search(self, encout_out, encoder_hidden):
        pass
    
    def greedy_search(self, encoder_out, encoder_hidden):
        translation = [self.sos_idx]
        attns = []
        hidden = encoder_hidden
        
        with torch.no_grad():
            for i in tqdm(range(self.max_output_length), desc = "Decoding", unit = "token"):
                inp = torch.tensor([translation[-1]]).long().view(1,1)
                if self.decoder.has_attention:
                    preds, hidden, attn = self.decoder(inp, hidden, encoder_out)
                    attns.append(attn)
                else:
                    preds, hidden = self.decoder(inp, hidden, encoder_out)

                pred_prob, pred_idx = torch.max(preds, 0)
                translation.append(pred_idx.item())
                
        return translation, attns