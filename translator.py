from tqdm import tqdm, tqdm_notebook
import torch
import sacrebleu

class Translator:
    def __init__(self, encoder, decoder, lang_pair, device = torch.device("cpu"),  max_output_length = 150, is_notebook = False):
        self.encoder = encoder
        self.decoder = decoder
        
        self.max_output_length = max_output_length
        self.sos_idx = lang_pair.lang2_vocab.sos_idx
        self.lang_pair = lang_pair
        
        self.device = device
        if is_notebook:
            self.tqdm = tqdm_notebook
        else:
            self.tqdm = tqdm
    
    def score_translations(self, inputs, targets):
        return sacrebleu.corpus_bleu(inputs, [targets]).score * 100
    
    def score_corpus(self, inputs, targets, n):
        inputs = [self.lang_pair.lang1_vocab.to_idxs(input) for input in inputs]
        targets_idxs = [self.lang_pair.lang1_vocab.to_idxs(target) for target in targets]
        
        batches = self.lang_pair.batchify(lang1 = inputs, lang2 = targets_idxs)
        
        translations, targets, attns = self.translate(batches[:n])
        score = self.score_translations(translations, targets)
        return score, translations, targets, attns
    
    def translate(self, sentence_batches, method = "greedy"):
        translations = []
        targets = []
        for inputs, targets_ in self.tqdm(sentence_batches, "Corpus Score", leave = False, unit = "batch"):
            enc_outs, enc_hidden = self.encode(inputs.to(self.device))

            if method == "greedy":
                batch_translations, attns = self.greedy_search(enc_outs, enc_hidden)
            elif method == "beam":
                translations, attns = self.beam_search(enc_outs, enc_hidden)
            else:
                raise "No such method '{}'".forat(method)

            batch_translations = batch_translations.cpu().numpy()            
            batch_translations = [" ".join(self.lang_pair.lang2_vocab.from_idxs(translation)) for translation in batch_translations]
            
            targets_ =  [" ".join(self.lang_pair.lang2_vocab.from_idxs(p)) for p in targets_]
            translations += batch_translations
            targets += targets_
        
        
        return translations, targets, attns
    
    def encode(self, inputs):
        with torch.no_grad():
            batch_size, seq_len = inputs.shape
            hidden = self.encoder.init_hidden(batch_size).to(self.device)
            outs = torch.zeros(batch_size, seq_len, self.encoder.hidden_size, device = self.device)
            
            for i in range(seq_len):
                out, hidden, _ = self.encoder(inputs[:, i], hidden)
                outs[:, i] = out[:, 0]
            
            return outs, hidden
    def beam_search(self, encout_out, encoder_hidden):
        translations = torch.tensor([self.sos_idx for i in range(encoder_out.shape[0])], device = self.device).view(encoder_out.shape[0], -1)
        attns = []
        hidden = encoder_hidden[:self.decoder.n_layers]
        while i < self.max_output_length and (translations == self.lang_pair.lang2_vocab.eos_idx).sum().item() != encoder_out.shape[0]:
                if self.decoder.has_attention:
                    preds, hidden, attn = self.decoder(translations[:, i], hidden, encoder_out)
                    attns.append(attn)
                else:
                    preds, hidden = self.decoder(translations[:, i], hidden, encoder_out)
    
    def greedy_search(self, encoder_out, encoder_hidden):
        translations = torch.tensor([self.sos_idx for i in range(encoder_out.shape[0])], device = self.device).view(encoder_out.shape[0], -1)
        attns = []
        hidden = encoder_hidden[:self.decoder.n_layers]
        
        with torch.no_grad():
            i = 0
            while i < self.max_output_length and (translations == self.lang_pair.lang2_vocab.eos_idx).sum().item() != encoder_out.shape[0]:
                if self.decoder.has_attention:
                    preds, hidden, attn = self.decoder(translations[:, i], hidden, encoder_out)
                    attns.append(attn)
                else:
                    preds, hidden = self.decoder(translations[:, i], hidden, encoder_out)
                
                dim = 1 if preds.dim() == 2 else 0
                pred_prob, pred_idxs = torch.max(preds, dim)
                translations = torch.cat([translations, pred_idxs.view(translations.shape[0], -1)], dim = 1)
                i += 1
                
        return translations, attns