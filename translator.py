import torch

class TranslatorNetwork:
    def __init__(self, encoder, decoder, output_sos_token, max_output_length):
        self.encoder = encoder
        self.decoder = decoder
        self.max_output_length = max_output_length
        self.sos_token = self.output_sos_token
    
    # note: make sure @sentence has a start-of-sentence token
    def translate(self, sentence):
        with torch.no_grad():
            encoder_hiddens = torch.zeros(1, sentence.shape[0], self.encoder.hidden_size)
            for tqdm(range(len(sentence)), desc = "Encoding", unit = "token"):
                encoder_out, encoder_hidden = self.encoder(sentence.view(1, -1))
                encoder_hiddens[:, i] = encoder_hidden
    
    def beam_search(self, encoder_hiddens):
        pass
    
    def greedy_search(self, encoder_hiddens):
        inp = torch.tensor([self.sos_token])
        hidden = self.decoder.init_hidden()
        translation = []
        attns = []
        
        for i in range(self.max_output_length):
            preds, hidden, attn = decoder(
                inp, hidden, encoder_hiddens)
            
            next_token = preds.max()
            
            inp = torch.tensor([next_token])
            attns.append(attn)
            translation.append(next_token)
        
        return translation, attns