from tqdm import tqdm
import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn

class Coach:
    def __init__(self, lang_pair, encoder, enc_optimizer, decoder, dec_optimizer, loss_fn):
        self.lang_pair = lang_pair
        
        self.encoder = encoder
        self.enc_optim = enc_optimizer
        
        self.decoder = decoder
        self.dec_optim = dec_optimizer
        
        self.loss_fn = loss_fn
    
    def train_encoder(self, input_batch):
        self.enc_optim.zero_grad()
        
        batch_size, input_len = input_batch.shape
        encoder_hiddens = torch.zeros(batch_size, input_len, self.encoder.hidden_size)
            
        for i in tqdm(range(input_len), desc = "Target Sample ({})".format(input_len), leave = False, unit = "token"):
            encoder_out, encoder_hidden = self.encoder(input_batch[:, i])
            encoder_hiddens[:, i] = encoder_hidden
            
        self.enc_optim.step()
        return encoder_hiddens
    
    def train_decoder(self, target_batch):
        self.dec_optim.zero_grad()
        loss = 0
        attns = []
        
        target_len = target_batch.shape[1]
        decoder_hidden = encoder_hiddens
        
        for i in tqdm(range(target_len), desc = "Decoder Sample ({})".format(target_len, leave = False, unit = "token"):
            dec_input = target_batch[:, i]
            decoder_out, decoder_hidden, att = decoder(dec_input, decoder_hidden, encoder_hiddens)
            
            loss += self.loss_fn(decoder_out, token)
            attns.append(attn)
        
        self.dec_optim.step()
        
        loss /= target_batch.shape[1]
        loss.backward()

        return loss.item(), att
        
    def train(self, iterations = 75000, print_interval = 1500, learning_rate = .01, batch_size = 32):
        losses = []
        
        for i in tqdm(range(1, iterations+1), desc = "Training Iterations", unit = "sample"):
            input_batch, target_batch = self.lang_pair.get_rand_batch(batch_size = batch_size)
            
            encoder_hiddens = self.train_encoder(input_batch)
            loss, attns = self.train_decoder(target_batch)
            
            if i % print_interval == 0:
                interval = i / print_interval
                
                interval_start = (interval - 1) * print_interval
                interval_loss = sum(losses[interval_start:i])
                avg_interval_loss = interval_loss / print_interval
                      
                print("Interval: {}\tAverage Loss: {}".format(interval, avg_interval_loss))
        
        return losses
         