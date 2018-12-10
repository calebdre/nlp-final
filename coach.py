from tqdm import tqdm
import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn

class Coach:
    def __init__(self, lang_pair, encoder, enc_optimizer, decoder, dec_optimizer, loss_fn, device = torch.device("cpu")):
        self.lang_pair = lang_pair
        
        self.encoder = encoder
        self.enc_optim = enc_optimizer
        
        self.decoder = decoder
        self.dec_optim = dec_optimizer
        
        self.loss_fn = loss_fn
        self.device = device
    
    def train(self, iterations = 75000, print_interval = 1500, learning_rate = .01, batch_size = 32):
        losses = []
        interval_losses = []
        iterations = int(iterations / batch_size)
        print_interval = int(print_interval / batch_size)
        
        for i in tqdm(range(1, iterations+1), desc = "Training Iterations", unit = "batch"):
            input_batch, input_batch_lengths, target_batch, target_batch_lengths = self.lang_pair.get_rand_batch(size = batch_size)
            encoder_out, encoder_hidden = self.train_encoder(input_batch, input_batch_lengths)
#             loss, attns = self.train_decoder(target_batch, encoder_hidden, encoder_out)
            loss = self.train_decoder(target_batch, encoder_hidden, encoder_out)
            losses.append(loss)
            interval_losses.append(loss)
            
            if i % print_interval == 0:
                interval = int(i / print_interval)
                total_intervals = int(iterations / print_interval)
                avg_interval_loss = sum(interval_losses) / len(interval_losses)
                m = "Interval ({}/{}) average loss: {:.4f}".format(interval, total_intervals, avg_interval_loss)
                tqdm.write(m)
                interval_losses = []
        
        return losses
    
    def train_encoder(self, input_batch, input_batch_lengths):
        self.enc_optim.zero_grad()
        
        batch_size, input_len = input_batch.shape
        outs = torch.zeros(batch_size, input_len, self.encoder.output_size, device = self.device)
        hidden = self.encoder.init_hidden()
        for i in range(input_len):
            out, hidden = self.encoder(input_batch[:, i], hidden)
            outs[:, i] = out[:, 0]
            
        self.enc_optim.step()
        return outs, hidden
    
    def train_decoder(self, target_batch, encoder_hidden, encoder_out):
        self.dec_optim.zero_grad()
        loss = 0
#         attns = []
        
        target_len = target_batch.shape[1]
        hidden = encoder_hidden
        
        for i in range(target_len):
            dec_input = target_batch[:, i]
            out, hidden = self.decoder(dec_input, hidden, encoder_out)
            
            loss += self.loss_fn(out, target_batch[:, i])
#             attns.append(attn)
        
        self.dec_optim.step()
        loss = loss / target_batch.shape[1]
        loss.backward()

#         return loss.item(), att
        return loss.item()
    