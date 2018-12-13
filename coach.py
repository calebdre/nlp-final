from tqdm import tqdm as _tqdm, tqdm_notebook
import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import random

class Coach:
    def __init__(self, lang_pair, encoder, enc_optimizer, decoder, dec_optimizer, loss_fn, device = torch.device("cpu"), is_notebook = False):
        self.lang_pair = lang_pair
        
        self.encoder = encoder
        self.enc_optim = enc_optimizer
        
        self.decoder = decoder
        self.dec_optim = dec_optimizer
        
        self.loss_fn = loss_fn
        self.device = device
        
        if is_notebook:
            self.tqdm = tqdm_notebook
        else:
            self.tqdm = _tqdm
    
    def train_random(self, iterations = 75000, print_interval = 1500, learning_rate = .01, batch_size = 32):
        losses = []
        batch_attentions = []
        interval_losses = []
        iterations = int(iterations / batch_size)
        print_interval = int(print_interval / batch_size)
        
        print("Fetching batches...\n")
        batches = self.lang_pair.batchify(size = batch_size)
        batches = random.choices(batches, k = iterations)  
        for i, (input_batch, target_batch) in enumerate(self.tqdm(batches, desc = "Training Iterations", unit = " batch")):
            self.dec_optim.zero_grad()
            self.enc_optim.zero_grad()
                
            encoder_out, encoder_hidden = self.train_encoder(input_batch)
            loss, attns = self.train_decoder(target_batch, encoder_hidden, encoder_out)
    
            loss = loss / target_batch.shape[1]
            loss.backward()
                
            self.dec_optim.step()
            self.enc_optim.step()

            losses.append(loss)
            interval_losses.append(loss)
            batch_attentions.append(attns)
            
            if i > 0 and i % print_interval == 0:
                interval = int(i / print_interval)
                total_intervals = int(iterations / print_interval)
                avg_interval_loss = sum(interval_losses) / len(interval_losses)
                m = "Interval ({}/{}) average loss: {:.4f}".format(interval, total_intervals, avg_interval_loss)
                self.tqdm.write(m)
                interval_losses = []
        
        return losses, batch_attentions
    
    def train_epochs(self, num_epochs = 10, print_interval = 1500, learning_rate = .01, batch_size = 32, percent_of_data = .6):
        losses = []
        batch_attentions = []
        interval_losses = []
        iterations = self.lang_pair.data_len * num_epochs
        num_intervals = iterations / print_interval
        
        print("Fetching batches...\n")
        batches = self.lang_pair.batchify(size = batch_size)
        
        for epoch in self.tqdm(range(num_epochs), desc = "Epochs", unit = " epoch", leave = False):
            sampled_batches = random.sample(batches, k = int(len(batches) * percent_of_data))
            
            for i, (input_batch, target_batch) in enumerate(self.tqdm(sampled_batches, leave = False, desc = "Batches", unit = " batch")):
                self.dec_optim.zero_grad()
                self.enc_optim.zero_grad()
                
                encoder_out, encoder_hidden = self.train_encoder(input_batch)
                loss, attns = self.train_decoder(target_batch, encoder_hidden, encoder_out)
                
                loss = loss / target_batch.shape[1]
                loss.backward()
                
                self.dec_optim.step()
                self.enc_optim.step()

                losses.append(loss)
                interval_losses.append(loss)
                batch_attentions.append(attns)
                
                if i > 0 and i % print_interval == 0:
                    interval = int(i / print_interval)
                    avg_interval_loss = sum(interval_losses) / len(interval_losses)
                    
                    m = "Epoch [{}/{}]\tInterval [{}/{}]\t Average Loss: {}".format(
                       epoch, num_epochs, interval, num_intervals, avg_interval_loss 
                    )
                    self.tqdm.write(m)
                    interval_losses = []
        return losses, batch_attentions
    
    def train_encoder(self, input_batch):
        batch_size, input_len = input_batch.shape
        outs = torch.zeros(batch_size, input_len, self.encoder.output_size, device = self.device)
        hidden = self.encoder.init_hidden(batch_size).to(self.device)
        
        for i in range(input_len):
            out, hidden = self.encoder(input_batch[:, i], hidden)
            outs[:, i] = out[:, 0]

        return outs, hidden[:self.encoder.n_layers]
    
    def train_decoder(self, target_batch, encoder_hidden, encoder_out):
        loss = 0
        attns = []
        
        target_len = target_batch.shape[1]
        hidden = encoder_hidden
        
        for i in range(target_len):
            dec_input = target_batch[:, i]
            if self.decoder.has_attention:
                out, hidden, attn_weights = self.decoder(dec_input, hidden, encoder_out)
                attns.append(attn_weights)
            else:
                out, hidden = self.decoder(dec_input, hidden, encoder_out)
                
            if len(out.shape) == 1:
                out = out.view(1, -1)
            
            loss += self.loss_fn(out, target_batch[:, i])
            
        return loss, attns
    