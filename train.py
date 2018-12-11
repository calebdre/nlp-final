from pipeline import Pipeline
from lang_pair import LangPair

from models.encoder import Encoder
from models.decoder import Decoder
from models.attn import Attn

from coach import Coach
import torch.optim as optim
import torch.nn as nn
import torch

import pandas as pd
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("vi_en_lang_pair.pkl", "rb") as f:
    vi_en_pair = torch.load(f)
    
batch_size = 25
learning_rate = .01
hidden_size = 150
embed_size = 300

enc_params = {
    "input_vocab_size": vi_en_pair.lang1_vocab.size,
    "hidden_size": hidden_size,
    "n_layers": 2,
    "dropout": .15,
    "embed_size": embed_size
}

dec_params = {
    "target_vocab_size": vi_en_pair.lang2_vocab.size,
    "hidden_size": hidden_size,
    "n_layers": 2,
    "dropout": .15,
}

attn_params = {
    "hidden_size": hidden_size,
    "method": "dot"
}

attn = Attn(**attn_params).to(device)
encoder = Encoder(**enc_params).to(device)
decoder = Decoder(**dec_params).to(device)
decoder_attn = Decoder(**dec_params, attn = attn).to(device)

enc_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
dec_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)
dec_attn_optimizer = optim.SGD(decoder_attn.parameters(), lr = learning_rate)
loss_fn = nn.NLLLoss()

coach_params = {
    "lang_pair": vi_en_pair, 
    "encoder": encoder, 
    "enc_optimizer": enc_optimizer, 
    "decoder": decoder, 
    "dec_optimizer": dec_optimizer, 
    "loss_fn": loss_fn
}

coach_attn_params = {
    **coach_params,
    "decoder": decoder_attn,
    "dec_optimizer": dec_attn_optimizer, 
}

coach = Coach(**coach_params)
coach_attn = Coach(**coach_attn_params)

rand_training_params = {
    "learning_rate": learning_rate,
    "iterations": 10,
    "print_interval": 10000,
    "batch_size": batch_size
}

epoch_training_params = {
    "num_epochs": 2,
    "print_interval": 5000,
    "learning_rate": learning_rate,
    "batch_size": batch_size
}

print("***************\nTraining w/o attention\n***************\n")
losses, _ = coach.train_random(**rand_training_params)

print("\n\n***************\nTraining with attention\n***************\n")
attn_losses, attns = coach.train_random(**rand_training_params)

info = {
    "coach": coach,
    "attn_coach": coach_attn,
    "losses": losses,
    "attn_losses":attn_losses,
    "params": {
        "enc": enc_params,
        "dec": dec_params,
        "attn": attn_params,
        "training": rand_training_params,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "embeding_size": embed_size
    }
}

print("Saving...")
with open("training_run_{}.pkl".format(time.time()), "wb") as f:
    torch.save(info, f)