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

def main(
    lang_pair_path = "vi_en_lang_pair.pkl", 
    lang_pair_valid_path = "vi_en_valid_lang_pair.pkl",
    batch_size = 25,
    learning_rate = .01,
    hidden_size = 150,
    embed_size = 300,
    validate_first = 200,
    enc_layers = 1,
    dec_layers = 1,
    enc_dropout = 0,
    dec_dropout = 0,
    use_attn = False,
    use_self_attn = False,
    save_filename = "training_run_{}.pkl".format(int(time.time())),
    iterations = 200000,
    print_interval = 25000,
    num_epochs = None,
    is_notebook = False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(lang_pair_path, "rb") as f:
        lang_pair = torch.load(f)
    
    with open(lang_pair_valid_path, "rb") as f:
        lang_pair_valid = torch.load(f)

    lang_pair.device = device

    enc_params = {
        "input_vocab_size": lang_pair.lang1_vocab.size,
        "hidden_size": hidden_size,
        "use_self_attn": use_self_attn,
        "n_layers": enc_layers,
        "dropout": enc_dropout,
        "embed_size": embed_size
    }

    dec_params = {
        "target_vocab_size": lang_pair.lang2_vocab.size,
        "hidden_size": hidden_size,
        "n_layers": dec_layers,
        "dropout": dec_dropout,
    }

    attn_params = {
        "hidden_size": hidden_size,
        "method": "general"
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
        "lang_pair": lang_pair, 
        "lang_pair_valid": lang_pair_valid,
        "encoder": encoder, 
        "enc_optimizer": enc_optimizer, 
        "decoder": decoder, 
        "dec_optimizer": dec_optimizer, 
        "loss_fn": loss_fn,
        "device": device,
        "is_notebook": is_notebook
    }

    coach_attn_params = {
        **coach_params,
        "decoder": decoder_attn,
        "dec_optimizer": dec_attn_optimizer, 
    }

    rand_training_params = {
        "iterations": iterations,
        "print_interval": print_interval,
        "batch_size": batch_size,
        "validate_first": validate_first
    }

    epoch_training_params = {
        "num_epochs": 10,
        "print_interval": print_interval,
        "batch_size": batch_size,
        "percent_of_data": 1,
        "validate_first":validate_first
    }

    info = {
        "use_attn": use_attn,
        "enc": enc_params,
        "dec": dec_params,
        "training": epoch_training_params,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "embeding_size": embed_size
    }

    if use_attn:
        print("***************\nTraining with attention\n***************\n")
        coach = Coach(**coach_attn_params)
    else:
        print("***************\nTraining w/o attention\n***************\n")
        coach = Coach(**coach_params)

    
    if num_epochs is not None:
        train_func = coach.train_epochs
        params = epoch_training_params
        params["num_epochs"] = num_epochs
    else:
        train_func = coach.train_random
        params = rand_training_params
        params["iterations"] = iterations
    
    params["print_interval"] = print_interval
    losses, bleu_scores, attns = train_func(**params)

    info["coach"] = coach
    info["losses"] = losses
    info["bleu_scores"] = bleu_scores

    if len(attns) > 0:
        info["attns"] = attns

    print("Saving...")
    with open(save_filename, "wb") as f:
        torch.save(info, f)
    
    return losses, attns, info

if __name__ == "__main__" :
    main()