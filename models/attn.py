import torch
import torch.nn as nn
import torch.nn.functional as F
"""
[Effective Approaches to Attention-based Neural Machine Translation by Luong et al.](https://arxiv.org/pdf/1508.04025.pdf) describe a few more attention models that offer improvements and simplifications. They describe a few "global attention" models, the distinction between them being the way the attention scores are calculated.

The general form of the attention calculation relies on the target (decoder) side hidden state and corresponding source (encoder) side state, normalized over all states to get values summing to 1.

The specific "score" function that compares two states is either ; ; or 

The modular definition of these scoring functions gives us an opportunity to build specific attention module that can switch between the different score methods. The input to this module is always the hidden state (of the decoder RNN) and set of encoder outputs.

"""
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        self.cat = nn.Linear(hidden_size * 2, hidden_size)
        
        if self.method == 'general':
            self.score = self.score_general
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.score = self.score_concat
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
            
        elif self.method == "dot":
            self.score = self.score_dot

    def forward(self, hidden, encoder_outputs):
        # given...
        # input_len -> 6
        # target_len -> 7
        
        # then...
        # hidden   -> (target_len) 7 x 256
        # enc_outs -> (input_len)  6 x 256
        
                               #  7 x 256 @ 256 x 6        -> 7 x 6
        attn_weights = self.score(hidden  , encoder_outputs)
        attn_weights = F.log_softmax(attn_weights, dim = 1)

        #         256 x 6                        @  6 x 7                      -> 256 x 7 (or 7 x 256)
        context = encoder_outputs.transpose(1,2) @  attn_weights.transpose(1,2)
        attn = torch.cat([hidden, context.transpose(1,2)], dim = -1).squeeze(1)
        return attn, attn_weights
        
    # "dot, a simple dot product between the states"
    def score_dot(self, hidden, enc_out):
        return hidden @ enc_out.transpose(1,2)
    
    # "general, a dot product between the decoder hidden state and a linear transform of the encoder state"
    def score_general(self, hidden, enc_out):
        return hidden @ self.attn(enc_out).transpose(1,2)
    
    # "concat, a dot product between a new parameter $v_a$ and a linear transform of the states concatenated together"
    def score_concat(self, hidden, enc_out):
        catted = torch.cat((hidden, enc_out), 1)
        catted_transf = self.attn(catted)
        return self.v @ catted_transf