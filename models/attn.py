import torch
import torch.nn as nn
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
        """
        First we calculate a set of attention weights. 
        Calculating the attention weights is done with another feed-forward layer attn, using the decoder's input and hidden state as inputs.
        
        These will be multiplied by the encoder output vectors to create a weighted combination. 
        The result (called attn_applied in the code) should contain information about that specific part of the input sequence, and thus help the decoder choose the right output words.
        """ 


        
        # Create variable to store attention energies

        # For each batch of encoder outputs
        # Calculate energy for each encoder output
        
        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        
        # Return context vectors
        return None
    
    # "dot, a simple dot product between the states"
    def score_dot(self, hidden, enc_out):
        return hidden @ enc_out
    
    # "general, a dot product between the decoder hidden state and a linear transform of the encoder state"
    def score_general(self, hidden, enc_out):
        return hidden @ self.attn(enc_out)
    
    # "concat, a dot product between a new parameter $v_a$ and a linear transform of the states concatenated together"
    def score_concat(self, hidden, enc_out):
        catted = torch.cat((hidden, enc_out), 1)
        catted_transf = self.attn(catted)
        return self.v @ catted_transf