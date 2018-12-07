class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):

        # Create variable to store attention energies

        # For each batch of encoder outputs
        # Calculate energy for each encoder output
        
        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        
        # Return context vectors
        return None
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            ## TODO implement
            return energy
        
        elif self.method == 'general':
            energy = None
            ## TODO implement 
            return energy
        
        elif self.method == 'concat':
            energy = None
            ## TODO implement 
            return energy