# event (x, y, p, t)
import torch
import torch.nn as nn
import torch.nn.functional as F 

# implement a transformer for event data
class EventTransformer(nn.Module):
    def __init__(self, d_event, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(EventTransformer, self).__init__()
        self.d_event = d_event
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.embedding = nn.Linear(d_event, d_model)  
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (seq_len, batch_size, d_event)
        # output: (batch_size, d_model)
        output = self.embedding(x)  # (seq_len, batch_size, d_model)
        output = self.encoder(output)  # (seq_len, batch_size, d_model)
        output = torch.sum(output, dim=0)  # (batch_size, d_model)
        output = self.linear(output)  # (batch_size, d_model)
        return output