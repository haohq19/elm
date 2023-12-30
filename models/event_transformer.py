import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import resnet18

# implement a transformer for event data
class EventFrameEmbedding(nn.Module):
    def __init__(self, in_channels, d_model):
        super(EventFrameEmbedding, self).__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.embedding = resnet18(num_classes=d_model)

    def forward(self, x):
        # x: (batch_size, nsteps, in_channels, height, width)
        # output: (batch_size, nsteps, d_model)
        batch_size = x.shape[0]
        nsteps = x.shape[1]
        x = x.reshape(nsteps * batch_size, x.shape[2], x.shape[3], x.shape[4])  # (batch_size * nsteps, in_channels, height, width) 
        x = self.embedding(x)                                                 # (batch_size * nsteps, d_model)
        return x.reshape(batch_size, nsteps, -1)                                # (batch_size, nsteps, d_model)


class EventTransformer(nn.Module):
    def __init__(self, in_channels, d_model, nhead, num_layers, dim_feedforward, seq_len, num_classes, dropout=0.1):
        super(EventTransformer, self).__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.dropout = dropout

        self.embedding = EventFrameEmbedding(in_channels, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (seq_len, batch_size, d_event, height, width)
        # output: (batch_size, d_model)
        output = self.embedding(x)  # (seq_len, batch_size, d_model)
        # output = output.mean(dim=0).squeeze()  # (seq_len, batch_size, d_model)
        output = self.encoder(output)  # (seq_len, batch_size, d_model)
        output = torch.mean(output, dim=0)  # (batch_size, d_model)
        output = self.linear(output)  # (batch_size, num_classes)
        return output
    

class EventResNet(nn.Module):
    def __init__(self, in_channels, d_model, num_classes):
        super(EventResNet, self).__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_classes = num_classes

        self.embedding = EventFrameEmbedding(in_channels, d_model)
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (seq_len, batch_size, d_event, height, width)
        # output: (batch_size, d_model)
        output = self.embedding(x)  # (seq_len, batch_size, d_model)
        output = output.mean(dim=0).squeeze()  # (seq_len, batch_size, d_model)
        output = self.linear(output)  # (batch_size, num_classes)
        return output