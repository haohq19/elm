# event (x, y, p, t)
import torch
import torch.nn as nn
import torch.nn.functional as F 

# implement a transformer for event data
class EventFrameEmbedding(nn.Module):
    def __init__(self, d_event, d_model, height, width):
        super(EventFrameEmbedding, self).__init__()
        self.d_event = d_event
        self.d_model = d_model
        self.height = height
        self.width = width
        self.conv0 = nn.Conv2d(in_channels=d_event, out_channels=d_event*2, kernel_size=3, stride=2, padding=1) 
        self.conv1 = nn.Conv2d(in_channels=d_event*2, out_channels=d_event*2, kernel_size=3, stride=2, padding=1)
        self.fc0 = nn.Linear(2 * (height // 4) * (width // 4), d_model)

    def forward(self, x):
        # x: (seq_len, batch_size, d_event, height, width)
        # output: (seq_len, batch_size, d_model)
        output = self.conv0(x)                                      # (seq_len, batch_size, d_event*2, height//2, width//2)
        output = F.relu(output)  
        output = self.conv1(output)                                 # (seq_len, batch_size, d_event*2, height//4, width//4)  
        output = F.relu(output)
        output = output.view(output.shape[0], output.shape[1], -1)  # (seq_len, batch_size, 2 * (height // 4) * (width // 4))
        output = self.fc0(output)                                   # (seq_len, batch_size, d_model)
        return output  

class EventTransformer(nn.Module):
    def __init__(self, d_event, d_model, height, width, nhead, num_layers, dim_feedforward, num_classes, dropout=0.1):
        super(EventTransformer, self).__init__()
        self.d_event = d_event
        self.d_model = d_model
        self.height = height
        self.width = width
        
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward

        self.num_classes = num_classes
        self.dropout = dropout

        self.embedding = EventFrameEmbedding(d_event, d_model, height, width)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (seq_len, batch_size, d_event, height, width)
        # output: (batch_size, d_model)
        output = self.embedding(x)  # (seq_len, batch_size, d_model)
        output = self.encoder(output)  # (seq_len, batch_size, d_model)
        output = torch.mean(output, dim=0)  # (batch_size, d_model)
        output = self.linear(output)  # (batch_size, num_classes)
        return output