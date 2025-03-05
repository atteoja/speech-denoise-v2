import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderLayer, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)

        return x

class CleanUNet(nn.Module):
    def __init__(self, coder_layers=3):
        super(CleanUNet, self).__init__()

        # Encoder
        self.encoder = nn.ModuleList()
        for i in range(coder_layers):
            self.encoder.append(EncoderLayer(1, 64))

        # Bottleneck with self-attention

        # Decoder
        self.decoder = nn.ModuleList()


    def forward(self, x):
        

        return x
