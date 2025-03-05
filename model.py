import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(EncoderLayer, self).__init__()

        self.bn = nn.BatchNorm1d(num_features=in_channels)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1)
        self.relu = nn.ReLU()

        self.conv1x1 = nn.Conv2d(in_channels=out_channels, out_channels=2*out_channels, kernel_size=1)
        self.glu = nn.GLU(dim=1) # the channel dim


    def forward(self, x):
        x = self.bn(x)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.conv1x1(x.unsqueeze(-1))
        x = self.glu(x)

        return x.squeeze(-1)
    
class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DecoderLayer, self).__init__()
        
        self.bn = nn.BatchNorm1d(num_features=in_channels)
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=2*in_channels, kernel_size=1)
        self.glu = nn.GLU(dim=1) # the channel dim

        self.up_conv1d = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(x)
        x = self.conv1x1(x.unsqueeze(-1))
        x = self.glu(x)
        x = self.up_conv1d(x.squeeze(-1))
        x = self.relu(x)

        return x

    
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.linear = nn.Linear(embed_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)  # T, B, C

        attn_output, _ = self.attention(x, x, x)  # T, B, C
        x = self.norm1(x + attn_output)

        linear_output = self.linear(x)  # T, B, C
        x = self.norm2(x + linear_output)

        output = x.permute(1, 2, 0)  # B, C, T

        return output

class SmallCleanUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=4, kernel_size=3):
        super(SmallCleanUNet, self).__init__()

        # Encoder
        self.encoder1 = EncoderLayer(in_channels= in_channels, out_channels= depth*in_channels, kernel_size= kernel_size+2)
        self.encoder2 = EncoderLayer(in_channels= depth*in_channels, out_channels= 2*depth*in_channels, kernel_size= kernel_size+2)
        self.encoder3 = EncoderLayer(in_channels= 2*depth*in_channels, out_channels= 4*depth*in_channels, kernel_size= kernel_size)

        # Bottleneck with self-attention
        #self.bottleneck = SelfAttention(embed_dim=4*depth*in_channels, num_heads=8)
        self.bottleneck = nn.Conv1d(in_channels= 4*depth*in_channels, out_channels= 4*depth*in_channels, kernel_size= 3, padding= 3//2)

        # Decoder
        self.decoder1 = DecoderLayer(in_channels= 8*depth*in_channels, out_channels= 2*depth*in_channels, kernel_size= kernel_size)
        self.decoder2 = DecoderLayer(in_channels= 4*depth*in_channels, out_channels= depth*in_channels, kernel_size= kernel_size+2)
        self.decoder3 = DecoderLayer(in_channels= 2*depth*in_channels, out_channels= 2*in_channels, kernel_size= kernel_size+2)

        self.out_conv = nn.Sequential(
            nn.Conv1d(in_channels=2*in_channels, out_channels=2*in_channels, kernel_size=3, padding=3//2),
            nn.Conv1d(in_channels=2*in_channels, out_channels=out_channels, kernel_size=1)
        )


    def forward(self, x):
        x1 = self.encoder1(x.float())
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        b = self.bottleneck(x3)

        x4 = torch.cat([x3, b], dim=1)
        x5 = self.decoder1(x4)
        
        x6 = torch.cat([x2, x5], dim=1)
        x7 = self.decoder2(x6)

        x8 = torch.cat([x1, x7], dim=1)
        x9 = self.decoder3(x8)

        out = self.out_conv(x9)

        return out


if __name__ == "__main__":
    from random import randint
    model = SmallCleanUNet(in_channels=1, out_channels=1, depth=2, kernel_size=3)
    sample_input = torch.randn(8, 1, 22250*2)
    print("Input:", sample_input.shape)
    output = model(sample_input)
    print("Output:", output.shape, "\n")