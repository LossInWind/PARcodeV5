"""
下面是一个Seq2Seq模型架构：
编码器：采用Transformer结构对前三帧的多普勒-时间图进行编码。在编码器的输入端，可以使用一个位置编码器（Positional Encoding）将时间维度的信息编码到输入序列中。

解码器：采用DNN结构，将编码器的输出解码为第四帧的多普勒-时间图。在解码器的输入端，也可以使用一个位置编码器将时间维度的信息编码到输入序列中。解码器需要逐步地生成第四帧多普勒-时间图，每个时间步生成一个像素值，直到生成整个第四帧。

训练：在训练时，可以采用Teacher-Forcing策略，即将真实的第四帧多普勒-时间图作为解码器每个时间步的输入，来加速模型的训练。损失函数可以使用像素级别的MSE损失函数来评估生成的第四帧多普勒-时间图与真实第四帧多普勒-时间图之间的差异。

推断：在推断时，可以使用自回归（Autoregressive）策略，即将解码器的前一个时间步的输出作为下一个时间步的输入，逐步生成第四帧多普勒-时间图。可以采用Beam Search等技术来搜索最优的输出序列。
"""
# 将前三帧的多普勒-时间图看作输入序列，将第四帧的多普勒-时间图看作输出序列

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :]
        x = x + pe.to(x.device)
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, dropout):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, n_heads, hidden_dim * 4, dropout), n_layers)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = F.relu(self.fc(output))
        return output


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, n_heads, dropout):
        # super(EncoderDecoder, self).init()
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.encoder = Encoder(input_dim, hidden_dim, n_layers, n_heads, dropout)
        self.decoder = Decoder(hidden_dim, hidden_dim, output_dim)

    def forward(self, src):
        # Split input into the first three frames and the fourth frame
        encoder_input = src[:, :3, :]
        decoder_input = src[:, 3, :]

        # Encoder
        encoder_output = self.encoder(encoder_input)

        # Decoder
        outputs = []
        for t in range(src.size(1)):
            decoder_output = self.decoder(decoder_input)
            outputs.append(decoder_output.unsqueeze(1))
            decoder_input = decoder_output

        # Concatenate the encoder and decoder output and pass it through a linear layer
        output = torch.cat(outputs, dim=1)
        output = F.relu(self.fc(output))

        return output



if __name__ == "__main__":
    #测试网络结构是否合理
    rand_input = torch.rand((8, 3, 16, 256, 256))
