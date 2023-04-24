
import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
            nn.BatchNorm2d(in_channel),

        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class Encoder(nn.Module):  # 输入形状为(64,1,224,224) 输出为[64, 128, 56, 56]
    # in_channel: 输入通道数 1
    # channel: 输出通道数 128
    # n_res_block: ResBlock的数量 2
    # n_res_channel: ResBlock中间层的通道数 32
    # stride: 步长 4
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()


        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3, stride=1, padding=1),  # 输入形状为(64,1,224,224) 输出为[64, 16, 224, 224]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出为[64, 16, 112, 112]

            nn.Conv2d(16, 32, 3, 1, 1),  # 输出为[64, 32, 112, 112]
            nn.BatchNorm2d(32),  # 输出为[64, 32, 112, 112]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出为[64, 32, 56, 56]

            nn.Conv2d(32, 64, 3, 1, 1),  # 输出为[64, 64, 56, 56]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2), #输出为[64, 64, 28, 28]

            nn.Conv2d(64, channel, 3, 1, 1),  # 输出为[64, 128, 56, 56]
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2), #输出为[64, 128, 14, 14]
        )


    def forward(self, input):
        input = self.cnn(input)
        return input

class Quantizer(nn.Module)  :
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim               # 输入的维度
        self.n_embed = n_embed       # 嵌入向量的数量
        self.decay = decay           # 衰减因子
        self.eps = eps               # 聚类中防止分母为零的平滑因子

        # 初始化嵌入向量
        embed = torch.randn(dim, n_embed)
        # 注册嵌入向量、簇大小、嵌入向量的均值向量为 buffer
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input): #输入形状为[64, 56, 56，64]
        # 将输入展平为二维张量
        flatten = input.reshape(-1, self.dim) # [B, C, H, W] -> [B*H*W, C]  [64, 56, 56，64] -> [64*56*56, 64]
        # 计算每个向量和嵌入向量的欧氏距离
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        # 找到与输入向量距离最近的嵌入向量的索引
        _, embed_ind = (-dist).max(1)
        # 将嵌入向量索引转化为 one-hot 表示
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        # 将嵌入向量索引重构为与输入相同的形状
        embed_ind = embed_ind.view(*input.shape[:-1])
        # 获取与输入最接近的嵌入向量
        quantize = self.embed_code(embed_ind)

        if self.training:
            # 统计每个嵌入向量被使用的次数
            embed_onehot_sum = embed_onehot.sum(0)
            # 计算每个嵌入向量的权重
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # 更新簇大小和均值向量
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            # 计算每个簇的相对大小
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )

            # 计算嵌入向量的归一化版本
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        # 计算输入与最接近嵌入向量的差距

        diff1 = (quantize.detach() - input).pow(2).mean()
        diff2 = (quantize - input.detach()).pow(2).mean()
        diff = diff1 * 0.25 + diff2
        # 更新量化后的向量
        quantize = input + (quantize - input).detach()

        # 返回量化后的向量、差距和嵌入向量索引
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))
        # 首先，使用 PyTorch 的 F.embedding 函数将嵌入ID转换为嵌入向量，其中 self.embed.transpose(0, 1) 将嵌入向量的维度从 [dim, n_embed] 转换为 [n_embed, dim]。返回转换后的嵌入向量张量。
class VQE(nn.Module):
    def __init__(
            self,
            in_channel=1,         # 输入图像的通道数，默认为1
            channel=128,          # 编码器和解码器中的通道数，默认为128
            n_res_block=1,        # 残差块的数量，默认为2
            n_res_channel=32,     # 残差块的通道数，默认为32
            embed_dim=128,         # 嵌入向量的维度，默认为64
            n_embed=1024,          # 嵌入向量的数量，默认为512
    ):
        super().__init__()

        # 定义模型中的各个模块
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.quantize_conv_b = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_b = Quantizer(embed_dim, n_embed)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim * 56 * 56, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 11),
        )



    def encode(self, input):
        # 输入为图像，返回编码后的高层特征和低层特征，以及两者的距离和索引
        # print(input.shape)
        enc_b = self.enc_b(input)       # 特征提取 [64, 128, 56, 56]
        # print(enc_b.shape)
        # enc_b = enc_b.permute(0, 2, 3, 1)  # 调整形状为 [64, 56, 56, 128]
        quant_b = self.quantize_conv_b(enc_b).permute(0,2,3,1) # 经过一层卷积，将通道数从128降低到embed_dim，即64 此时输入为[64, 56, 56, 64]

        # 量化操作
        quant_b, diff_b, id_b = self.quantize_b(quant_b)              # 计算距离和索引
        quant_b = quant_b.permute(0, 3, 1, 2)                       # 调整形状
        diff_b = diff_b.unsqueeze(0)
        return quant_b, diff_b, id_b


    def forward(self, input):
        # 输入为信号，返回经过解码器得到的重构信号和损失

        quant_b, diff_b, id_b = self.encode(input)
        x = self.fc(quant_b)
        x = F.log_softmax(x, dim=1)

        return x ,diff_b



if __name__ == "__main__":
    # 创建一个形状为 (64,1,224,224) 的随机输入张量
    rand_input = torch.rand((32, 1, 224, 224))

    # 创建一个模型
    model = VQE()

    # 将输入张量传入模型中
    output = model(rand_input)

    # 输出模型的输出
    print(output)


