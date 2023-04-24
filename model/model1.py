"""
下面是一个模型架构：
掩膜：

编码器：

量化器：

解码器：
"""
import torch
from torch import nn
from torch.nn import functional as F
import random
import math
import numpy as np

class CnnEncoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, kernel_size=(8, 8), stride=(4, 4
), padding=(2, 2))]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        #先将一个patch转化成一个向量

        #再使用自注意力网络
        return self.blocks(input)


# 定义 MaskingGenerator 类
class MaskingGenerator(nn.Module):
    def __init__(
            self,
            input_size,  # 输入图像的尺寸,默认为(40, 124)
            num_masking_patches,  # 要生成的遮蔽位置数量(有多少mask块),默认为 2
            patches_size,  # 遮蔽块的尺寸,默认为 (4,31)
            num_masking,  # 要生成的遮蔽数量(生成具体多少个mask),默认为 10
            masking_way,  # 遮蔽方式(random/adjunction),默认为 'adjunction'
            mask_shape,  # mask的形状(rectangle/list),默认为 'rectangle'

    ):
        super().__init__()

        # 如果输入的不是元组，则将其转换为元组类型
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        # 获取输入的宽度和高度
        self.height, self.width = input_size

        ##get_rectangle参数
        # patch 的尺寸
        self.patches_size = patches_size
        # 计算图像中 patch 的数量
        self.num_patches = (self.height // patches_size[0]) * (self.width // patches_size[1])

        ##get_row参数
        # 需要生成的 mask 的 patch 的数量（有几团mask）
        self.num_masking_patches = num_masking_patches
        # 需要生成的 mask 的 列数
        self.num_masking = num_masking

        ## 遮蔽方式
        self.masking_way = masking_way
        # mask的形状
        self.mask_shape = mask_shape

    # def __repr__(self):  # ！！！！！！！！！！
    #     # 返回一个表示对象的字符串
    #     repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
    #         self.height, self.width, self.min_num_patches, self.max_num_patches,
    #         self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
    #     return repr_str

    def get_rectangle(self, input, num_masking, patches_size):
        num_patches_h = input.shape[2] // patches_size[0]
        num_patches_w = input.shape[3] // patches_size[1]
        num_patches = num_patches_h * num_patches_w

        # ensure num_masking does not exceed available patches
        num_masking = min(num_masking, num_patches)

        # randomly choose patches to mask
        masking_indices = np.random.choice(num_patches, size=num_masking, replace=False)

        # convert 1D indices to 2D coordinates
        masking_cords = np.unravel_index(masking_indices, (num_patches_h, num_patches_w))

        # calculate the number of changed pixels
        changed_pixels = num_masking * patches_size[0] * patches_size[1] * input.shape[1]

        # create a mask for the chosen patches
        # mask = np.ones_like(input)
        for i, j in zip(*masking_cords):
            input[:, :, i * patches_size[0]:(i + 1) * patches_size[0],
            j * patches_size[1]:(j + 1) * patches_size[1]] = 0

        # apply the mask to the input image
        masked_input = input

        slice_tensor = masked_input[0, 0, :, :].numpy()
        print(slice_tensor)

        return changed_pixels, masking_indices, masked_input

    def get_row(self, input, num_masking_patches, num_masking):
        # 随机生成num_masking_patches个数,范围在0~height-1，同时要求生成的数间隔至少为2
        positions = np.random.choice(np.arange(0, self.height, 2), size=num_masking_patches, replace=False)
        # 初始化一个全1的行，让随机位置的1变为0，要求这些0不能相邻，同时0的个数为num_masking_patches个
        row = np.ones(self.height, dtype=int)
        for p in positions:
            row[p] = 0

        # 从每个为0的位置开始向周围扩散，将相邻的1都变成0，直到0的数目等于num_masking为止
        n = 0  # 计数
        while n != num_masking - num_masking_patches:
            for i in range(len(row)):
                if row[i] == 0:
                    # 随机选择扩散方向
                    direction = random.choice([-1, 1])
                    # 向左或向右扩散一格
                    if row[i + direction] == 1 and 0 <= i + direction < self.height:
                        row[i + direction] = 0
                        n += 1
                        if n == num_masking - num_masking_patches:
                            break

        # 获取input第一维的维数
        batch_size = input.shape[0]

        # 将行向量转化为列向量，形状为 (40, 1)
        row = torch.tensor(row, dtype=torch.float32)
        row = torch.unsqueeze(row, 1)
        # 将列向量扩展为 (16, 1, 40, 40) 的 tensor
        row = row.expand(batch_size, 1, 40, 40)
        # slice_tensor = row[0, 0, :, :]
        # print(slice_tensor.numpy())
        # 进行矩阵相乘操作
        masked_input = torch.matmul(row, input)
        slice_tensor = row[0, 0, :, :].numpy()
        print(slice_tensor)
        return masked_input

    def forward(self, input):
        if self.mask_shape == "list":
            masked_input = self.get_row(input, self.num_masking_patches, self.num_masking)
            return masked_input
        elif self.mask_shape == "rectangle":
            changed_pixels, masking_indices, masked_input = self.get_rectangle(input, self.num_masking,
                                                                               self.patches_size)
            return masked_input


class NET(nn.Module):
    def __init__(
            self,
            in_channel=1,  # 输入图像的通道数，默认为1
            channel=128,  # 编码器和解码器中的通道数，默认为128
            n_res_block=1,  # 残差块的数量，默认为2
            n_res_channel=32,  # 残差块的通道数，默认为32
            embed_dim=8,  # 嵌入向量的维度，默认为64
            n_embed=11,  # 嵌入向量的数量，默认为512

            input_size=(40, 124),
            num_masking_patches=2,  # 要生成的遮蔽位置数量(有多少mask块)(针对列掩膜),默认为 2
            patches_size=(4, 31),  # 遮蔽块的尺寸,默认为 (4,31)
            num_masking=10,  # 要生成的遮蔽数量(生成具体多少个mask),默认为 10
            masking_way="adjunction",  # 遮蔽方式(random/adjunction),默认为 'adjunction'
            mask_shape="rectangle",  # mask的形状(rectangle/list),默认为 'rectangle'



    ):
        super().__init__()

        # 定义模型中的各个模块
        # mask:先分块，再mask
        self.mask_cnn = MaskingGenerator(input_size=input_size, num_masking_patches=num_masking_patches,
                                         patches_size=patches_size,
                                         num_masking=num_masking, mask_shape=mask_shape, masking_way=masking_way)
        # # 编码器：分两个结构，一个是类似于unet的cnn卷积结构，一个是attention结构
        self.enc_cnn = CnnEncoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        # self.enc_attn = AttnEncoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        # # 量化器
        # # self.quantize_conv_b = nn.Conv2d(channel, embed_dim, 1)
        # self.quantize = Quantizer(embed_dim, n_embed)
        # # 解码器:分两个结构，一个是类似于unet的cnn卷积结构，一个是attention结构
        # self.dec_cnn = CnnDecoder()
        # self.dec_attn = AttnDecoder()

    def forward(self, input):
        # 输入为图像，返回经过解码器得到的重构图像和损失
        # 1. mask
        masked_input = self.mask_cnn(input)
        # 2. encoder
        enc_cnn_out = self.enc_cnn(masked_input)

        return masked_input


if __name__ == "__main__":
    rand_input = np.ones((16, 1, 40, 124))
    rand_input = torch.tensor(rand_input, dtype=torch.float32)
    net = NET()
    print(net(rand_input).shape)

    # masked_input = masking_generator.forward(input_image)

    # bot_out = bot_test(rand_input)
    # decoder = Decoder(4, 3, 128, 5, 10)
    # print(bot_out.shape)
    # print(decoder(bot_out).shape)
    # print("VQVAE")
    # final = VQVAE()
    # print(final.encode(rand_input)[-1].shape, "hello")
    # print(final(rand_input)[0].shape)
