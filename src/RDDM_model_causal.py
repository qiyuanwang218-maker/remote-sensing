# coding=utf8
#实现基本的causal功能
import torch
import torch.nn as nn
import torch.nn.functional as F
# 引入模型构建所需的组件
from .RDDM_parts import (default, partial, ResnetBlock, SinusoidalPosEmb, Residual,
                         PreNorm, LinearAttention, Attention, Upsample, Downsample)
import pytorch_lightning as pl
import random
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU

#因果推理模块定义（中间嵌入）
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class CausalReasoningBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = 64  # 定义隐藏层维度

        # 用于生成节点 attention（context/object）
        self.node_att_mlp = nn.Linear(dim, dim)

        # context 和 object 分支的卷积层，保持输入和输出通道一致
        self.context_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)  # context 分支卷积
        self.object_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)  # object 分支卷积

        # 批归一化
        self.norm = nn.BatchNorm2d(dim)

        # 使用卷积代替全连接层操作
        self.context_conv_fc = nn.Conv2d(dim, dim, kernel_size=1)  # context 输出卷积
        self.object_conv_fc = nn.Conv2d(dim, dim, kernel_size=1)  # object 输出卷积

    def forward(self, x):
        B, C, H, W = x.shape  # 获取输入特征维度

        # 展平空间维度后变换为 [B, HW, C]
        x_flat = x.view(B, C, -1).permute(0, 2, 1)

        # MLP 输出 attention 分数并归一化（每个空间位置有两个得分：context/object）
        att = torch.softmax(self.node_att_mlp(x_flat), dim=-1)  # [B, HW, 2]

        att_context = att[..., 0].unsqueeze(-1)  # context 分支注意力权重 [B, HW, 1]
        att_object = att[..., 1].unsqueeze(-1)  # object[B, HW, 1]

        # 重构为原始形状 [B, C, H, W]，并乘上各自分支的 attention 权重
        x_context = (att_context * x_flat).permute(0, 2, 1).view(B, C, H, W)
        x_object = (att_object * x_flat).permute(0, 2, 1).view(B, C, H, W)

        # 各分支卷积后并通过卷积处理增强反向传播优化能力
        xc = F.relu(self.context_conv(self.norm(x_context)))
        xo = F.relu(self.object_conv(self.norm(x_object)))

        # 使用卷积处理 context 和 object 特征，保持特征维度一致
        xc = self.context_conv_fc(xc)  # 使用卷积层处理 context 特征
        xo = self.object_conv_fc(xo)  # 使用卷积层处理 object 特征

        # 随机打乱 context 特征后与 object 特征融合
        num = xc.shape[0]
        l = list(range(num))
        random.shuffle(l)  # 打乱 context 的顺序，引入扰动
        rand_idx = torch.tensor(l, device=x.device)

        # 将上下文和目标特征混合在一起
        xco_logis = xc[rand_idx] + xo
        x_combined=0.3*xc+0.3*xo+0.4*xco_logis
        # 返回与输入相同的形状
        return x_combined


#主干 U-Net 模型结构
class RDDM_Unet(pl.LightningModule):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            input_condition=False,
            input_condition_channels=0,
            mask_condition=False,
            mask_condition_channels=0,
            resnet_block_groups=8
    ):
        super().__init__()

        # 条件输入和掩码输入控制
        self.input_condition = input_condition
        self.mask_condition = mask_condition

        # 输入通道数 = 原图 + 条件图 + 掩码图（可选）
        input_channels = channels + input_condition_channels * (1 if self.input_condition else 0) + \
                         mask_condition_channels * (1 if self.mask_condition else 0)

        init_dim = default(init_dim, dim)# 初始维度
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)# 首层大卷积

        #构建 U-Net 编码器和解码器的通道配置。时间编码模块（扩散模型关键）
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]# 构建多层维度
        in_out = list(zip(dims[:-1], dims[1:]))# 构建每层的输入输出对
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        #编码器（下采样路径）
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim), # ResBlock1
                block_klass(dim_in, dim_in, time_emb_dim=time_dim), # ResBlock2
                Residual(PreNorm(dim_in, LinearAttention(dim_in))), # Linear Attention
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        #中间 Bottleneck + CausalBlock
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_causal = CausalReasoningBlock(mid_dim)#中间插入因果推理
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        #解码器（上采样路径）
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        #最终输出头
        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    #前向传播逻辑
    def forward(self, x, time, input_cond=None, mask_cond=None):
        # 拼接条件图/掩码图
        if self.input_condition:
            x = torch.cat((x, input_cond), dim=1)
        if self.mask_condition:
            x = torch.cat((x, mask_cond), dim=1)

        x = self.init_conv(x)# 首层大卷积
        r = x.clone()           # 保存原图用于最终拼接
        t = self.time_mlp(time)# 时间嵌入

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x) # 下采样

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_causal(x)  # ⭐ 插入因果推断模块
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)# 最终拼接跳跃连接
        x = self.final_res_block(x, t)
        return self.final_conv(x)# 输出 denoised 图像


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(4, 3, 64, 64).to(device)# 输入图像
    t = torch.randint(0, 1000, (4,), dtype=torch.long).to(device)# 时间步（4个样本）

    model = RDDM_Unet(
        dim=32,
        dim_mults=(1, 2),
        input_condition=False,
        mask_condition=False
    ).to(device)

    y = model(x, t)
    print('RDDM_Unet Output Shape:', y.shape)