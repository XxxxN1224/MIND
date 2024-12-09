import math
import torch
import numpy as np
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention, TransformerEncoderLayer, TransformerEncoder
from typing import Dict, List, Tuple, Optional
from planners.mind.utils import gpu
from planners.mind.networks.layers import Conv1d, Res1d


'''
ActorNet

Input: (batch_size, 3, seq_len)

+------------------------+      +---------------------------+      +---------------------------+       +---------------------------+
| Group 1                |      | Group 2                   |      | Group 3                   |       | Group 4                   |
|                        |      |                           |      |                           |       |                           |
| Res1d(3, 32)           |      | Res1d(32, 64, 2)          |      | Res1d(64, 128, 2)         |       | Res1d(128, 256, 2)        |
| Res1d(32, 32)          |      | Res1d(64, 64)             |      | Res1d(128, 128)           |       | Res1d(256, 256)           |
+------------------------+      +---------------------------+      +---------------------------+       +---------------------------+
| Output: (32, seq_len)  |      | Output: (64, seq_len//2)  |      | Output: (128, seq_len//4) |       | Output: (256, seq_len//8) |

+------------------------+      +---------------------------+      +---------------------------+       +---------------------------+
| Lateral 1              |      | Lateral 2                 |      | Lateral 3                 |       | Lateral 4                 |
| Conv1d(32, 128)        |      | Conv1d(64, 128)           |      | Conv1d(128, 128)          |       | Conv1d(256, 128)          |
+------------------------+      +---------------------------+      +---------------------------+       +---------------------------+
| Output: (128, seq_len) |      | Output: (128, seq_len//2) |      | Output: (128, seq_len//4) |       | Output: (128, seq_len//8) |

+------------------------+
| Interpolate and Add    |
|                        |
| Interpolate x2         |
| Add Lateral 3          |
| Interpolate x2         |
| Add Lateral 2          |
| Interpolate x2         |
| Add Lateral 1          |
+------------------------+
| Output: (128, seq_len) |

+---------------------+
| Final Output        |
|                     |
| Res1d(128, 128)     |
| Output: (128, 1)    |
+---------------------+

1. 为什么要使用4个 group?
    多尺度特征提取
        多尺度特征：每个 group 通过不同的卷积操作（尤其是第一个 Res1d 块的步长为2）生成不同尺度的特征图。这些不同尺度的特征图能够捕捉不同层次的细节信息，这对于许多任务（如目标检测、语义分割等）非常重要。
        层次结构：通过4个 group，网络可以构建一个层次结构，从低级特征到高级特征逐步抽象。这种层次结构有助于网络学习到更丰富的特征表示。
    特征层次
        低级特征：第一个 group 主要捕捉低级特征，如边缘和纹理。
        中级特征：中间的 group 捕捉中级特征，如形状和局部结构。
        高级特征：最后一个 group 捕捉高级特征，如对象类别和全局结构。

2. 每个 group 的输入是否一样？
    输入变化：每个 group 的输入并不完全一样。第一个 group 的输入是原始输入数据，而后续的 group 的输入是前一个 group 的输出。具体来说：
        group[0]：输入是 (batch_size, 3, seq_len)，输出是 (batch_size, 32, seq_len)。
        group[1]：输入是 group[0] 的输出 (batch_size, 32, seq_len)，输出是 (batch_size, 64, seq_len // 2),.步幅为2, 可以控制特征图的空间尺寸变化即序列长度减半
        group[2]：输入是 group[1] 的输出 (batch_size, 64, seq_len // 2)，输出是 (batch_size, 128, seq_len // 4)。
        group[3]：输入是 group[2] 的输出 (batch_size, 128, seq_len // 4)，输出是 (batch_size, 256, seq_len // 8)。

3.  为什么要使用4个 lateral?
    特征融合
        特征融合：lateral 模块通过1x1卷积将不同尺度的特征图转换为统一的通道数（hidden_size），以便上采样和加法操作将高层次的特征图与低层次的特征图进行融合。这种融合有助于结合不同尺度的特征，增强特征表示的丰富性。
    多尺度输出
        多尺度输出：虽然 ActorNet 最终输出的是一个固定大小的特征向量，但在融合过程中，lateral 模块确保了不同尺度的特征信息都能被充分利用。这种设计有助于网络在不同尺度上做出更准确的预测。

4. 4个 lateral的输入是一样的么？不一样的话那么分别是什么？
    lateral[0]：输入是 group[0] 的输出 (batch_size, 32, seq_len)，输出是 (batch_size, 128, seq_len)。
    lateral[1]：输入是 group[1] 的输出 (batch_size, 64, seq_len // 2)，输出是 (batch_size, 128, seq_len // 2)。
    lateral[2]：输入是 group[2] 的输出 (batch_size, 128, seq_len // 4)，输出是 (batch_size, 128, seq_len // 4)。
    lateral[3]：输入是 group[3] 的输出 (batch_size, 256, seq_len // 8)，输出是 (batch_size, 128, seq_len // 8)。

5. Interpolate and Add 的作用
    插值（Interpolate）
        上采样：插值操作用于将高层次的特征图（低分辨率）上采样到低层次的特征图（高分辨率）的尺寸。这通常通过线性插值（F.interpolate）实现。
        对齐：插值操作确保不同尺度的特征图在空间维度上对齐，以便进行逐元素的加法操作。
    加法（Add）
        特征融合：加法操作将上采样后的高层次特征图与低层次特征图逐元素相加，实现特征融合。这种融合有助于结合不同尺度的特征信息，增强模型的表示能力。

6. Interpolate and Add 的输入输出
    初始特征
        group[3] 的输出：(batch_size, 256, seq_len // 8)
        lateral[3] 的输出：(batch_size, 128, seq_len // 8)
    第一层 Interpolate and Add
        输入：lateral[3] 的输出 (batch_size, 128, seq_len // 8)
        插值操作：将 (batch_size, 128, seq_len // 8) 上采样到 (batch_size, 128, seq_len // 4)
        lateral[2] 的输出：(batch_size, 128, seq_len // 4)
        加法操作：(batch_size, 128, seq_len // 4) + (batch_size, 128, seq_len // 4) = (batch_size, 128, seq_len // 4)
    第二层 Interpolate and Add
        输入：第一层 Interpolate and Add 的输出 (batch_size, 128, seq_len // 4)
        插值操作：将 (batch_size, 128, seq_len // 4) 上采样到 (batch_size, 128, seq_len // 2)
        lateral[1] 的输出：(batch_size, 128, seq_len // 2)
        加法操作：(batch_size, 128, seq_len // 2) + (batch_size, 128, seq_len // 2) = (batch_size, 128, seq_len // 2)
    第三层 Interpolate and Add
        输入：第二层 Interpolate and Add 的输出 (batch_size, 128, seq_len // 2)
        插值操作：将 (batch_size, 128, seq_len // 2) 上采样到 (batch_size, 128, seq_len)
        lateral[0] 的输出：(batch_size, 128, seq_len)
        加法操作：(batch_size, 128, seq_len) + (batch_size, 128, seq_len) = (batch_size, 128, seq_len)
'''
# ActorNet 是一个多尺度的卷积神经网络（借鉴了FPN），通过特征金字塔结构提取不同层次的特征，结合了残差连接和上采样机制，适用于处理时序数据或一维信号特征提取
# 用来建模这些交通参与者的动态行为, 识别其他交通参与者的动作和意图，预测它们在未来几秒内的可能移动路径
class ActorNet(nn.Module):
    """
    Actor feature extractor with Conv1D
    """

    def __init__(self, n_in=3, hidden_size=128, n_fpn_scale=4): # n_in 是输入通道数，FPN隐藏层大小，FPN的尺度数
        super(ActorNet, self).__init__()
        norm = "GN" # 归一化方法，这里是“GN”（Group Normalization）
        ng = 1

        n_out = [2 ** (5 + s) for s in range(n_fpn_scale)]  # [32, 64, 128, 256]
        blocks = [Res1d] * n_fpn_scale
        num_blocks = [2] * n_fpn_scale

        groups = [] # groups 变量是一个列表，其中每个元素都是一个 nn.Sequential 容器。每个 nn.Sequential 容器包含一组 Res1d 块，这些块在同一个 FPN（Feature Pyramid Network）尺度上工作
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))  # seq_len_out = (seq_len_in + 2 * padding - kernel_size) / stride + 1

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))    # nn.Sequential 是一个容器，它按照添加的顺序执行内部包含的所有模块，适合于线性堆叠的层结构
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups) # nn.ModuleList 是一个特殊的列表，用于存储 nn.Module 子类的实例

        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], hidden_size, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(hidden_size, hidden_size, norm=norm, ng=ng)

    def forward(self, actors: Tensor) -> Tensor:
        out = actors

        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        # self.lateral[-1]：这是一个 nn.Conv1d 层，位于 self.lateral 列表的最后一个位置
        # outputs[-1]：这是 outputs 列表的最后一个元素，即最高层的特征图
        # 将最高层的特征图 outputs[-1] 通过 self.lateral[-1] 层进行特征变换，得到新的特征图 out
        out = self.lateral[-1](outputs[-1])

        for i in range(len(outputs) - 2, -1, -1):   # range(len(outputs) - 2, -1, -1)：这是一个 range 函数，生成一个从 len(outputs) - 2 到 0 的递减序列
            # F.interpolate：这是 PyTorch 中的一个函数，用于对张量进行上采样（upsampling）。它会将输入张量的分辨率放大指定的倍数
            # scale_factor=2：上采样的比例因子，即将特征图的分辨率放大 2 倍
            # mode="linear"：上采样的模式，这里使用线性插值
            # align_corners=False：是否对齐角点，这里设置为 False，意味着插值时不考虑角点对齐
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)    # 上采样操作通过 F.interpolate 将特征图的分辨率恢复到上一层的分辨率
            # self.lateral[i]：这是 self.lateral 列表中的第 i 个 nn.Conv1d 层
            # outputs[i]：这是 outputs 列表中的第 i 个特征图
            # 操作：将 outputs[i] 通过 self.lateral[i] 层进行特征变换，然后将结果加到当前的特征图 out 上，实现特征融合
            out += self.lateral[i](outputs[i])  # 横向连接通过 self.lateral[i] 将自底向上路径中的特征图与自顶向下路径中的特征图进行融合

        # [:, :, -1]：选择最后一个时间步的特征。假设 out 的形状为 (batch_size, channels, seq_len)，[:, :, -1] 会选择 out 的最后一个时间步，即 out[:, :, -1] 的形状为 (batch_size, channels)
        out = self.output(out)[:, :, -1]    # 通过 self.output 层对融合后的特征图进行最终的特征提取, 选择最后一个时间步的特征作为输出
        return out

# 在自动驾驶任务中，特别是涉及车道检测和轨迹预测时，每个车道上的特征点（如车道线上的点）包含了丰富的信息。
# 这些特征点的分布和动态变化对于理解车道的状态和预测未来的行为非常重要。
# 然而，单纯依赖每个特征点的局部信息可能不足以捕捉全局的车道特征，因此需要一种方法来聚合这些局部特征，提取出更高级的全局信息
# PointAggregateBlock 的设计目的是通过对车道上所有特征点的特征进行聚合，利用全连接层和最大池化操作提取出车道的全局特征。
class PointAggregateBlock(nn.Module):
    def __init__(self, hidden_size: int, aggre_out: bool, dropout: float = 0.1) -> None:
        super(PointAggregateBlock, self).__init__()
        self.aggre_out = aggre_out

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.norm = nn.LayerNorm(hidden_size)

    # 通过全局最大池化（Global Max Pooling）将每个车道的所有特征点聚合为一个全局特征向量：得到每个车道的一个全局特征向量
    def _global_maxpool_aggre(self, feat):
        # 输入：方法接受一个张量feat，通常是一个三维张量，形状为 [N_{lane}, 10, hidden_size]，其中 N_{lane} 是车道的数量，10 是每个车道的特征点数量，hidden_size 是每个特征点的特征维度
        # 维度调整：feat.permute(0, 2, 1)：将特征张量的维度顺序从 [N_{lane}, 10, hidden_size] 调整为 [N_{lane}, hidden_size, 10]。这样做是为了使得adaptive_max_pool1d函数的输入格式符合要求
        # 全局最大池化：F.adaptive_max_pool1d(..., 1) 对重新排列后的张量进行全局最大池化操作，将每个特征的时间维度压缩为1，输出形状为 (batch_size, 1, features)。这意味着它为每个特征通道计算了一个最大值
        # 维度恢复：最后，permute(0, 2, 1) 将池化后的特征张量的维度恢复为 [N_{lane}, 1, hidden_size]，使得输出与原始特征的维度格式一致
        return F.adaptive_max_pool1d(feat.permute(0, 2, 1), 1).permute(0, 2, 1) # 重点是F.adaptive_max_pool1d(..., 1) 

    def forward(self, x_inp):
        # 1. fc1层目的是对输入特征进行非线性变换，提取出更高级的特征表示
        x = self.fc1(x_inp)  # [N_{lane}, 10, hidden_size]

        # 2. 将原始特征和全局特征向量拼接在一起，形成新的特征表示。这样可以保留局部特征的同时，加入全局特征的信息
        x_aggre = self._global_maxpool_aggre(x)
        # x形状为 [N_{lane}, 10, hidden_size], x_aggre形状为 [N_{lane}, 1, hidden_size]
        # x_aggre.repeat([1, x.shape[1], 1])将全局池化结果沿时间维度（第二维）重复10次，以使其形状为[N_lane, 10, hidden_size]，与局部特征x一致。
        # torch.cat(..., dim=-1)将局部特征x和全局特征x_aggre拼接在一起，形成一个新的张量，包含了局部和全局特征的信息。
        x_aggre = torch.cat([x, x_aggre.repeat([1, x.shape[1], 1])], dim=-1)

        # 3. fc2层的目的是将局部特征和全局特征有效融合，生成更鲁棒的特征表示
        # self.fc2(x_aggre)：通过 fc2 层进行两次线性变换和层归一化，中间使用 ReLU 激活函数，将特征维度从 hidden_size * 2 降回到 hidden_size。
        # x_inp + self.fc2(x_aggre)：将原始输入特征 x_inp 与 fc2 层的输出相加，实现残差连接。
        # self.norm(...)：通过层归一化对相加后的特征进行归一化，稳定训练过程。
        out = self.norm(x_inp + self.fc2(x_aggre))
        if self.aggre_out:
            return self._global_maxpool_aggre(out).squeeze()    # 通过 _global_maxpool_aggre 方法将 out 聚合为全局特征向量，并使用 squeeze 去除多余的维度，形状为 [N_{lane}, hidden_size]
        else:
            return out  # 形状为 [N_{lane}, 10, hidden_size]

'''
LaneNet

+-------------------+
| Input: [N_lane, 10, in_size] |
+-------------------+
              |
              v
+-------------------+
| proj (Linear, LN, ReLU) |
| [N_lane, 10, in_size] -> [N_lane, 10, hidden_size] |
+-------------------+
              |
              v
+-------------------+
| aggre1 (PointAggregateBlock) |
| [N_lane, 10, hidden_size] -> [N_lane, 10, hidden_size] |
+-------------------+
              |
              v
+-------------------+
| aggre2 (PointAggregateBlock) |
| [N_lane, 10, hidden_size] -> [N_lane, hidden_size] |
+-------------------+
              |
              v
+-------------------+
| Output: [N_lane, hidden_size] |
+-------------------+


输入层：[N_{lane}, 10, in_size], N_{lane} 是车道的数量，10 是每个车道的采样点数，in_size 是每个采样点的特征维度
投影层：proj 将输入特征映射到 hidden_size 维度，输出形状为 [N_{lane}, 10, hidden_size]。
第一聚合块：aggre1 进行特征聚合和变换，输出形状为 [N_{lane}, 10, hidden_size]。
第二聚合块：aggre2 再次进行特征聚合和变换，并最终输出形状为 [N_{lane}, hidden_size]。
'''
# LaneNet其实借鉴了PointNet针对向量点进行聚合，然后RelaFusionLayer借鉴VectorNet进行图神经关联
class LaneNet(nn.Module):
    def __init__(self, device, in_size=10, hidden_size=128, dropout=0.1):
        super(LaneNet, self).__init__()

        self.device = device

        self.proj = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.aggre1 = PointAggregateBlock(hidden_size=hidden_size, aggre_out=False, dropout=dropout)
        self.aggre2 = PointAggregateBlock(hidden_size=hidden_size, aggre_out=True, dropout=dropout)

    # for av2
    def forward(self, feats):
        x = self.proj(feats)  # [N_{lane}, 10, hidden_size]
        x = self.aggre1(x)  # 在 PointAggregateBlock 中，聚合层指的是将每个车道的所有特征点聚合为一个全局特征向量，并将这个全局特征向量与局部特征点进行融合
        x = self.aggre2(x)  # [N_{lane}, hidden_size]
        return x

'''
RelaFusionLayer

+-------------------+
| Input:            |
| node: [N, d_model]|
| edge: [N, N, d_edge]|
| edge_mask: [N, N] |
+-------------------+
              |
              v
+-------------------+
| forward           |
| 1. x, edge, memory = _build_memory(node, edge) |
| 2. x_prime, _ = _mha_block(x, memory, attn_mask=None, key_padding_mask=edge_mask) |
| 3. x = norm2(x + x_prime).squeeze()  # [N, d_model] |
| 4. x = norm3(x + _ff_block(x))  # [N, d_model] |
| 5. return x, edge, None |
+-------------------+
              |
              v
+-------------------+
| _build_memory     |
| 1. src_x: [N, N, d_model]|
| 2. tar_x: [N, N, d_model]|
| 3. memory: [N, N, d_model]|
| 4. if update_edge:|
|    edge: [N, N, d_edge]  |
| 5. return [1, N, d_model], [N, N, d_edge], [N, N, d_model] |
+-------------------+
              |
              v
+-------------------+
| _mha_block        |
| 1. x: [1, N, d_model]|
| 2. return [1, N, d_model], None |
+-------------------+
              |
              v
+-------------------+
| _ff_block         |
| 1. x: [N, d_model]|
| 2. return [N, d_model] |
+-------------------+
              |
              v
+-------------------+
| Output:           |
| x: [N, d_model]   |
| edge: [N, N, d_edge]|
| None              |
+-------------------+
'''
# RelaFusionLayer(Relation Fusion Layer) 是一个用于处理图结构数据（如交通网络中的节点和边）的神经网络层。
# 它的主要功能是在图结构中融合节点和边的信息，通过多头注意力机制和前馈网络来更新节点特征，并可选地更新边特征。
# 这种设计特别适用于需要理解节点之间的关系和动态变化的任务，如自动驾驶中的交通参与者轨迹预测和行为识别
class RelaFusionLayer(nn.Module):
    def __init__(self,
                 device,
                 d_edge: int = 128,
                 d_model: int = 128,
                 d_ffn: int = 2048,
                 n_head: int = 8,
                 dropout: float = 0.1,
                 update_edge: bool = True) -> None:
        super(RelaFusionLayer, self).__init__()
        # 1. 设置device和edge更新标志
        self.device = device
        self.update_edge = update_edge

        # 2. proj_memory 的目的是构建一个记忆表示，这个记忆表示融合了节点特征和边特征。
        # 具体来说，proj_memory 将每个节点对之间的边特征和两个节点的特征拼接在一起，然后通过一个线性层投影到 d_model 维度，可以减少特征的维度，使得后续的计算更加高效
        self.proj_memory = nn.Sequential(
            nn.Linear(d_model + d_model + d_edge, d_model), # 对于每个节点，取其自身的特征（d_model）、邻接节点的特征（d_model），以及与这些边相关的边特征（d_edge）
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True)
        )

        # 3. 如果 update_edge 为 True，则通过 proj_edge 将 d_model 维度的特征投影回 d_edge 维度，并进行归一化
        if self.update_edge:
            self.proj_edge = nn.Sequential(
                nn.Linear(d_model, d_edge),
                nn.LayerNorm(d_edge),
                nn.ReLU(inplace=True)
            )
            self.norm_edge = nn.LayerNorm(d_edge)

        # 4. 初始化多头注意力机制
        self.multihead_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=n_head, dropout=dropout, batch_first=False)

        # 5. Feedforward model
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)

        # 6. 初始化归一化层、丢弃层和初始化激活函数
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self,
                node: Tensor,
                edge: Tensor,
                edge_mask: Optional[Tensor]) -> Tensor:
        '''
            input:
                node:       (N, d_model)，表示 N 个节点的特征向量
                edge:       (N, N, d_model)，表示 N 个节点之间的边特征，通常是一个邻接矩阵
                edge_mask:  (N, N)，用于指示边的有效性
        '''
        # update node
        x, edge, memory = self._build_memory(node, edge)    #  更新节点和边的信息
        x_prime, _ = self._mha_block(x, memory, attn_mask=None, key_padding_mask=edge_mask) # 实现多头注意力机制
        x = self.norm2(x + x_prime).squeeze()
        x = self.norm3(x + self._ff_block(x))   # 更新节点特征并返回
        return x, edge, None    # 这个x包含actor、lane、cls 这3类信息

    def _build_memory(self,
                      node: Tensor,
                      edge: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        '''
            _build_memory是个node pair，在节点特征和边特征之间建立联系，以便后续的图神经网络操作

            input:
                node:   (N, d_model)
                edge:   (N, N, d_edge)
            output:
                :节点特征张量  (1, N, d_model)
                :边特征张量  (N, N, d_edge)
                :记忆张量  (N, N, d_model)
        '''
        n_token = node.shape[0] # n_token：节点的数量，即 N

        # 1. build memory
        # unsqueeze(dim=0)：在第0维上增加一个维度，形状变为 (1, N, d_model)。
        # repeat([n_token, 1, 1])：在第0维上重复 n_token 次，形状变为 (N, N, d_model)
        src_x = node.unsqueeze(dim=0).repeat([n_token, 1, 1])  # (N, N, d_model)
        # unsqueeze(dim=1)：在第1维上增加一个维度，形状变为 (N, 1, d_model)。
        # repeat([1, n_token, 1])：在第1维上重复 n_token 次，形状变为 (N, N, d_model)。
        tar_x = node.unsqueeze(dim=1).repeat([1, n_token, 1])  # (N, N, d_model)
        # torch.cat([edge, src_x, tar_x], dim=-1)：将边特征 edge 和扩展后的节点特征 src_x 和 tar_x 拼接在一起，形状变为 (N, N, d_model + d_model + d_edge)
        # torch.cat：沿着最后一个维度（特征维度）拼接张量
        # self.proj_memory：通过一个线性层将拼接后的特征投影到 d_model 维度
        memory = self.proj_memory(torch.cat([edge, src_x, tar_x], dim=-1))  # (N, N, d_model)
        
        # 2. 如果 self.update_edge 为真，则更新边特征 edge (with residual)
        if self.update_edge:
            # self.proj_edge(memory)：通过一个线性层将记忆表示 memory 投影到 d_edge 维度。
            # edge + self.proj_edge(memory)：将原始边特征 edge 与投影后的记忆表示相加，实现残差连接
            edge = self.norm_edge(edge + self.proj_edge(memory))  # (N, N, d_edge)

        # node.unsqueeze(dim=0)：将节点特征 node 在第0维上增加一个维度，形状变为 (1, N, d_model)
        # edge：更新后的边特征，形状为 (N, N, d_edge)
        # memory：记忆表示，形状为 (N, N, d_model)
        return node.unsqueeze(dim=0), edge, memory

    # multihead attention block
    '''
    multihead attention block

    [Input x] ---> [Multihead Attention] ---> [Dropout] ---> [Output x]
                    ↑
               [Memory mem]

    +-------------------+
    | Input:            |
    | x: [1, N, d_model]|
    | mem: [N, N, d_model]|
    | attn_mask: [N, N] (optional)|
    | key_padding_mask: [N, N] (optional)|
    +-------------------+
                |
                v
    +-------------------+
    | Multi-Head Attention |
    | query: x [1, N, d_model] |
    | key: mem [N, N, d_model] |
    | value: mem [N, N, d_model] |
    | attn_mask: [N, N] (optional)|
    | key_padding_mask: [N, N] (optional)|
    | need_weights: False |
    | 1. Compute attention scores |
    | 2. Apply attention mask |
    | 3. Apply key padding mask |
    | 4. Compute weighted sum |
    | 5. Output x [1, N, d_model] |
    | 6. Output attention_weights [1, N, N] (not returned) |
    +-------------------+
                |
                v
    +-------------------+
    | Dropout Layer     |
    | input: x [1, N, d_model]|
    | 1. Randomly drop neurons |
    | 2. Output x [1, N, d_model]|
    +-------------------+
                |
                v
    +-------------------+
    | Output:           |
    | x: [1, N, d_model]|
    | None              |
    +-------------------+
    '''
    def _mha_block(self,
                   x: Tensor,
                   mem: Tensor,
                   attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor]) -> Tensor:
        '''
            input:
                x:                  [1, N, d_model], 表示输入的节点特征 [batch_size, seq_len_q, d_model]
                mem:                [N, N, d_model], 表示记忆表示
                attn_mask:          [N, N]，表示注意力掩码，用于防止某些位置之间的注意力计算
                key_padding_mask:   [N, N]，表示键填充掩码，用于处理序列中的填充部分
            output:
                :param      [1, N, d_model]
                :param      [N, N]
        '''

        # x：更新后的节点特征，形状为 [1, N, d_model]。
        # _：注意力权重，由于 need_weights=False，这里返回 None。
        x, _ = self.multihead_attn(x, mem, mem, # x 是查询（Query）,mem 同时作为键（Key）和值（Value）
                                   attn_mask=attn_mask,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=False)  # return average attention weights
        return self.dropout2(x), None

    # feed forward block
    def _ff_block(self,
                  x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x) # 这个x包含actor、lane、context3类信息


class RelaFusionNet(nn.Module):
    def __init__(self,
                 device,
                 d_model: int = 128,
                 d_edge: int = 128,
                 n_head: int = 8,
                 n_layer: int = 6,  # 创建 n_layer 个 RelaFusionLayer 层
                 dropout: float = 0.1,
                 update_edge: bool = True):
        super(RelaFusionNet, self).__init__()
        self.device = device

        fusion = []
        for i in range(n_layer):
            need_update_edge = False if i == n_layer - 1 else update_edge
            fusion.append(RelaFusionLayer(device=device,
                                          d_edge=d_edge,
                                          d_model=d_model,
                                          d_ffn=d_model * 2,
                                          n_head=n_head,
                                          dropout=dropout,
                                          update_edge=need_update_edge))
        self.fusion = nn.ModuleList(fusion)

    def forward(self, x: Tensor, edge: Tensor, edge_mask: Tensor) -> Tensor:
        '''
            该forward会在out, _ = self.fuse_scene(tokens_with_cls, rpe_with_cls, edge_mask=None)被调用
            x: (N, d_model)，表示每个节点的特征向量: tokens_with_cls
            edge: (d_model, N, N)，表示每条边的特征矩阵。d_model 是特征维度，N 是节点数: rpe_with_cls
            edge_mask: (N, N)，用于指示哪些边是有效的。通常是一个布尔矩阵，有效边的位置为 True，无效边的位置为 False: edge_mask=None
        '''
        # attn_multilayer = []
        for mod in self.fusion:
            x, edge, _ = mod(x, edge, edge_mask)    # mod是self.fusion里的元素，即RelaFusionLayer()，共有n_layer个，这是多层串联
        return x, None  # 这个x包含actor、lane、cls 3类信息


class FusionNet(nn.Module):
    def __init__(self, device, config):
        """
        初始化FusionNet模型类。

        该构造函数初始化了模型所需的设备信息、嵌入维度、相对位置编码维度等配置，
        并定义了用于对演员（actor）和车道（lane）特征进行投影的线性层，以及场景融合网络。

        参数:
        - device: 字符串，指示模型在哪个设备（如'cpu'或'cuda'）上运行。
        - config: 字典，包含模型的各种配置参数，如嵌入维度d_embed，相对位置编码维度d_rpe，
            dropout率，是否更新边等。

        返回值:
        无
        """
        super(FusionNet, self).__init__()  # 调用父类构造方法
        self.device = device  # 设置模型运行的设备

        # 从配置字典中提取相关维度和参数
        self.d_embed = config['d_embed']
        self.d_rpe = config['d_rpe']
        self.d_model = config['d_embed']  # 设置模型维度
        dropout = config['dropout']  # 提取dropout率
        update_edge = config['update_edge']  # 提取是否更新边的信息

        # 1. 定义用于演员特征投影的线性层和归一化层
        self.proj_actor = nn.Sequential(
            nn.Linear(config['d_actor'], self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(inplace=True)
        )
        # 2. 定义用于车道特征投影的线性层和归一化层
        self.proj_lane = nn.Sequential(
            nn.Linear(config['d_lane'], self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(inplace=True)
        )
        # 3. 定义用于相对位置编码投影的线性层和归一化层
        self.proj_rpe_scene = nn.Sequential(
            nn.Linear(config['d_rpe_in'], config['d_rpe']),
            nn.LayerNorm(config['d_rpe']),
            nn.ReLU(inplace=True)
        )

        # 4. 初始化场景融合网络
        self.fuse_scene = RelaFusionNet(self.device,
                                        d_model=self.d_model,
                                        d_edge=config['d_rpe'],
                                        n_head=config['n_scene_head'],
                                        n_layer=config['n_scene_layer'],
                                        dropout=dropout,
                                        update_edge=update_edge)

    def forward(self,
                actors: Tensor,
                actor_idcs: List[Tensor],
                lanes: Tensor,
                lane_idcs: List[Tensor],
                rpe_prep: Dict[str, Tensor]):
        """
        前向传播函数，用于处理输入的演员和车道数据，并融合场景信息。
        这里需要注意的是cls: 这种在序列模型中使用分类 token 的做法，主要是借鉴了 BERT，旨在通过引入全局表示来提高模型的效果，特别是在需要进行分类或序列理解的任务上

        :param actors: 输入的演员数据张量
        :param actor_idcs: 演员数据的索引列表
        :param lanes: 输入的车道数据张量
        :param lane_idcs: 车道数据的索引列表
        :param rpe_prep: 相对位置嵌入的预处理数据字典
        :return: 处理后的actors, lanes, cls
        """
        # 对演员和车道数据进行投影
        actors = self.proj_actor(actors)
        lanes = self.proj_lane(lanes)

        # 初始化新的数据列表
        actors_new, lanes_new, cls_new = list(), list(), list() # 其中cls_new用于存储分类 token 的特征

        # 遍历每个场景中的演员和车道数据，以便在后续的处理中对每个场景进行单独的特征提取和融合
        # 这里有疑问：输入的actor、lane和rpe是以场景为单位打包的？
        for a_idcs, l_idcs, rpes in zip(actor_idcs, lane_idcs, rpe_prep):
            # 获取当前样本的演员和车道数据
            _actors = actors[a_idcs]
            _lanes = lanes[l_idcs]
            # 将演员和车道数据合并，完全是SIMPL的架构
            tokens = torch.cat([_actors, _lanes], dim=0)

            # 创建CLS标记(Classification Token)，并将其添加到合并的数据中, 此处借鉴BERT的架构
            cls_token = torch.zeros((1, self.d_model), device=self.device)  # 创建一个形状为 (1, d_model) 的零张量，作为分类 token
            tokens_with_cls = torch.cat([tokens, cls_token], dim=0) # 将分类 token 拼接到 tokens 的末尾

            # 对相对位置嵌入进行投影，并调整形状
            rpe = self.proj_rpe_scene(rpes['scene'].permute(1, 2, 0))
            # 创建包含CLS的相对位置嵌入矩阵
            # 这段代码的主要目的是扩展相对位置编码（Relative Positional Encoding, RPE）以包含CLS标记，因为：
            # 1. 在序列tokens中加入CLS标记后，整个序列tokens_with_cls的长度变长了。
            # 2. 相对位置编码也需要相应地扩展，以包含CLS标记的位置信息。
            rpe_with_cls = torch.zeros(
                (tokens_with_cls.shape[0], tokens_with_cls.shape[0], self.d_rpe),
                device=self.device)
            rpe_with_cls[:tokens.shape[0], :tokens.shape[0], :] = rpe

            # 使用融合模块处理数据和相对位置嵌入
            out, _ = self.fuse_scene(tokens_with_cls, rpe_with_cls, edge_mask=None) # 实际对应RelaFusionLayer中forward的node、edge、edge_mask

            # 将处理结果分别添加到对应的列表中
            actors_new.append(out[:len(a_idcs)])    # actors_new: 包含了更新后的代理（例如车辆、行人等）的状态。这部分是从输出的开始到a_idcs长度的部分
            lanes_new.append(out[len(a_idcs):-1])   # lanes_new: 包含了更新后的车道状态。这部分是从a_idcs长度的位置到倒数第二个元素
            cls_new.append(out[-1].unsqueeze(0))    # cls_new: 包含了分类标记（cls）的信息。这部分是输出的最后一个元素，并且通过unsqueeze(0)操作增加了一个维度，使其形状变为(1, hidden_size)

        # 将所有样本的数据合并
        actors = torch.cat(actors_new, dim=0)
        lanes = torch.cat(lanes_new, dim=0)
        cls = torch.cat(cls_new, dim=0) # cls会被用作SceneDecoder中forward里的context

        return actors, lanes, cls


'''
SceneDecoder

+-------------------+
|    输入           |
|                   |
|  ctx: (batch_size, hidden_size)  |
|  actors: (num_actors, hidden_size)  |
|  actor_idcs: List[Tensor]  |
|  tgt_feat: (batch_size, hidden_size)  |
|  tgt_rpes: (batch_size, 20)  |
+-------------------+
          |
          v
+-------------------+
| RPE 嵌入投影       |
| self.proj_rpe     |
|                   |
| 输入: (batch_size, 20)  |
| 输出: (batch_size, hidden_size)  |
+-------------------+
          |
          v
+-------------------+
| 目标特征投影       |
| self.proj_tgt     |
|                   |
| 输入: (batch_size, 2 * hidden_size)  |
| 输出: (batch_size, hidden_size)  |
+-------------------+
          |
          v
+-------------------+
| 遍历每个 agent     |
| for idx, a_idcs in enumerate(actor_idcs)  |
|                   |
| _ctx: (1, hidden_size)  |
| _actors: (num_actors, hidden_size)  |
+-------------------+
          |
          v
+-------------------+
| 上下文投影         |
| self.ctx_proj     |
|                   |
| 输入: (1, hidden_size)  |
| 输出: (num_modes, 1, hidden_size)  |
+-------------------+
          |
          v
+-------------------+
| 上下文饱和         |
| self.ctx_sat      |
|                   |
| 输入: (num_modes, 1, hidden_size)  |
| 输出: (num_modes, 1, hidden_size)  |
+-------------------+
          |
          v
+-------------------+
| Agent 投影         |
| self.actor_proj   |
|                   |
| 输入: (num_actors, hidden_size)  |
| 输出: (num_modes, num_actors, hidden_size)  |
+-------------------+
          |
          v
+-------------------+
| 目标嵌入           |
| tgt_embed         |
|                   |
| 形状: (num_modes, num_actors, hidden_size)  |
+-------------------+
          |
          v
+-------------------+
| 特征融合           |
|                   |
| 输入: (num_modes, num_actors, hidden_size)  |
| 输出: (num_modes, num_actors, hidden_size)  |
+-------------------+
          |
          v
+-------------------+
| 分类和回归        |
|                   |
| cls_embed: (num_modes, 1, hidden_size)  |
| actor_embed: (num_modes, num_actors, hidden_size)  |
| tgt_embed: (num_modes, num_actors, hidden_size)  |
| embed: (num_modes, num_actors, hidden_size)  |
+-------------------+
          |
          +------------------------------------+
          |                                    |
          v                                    v
+-------------------+                  +-------------------+
| 分类               |                  | 回归               |
| self.cls          |                  | self.reg          |
|                   |                  |                   |
| 输入: (num_modes, 1, hidden_size)  |  | 输入: (num_modes, num_actors, hidden_size)  |
| 输出: (num_modes, 1)  |  | 输出: (num_modes, num_actors, N_ORDER + 1, 5)  |
+-------------------+                  +-------------------+
          |                                    |
          v                                    v
+-------------------+                  +-------------------+
| 结果收集           |                  | 结果收集           |
|                   |                  |                   |
| res_cls.append(cls)|  | res_reg.append(reg)|
| res_aux.append((vel, cov_vel, param))|  | res_aux.append((vel, cov_vel, param))|
+-------------------+                  +-------------------+
'''
class SceneDecoder(nn.Module):
    def __init__(self,
                    device,
                    param_out='none',
                    hidden_size=128,
                    future_steps=30,
                    num_modes=6) -> None:
        """
        SceneDecoder类的初始化方法。

        参数:
        - device: 用于指定运行设备（如CPU或GPU）。
        - param_out: 指定输出参数的类型，默认为'none'。也可以是 'bezier'或'monomial'
        - hidden_size: 隐藏层的大小，默认为128。
        - future_steps: 预测的未来步数，默认为30。
        - num_modes: 模式数量，默认为6。

        返回:
        无返回值。
        """
        super(SceneDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.device = device
        self.param_out = param_out

        # 1. Actor 投影
        dim_mm = self.hidden_size * num_modes
        dim_inter = dim_mm // 2
        self.actor_proj = nn.Sequential(
            nn.Linear(self.hidden_size, dim_inter), # 第一层线性变换
            nn.LayerNorm(dim_inter),    # 层归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Linear(dim_inter, dim_mm),   # 第二层线性变换
            nn.LayerNorm(dim_mm),   # 层归一化
            nn.ReLU(inplace=True)   # ReLU激活函数
        )

        # 2. Context 投影: 通常包括道路布局、其他车辆的位置和速度等
        self.ctx_proj = nn.Sequential(
            nn.Linear(self.hidden_size, dim_inter), # 第一层线性变换
            nn.LayerNorm(dim_inter),    # 层归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Linear(dim_inter, dim_mm),   # 第二层线性变换
            nn.LayerNorm(dim_mm),   # 层归一化
            nn.ReLU(inplace=True)   # ReLU激活函数
        )

        # Transformer 编码器：several layers of transformer encoder
        enc_layer = TransformerEncoderLayer(d_model=self.hidden_size,
                                            nhead=4, dim_feedforward=self.hidden_size * 12)
        self.ctx_sat = TransformerEncoder(enc_layer, num_layers=2, enable_nested_tensor=False)	# context saturation上下文饱和层，通过多层 Transformer 编码器对上下文特征进行深度处理以增强上下文信息的表示能力


        # 4. RPE 嵌入投影：linear projection for rpe embedding rpe_dim = 11
        self.proj_rpe = nn.Sequential(
            nn.Linear(5 * 2 * 2, self.hidden_size), # 输入维度为20，输出维度为hidden_size
            nn.LayerNorm(self.hidden_size), # 层归一化
            nn.ReLU(inplace=True)   # ReLU激活函数
        )

        # 5. Target 特征投影
        self.proj_tgt = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True)
        )

        # 6. 分类头
        self.cls = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1)
        )

        # 7. 回归头
        if self.param_out == 'bezier':
            self.N_ORDER = 7
            self.mat_T = self._get_T_matrix_bezier(n_order=self.N_ORDER, n_step=future_steps).to(self.device)   # 生成贝塞尔曲线的参数矩阵 T
            self.mat_Tp = self._get_Tp_matrix_bezier(n_order=self.N_ORDER, n_step=future_steps).to(self.device) # 生成贝塞尔曲线的一阶导数参数矩阵 Tp
            self.reg = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, (self.N_ORDER + 1) * 5)
            )
        elif self.param_out == 'monomial':
            self.N_ORDER = 7
            self.mat_T = self._get_T_matrix_monomial(n_order=self.N_ORDER, n_step=future_steps).to(self.device) # 生成多项式的参数矩阵 T
            self.mat_Tp = self._get_Tp_matrix_monomial(n_order=self.N_ORDER, n_step=future_steps).to(self.device)   # 生成多项式的一阶导数参数矩阵 Tp
            self.reg = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, (self.N_ORDER + 1) * 5)
            )
        elif self.param_out == 'none':
            self.reg = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.future_steps * 5)
            )
        else:
            raise NotImplementedError

    def _get_T_matrix_bezier(self, n_order, n_step):
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        T = []
        for i in range(n_order + 1):
            coeff = math.comb(n_order, i) * (1.0 - ts) ** (n_order - i) * ts ** i
            T.append(coeff)
        return torch.Tensor(np.array(T).T)

    def _get_Tp_matrix_bezier(self, n_order, n_step):
        # ~ 1st derivatives
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        Tp = []
        for i in range(n_order):
            coeff = n_order * math.comb(n_order - 1, i) * (1.0 - ts) ** (n_order - 1 - i) * ts ** i
            Tp.append(coeff)
        return torch.Tensor(np.array(Tp).T)

    def _get_T_matrix_monomial(self, n_order, n_step):
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        T = []
        for i in range(n_order + 1):
            coeff = ts ** i
            T.append(coeff)
        return torch.Tensor(np.array(T).T)

    def _get_Tp_matrix_monomial(self, n_order, n_step):
        # ~ 1st derivatives
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        Tp = []
        for i in range(n_order):
            coeff = (i + 1) * (ts ** i)
            Tp.append(coeff)
        return torch.Tensor(np.array(Tp).T)

    def forward(self,
                ctx: torch.Tensor,  # 交通上下文特征
                actors: torch.Tensor,   # agent特征
                actor_idcs: List[Tensor],   # agent索引列表
                tgt_feat: torch.Tensor, # 目标特征
                tgt_rpes: torch.Tensor):    # 目标相对于位置的嵌入
        """
        模型的前向传播函数。

        参数:
        - ctx: 交通上下文特征，类型为torch.Tensor
        - actors: agent特征，类型为torch.Tensor
        - actor_idcs: agent索引列表，类型为List[Tensor]
        - tgt_feat: 目标特征，类型为torch.Tensor
        - tgt_rpes: 目标相对于位置的嵌入，类型为torch.Tensor

        返回值:
        - res_cls: 分类结果列表
        - res_reg: 回归结果列表
        - res_aux: 辅助结果列表
        """
        res_cls, res_reg, res_aux = [], [], []
        # 1. RPE 嵌入投影: high-level commands的target node
        tgt_rpes = self.proj_rpe(tgt_rpes)  # [n_av, 128]
        # 2. 目标特征投影
        if len(tgt_feat.shape) == 1:
            tgt_feat = tgt_feat.unsqueeze(0)    # 如果目标特征维度为1，增加一个维度

        # 对目标特征和相对位置嵌入的组合进行投影
        tgt = self.proj_tgt(torch.cat([tgt_feat, tgt_rpes], dim=-1))

        # 3. 遍历每个agent索引，进行特征嵌入和预测
        for idx, a_idcs in enumerate(actor_idcs):
            _ctx = ctx[idx].unsqueeze(0)    # 获取当前交通上下文并调整维度
            _actors = actors[a_idcs]    # 获取当前agent特征

            # 4. 上下文投影：对交通上下文进行投影并调整维度，然后进行饱和处理
            cls_embed = self.ctx_proj(_ctx).view(-1, self.num_modes, self.hidden_size).permute(1, 0, 2)
            cls_embed = self.ctx_sat(cls_embed)

            # 5. Agent 投影：对agent特征进行投影并调整维度
            actor_embed = self.actor_proj(_actors).view(-1, self.num_modes, self.hidden_size).permute(1, 0, 2)

            # 6. 目标嵌入：初始化目标嵌入，并将第一个目标嵌入设置为投影后的目标特征
            # 猜测：high-level commands的target node
            tgt_embed = torch.zeros_like(actor_embed)
            tgt_embed[0] = tgt[idx].unsqueeze(0)

            # 7. 特征融合：进行分类和回归预测
            embed = cls_embed + actor_embed + tgt_embed # 结合交通上下文、agent和目标嵌入

            # 8. 分类：分类头的主要作用是确定每个预测模式的概率分布。具体来说，分类头会输出一个概率值，表示每个预测模式的可能性
            cls = self.cls(cls_embed).view(self.num_modes, -1)

            # 9. 回归：回归头的主要作用是生成具体的未来轨迹。具体来说，回归头会输出每个预测模式下的具体轨迹点坐标及其相关属性（如速度、协方差等）
            if self.param_out == 'bezier':
                # 通过 self.reg 层处理 embed 向量，然后将其重塑为形状 (num_modes, -1, N_ORDER + 1, 5) 的张量
                # param 包含了不同模式下的贝塞尔曲线参数，每个模式有 N_ORDER + 1 个控制点，每个控制点有 5 个参数
                param = self.reg(embed).view(self.num_modes, -1, self.N_ORDER + 1, 5)

                # 作用：提取 param 中前两个维度的数据，即位置参数，并重新排列维度顺序。
                # 物理意义：reg_param 包含了位置参数，每个控制点有 2 个位置坐标。
                reg_param = param[..., :2]
                reg_param = reg_param.permute(1, 0, 2, 3)

                # reg 可以被看作是未来位置的一个预测或表示
                reg = torch.matmul(self.mat_T, reg_param)   # 矩阵乘法 torch.matmu
                vel = torch.matmul(self.mat_Tp, torch.diff(reg_param, dim=2)) / (self.future_steps * 0.1)   # 通过矩阵乘法 torch.matmul 和差分 torch.diff 计算速度 vel

                # 作用：提取 param 中从第三个维度开始的数据，即协方差参数，并重新排列维度顺序。
                # 物理意义：cov_param 包含了协方差参数，每个控制点有 3 个协方差参数
                cov_param = param[..., 2:]
                cov_param = cov_param.permute(1, 0, 2, 3)

                # 计算 协方差cov 和 协方差速度cov_vel
                cov = torch.matmul(self.mat_T, cov_param)
                cov_vel = torch.matmul(self.mat_Tp, torch.diff(cov_param, dim=2)) / (self.future_steps * 0.1)

            elif self.param_out == 'monomial':
                param = self.reg(embed).view(self.num_modes, -1, self.N_ORDER + 1, 5)
                reg_param = param[..., :2]
                reg_param = reg_param.permute(1, 0, 2, 3)
                reg = torch.matmul(self.mat_T, reg_param)
                vel = torch.matmul(self.mat_Tp, reg_param[:, :, 1:, :]) / (self.future_steps * 0.1)
                cov_param = param[..., 2:]
                cov_param = cov_param.permute(1, 0, 2, 3)
                cov = torch.matmul(self.mat_T, cov_param)
                cov_vel = torch.matmul(self.mat_Tp, torch.diff(cov_param, dim=2)) / (self.future_steps * 0.1)

            elif self.param_out == 'none':
                param = self.reg(embed).view(self.num_modes, -1, self.N_ORDER + 1, 5)
                reg = param[..., :2]
                reg = reg.permute(1, 0, 2, 3)
                vel = torch.gradient(reg, dim=-2)[0] / 0.1
                cov = param[..., 2:]
                cov = cov.permute(1, 0, 2, 3)
                cov_vel = torch.gradient(cov, dim=-2)[0] / 0.1

            # 将回归结果和辅助信息组合并存储
            reg = torch.cat([reg, torch.exp(cov)], dim=-1)

            cls = cls.permute(1, 0)
            cls = F.softmax(cls * 1.0, dim=1)
            res_cls.append(cls)
            res_reg.append(reg)
            if self.param_out == 'none':
                res_aux.append((vel, cov_vel, None))  # ! None is a placeholder
            else:
                res_aux.append((vel, cov_vel, param))

        # res_cls：表示对于每个模式（mode）的分类概率。具体来说，它包含了模型对不同未来轨迹模式的概率分布估计。这些概率可以用来评估不同预测路径的可能性大小，帮助决策系统选择最有可能的未来轨迹
        # res_reg：表示对于每个预测模式的具体轨迹参数。这些参数描述了预测轨迹的具体形状或位置信息，例如位置坐标、速度等。根据不同的 param_out 设置，回归结果可能包含贝塞尔曲线参数、多项式系数或其他形式的轨迹描述
        # res_aux：提供了额外的信息来辅助理解或评估回归结果。具体来说，它包含了预测轨迹的速度（vel）、协方差（cov_vel）以及在某些情况下还包含了原始的回归参数（param）
        return res_cls, res_reg, res_aux


class ScenePredNet(nn.Module):
    # Initialization
    def __init__(self, cfg, device):
        super(ScenePredNet, self).__init__()    # 调用父类的初始化方法
        self.device = device

        # 初始化演员网络，用于处理演员相关的数据和行为
        self.actor_net = ActorNet(n_in=cfg['in_actor'],
                                  hidden_size=cfg['d_actor'],
                                  n_fpn_scale=cfg['n_fpn_scale'])

        # 初始化车道网络，用于处理车道相关的数据和行为
        self.lane_net = LaneNet(device=self.device,
                                in_size=cfg['in_lane'],
                                hidden_size=cfg['d_lane'],
                                dropout=cfg['dropout'])

        # 初始化融合网络，用于融合演员网络和车道网络的信息
        self.fusion_net = FusionNet(device=self.device, config=cfg)

        # 初始化场景解码器，用于根据融合的信息预测未来的场景
        self.pred_scene = SceneDecoder(device=self.device,
                                       param_out=cfg['param_out'],
                                       hidden_size=cfg['d_embed'],
                                       future_steps=cfg['g_pred_len'],
                                       num_modes=cfg['g_num_modes'])

    def forward(self, data):
        """
        前向传播函数，用于处理输入数据并生成预测结果。

        参数:
        - data: 包含多个数据部分的元组，具体包括：
            - actors: 表示演员节点的特征
            - actor_idcs: 演员节点的索引
            - lanes: 表示车道节点的特征
            - lane_idcs: 车道节点的索引
            - rpe: relative positional embedding，相对位置误差，用于演员和车道之间的关联
            - tgt_nodes: 目标节点的特征
            - tgt_rpe: 目标节点的相对位置误差

        返回:
        - out: 预测的场景图模型输出
        """

        # 解包输入数据
        actors, actor_idcs, lanes, lane_idcs, rpe, tgt_nodes, tgt_rpe = data

        # 1.actors/lanes encoding，基于FPN和PointNet
        actors = self.actor_net(actors)  # output: [N_{actor}, 128]
        lanes = self.lane_net(lanes)  # output: [N_{lane}, 128]
        # 2.tgt encode, 这东西是high-level commands的出来的Target node。tgt_feat在pred_scene使用相当于high-level commands仅放在了decoder里
        tgt_feat = self.lane_net(tgt_nodes)  # output: [1, 128]
        # 3.fusion，这里是基于GNN的transformer encoding。actor和lane是node，rpe是edge
        actors, lanes, cls = self.fusion_net(actors, actor_idcs, lanes, lane_idcs, rpe)
        # 4.decoding，输入竟然没有lanes feature（但是有cls，相当于env feature了）
        out = self.pred_scene(cls, actors, actor_idcs, tgt_feat, tgt_rpe)   # cls就是SceneDecoder中forward里的context，是从FusionNet里来的

        return out

    def pre_process(self, data):
        actors = gpu(data['ACTORS'], self.device)
        actor_idcs = gpu(data['ACTOR_IDCS'], self.device)
        lanes = gpu(data['LANES'], self.device)
        lane_idcs = gpu(data['LANE_IDCS'], self.device)
        rpe = gpu(data['RPE'], self.device) # rpe: relative positional embedding
        tgt_nodes = gpu(data['TGT_NODES'], self.device)
        tgt_rpe = gpu(data['TGT_RPE'], self.device)

        return actors, actor_idcs, lanes, lane_idcs, rpe, tgt_nodes, tgt_rpe
