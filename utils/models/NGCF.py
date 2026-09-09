import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List
from .BaseModel import BaseRecommendationModel, BaseModelConfig

@dataclass
class NGCFConfig(BaseModelConfig):
    """NGCF模型配置"""
    emb_dim: int = 32          # 初始嵌入维度
    layer_sizes: List[int] = field(default_factory=lambda: [32, 32, 32]) # 每层的输出维度
    node_dropout: float = 0.0    # 邻接矩阵 Dropout (Edge Dropout)
    message_dropout: float = 0.0 # 特征传播过程中的 Dropout

class NGCF(BaseRecommendationModel):
    """
    NGCF: Neural Graph Collaborative Filtering
    Paper: SIGIR 2019
    """
    
    def __init__(self, config, user_num, item_num, device, train_dict=None):
        super(NGCF, self).__init__(config, user_num, item_num, device)

        self.emb_dim = self.config['emb_dim']
        self.layer_sizes = self.config['layer_sizes']
        self.node_dropout_rate = self.config.get('node_dropout', 0.0)
        self.mess_dropout_rate = self.config.get('message_dropout', 0.0)

        # 1. 初始化 Embedding
        self.user_embedding = self._init_embedding(self.num_users, self.emb_dim)
        self.item_embedding = self._init_embedding(self.num_items, self.emb_dim)

        # 2. 初始化权重矩阵 (W1, W2)
        # NGCF 的传播公式: E = LeakyReLU( (L+I)E W1 + (L E ⊙ E) W2 )
        self.GC_Linear_list = nn.ModuleList() # W1: 处理线性聚合部分
        self.Bi_Linear_list = nn.ModuleList() # W2: 处理特征交互部分
        
        input_size = self.emb_dim
        for out_size in self.layer_sizes:
            self.GC_Linear_list.append(nn.Linear(input_size, out_size))
            self.Bi_Linear_list.append(nn.Linear(input_size, out_size))
            input_size = out_size # 下一层的输入是上一层的输出

        # 3. 激活与 Dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.mess_dropout = nn.Dropout(self.mess_dropout_rate)

        # 4. 预计算归一化邻接矩阵
        self.norm_adj = self._create_norm_adj(train_dict).to(device)
        
        # 初始化权重参数 (Xavier)
        self._init_weights()

    def _init_weights(self):
        """对线性层进行 Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _sparse_dropout(self, x, rate):
        """针对稀疏矩阵的 Edge Dropout"""
        if rate == 0.0:
            return x
            
        noise_shape = x._nnz()
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]
        
        # Rescale 以保持数值期望一致
        v = v * (1.0 / (1.0 - rate))

        return torch.sparse_coo_tensor(i, v, x.shape).to(x.device)

    def _create_ngcf_embed(self, is_train=False):
        """
        NGCF 图卷积传播
        """
        # 初始 Embedding (Layer 0)
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        ego_embeddings = torch.cat([u_emb, i_emb], dim=0)
        
        all_embeddings = [ego_embeddings]

        # 决定是否对邻接矩阵进行 Dropout
        if is_train and self.node_dropout_rate > 0:
            A_hat = self._sparse_dropout(self.norm_adj, self.node_dropout_rate)
        else:
            A_hat = self.norm_adj

        # 逐层传播
        for k in range(len(self.layer_sizes)):
            # 1. 邻居聚合: L * E
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            # 2. 计算两条路径
            # 路径A: (L * E + I * E) * W1 -> Sum Aggregation
            # 这里的 ego_embeddings 相当于 self-connection (I*E)
            sum_embeddings = self.GC_Linear_list[k](side_embeddings + ego_embeddings)

            # 路径B: (L * E ⊙ I * E) * W2 -> Bi-Interaction Aggregation
            # element-wise product 增强特征交互
            bi_embeddings = self.Bi_Linear_list[k](side_embeddings * ego_embeddings)

            # 3. 合并与激活: LeakyReLU(Sum + Bi)
            ego_embeddings = self.leaky_relu(sum_embeddings + bi_embeddings)

            # 4. Message Dropout
            if is_train:
                ego_embeddings = self.mess_dropout(ego_embeddings)
            
            ego_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings.append(ego_embeddings)

        # NGCF 的做法是将所有层的 embedding 拼接 (Concatenation)
        # 维度变成: emb_dim + layer1 + layer2 + ...
        # final_embeddings = torch.cat(all_embeddings, dim=1)
        # 不要拼接，取平均值
        all_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(final_embeddings, [self.num_users, self.num_items], dim=0)
        
        return u_g_embeddings, i_g_embeddings

    def forward(self, user_indices, item_indices, return_emb=False, is_train=True):
        """前向传播"""
        u_g_embeddings, i_g_embeddings = self._create_ngcf_embed(is_train=is_train)
        
        user_emb = u_g_embeddings[user_indices]
        item_emb = i_g_embeddings[item_indices]

        if return_emb:
            return user_emb, item_emb
        
        scores = self.compute_scores(user_emb, item_emb)
        return scores
    
    def get_embeddings_for_fair_loss(self, samples: torch.LongTensor) -> tuple:
        """用于公平性Loss"""
        u_g_embeddings, _ = self._create_ngcf_embed(is_train=False)
        samples = samples.to(u_g_embeddings.device)
        inactive_emb = u_g_embeddings[samples[:, 0]]
        neighbor_emb = u_g_embeddings[samples[:, 1]]
        return inactive_emb, neighbor_emb
    
    @torch.no_grad()
    def get_embedding(self) -> tuple:
        """推理阶段获取 Embedding"""
        user_emb, item_emb = self._create_ngcf_embed(is_train=False)
        return user_emb, item_emb