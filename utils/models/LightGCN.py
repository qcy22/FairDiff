import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .BaseModel import BaseRecommendationModel, BaseModelConfig

@dataclass
class LightGCNConfig(BaseModelConfig):
    """LightGCN模型配置"""
    emb_dim: int = 32        # 嵌入维度
    n_layers: int = 3        # GCN层数
    node_dropout: float = 0.0    # [图结构] 邻接矩阵 Dropout (Edge Dropout)
    message_dropout: float = 0.0 # [特征层] 特征传播过程中的 Dropout

class LightGCN(BaseRecommendationModel):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
    [增强版] 仿照 NGCF 实现了双重 Dropout 机制
    """
    
    def __init__(self, config, user_num, item_num, device, train_dict=None):
        super(LightGCN, self).__init__(config, user_num, item_num, device)
        
        self.emb_dim = self.config['emb_dim']
        self.n_layers = self.config['n_layers']
        
        self.node_dropout_rate = self.config.get('node_dropout', 0.0)
        self.mess_dropout_rate = self.config.get('message_dropout', 0.0)

        self.user_embedding = self._init_embedding(self.num_users, self.emb_dim)
        self.item_embedding = self._init_embedding(self.num_items, self.emb_dim)
        
        self.mess_dropout = nn.Dropout(self.mess_dropout_rate)

        self.norm_adj = self._create_norm_adj(train_dict).to(device)

        self.cached_user_embedding = None
        self.cached_item_embedding = None

    def _sparse_dropout(self, x, rate):
        """
        [Node Dropout] 针对稀疏邻接矩阵的 Dropout
        x: torch.sparse.FloatTensor
        rate: float, drop 概率
        """
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
        
        # Rescale 以保持数值期望一致 (Scale up)
        v = v * (1.0 / (1.0 - rate))

        return torch.sparse_coo_tensor(i, v, x.shape).to(x.device)

    def _create_lightgcn_embed(self, is_train=False):
        """
        LightGCN 图卷积传播 (含 Node Dropout 和 Message Dropout)
        """
        if self.cached_user_embedding is not None and self.cached_item_embedding is not None:
            return self.cached_user_embedding.weight, self.cached_item_embedding.weight

        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        # --- 1. Node Dropout (针对图结构) ---
        # 只有在训练模式且设定了 rate 时才启用
        if is_train and self.node_dropout_rate > 0:
            A_hat = self._sparse_dropout(self.norm_adj, self.node_dropout_rate)
        else:
            A_hat = self.norm_adj
        
        # --- 2. 图卷积传播 ---
        for k in range(self.n_layers):
            # 聚合邻居信息
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            
            # --- 3. Message Dropout (针对特征) ---
            # 仿照 NGCF，在每一层聚合后，对特征向量进行 Dropout
            if is_train and self.mess_dropout_rate > 0:
                ego_embeddings = self.mess_dropout(ego_embeddings)

            ego_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            
            all_embeddings.append(ego_embeddings)
            
        # 聚合所有层 (LightGCN 使用 Mean, NGCF 使用 Concat)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(final_embeddings, [self.num_users, self.num_items], dim=0)
        
        return u_g_embeddings, i_g_embeddings

    def forward(self, user_indices, item_indices, return_emb=False, is_train=True):
        """前向传播计算预测分数"""
        u_g_embeddings, i_g_embeddings = self._create_lightgcn_embed(is_train=is_train)
        
        user_emb = u_g_embeddings[user_indices]
        item_emb = i_g_embeddings[item_indices]

        if return_emb:
            return user_emb, item_emb
        
        scores = self.compute_scores(user_emb, item_emb)
        return scores
    
    def get_embeddings_for_fair_loss(self, samples: torch.LongTensor) -> tuple:
        """获取目标用户和邻居用户的嵌入（用于公平性损失）"""
        u_g_embeddings, _ = self._create_lightgcn_embed(is_train=False)
        samples = samples.to(u_g_embeddings.device)
        inactive_emb = u_g_embeddings[samples[:, 0]]
        neighbor_emb = u_g_embeddings[samples[:, 1]]

        return inactive_emb, neighbor_emb
    
    @torch.no_grad()
    def get_embedding(self) -> tuple:
        """获取用户和物品的嵌入表示"""
        user_emb, item_emb = self._create_lightgcn_embed(is_train=False)
        return user_emb, item_emb

    @torch.no_grad()
    def load_embedding(self, embedding_path: str, device):
        """加载预训练的嵌入向量"""

        self.cached_user_embedding = nn.Embedding(self.num_users, self.emb_dim).to(device)
        self.cached_item_embedding = nn.Embedding(self.num_items, self.emb_dim).to(device)

        data = torch.load(embedding_path, map_location=device)
        user_data = data['user_data']
        item_data = data['item_data']

        for user_id, user_data in user_data.items():
            self.cached_user_embedding.weight.data[user_id] = user_data['cached_user_emb'].to(device)

        for item_id, item_data in item_data.items():
            self.cached_item_embedding.weight.data[item_id] = item_data['cached_item_emb'].to(device)
