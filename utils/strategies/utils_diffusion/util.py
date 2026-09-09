import os

import numpy as np
import torch
import torch.nn.functional as F


def load_and_normalize_user_embeddings(pth_path: str, normalize: bool = False):
    print(f"Loading data from {pth_path}...")
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"文件未找到: {pth_path}")

    checkpoint = torch.load(pth_path, map_location="cpu")
    user_data = checkpoint.get("user_data", {})
    user_ids = list(user_data.keys())
    user_emb_list = []
    user_counts = []
    auc_list = []

    for uid in user_ids:
        embedding = user_data[uid]["embedding"]
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        user_emb_list.append(embedding.float())
        user_counts.append(len(user_data[uid].get("train_interactions", [])))
        auc_list.append(user_data[uid]["auc"])

    if not user_emb_list:
        raise ValueError("未找到任何用户 Embedding 数据")

    user_mat = torch.stack(user_emb_list, dim=0).numpy()
    user_counts = np.array(user_counts)
    if normalize:
        norms = np.linalg.norm(user_mat, axis=1, keepdims=True)
        user_mat = user_mat / np.clip(norms, 1e-12, None)

    print(
        f"Data Loaded. Users: {user_mat.shape[0]}, Dim: {user_mat.shape[1]}, "
        f"normalized={normalize}"
    )
    return user_mat, user_counts, user_ids, np.array(auc_list)


def build_groups(user_counts: np.ndarray, n_groups: int):
    sort_idx = np.argsort(user_counts)[::-1]
    return np.array_split(sort_idx, n_groups)


@torch.no_grad()
def compute_user_cond_vectors(train_dict, user_embs, item_embs, device):
    """根据用户历史交互构造条件向量。"""
    num_users, emb_dim = user_embs.shape
    user_cond_emb = torch.zeros(num_users, emb_dim, device=device)
    user_embs_norm = F.normalize(user_embs, p=2, dim=1)
    item_embs_norm = F.normalize(item_embs, p=2, dim=1)

    for user_id, pos_items in train_dict.items():
        if not pos_items:
            continue

        item_idx = torch.as_tensor(list(pos_items), dtype=torch.long, device=device)
        pos_item_emb_norm = item_embs_norm[item_idx]
        weights = torch.matmul(pos_item_emb_norm, user_embs_norm[user_id])
        user_cond_emb[user_id] = torch.sum(
            weights.unsqueeze(-1) * pos_item_emb_norm, dim=0
        )

    return F.normalize(user_cond_emb, p=2, dim=1)
