import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any
from abc import ABC
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler

@dataclass 
class TrainingResult:
    total_loss: float
    metrics: Dict[str, float] = field(default_factory=dict)

    # 返回一个初始为 0 的 TrainingResult
    @classmethod
    def zero(cls) -> "TrainingResult":
        return cls(0.0, {})

    # 就地累加另一个 TrainingResult（保留所有 metric key）
    def __iadd__(self, other: "TrainingResult") -> "TrainingResult":
        if not isinstance(other, TrainingResult):
            return NotImplemented
        self.total_loss += float(other.total_loss)
        for k, v in other.metrics.items():
            self.metrics[k] = self.metrics.get(k, 0.0) + float(v)
        return self

    # 返回按 n 平均后的新 TrainingResult（n 应>0）
    def mean(self, n: int) -> "TrainingResult":
        if n <= 0:
            return TrainingResult(self.total_loss, dict(self.metrics))
        avg_total = float(self.total_loss) / n
        avg_metrics = {k: float(v) / n for k, v in self.metrics.items()}
        return TrainingResult(avg_total, avg_metrics)

    # 格式化为可读字符串
    def formatted(self, precision: int = 4) -> str:
        metrics_str = ", ".join([f"{k}: {v:.{precision}f}" for k, v in self.metrics.items()])
        return f"total_loss: {self.total_loss:.{precision}f}" + (f", {metrics_str}" if metrics_str else "")
    
class BPRTrainDataset(Dataset):
    """BPR 训练数据集: 返回 (user, pos_item, neg_item, user_percentile, user_activity_group)
    其中 neg_item 在 __getitem__ 中按用户的负样本集合随机采样
    """
    def __init__(
        self,
        user_tensor,
        pos_item_tensor,
        user_percentile_tensor,
        user_activity_group_tensor,
        user_neg_items_dict,
    ):
        self.user_tensor = user_tensor
        self.pos_item_tensor = pos_item_tensor
        self.user_percentile_tensor = user_percentile_tensor
        self.user_activity_group_tensor = user_activity_group_tensor
        self.user_neg_items_dict = {
            int(u): np.asarray(items, dtype=np.int64)
            for u, items in user_neg_items_dict.items()
            if items
        }

    def __getitem__(self, index):
        user = int(self.user_tensor[index].item())
        pos_item = self.pos_item_tensor[index]
        user_percentile = self.user_percentile_tensor[index]
        user_activity_group = self.user_activity_group_tensor[index]

        neg_candidates = self.user_neg_items_dict.get(user)
        if neg_candidates is None or len(neg_candidates) == 0:
            neg_item = int(pos_item.item())
        else:
            neg_item = int(np.random.choice(neg_candidates))

        neg_item_tensor = torch.tensor(neg_item, dtype=torch.long)

        return (
            self.user_tensor[index],
            pos_item,
            neg_item_tensor,
            user_percentile,
            user_activity_group,
        )

    def __len__(self):
        return self.user_tensor.size(0)

class BaseStrategy(ABC):
    
    def __init__(self, 
                 display_name: str, 
                 model: torch.nn.Module, 
                 lr: float,
                 weight_decay: float,
                 dataset,
                 ):
        self.display_name = display_name
        self.model = model
        self.custom_weight = 1.0
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = self._get_optimizer(
            self.model.parameters(), lr, weight_decay
        )
        self.dataset = dataset

        self.dataloader = None

    def _get_optimizer(self, parameters, lr: float, weight_decay: float):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, parameters),
            lr=lr,
            weight_decay=weight_decay
        )

        return optimizer
    

    def build_dataloader(self, train_data, batch_size):
        train_pos_dict = train_data["train_pos_dict"]
        train_neg_dict = train_data["train_neg_dict"]
        user_activity_percentiles = train_data["user_activity_percentiles"]
        user_activity_tiers = train_data["user_activity_tiers"]
        
        out_users, out_pos = [], []
        for u, pos_items_u in train_pos_dict.items():
            neg_items_u = train_neg_dict.get(u)
            if not neg_items_u:
                continue
            for pi in pos_items_u:
                for _ in range(3):
                    out_users.append(u)
                    out_pos.append(pi)

        user_percentiles = [user_activity_percentiles[u] for u in out_users]
        user_activity_groups = [user_activity_tiers[u] for u in out_users]

        dataset = BPRTrainDataset(
            torch.LongTensor(out_users),
            torch.LongTensor(out_pos),
            torch.FloatTensor(user_percentiles),
            torch.LongTensor(user_activity_groups),
            train_neg_dict,
        )

        g = torch.Generator()
        g.manual_seed(42)

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=RandomSampler(dataset, generator=g),
            num_workers=24,
            pin_memory=torch.cuda.is_available(),
        )

    def train_epoch(self, epoch: int, device):
        """训练一个epoch的所有模型"""
        if self.dataloader is None:
            raise ValueError("Dataloader 未设置，无法进行训练。请在外部设置 self.dataloader。")

        results = TrainingResult.zero()
        
        batch_count = 0
        pbar = tqdm(total=len(self.dataloader), desc=f"Epoch {epoch} for {self.display_name}", leave=False)

        for batch_idx, data in enumerate(self.dataloader):
            batch_count += 1
            batch_data = [x.to(device) for x in data]
            
            result = self._train_step(batch_data, epoch)
            results += result
            pbar.update(1)
        pbar.close()

        denom = batch_count if batch_count > 0 else 1
        avg = results.mean(denom)
        print(f"{self.display_name} - {avg.formatted()}")

    def _train_step(self, batch_data: Tuple, epoch: int) -> TrainingResult:
        # 批格式：(user, pos_item, neg_item, user_percentile, user_activity_group)
        users, pos_items, neg_items, user_percentile, user_activity_group = batch_data
        compute_loss_extras = {
            "user_percentile": user_percentile,
            "user_activity_group": user_activity_group,
            "device": users.device,
        }

        self.model.train()
        self.optimizer.zero_grad()

        loss, metrics = self._compute_loss(users, pos_items, neg_items, compute_loss_extras)
        loss.backward()
        self.optimizer.step()

        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])

        return TrainingResult(
            total_loss=float(loss.item()),
            metrics=metrics
        )

    def _compute_loss(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor, extras: Dict[str, torch.Tensor]):
        """使用 BPR Loss: -log(sigmoid(s_pos - s_neg))"""
        # 打分
        pos_scores = self.model(users, pos_items)   # 形状 (N,)
        neg_scores = self.model(users, neg_items)   # 形状 (N,)

        # BPR 损失
        diff = pos_scores - neg_scores
        loss = F.softplus(-diff).mean()

        metrics = {
            "bpr_loss": float(loss.item()),
        }
        return loss, metrics



    @torch.no_grad()
    def infer(self, user_set, pos_item_set, neg_item_set, user_tiers, batch_size, epoch):
        self.model.eval()
        pos_predictions = []
        neg_predictions = []

        user_embeddings, item_embeddings = self.model.get_embedding()

        for start_idx in (range(0, len(user_set), batch_size)):
            end_idx = min(start_idx + batch_size, len(user_set))

            batch_user_embedding = user_embeddings[user_set[start_idx:end_idx]]
            pos_batch_item_embedding = item_embeddings[pos_item_set[start_idx:end_idx]]
            neg_batch_item_embedding = item_embeddings[neg_item_set[start_idx:end_idx]]

            pos_batch_predict = self.model.compute_scores(batch_user_embedding, pos_batch_item_embedding)
            neg_batch_predict = self.model.compute_scores(batch_user_embedding, neg_batch_item_embedding)

            pos_predictions.append(pos_batch_predict.cpu())
            neg_predictions.append(neg_batch_predict.cpu())
        
        return torch.cat(pos_predictions, dim=0), torch.cat(neg_predictions, dim=0), user_embeddings, item_embeddings


