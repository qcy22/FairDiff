import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import math
from typing import List, Dict, Optional
from torch.utils.data import Dataset, DataLoader
from .base import BaseStrategy, TrainingResult
from .utils_diffusion import Diffusion, AmbientLoss, fit_timestamps, TinyUNet1D, compute_user_cond_vectors

class DiffusionDataset(Dataset):
    """
    增加了课程学习功能的扩散数据集
    """
    def __init__(
        self,
        xtn_all: torch.Tensor,
        sigma_all: torch.Tensor,
        user_cond_emb: torch.Tensor,
        sigma_min: float = 0.01,
        sigma_max: float = 10.0,
        sigma_safety_factor: float = 0.9,
        dataset_len: int = 1,
        min_curriculum_ratio: float = 0.1, # 新增：最小课程比例，防止初期样本太少
    ):
        assert xtn_all.size(0) == sigma_all.size(0) == user_cond_emb.size(0), \
            "xtn_all, sigma_all, user_cond_emb 必须在第 0 维长度一致"
        
        self.xtn_all = xtn_all
        self.sigma_all = sigma_all
        self.user_cond_emb = user_cond_emb

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_sigma_min = math.log(sigma_min)
        self.log_sigma_max = math.log(sigma_max)
        self.sigma_safety_factor = sigma_safety_factor
        self.dataset_len = dataset_len

        # --- 排序逻辑保持不变 (从小到大) ---
        sigma_1d = self.sigma_all.view(-1)
        sorted_sigma, sorted_idx = torch.sort(sigma_1d)
        self.sigma_all = sorted_sigma.view(-1, 1)
        self.sigma_all_d1 = self.sigma_all.view(-1)
        self.xtn_all = self.xtn_all[sorted_idx]
        self.user_cond_emb = self.user_cond_emb[sorted_idx]

        # --- 新增：课程学习控制变量 ---
        self.total_samples = self.sigma_all.size(0)
        self.curriculum_ratio = 1.0 # 默认为 1.0 (全量)，由外部策略控制更新
        self.min_curriculum_ratio = min_curriculum_ratio

    def set_curriculum_ratio(self, ratio: float):
        """
        外部调用此方法更新课程进度
        ratio: 0.0 ~ 1.0
        """
        # 保证至少有一定比例的样本，避免 crash
        self.curriculum_ratio = max(ratio, self.min_curriculum_ratio)

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, idx: int):
        # 1. 采样目标 sigma
        log_sigma = torch.empty(1).uniform_(self.log_sigma_min, self.log_sigma_max)
        sigma = log_sigma.exp()
        threshold = sigma * self.sigma_safety_factor

        # 2. 二分查找满足条件的物理边界
        # 找到所有 sigma_all <= threshold 的位置
        upper_bound = torch.searchsorted(self.sigma_all_d1, threshold.to(self.sigma_all.device)).item()

        # 3. 【核心修改】应用课程学习限制
        # 计算当前课程允许的最大索引（例如只允许前 30% 的简单样本）
        curriculum_limit = int(self.total_samples * self.curriculum_ratio)
        
        # 实际的上界是：物理阈值 与 课程限制 的交集（取最小值）
        # 这样既保证了 sigma 采样的合法性，又保证了不超出当前课程的难度
        effective_upper_bound = min(upper_bound, curriculum_limit)

        # 4. 边界处理 (防止为 0)
        if effective_upper_bound <= 0:
            effective_upper_bound = 1

        # 5. 随机选择
        chosen_idx = torch.randint(0, effective_upper_bound, (1,)).item()

        return (
            sigma.view_as(self.sigma_all[chosen_idx]),
            self.xtn_all[chosen_idx],
            self.sigma_all[chosen_idx],
            self.user_cond_emb[chosen_idx],
        )


# ----------------- Diffusion Strategy -----------------
class DiffusionStrategy(BaseStrategy):
    """双阶段训练策略：
       阶段1：仅底层 BPR
       阶段2：冻结基础 embedding，训练扩散增强用户表征
    """
    def __init__(self, 
                 display_name: str, 
                 model: torch.nn.Module, 
                 lr: float, 
                 weight_decay: float,
                 dataset,
                 diffusion_weight: float = 1.0,
                 freeze_epoch: int = 30,
                 sigma_min: float = 0.02,
                 sigma_max: float = 3.0,
                 sigma_data: float = 0.2,
                 trial_times: int = 30,
                 use_curriculum: bool = False,
                 use_ambient: bool = True,
                 sample_from_gaussian: bool = False,
                 guidance_scale: float = 1.0,
                 curriculum_epochs: int = 50
        ):

        super().__init__(display_name, model, lr, weight_decay, dataset)
        
        self.diffusion_weight = diffusion_weight
        self._phase2_ready = False
        self.denoiser = None
        self.diffusion = None
        self.freeze_epoch = freeze_epoch
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.trial_times = trial_times
        self.use_curriculum = use_curriculum
        self.use_ambient = use_ambient
        self.sample_from_gaussian = sample_from_gaussian
        self.guidance_scale = guidance_scale
        self.curriculum_epochs = curriculum_epochs
        self._cached_user_emb = None
        self._cached_item_emb = None
        self.sigma_all = None
        self.xtn_all = None
        self.user_cond_emb = None

        self.last_infer_epoch = -1
        self.last_infer_user_emb = None

    def train_epoch(self, epoch: int, device):
        # 检查是否进入第二阶段
        if epoch >= self.freeze_epoch:
            self._init_phase2()
            
            # --- 新增：计算并更新课程进度 ---
            # 计算相对于 freeze_epoch 的偏移量
            phase2_epoch = epoch - self.freeze_epoch
            
            if (phase2_epoch < self.curriculum_epochs) and self.use_curriculum:
                # 线性增长：从 min_ratio 增长到 1.0
                # 假设 min_ratio 在 dataset 里默认 0.1，这里计算 0.1 + 0.9 * 进度
                progress = phase2_epoch / self.curriculum_epochs
                # 简单的线性策略： ratio = progress (dataset 内部会 handle 最小值)
                # 或者更平滑的策略： ratio = 0.2 + 0.8 * progress
                current_ratio = 0.1 + 0.9 * progress
            else:
                current_ratio = 1.0

            # 获取 dataset 并更新
            # 注意：self.dataloader.dataset 是 DiffusionDataset
            self.dataloader.dataset.set_curriculum_ratio(current_ratio)
                
            if phase2_epoch % 5 == 0:
                print(f"[Curriculum] Epoch {epoch}: Ratio={current_ratio:.4f}, "
                      f"Max Index={int(self.dataloader.dataset.total_samples * current_ratio)}")

        super().train_epoch(epoch, device)

    def _train_step(self, batch_data: Tuple, epoch) -> TrainingResult:
        if epoch < self.freeze_epoch:
            return super()._train_step(batch_data, epoch)
        else:
            return self._phase2_train(batch_data)
    
    def _init_phase2(self):

        if self._phase2_ready:
            return

        # 冻结底层 embedding
        self.model.user_embedding.weight.requires_grad_(False)
        self.model.item_embedding.weight.requires_grad_(False)

        device = self.model.device
        
        all_user_embedding, all_item_embedding = self.model.get_embedding()
        all_user_embedding_norm = F.normalize(all_user_embedding, p=2, dim=1)
        # all_user_embedding_norm = all_user_embedding
        self._cached_user_emb = all_user_embedding_norm.to(device).detach()
        self._cached_item_emb = all_item_embedding.to(device).detach()

        user_counts = self.dataset.get_user_train_nums()

        self.denoiser = TinyUNet1D(dim=self.model.emb_dim, depth=5, hidden=256).to(device)
        self.loss_fn = AmbientLoss(sigma_data=self.sigma_data, cond_drop_prob=0.1)
        self.diffusion = Diffusion(sigma_min=self.sigma_min, sigma_max=self.sigma_max)

        self.optimizer = self._get_optimizer(
            list(self.denoiser.parameters()), 2e-4, self.weight_decay
        )

        train_dict = self.dataset.train_pos_dict

        self.user_cond_emb = compute_user_cond_vectors(
            train_dict=train_dict,
            user_embs=self._cached_user_emb,
            item_embs=self._cached_item_emb,
            device=device,
        )

        if self.use_ambient:

            # 从时间标定中获取每个用户的 sigma_all 和对应的含噪样本 xtn_all
            sigma_all, xtn_all, sigma_all_rep, xtn_all_rep = fit_timestamps(
                user_emb_list=self._cached_user_emb, 
                user_counts=user_counts, 
                normalize=False,  
                n_groups=self.dataset.user_tiers, 
                batch_size=4096, 
                epochs=300, 
                lr=5e-4,
                device=device, 
                assign_num_time_steps=50,
                scheduler=self.diffusion,
                repeat_times=self.trial_times,
            )

            self.sigma_all = sigma_all.unsqueeze(1).to(device)
            self.xtn_all = xtn_all.to(device)
            self.sigma_all_rep = sigma_all_rep.unsqueeze(1).to(device)
            self.xtn_all_rep = xtn_all_rep.to(device)
            self.user_cond_emb_rep = self.user_cond_emb.repeat_interleave(repeats=self.trial_times, dim=0).to(device)

            self.user_sigma_max = self.sigma_all.max()
            print(f"[DiffusionStrategy] 用户噪声水平范围: [{self.sigma_all.min().item():.4f}, {self.sigma_all.max().item():.4f}]")

        else:
            # 不使用 ambient 信息时，sigma_all 全为 0，xtn_all 即为原始用户嵌入
            num_users = self._cached_user_emb.size(0)
            self.sigma_all = torch.zeros((num_users, 1), device=device)
            self.xtn_all = self._cached_user_emb

        # 使用 ambient 信息构建扩散训练数据集和 DataLoader
        self._build_diffusion_dataset()

        self._phase2_ready = True

    def _build_diffusion_dataset(self):
        if self.use_ambient:
            xtn_dataset = self.xtn_all_rep
            sigma_dataset = self.sigma_all_rep
            user_cond_dataset = self.user_cond_emb_rep
        else:
            xtn_dataset = self._cached_user_emb
            sigma_dataset = torch.zeros_like(self.sigma_all)
            user_cond_dataset = self.user_cond_emb

        diffusion_dataset = DiffusionDataset(
            xtn_all=xtn_dataset.cpu(),
            sigma_all=sigma_dataset.cpu(),
            user_cond_emb=user_cond_dataset.cpu(),
            sigma_min=self.diffusion.sigma_min,
            sigma_max=self.diffusion.sigma_max,
            sigma_safety_factor=0.9,
            dataset_len=len(self.dataloader.dataset)
        )
        batch_size = self.dataloader.batch_size
        self.dataloader = DataLoader(
            diffusion_dataset,
            batch_size=batch_size,
            shuffle=True,       
            num_workers=24,
            pin_memory=torch.cuda.is_available(),
        )

    def _phase2_train(self, batch_data) -> TrainingResult:
        self.denoiser.train()
        self.optimizer.zero_grad()

        sampled_sigma_batch, xtn_batch, user_sigma_batch, user_cond_emb_batch = batch_data

        diff_loss, x0_user_hat = self.loss_fn(
            net=self.denoiser,
            embs=xtn_batch,
            current_sigma=user_sigma_batch,
            target_sigma=sampled_sigma_batch,
            cond_embs=user_cond_emb_batch,
        )
        diff_loss = (diff_loss.mean()) * self.diffusion_weight

        bpr_loss = torch.tensor(0.0, device=user_cond_emb_batch.device)

        total_loss = bpr_loss + diff_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), max_norm=1.0)
        self.optimizer.step()

        return TrainingResult(
            total_loss=float(total_loss.item()),
            metrics={
                "bpr_loss": float(bpr_loss.item()),
                "diff_loss": float(diff_loss.item()),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }
        )

    @torch.no_grad()
    def infer(self, user_set, pos_item_set, neg_item_set, user_tiers, batch_size, epoch):
        self.model.eval()
        if self.denoiser is not None:
            self.denoiser.eval()
        pos_predictions = []
        neg_predictions = []
        user_embeddings = None
        item_embeddings = None

        if not self._phase2_ready:
            user_embeddings, item_embeddings = self.model.get_embedding()

        else:
            item_embeddings = self._cached_item_emb.clone()

            start_idx_all = 0
            start_idx_num = 0

            if epoch == self.last_infer_epoch and self.last_infer_user_emb is not None:
                user_embeddings = self.last_infer_user_emb
            else:
                user_embeddings = self._cached_user_emb.clone()
                device = self._cached_user_emb.device
                user_set_tensor = user_set.to(device)
                unique_users, inverse_indices = torch.unique(user_set_tensor, return_inverse=True)

                for start in range(0, unique_users.size(0), batch_size):
                    end = min(start + batch_size, unique_users.size(0))
                    batch_users = unique_users[start:end]

                    user_cond_emb = self.user_cond_emb[batch_users]
                    if self.sample_from_gaussian:
                        sigma_max_tensor = torch.tensor(self.sigma_max)
                        user_sigma = sigma_max_tensor.expand(len(batch_users), 1).to(device)
                        user_emb = self._cached_user_emb[batch_users]
                        user_emb_noisy = user_emb + torch.randn_like(user_emb) * user_sigma
                    else:
                        user_sigma = self.sigma_all[batch_users]
                        user_emb_noisy = self.xtn_all[batch_users]

                    enhanced_batch,start_idx_sample = self.diffusion.sample(
                        net=self.denoiser,
                        embs=user_emb_noisy,
                        cond_embs=user_cond_emb,
                        num_steps=50,
                        cur_sigma=user_sigma,
                        deterministic=True,
                        guidance_scale=self.guidance_scale,
                    )
                    user_embeddings[batch_users] = enhanced_batch
                    start_idx_num += start_idx_sample.size(0)
                    start_idx_all += start_idx_sample.sum().item()

                self.last_infer_epoch = epoch
                self.last_infer_user_emb = user_embeddings.clone()

                print(f"[DiffusionStrategy] Inference avg start_idx: {start_idx_all / start_idx_num:.2f}")
            
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


