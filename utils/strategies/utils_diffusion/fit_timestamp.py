import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from .unet import TimeConditionalClassifier
from .diffusion import Diffusion
from .util import load_and_normalize_user_embeddings, build_groups


# ==========================================
# 4. 训练分类器 (Equation 3.1)
# ==========================================
def prepare_datasets(pth_path, n_groups=5, batch_size=256):
    # 1. 加载 (强制使用 normalize=True 以适配 sigma_max=50)
    user_mat, user_counts, _, _ = load_and_normalize_user_embeddings(pth_path, normalize=True)
    # 使用数组版封装，保证返回 indices
    return prepare_datasets_from_arrays(user_mat, user_counts, n_groups=n_groups)

def prepare_datasets_from_arrays(user_mat: np.ndarray,
                                 user_counts: np.ndarray,
                                 n_groups: int = 5):
    """
    基于内存中的数组构建训练/可视化数据集，并返回 indices 以供对齐与持久化。
    """
    group_indices = build_groups(user_counts, n_groups)
    sg_indices = group_indices[0]
    sb_indices = np.concatenate(group_indices[1:]) if len(group_indices) > 1 else np.array([], dtype=int)

    sg_data = torch.tensor(user_mat[sg_indices], dtype=torch.float32)
    sb_data = torch.tensor(user_mat[sb_indices], dtype=torch.float32)
    # SB 对应的统计属性（与 ds_sb 顺序一致）
    sb_user_counts = user_counts[sb_indices]

    if len(sg_data) > 0 and len(sb_data) > 0:
        repeat_times = int(np.ceil(len(sb_data) / len(sg_data)))
        sg_balanced = sg_data.repeat((repeat_times, 1))[:len(sb_data)]
    elif len(sg_data) == 0:
        raise ValueError("SG 数据为空，无法平衡数据集")
    else:
        # 全部样本都在 SG，构造最小可训练集（退化情况）
        sg_balanced = sg_data

    X = torch.cat([sg_balanced, sb_data], dim=0)
    Y = torch.cat([torch.ones(len(sg_balanced)), torch.zeros(len(sb_data))], dim=0)
    combined_ds = TensorDataset(X, Y)

    ds_sb = TensorDataset(sb_data, torch.zeros(len(sb_data)))
    ds_sg = TensorDataset(sg_data, torch.ones(len(sg_data)))

    print(f"Group Split Results:")
    print(f"  SG (Clean) Size: {len(sg_data)} (Label=1) -> Balanced to {len(sg_balanced)}")
    print(f"  SB (Corrupt) Size: {len(sb_data)} (Label=0)")
    print(f"  Combined dataset size: {len(combined_ds)}")

    input_dim = sg_data.shape[1] if sg_data.numel() > 0 else (sb_data.shape[1] if sb_data.numel() > 0 else 0)
    return combined_ds, ds_sg, ds_sb, input_dim, sb_user_counts, sb_indices, sg_indices

def train_classifier(dataset_combined, input_dim, device='cuda', epochs=50, batch_size=256,
                     lr=3e-3, weight_decay=1e-4, scheduler: Diffusion = None):
    dl = DataLoader(dataset_combined, batch_size=batch_size, shuffle=True, drop_last=True)

    model = TimeConditionalClassifier(dim=input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler is None:
        raise ValueError("需要从外部传入统一初始化的 Diffusion 调度器 scheduler")
    criterion = nn.BCEWithLogitsLoss()

    print("\n>>> 开始训练分类器...")
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0

        for x_batch, y_batch in dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            # 随机采样 t ~ U[0, 1]
            t = torch.rand(x_batch.shape[0], device=device)
            sigma_t = scheduler.get_sigma(t)
            x_t = x_batch + sigma_t.view(-1, 1) * torch.randn_like(x_batch)

            # 前向与损失
            logits = model(x_t, sigma_t)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/max(steps,1):.4f}")

    return model

# ==========================================
# 5. 样本级标注 (Sample-dependent Annotation)
# ==========================================
def assign_sample_wise_times(model, dataset_sb, device='cuda', 
                             num_time_steps=100,
                             scheduler: Diffusion = None,
                             user_counts=None,
                             tau_min: float = 0.25,
                             tau_max: float = 0.48):
    """
    对每个 SB 样本，找到最小的 t，使得分类器误判概率在“任意一个时间步” > tau。
    返回：
      - final_embeddings: 原始 SB embedding（无噪声）
      - final_xt: 对应 t_min 时刻的含噪声样本 x_t（即当时输入 classifier 的 xt）
      - final_sigmas: t_min 对应的 sigma
      - final_tmins: t_min
    """
    model.eval()
    if scheduler is None:
        raise ValueError("需要从外部传入统一初始化的 Diffusion 调度器 scheduler")
    
    # 基于 user_counts 计算每个样本的动态阈值 tau_per_sample（可选）
    user_counts_np = np.asarray(user_counts, dtype=np.float64)
    ranks_for_color = np.argsort(np.argsort(user_counts_np))
    n = len(user_counts_np)
    counts_norm = ranks_for_color / (n - 1)

    tmin, tmax = float(tau_min), float(tau_max)
    if tmin > tmax:
        tmin, tmax = tmax, tmin

    # user_count 越大（rank 越靠后），counts_norm 越大 -> tau 越小
    tau_per_sample_np = tmax - (tmax - tmin) * counts_norm
    tau_per_sample = torch.from_numpy(tau_per_sample_np).to(device).float()
    
    # 时间网格：在最前面增加 sigma=0（先不加噪声过一遍 classifier）
    sigmas = scheduler.get_sigmas(num_time_steps).to(device)
    # sigma_grid = torch.cat([torch.zeros(1, device=device), sigmas], dim=0)
    sigma_grid = sigmas
    
    all_embeddings = []
    all_sigma_mins = []
    all_xt_mins = []
    
    dataloader = DataLoader(dataset_sb, batch_size=128, shuffle=False)
    
    print(f"\n>>> 开始进行样本级时间标注 (Total Samples: {len(dataset_sb)})...",flush=True)
    
    start_idx = 0  # 用于从 tau_per_sample 中按顺序切片
    with torch.no_grad():
        for batch_idx, (x_batch, _) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            B = x_batch.shape[0]

            batch_tau = tau_per_sample[start_idx:start_idx + B]
            
            # 默认 sigma_min 为 1.0（若始终未超过阈值则保持为 1.0）
            batch_sigma_mins = torch.ones(B, device=device)
            found_mask = torch.zeros(B, dtype=torch.bool, device=device)
            batch_xt_mins = torch.zeros_like(x_batch)
            
            for sigma_val in sigma_grid:
                if found_mask.all():
                    break

                sigma_scalar = sigma_val.item()
                sigma_tensor = torch.full((B,), sigma_scalar, device=device)

                x_t = x_batch + sigma_tensor.view(-1, 1) * torch.randn_like(x_batch)
                
                logits = model(x_t, sigma_tensor)
                probs = torch.sigmoid(logits).squeeze(1)  # [B]
                
                # 判定当前时间步是否 > tau（逐样本动态阈值）
                above_tau = probs > batch_tau

                # 只对此前还没命中的样本进行记录
                newly_found = above_tau & (~found_mask)
                batch_sigma_mins[newly_found] = sigma_val
                batch_xt_mins[newly_found] = x_t[newly_found]
                found_mask = found_mask | newly_found
            
            all_embeddings.append(x_batch.cpu())
            all_sigma_mins.append(batch_sigma_mins.detach().cpu())
            all_xt_mins.append(batch_xt_mins.detach().cpu())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches...")

            start_idx += B  # 移动到下一个 batch 的起始位置

    final_embeddings = torch.cat(all_embeddings, dim=0)
    final_sigma_mins = torch.cat(all_sigma_mins, dim=0)
    final_xt = torch.cat(all_xt_mins, dim=0)
    
    print(f"标注完成。")
    print(f"t_min 统计: Mean={final_sigma_mins.mean():.4f}, Median={final_sigma_mins.median():.4f}, Min={final_sigma_mins.min():.4f}")

    return final_embeddings, final_xt, final_sigma_mins


# ==========================================
# 7. 抽象主流程：从内存数据训练/推理，输出每个 embedding 的 t_min
# ==========================================
def fit_timestamps(user_emb_list,
                   user_counts,
                   normalize: bool = True,
                   n_groups: int = 20,
                   batch_size: int = 1024,
                   epochs: int = 300,
                   lr: float = 3e-3,
                   weight_decay: float = 1e-4,
                   device: str | None = None,
                   tau: float = 0.48,
                   assign_num_time_steps: int = 20,
                   scheduler: Diffusion = None,
                   repeat_times: int = 1):
    """
    输入:
      - user_emb_list: [N, D]
      - user_counts:   [N]
    输出:
      - sigma_all:         [N]，每个 embedding 的最小 sigma（SG 默认 0，SB 取多次重复中的最小值）
      - xt_all:            [N, D]，SB 为对应最小 sigma 的含噪样本，SG 为原始 embedding
      - sigma_all_sb_all:  [N * repeat_times]，所有 user 的全部重复版本的 sigma，按 user 分组 (AAABBBCCC...)
      - xt_all_sb_all:     [N * repeat_times, D]，所有 user 的全部重复版本的含噪样本/原始样本
    """
    if scheduler is None:
        raise ValueError("需要从外部传入统一初始化的 Diffusion 调度器 scheduler")
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # to numpy
    if isinstance(user_emb_list, torch.Tensor):
        user_mat = user_emb_list.detach().cpu().numpy()
    else:
        user_mat = np.asarray(user_emb_list, dtype=np.float32)
    user_counts = np.asarray(user_counts)

    if normalize:
        norms = np.linalg.norm(user_mat, axis=1, keepdims=True)
        user_mat = user_mat / np.clip(norms, 1e-12, None)

    # 构建数据集与 indices
    ds_combined, ds_sg, ds_sb, input_dim, sb_user_counts, sb_indices, sg_indices = \
        prepare_datasets_from_arrays(user_mat, user_counts, n_groups=n_groups)

    N = user_mat.shape[0]
    sigma_all = torch.zeros(N, dtype=torch.float32)
    xt_all = torch.from_numpy(user_mat).float()

    # 重新训练并拟合
    classifier = train_classifier(
        ds_combined, input_dim, device=device, 
        epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        scheduler=scheduler
    )

    # _, annotated_xt, annotated_sigma = assign_sample_wise_times(
    #     classifier, ds_sb, device=device, 
    #     num_time_steps=assign_num_time_steps,
    #     scheduler=scheduler,
    #     user_counts=sb_user_counts
    # )
    # sigma_all[sb_indices] = annotated_sigma.view(-1).float().cpu()
    # xt_all[sb_indices] = annotated_xt.view(len(sb_indices), -1).float().cpu()

    # ========= 重复 ds_sb 做样本级标注 =========
    num_sb = len(sb_indices)

    sb_data = ds_sb.tensors[0]          # [Ns, D]
    sb_labels = ds_sb.tensors[1]        # 全 0，占位
    sb_data_rep = sb_data.repeat((repeat_times, 1))       # [Ns * repeat_times, D]
    sb_labels_rep = sb_labels.repeat((repeat_times,))     # [Ns * repeat_times]

    ds_sb_rep = TensorDataset(sb_data_rep, sb_labels_rep)

    # user_counts 同步重复，用于动态阈值
    sb_user_counts_rep = np.tile(sb_user_counts, repeat_times)  # [Ns * repeat_times]

    # 对重复后的 ds_sb 逐样本独立地扫描 sigma_grid，得到含噪样本和 sigma
    _, annotated_xt_rep, annotated_sigma_rep = assign_sample_wise_times(
        classifier, ds_sb_rep, device=device,
        num_time_steps=assign_num_time_steps,
        scheduler=scheduler,
        user_counts=sb_user_counts_rep
    )
    # annotated_xt_rep:    [Ns * repeat_times, D]
    # annotated_sigma_rep: [Ns * repeat_times]

    # ========= 对同一个 SB embedding，从多次重复中选出最小 sigma =========
    annotated_sigma_rep = annotated_sigma_rep.view(repeat_times, num_sb)  # [R, Ns]
    min_sigma, min_idx = annotated_sigma_rep.min(dim=0)                  # [Ns], [Ns]

    xt_rep_view = annotated_xt_rep.view(repeat_times, num_sb, -1)        # [R, Ns, D]
    D = xt_rep_view.shape[-1]
    best_xt = torch.empty(num_sb, D)
    for i in range(num_sb):
        best_xt[i] = xt_rep_view[min_idx[i], i]

    min_scheduler_sigma = scheduler.sigma_min

    # === 根据阈值修正：若 min_sigma < 1.2 * min_scheduler_sigma，则认为 sigma=0，xt 为原始值 ===
    min_sigma_cpu = min_sigma.view(-1).float().cpu()          # [Ns]
    best_xt_cpu = best_xt.view(num_sb, -1).float().cpu()      # [Ns, D]
    orig_sb_xt = torch.from_numpy(user_mat[sb_indices]).float()  # [Ns, D]

    threshold = 1.2 * float(min_scheduler_sigma)
    mask_close = min_sigma_cpu < threshold                    # [Ns] bool

    adjusted_sigma = min_sigma_cpu.clone()
    adjusted_xt = best_xt_cpu.clone()
    adjusted_sigma[mask_close] = 0.0
    adjusted_xt[mask_close] = orig_sb_xt[mask_close]

    # 将“最终 sigma / 对应 xt”写回到整体的 sigma_all / xt_all 中
    sigma_all[sb_indices] = adjusted_sigma
    xt_all[sb_indices] = adjusted_xt

    # ========= 组合“全部用户”的重复版本（AAABBBCCC...） =========
    N_total = N
    sigma_all_rep = torch.zeros(N_total * repeat_times, dtype=torch.float32)
    xt_all_rep = torch.zeros(N_total * repeat_times, D, dtype=torch.float32)

    # SB 用户：使用多次重复打标得到的噪声与含噪样本
    for local_idx, user_idx in enumerate(sb_indices):
        start = int(user_idx * repeat_times)
        end = start + repeat_times
        sigma_all_rep[start:end] = annotated_sigma_rep[:, local_idx].view(-1).float().cpu()
        xt_all_rep[start:end] = xt_rep_view[:, local_idx, :].view(repeat_times, D).float().cpu()

    # SG 用户：sigma=0，embedding 为原始值在第 0 维重复 repeat_times 次
    for user_idx in sg_indices:
        start = int(user_idx * repeat_times)
        end = start + repeat_times
        sigma_all_rep[start:end] = 0.0
        base_xt = xt_all[user_idx]
        xt_all_rep[start:end] = base_xt.view(1, D).repeat(repeat_times, 1)

    return sigma_all, xt_all, sigma_all_rep, xt_all_rep

