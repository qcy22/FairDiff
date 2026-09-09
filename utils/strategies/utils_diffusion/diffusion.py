import torch
import math
import torch.nn.functional as F

class Diffusion:
    """
    将 VE 调度与 EDM 采样合并在一起。
    - sigma(t) = sigma_min * (sigma_max / sigma_min)^t  (连续时间)
    - sample: EDM 风格的离散步长采样
    """
    def __init__(self, sigma_min=0.01, sigma_max=10.0):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.log_sigma_min = math.log(self.sigma_min)
        self.log_sigma_max = math.log(self.sigma_max)

    def get_sigma(self, t):
        # t: [Batch] 或标量，范围 [0, 1]
        return torch.exp(self.log_sigma_min + t * (self.log_sigma_max - self.log_sigma_min))

    def get_sigmas(self, num_steps):
        # 返回 num_steps 个均匀分布的噪声水平
        t_steps = torch.linspace(0.0, 1.0, steps=num_steps)
        sigmas = self.get_sigma(t_steps)
        return sigmas
    
    @torch.no_grad()
    def sample(self, net, embs, cond_embs, num_steps=20, cur_sigma=None, deterministic=True, guidance_scale=1.0):
        device = embs.device
        net.eval()

        # 连续时间 t ∈ [1, 0] 均匀采样，再映射到 sigma
        # t_cont = torch.linspace(1.0, 0.0, steps=num_steps, dtype=torch.float32, device=device)
        # sigma_schedule = self.get_sigma(t_cont)  # [N], 递减: sigma_schedule[0]=sigma_max, sigma_schedule[-1]=sigma_min
        sigma_schedule = torch.flip(self.get_sigmas(num_steps), dims=[0]).to(device)  # [N], 递减: sigma_schedule[0]=sigma_max, sigma_schedule[-1]=sigma_min

        x_next = embs.to(torch.float32)
        cond_embs = cond_embs.to(torch.float32)

        # 计算每个样本的起始步索引（最小的 i 使得 sigma_schedule[i] >= cur_sigma_i）
        if cur_sigma is None:
            start_idx = torch.zeros(embs.shape[0], dtype=torch.long, device=device)
        else:
            cur_sigma = torch.as_tensor(cur_sigma, device=device, dtype=torch.float32)
            if cur_sigma.ndim == 0:
                cur_sigma = cur_sigma.expand(embs.shape[0])
            sigma_schedule_rev = torch.flip(sigma_schedule, dims=[0])  
            pos_rev = torch.searchsorted(
                sigma_schedule_rev,
                cur_sigma.clamp(min=sigma_schedule.min(), max=sigma_schedule.max())
            )
            start_idx = (num_steps - 1) - pos_rev + 1
            start_idx = start_idx.clamp(min=0, max=num_steps - 1)

        for s in range(num_steps - 1):
            x_cur = x_next
            sigma_cur = sigma_schedule[s]
            sigma_next = sigma_schedule[s + 1]

            mask = (start_idx <= s)  # [B]
            if not mask.any():
                continue

            sigma_cur_for_net = torch.full(
                (x_cur.shape[0],),
                float(sigma_cur),
                device=device,
                dtype=torch.float32
            )
            sigma_next_for_net = torch.full(
                (x_cur.shape[0],),
                float(sigma_next),
                device=device,
                dtype=torch.float32
            )
            sigma_cur_b  = sigma_cur_for_net.view(-1, *([1] * (x_cur.ndim - 1)))
            sigma_next_b = sigma_next_for_net.view(-1, *([1] * (x_cur.ndim - 1)))
            mask_b = mask.view(-1, *([1] * (x_cur.ndim - 1)))

            # classifier-free guidance:
            #  无条件分支：cond = 0
            #  有条件分支：cond = cond_embs
            cond_zero = torch.zeros_like(cond_embs)
            x0_uncond = net(x_cur, sigma_cur_for_net, cond_zero).to(torch.float32)
            x0_cond = net(x_cur, sigma_cur_for_net, cond_embs).to(torch.float32)
            x0 = x0_uncond + guidance_scale * (x0_cond - x0_uncond)

            # EDM / VE 风格更新
            d_cur = (x_cur - x0) / sigma_cur_b

            if deterministic:
                x_update = x_cur + (sigma_next_b - sigma_cur_b) * d_cur
            else:
                x_update = x_cur + 2.0 * (sigma_next_b - sigma_cur_b) * d_cur + torch.sqrt(
                    2.0 * (sigma_cur_b - sigma_next_b).abs() * sigma_cur_b
                ) * torch.randn_like(x_cur)

            x_next = torch.where(mask_b, x_update, x_cur)

        return x_next.to(embs.dtype), start_idx



