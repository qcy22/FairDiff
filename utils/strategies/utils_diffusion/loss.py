import torch

class AmbientLoss:
    def __init__(self, sigma_data=0.5, cond_drop_prob=0.1):
        self.sigma_data = sigma_data
        self.cond_drop_prob = float(cond_drop_prob)

        self.last_loss = None


    def __call__(self, net, embs, current_sigma, target_sigma, cond_embs):
        """
        net: 条件去噪网络 TinyUNet1D
        embs: [B, D] 当前向量（可能已含 current_sigma 噪声）
        current_sigma: [B, 1]
        cond_embs: [B, D] 条件向量（用户的 user_cond_emb）
        """
        B = embs.shape[0]
        sigma = target_sigma
        y = embs

        # 额外加噪，把 current_sigma 提升到 sigma
        n = torch.randn_like(y) * torch.sqrt(sigma ** 2 - current_sigma ** 2)
        noisy_input = y + n

        # classifier-free 条件 dropout
        drop_mask = (torch.rand(B, 1, device=embs.device) < self.cond_drop_prob).float()
        cond_for_net = cond_embs * (1.0 - drop_mask)

        # 条件去噪预测 x0
        x0_pred = net(x=noisy_input, t=sigma, cond=cond_for_net)

        D_yn = (1 - (current_sigma / sigma) ** 2) * x0_pred + ((current_sigma / sigma) ** 2) * noisy_input

        # 为current_sigma设置下限min
        current_sigma_weight = torch.clamp(current_sigma, min=1e-2)
         
        ambient_factor1 = (sigma**4) / (((sigma**2 - current_sigma**2)**2))
        ambient_factor2 = (current_sigma_weight**2 + self.sigma_data**2) / ((current_sigma_weight**2) * (self.sigma_data**2))
        edm_weight = (sigma**2 + self.sigma_data**2) / ((sigma**2) * (self.sigma_data**2))
        weight = ambient_factor1 * ambient_factor2 * edm_weight

        # weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2
        # weight = 1.0 / ((sigma)**(1.5) * current_sigma)
        loss = weight * ((D_yn - y) ** 2) * 1e-3

        return loss, x0_pred
