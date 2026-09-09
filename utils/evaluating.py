import torch
import numpy as np
import copy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from utils.configs import EvaluationConfig

@dataclass
class ModelResults:
    """模型评估结果封装"""
    overall_auc: float
    active_auc: float
    inactive_auc: float
    ugf_gap_auc: float

    by_activity_tier: List[Dict[str, float]]
    normalized_ecd_wasserstein: float
    tier_auc_vals: List[float]
    tier_var: float
    
    def print_summary(self, model_name: str):
        """打印评估摘要"""
        print("\t\t\t\tAUC")
        print(f"{model_name}\t总体\t\t{self.overall_auc:.8f}")
        print(f"\t\t活跃用户\t{self.active_auc:.8f}")
        print(f"\t\t不活跃用户\t{self.inactive_auc:.8f}")
        print(f"\t\tUGF差距\t\t{self.ugf_gap_auc:.8f}")
        print("-" * 40)
        print(f"\t\tECD (Norm W1-Dist)\t{self.normalized_ecd_wasserstein:.8f}")
        print("-" * 40)

        tier_vals = np.array(self.tier_auc_vals, dtype=np.float32)
        if len(tier_vals) > 0:
            print(f"\t\t活跃度档次AUC:")
            print(f"\t\t  AUC方差:\t{self.tier_var:.8f}")
            
            per_line = 5
            for i in range(0, len(tier_vals), per_line):
                end = min(i + per_line, len(tier_vals))
                rng = f"档次{i}-{end-1}"
                vals = "\t".join(f"{v:.8f}" for v in tier_vals[i:end])
                print(f"\t\t  {rng}:\t{vals}")

        print()


class EvaluationProcessor:
    """评估处理器"""
    
    def __init__(self, batch_size: int = 1024, user_tiers: int = 10):
        self.batch_size = batch_size
        self.user_tiers = user_tiers

    def _compute_normalized_ecd_wasserstein(
        self,
        user_activity_list: np.ndarray,
        user_auc_list: np.ndarray,
    ) -> float:
        """计算按活跃度分组的归一化 1-Wasserstein ECD。"""
        acts = np.asarray(user_activity_list, dtype=np.float32)
        aucs = np.asarray(user_auc_list, dtype=np.float32)
        num_users = len(acts)
        if num_users == 0:
            return 0.0

        sorted_indices = np.argsort(acts)
        sorted_acts = acts[sorted_indices]
        sorted_aucs = aucs[sorted_indices]
        cumulative_auc = np.cumsum(sorted_aucs)
        total_auc = cumulative_auc[-1]
        if total_auc == 0:
            return 0.0

        _, group_counts = np.unique(sorted_acts, return_counts=True)
        group_end_indices = np.cumsum(group_counts) - 1
        actual_cumulative_auc = cumulative_auc[group_end_indices]
        user_counts = (group_end_indices + 1).astype(np.float32)
        ideal_cumulative_auc = (user_counts / num_users) * total_auc
        bias_area = np.sum(
            np.abs(actual_cumulative_auc - ideal_cumulative_auc) * group_counts
        )
        normalization_factor = np.sum(ideal_cumulative_auc * group_counts)

        return float(bias_area / normalization_factor) if normalization_factor else 0.0

    def _aggregate_user_auc(
        self,
        users_np: np.ndarray,
        tiers_np: np.ndarray,
        user_activity_np: np.ndarray,
        scores: np.ndarray,
    ):
        """聚合到用户级 AUC、Tier 和训练集活跃度。"""
        unique_uids, first_occurrence_idx, inverse_indices = np.unique(
            users_np, return_index=True, return_inverse=True
        )
        user_counts = np.bincount(inverse_indices)
        user_sums = np.bincount(inverse_indices, weights=scores)
        user_auc_arr = user_sums / user_counts
        unique_user_tiers = tiers_np[first_occurrence_idx]
        unique_user_activity = user_activity_np[first_occurrence_idx]
        return unique_uids, user_auc_arr, unique_user_tiers, unique_user_activity

    def _compute_tier_metrics(
        self,
        user_auc_arr: np.ndarray,
        unique_user_tiers: np.ndarray,
    ):
        """计算各 Tier 的 AUC 统计以及 overall/active/inactive/UGF"""
        tier_counts = np.bincount(unique_user_tiers, minlength=self.user_tiers)
        tier_auc_sums = np.bincount(
            unique_user_tiers, weights=user_auc_arr, minlength=self.user_tiers
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            tier_auc_avgs = tier_auc_sums / tier_counts
            tier_auc_avgs = np.nan_to_num(tier_auc_avgs, nan=0.0)

        tier_metrics = [{'auc': float(val)} for val in tier_auc_avgs]
        tier_vals = tier_auc_avgs

        overall_auc = float(np.mean(user_auc_arr)) if len(user_auc_arr) > 0 else 0.0
        active_auc = float(tier_auc_avgs[0]) if len(tier_auc_avgs) > 0 else 0.0

        inactive_mask = (unique_user_tiers != 0)
        if np.any(inactive_mask):
            inactive_auc = float(np.mean(user_auc_arr[inactive_mask]))
        else:
            inactive_auc = 0.0

        ugf_gap_auc = abs(round(active_auc - inactive_auc, 4))
        tier_var = float(np.var(tier_vals)) if len(tier_vals) > 0 else 0.0

        return tier_metrics, tier_vals, tier_var, overall_auc, active_auc, inactive_auc, ugf_gap_auc

    def evaluate_model_comprehensive(
        self,
        strategy,
        data_sets,
        epoch
    ) -> Tuple[ModelResults, Dict[int, float], np.ndarray, np.ndarray]:
        
        users = data_sets['user']
        pos_items = data_sets['pos_item']
        neg_items = data_sets['neg_item']
        user_tiers = data_sets['user_tier']
        user_train_activity = data_sets['user_train_activity']

        pos_scores, neg_scores, user_embeddings, item_embeddings = strategy.infer(
            user_set=users, pos_item_set=pos_items, neg_item_set=neg_items,
            batch_size=self.batch_size, user_tiers=user_tiers, epoch=epoch
        )

        pos_np = pos_scores.detach().cpu().numpy()
        neg_np = neg_scores.detach().cpu().numpy()
        tiers_np = user_tiers.detach().cpu().numpy()
        users_np = users.detach().cpu().numpy()
        user_activity_np = user_train_activity.detach().cpu().numpy()

        # Pairwise 分数
        scores = (pos_np > neg_np).astype(np.float32) + 0.5 * (pos_np == neg_np).astype(np.float32)

        # 1) 聚合到用户级
        unique_uids, user_auc_arr, unique_user_tiers, unique_user_activity = self._aggregate_user_auc(
            users_np, tiers_np, user_activity_np, scores
        )

        # 2) Tier 级统计与 overall/active/inactive/UGF
        (
            tier_metrics,
            tier_vals,
            tier_var,
            overall_auc,
            active_auc,
            inactive_auc,
            ugf_gap_auc,
        ) = self._compute_tier_metrics(user_auc_arr, unique_user_tiers)

        # 3) 归一化 Wasserstein ECD
        normalized_ecd_wasserstein = self._compute_normalized_ecd_wasserstein(
            unique_user_activity, user_auc_arr
        )

        # 4) 用户 AUC 字典
        user_auc_dict = dict(zip(unique_uids, user_auc_arr))

        # 5) 组装 ModelResults
        model_results = ModelResults(
            overall_auc=overall_auc,
            active_auc=active_auc,
            inactive_auc=inactive_auc,
            ugf_gap_auc=ugf_gap_auc,
            by_activity_tier=tier_metrics,
            normalized_ecd_wasserstein=normalized_ecd_wasserstein,
            tier_auc_vals=tier_vals.tolist(),
            tier_var=tier_var,
        )

        return model_results, user_auc_dict, user_embeddings, item_embeddings


def _copy_module_state_dict(module):
    """将模块权重快照保存到 CPU，避免后续训练覆盖最佳权重。"""
    return {
        name: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for name, value in module.state_dict().items()
    }


def _copy_strategy_state_dict(strategy):
    """保存策略参与推理的模块状态，包括 DiffusionStrategy 的 denoiser。"""
    state_dicts = {}
    for name in ("model", "denoiser"):
        if hasattr(strategy, name):
            module = getattr(strategy, name)
            state_dicts[name] = (
                _copy_module_state_dict(module)
                if isinstance(module, torch.nn.Module)
                else None
            )
    return state_dicts


def _restore_strategy_state_dict(strategy, state_dicts):
    """恢复验证集最优时参与推理的模块状态。"""
    for name, state_dict in state_dicts.items():
        if state_dict is None:
            setattr(strategy, name, None)
        else:
            getattr(strategy, name).load_state_dict(state_dict)


@torch.no_grad()
def tune_new(epoch, tune_data, eval_config: EvaluationConfig):
    """在验证集上评估并保存验证 AUC 最优的模型权重。"""
    processor = EvaluationProcessor(
        batch_size=eval_config.batch_size,
        user_tiers=eval_config.user_tiers
    )

    print("验证结果:")
    for model_instance in eval_config.model_instances:
        tune_model_results, user_auc, user_embeddings, item_embeddings = processor.evaluate_model_comprehensive(
            model_instance.strategy, tune_data, epoch
        )
        
        tune_model_results.print_summary(model_instance.display_name)
        if epoch > eval_config.recording_epoch:
            best_results = getattr(model_instance, "best_model_results", None)
            if (best_results is None) or (tune_model_results.overall_auc > best_results.overall_auc):
                model_instance.best_model_results = tune_model_results
                model_instance.best_epoch = epoch
                model_instance.best_strategy_state_dict = _copy_strategy_state_dict(model_instance.strategy)



@torch.no_grad()
def print_best_results(model_instances, test_data, eval_config: EvaluationConfig):
    """恢复验证集最优权重，并在测试集上进行一次最终评估。"""
    processor = EvaluationProcessor(
        batch_size=eval_config.batch_size,
        user_tiers=eval_config.user_tiers,
    )
    print("===== 各模型最佳验证结果 =====")
    for model_instance in model_instances:
        best_results = model_instance.best_model_results
        best_strategy_state_dict = model_instance.best_strategy_state_dict
        if best_results is None or best_strategy_state_dict is None:
            print(f"{model_instance.display_name} 没有可用的验证结果，跳过最终测试。")
            continue

        _restore_strategy_state_dict(model_instance.strategy, best_strategy_state_dict)
        display_name = model_instance.display_name 
        best_epoch = model_instance.best_epoch 
        best_name = f"{display_name} (best @ epoch {best_epoch})"
        best_results.print_summary(best_name)

        print("对应的测试结果:")
        test_model_results, _, _, _ = processor.evaluate_model_comprehensive(
            model_instance.strategy, test_data, best_epoch
        )
        test_model_results.print_summary(f"{best_name} (Test)")
