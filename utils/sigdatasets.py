import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Tuple, Dict, Set, Optional, Any
import multiprocessing as mp
import math

class myDatasetNew:
    """重构后的数据集类 - 优化版本"""
    
    def __init__(self, dataset: str,
                 user_tiers: int = 10,
                 item_tiers: int = 5,
                 use_multithread: bool = True,
                 dataset_prefix: str = "data",
                 file_prefix: str = "q_35",
                 random_seed: Optional[int] = None,
                 num_workers: Optional[int] = None):
        # 基础配置
        self.dataset_path = os.path.join(dataset_prefix, dataset)
        self.file_prefix = file_prefix

        # 参数配置
        self.user_tiers = user_tiers
        self.item_tiers = item_tiers
        self.use_multithread = use_multithread
        self.num_threads = num_workers or min(16, mp.cpu_count())
        self.rng = np.random.default_rng(random_seed)

        self._load_all_data()

        self._compute_activity_metrics()

        self._print_statistics()

    def _load_all_data(self):
        """加载所有数据"""
        print("加载数据集...")
        prefix = self.file_prefix

        def _load_interaction_data(dataset_path, filename: str):
            """内联自 DataProcessor.load_interaction_data"""
            filepath = os.path.join(dataset_path, filename)
            users, items, ratings = [], [], []
            user_dict: Dict[int, List[int]] = {}
            item_dict: Dict[int, List[int]] = {}

            with open(filepath, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc=f"读取{filename}"):
                    parts = line.strip().split()
                    user_id, item_id, rating = int(parts[0]), int(parts[1]), int(parts[2])

                    users.append(user_id)
                    items.append(item_id)
                    ratings.append(rating)
                    user_dict.setdefault(user_id, []).append(item_id)
                    item_dict.setdefault(item_id, []).append(user_id)

            return [users, items, ratings], user_dict, item_dict

        def _build_interactions(*dicts: Dict[int, List[int]]) -> Dict[int, Set[int]]:
            """合并若干 用户->物品 列表字典，得到每个用户的去重交互集合"""
            interactions: Dict[int, Set[int]] = {}
            for d in dicts:
                for u, items in d.items():
                    interactions.setdefault(u, set()).update(items)
            return interactions

        def _build_pos_neg_dicts(users: List[int], items: List[int], ratings: List[int]):
            """构建正负样本字典"""
            pos_dict: Dict[int, Set[int]] = {}
            neg_dict: Dict[int, Set[int]] = {}
            for u, i, r in zip(users, items, ratings):
                if r >= 1:
                    pos_dict.setdefault(u, []).append(i)
                else:
                    neg_dict.setdefault(u, []).append(i)
            return pos_dict, neg_dict

        # 使用内联的数据加载方法
        self.train_set, self.train_dict, self.train_item_dict = _load_interaction_data(self.dataset_path, f"{self.file_prefix}_train.txt")
        self.tune_set, self.tune_dict, self.tune_item_dict = _load_interaction_data(self.dataset_path, f"{self.file_prefix}_tune.txt")
        self.test_set, self.test_dict, self.test_item_dict = _load_interaction_data(self.dataset_path, f"{self.file_prefix}_test.txt")
        
        # 创建用户和物品池
        self.user_pool = set.union(set(self.train_set[0]), set(self.tune_set[0]), set(self.test_set[0]))
        self.item_pool = set.union(set(self.train_set[1]), set(self.tune_set[1]), set(self.test_set[1]))
        
        self.user_num = max(self.user_pool) + 1
        self.item_num = max(self.item_pool) + 1

        # 构建交互记录
        self.all_interactions = _build_interactions(self.train_dict, self.tune_dict, self.test_dict)
        self.train_interactions = _build_interactions(self.train_dict)

        # 使用内部函数构建正负样本字典
        self.train_pos_dict, self.train_neg_dict = _build_pos_neg_dicts(*self.train_set)
        self.tune_pos_dict, self.tune_neg_dict = _build_pos_neg_dicts(*self.tune_set)
        self.test_pos_dict, self.test_neg_dict = _build_pos_neg_dicts(*self.test_set)

    def _compute_activity_metrics(self):
        """计算用户与物品活跃度相关指标并按 tier 划分活跃/不活跃实体"""
        print("计算用户/物品活跃度指标...")

        def _compute_entity_tiers(activity_dict, tiers_num):
            """通用：根据交互次数为实体计算 tier / 百分位 / 活跃集合"""
            activity_pairs = sorted(activity_dict.items(), key=lambda x: x[1], reverse=True)
            total = len(activity_pairs)
            tier_size = max(1, math.ceil(total / tiers_num))

            activity_percentiles = {}
            activity_tiers = {}
            entities_by_tier = [[] for _ in range(tiers_num)]

            for i, (ent, cnt) in enumerate(activity_pairs):
                tier = min(i // tier_size, tiers_num - 1)
                activity_tiers[ent] = tier
                entities_by_tier[tier].append(ent)
                percentile = (total - i) / total if total > 0 else 0.0
                activity_percentiles[ent] = percentile

            active_entities = list(entities_by_tier[0])
            inactive_entities = [ent for ent, _ in activity_pairs if ent not in active_entities]
            return {
                "pairs": activity_pairs,
                "percentiles": activity_percentiles,
                "tiers": activity_tiers,
                "by_tier": entities_by_tier,
                "active": active_entities,
                "inactive": inactive_entities
            }

        # 用户活跃度
        train_user_activity = {user: len(items) for user, items in self.train_pos_dict.items()}
        user_res = _compute_entity_tiers(train_user_activity, self.user_tiers)
        self.user_activity_percentiles = user_res["percentiles"]
        self.user_activity_tiers = user_res["tiers"]
        self.users_by_tier = user_res["by_tier"]
        self.active_users = user_res["active"]
        self.inactive_users = user_res["inactive"]
        self.train_user_activity = train_user_activity

        # 物品活跃度（基于训练集中被多少用户点击 / 交互）
        train_item_activity = {item: len(users) for item, users in self.train_item_dict.items()}
        item_res = _compute_entity_tiers(train_item_activity, self.item_tiers)
        self.item_activity_percentiles = item_res["percentiles"]
        self.item_activity_tiers = item_res["tiers"]
        self.items_by_tier = item_res["by_tier"]
        self.active_items = item_res["active"]
        self.inactive_items = item_res["inactive"]
        self.train_item_activity = train_item_activity

        print(f"自动划分结果: 用户 活跃 {len(self.active_users)} / 不活跃 {len(self.inactive_users)} | 物品 活跃 {len(self.active_items)} / 不活跃 {len(self.inactive_items)}")

    def instance_tune_test_data(self, type, device):
        if type == 'tune':
            pos_dict = self.tune_pos_dict
            neg_dict = self.tune_neg_dict
        else:
            pos_dict = self.test_pos_dict
            neg_dict = self.test_neg_dict

        out_users, pos_items, neg_items = [], [], []
        for u, pos_items_u in pos_dict.items():
            neg_items_u = neg_dict.get(u, [])
            if not pos_items_u or not neg_items_u:
                continue
            for pi in pos_items_u:
                for ni in neg_items_u:
                    out_users.append(u)
                    pos_items.append(pi)
                    neg_items.append(ni)

        print(f"构建 {type} 数据集: 样本数量 = {len(pos_items)}")

        user_tiers = [self.user_activity_tiers.get(u, self.user_tiers - 1) for u in out_users]
        user_train_activities = [self.train_user_activity[u] for u in out_users]

        all_data = {
            'user': torch.LongTensor(out_users).to(device),
            'pos_item': torch.LongTensor(pos_items).to(device),
            'neg_item': torch.LongTensor(neg_items).to(device),
            'user_tier': torch.LongTensor(user_tiers).to(device),
            'user_train_activity': torch.LongTensor(user_train_activities).to(device),
        }
        return all_data

    def _print_statistics(self):
        """打印数据集统计信息"""
        interaction_num = sum(len(interactions) for interactions in self.all_interactions.values())
        sparsity = round((1 - interaction_num / (self.user_num * self.item_num)) * 100, 2)
        print(f"用户数量: {self.user_num}, 物品数量: {self.item_num}")
        print(f"交互数量: {interaction_num}, 稀疏度 = {sparsity}%")
        print(f"用户: 活跃 {len(self.active_users)}, 不活跃 {len(self.inactive_users)}")
        print(f"物品: 活跃 {len(self.active_items)}, 不活跃 {len(self.inactive_items)}")

    def get_user_train_nums(self):
        """获取训练集中用户的活跃度字典"""
        train_nums = []
        for i in range(self.user_num):
            train_nums.append(self.train_user_activity[i])
        return np.array(train_nums)
    
    def get_train_data(self):
        res = {
            "train_pos_dict": self.train_pos_dict,
            "train_neg_dict": self.train_neg_dict,
            "user_activity_percentiles": self.user_activity_percentiles,
            "user_activity_tiers": self.user_activity_tiers,
        }
        return res




