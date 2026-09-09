import os
from typing import Dict, List, Tuple
from tqdm import tqdm 
import random

def load_data(input_file: str, delimiter: str = '\t', rating_threshold: float = 3.5) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, int], float], int]:
    print(f"加载数据: {os.path.basename(input_file)}, 评分阈值>={rating_threshold}")
    user_items = {}
    ratings = {}
    total_interactions = 0
    positive_interactions = 0

    with open(input_file, "r") as f:
        for line in tqdm(f, desc="加载数据"):
            parts = line.strip().split(delimiter)
            user_id, item_id = int(parts[0]) - 1, int(parts[1]) - 1
            rating = float(parts[2])
            
            if rating > rating_threshold:
                user_items.setdefault(user_id, []).append(item_id)
                total_interactions += 1
                positive_interactions += 1
                ratings[(user_id, item_id)] = 1

            if rating < rating_threshold:
                user_items.setdefault(user_id, []).append(item_id)
                total_interactions += 1
                ratings[(user_id, item_id)] = 0

    print(f"初始统计: 总交互={total_interactions}, 正反馈={positive_interactions}，占比{(positive_interactions/total_interactions)*100:.2f}%")
    return user_items, ratings, total_interactions

def save_data(filename: str, data: List[Tuple[int, int, float]]) -> None:
    """保存数据集到文件"""
    with open(filename, 'w') as f:
        for user_id, item_id, rating in sorted(data):
            f.write(f"{user_id}\t{item_id}\t{rating}\n")

def split_train_test(
    user_items: Dict[int, List[int]],
    ratings: Dict[Tuple[int, int], float],
    tune_num: int = 0,
    test_num: int = 4,
    train_min: int = 2,
    seed: int = 42
) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    """
    按用户划分训练/验证/测试。每个用户要求：
    - 测试集恰含 test_num 个正样本与 test_num 个负样本
    - 验证集恰含 tune_num 个正样本与 tune_num 个负样本
    - 训练集至少 train_min 个正样本与 train_min 个负样本
    若无法满足，则删除该用户的所有交互。
    全局清洗：移除仅在验证/测试集中出现的物品；若破坏用户约束则删除该用户，迭代直到稳定。
    """
    rng = random.Random(seed)

    def cls(u: int, i: int) -> int:
        return int(ratings[(u, i)])

    kept_users_train: Dict[int, List[int]] = {}
    kept_users_tune: Dict[int, List[int]] = {}
    kept_users_test: Dict[int, List[int]] = {}
    dropped_users_stage1 = 0

    for u, items in user_items.items():
        # 不去重，交互被认为不重复
        pos_items = [i for i in items if cls(u, i) == 1]
        neg_items = [i for i in items if cls(u, i) == 0]

        # 需满足：测试集各取 test_num，验证集各取 tune_num，训练端至少各留 train_min
        if len(pos_items) < test_num + tune_num + train_min or len(neg_items) < test_num + tune_num + train_min:
            dropped_users_stage1 += 1
            continue

        rng.shuffle(pos_items)
        rng.shuffle(neg_items)

        # 先划测试，再划验证
        test_pos = pos_items[:test_num]
        tune_pos = pos_items[test_num:test_num + tune_num]
        test_neg = neg_items[:test_num]
        tune_neg = neg_items[test_num:test_num + tune_num]

        test_sel = test_pos + test_neg
        tune_sel = tune_pos + tune_neg
        used = set(test_sel + tune_sel)

        # 剩余即训练集
        train_sel = [i for i in items if i not in used]

        # 最终快速校验（不做修复，失败则丢弃该用户）
        def count_pos_neg(L: List[int]) -> Tuple[int, int]:
            p = sum(1 for x in L if cls(u, x) == 1)
            return p, len(L) - p

        tr_p, tr_n = count_pos_neg(train_sel)
        tu_p, tu_n = count_pos_neg(tune_sel)
        te_p, te_n = count_pos_neg(test_sel)

        cond_train = tr_p >= train_min and tr_n >= train_min
        cond_tune = (tune_num == 0) or (tu_p == tune_num and tu_n == tune_num)
        cond_test = te_p == test_num and te_n == test_num

        if not (cond_train and cond_tune and cond_test):
            dropped_users_stage1 += 1
            continue

        kept_users_train[u] = train_sel
        if tune_num > 0:
            kept_users_tune[u] = tune_sel
        kept_users_test[u] = test_sel

    # 全局清洗：移除仅在验证/测试集中出现的物品，并剔除不满足约束的用户（迭代）
    def valid_user(u: int, tr_items: List[int], tune_items: List[int], te_items: List[int]) -> bool:
        if len(tr_items) < 2 * train_min:
            return False
        if test_num > 0 and len(te_items) < 2 * test_num:
            return False
        if tune_num > 0 and len(tune_items) < 2 * tune_num:
            return False

        tp = sum(1 for i in tr_items if cls(u, i) == 1)
        tn = len(tr_items) - tp

        ep = sum(1 for i in te_items if cls(u, i) == 1)
        en = len(te_items) - ep

        vp = sum(1 for i in tune_items if cls(u, i) == 1)
        vn = len(tune_items) - vp

        if tp < train_min or tn < train_min:
            return False
        if test_num > 0 and (ep < test_num or en < test_num):
            return False
        if tune_num > 0 and (vp < tune_num or vn < tune_num):
            return False
        return True

    removed_test_interactions = 0
    removed_tune_interactions = 0
    dropped_users_stage2 = 0

    while True:
        changed = False
        items_in_train = set(i for lst in kept_users_train.values() for i in lst)

        # 移除验证/测试集中不在训练物品集合中的交互
        for u in list(kept_users_test.keys()):
            old_len = len(kept_users_test[u])
            kept_users_test[u] = [i for i in kept_users_test[u] if i in items_in_train]
            if len(kept_users_test[u]) != old_len:
                removed_test_interactions += (old_len - len(kept_users_test[u]))
                changed = True

        for u in list(kept_users_tune.keys()):
            old_len = len(kept_users_tune[u])
            kept_users_tune[u] = [i for i in kept_users_tune[u] if i in items_in_train]
            if len(kept_users_tune[u]) != old_len:
                removed_tune_interactions += (old_len - len(kept_users_tune[u]))
                changed = True

        # 剔除不满足用户级别正负覆盖的用户
        to_drop = []
        for u in list(kept_users_train.keys()):
            tr = kept_users_train[u]
            te = kept_users_test.get(u, [])
            tu = kept_users_tune.get(u, [])
            if not valid_user(u, tr, tu, te):
                to_drop.append(u)
        if to_drop:
            for u in to_drop:
                kept_users_train.pop(u, None)
                kept_users_test.pop(u, None)
                kept_users_tune.pop(u, None)
            dropped_users_stage2 += len(to_drop)
            changed = True

        if not changed:
            break

    train = [(u, i, ratings[(u, i)]) for u, lst in kept_users_train.items() for i in lst]
    tune = [(u, i, ratings[(u, i)]) for u, lst in kept_users_tune.items() for i in lst]
    test = [(u, i, ratings[(u, i)]) for u, lst in kept_users_test.items() for i in lst]

    print(
        f"划分完成: 训练集={len(train)}, 验证集={len(tune)}, 测试集(最终)={len(test)}, "
        f"首轮剔除用户={dropped_users_stage1}, 清洗迭代剔除用户={dropped_users_stage2}, "
        f"因物品不在训练而删除的验证交互={removed_tune_interactions}，测试交互={removed_test_interactions}"
    )
    return train, tune, test

def create_id_mappings(
    all_data: List[Tuple[int, int, float]],
    seed: int = 42
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    对 all_data 中出现的用户与物品ID进行随机打乱并重映射到 [0..n) / [0..m)
    """
    rng = random.Random(seed)
    all_users = list({user_id for user_id, _, _ in all_data})
    all_items = list({item_id for _, item_id, _ in all_data})
    rng.shuffle(all_users)
    rng.shuffle(all_items)
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(all_users)}
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(all_items)}
    return user_id_map, item_id_map

# 新增：过滤少交互（基于去重计数），迭代直至稳定
def filter_popular_items_and_active_users(
    user_items: Dict[int, List[int]],
    ratings: Dict[Tuple[int, int], float],
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, int], float], int, int]:
    current_user_items = {u: lst[:] for u, lst in user_items.items()}
    total_removed_items = 0
    total_removed_users = 0

    while True:
        n_users = len(current_user_items)
        if n_users == 0:
            current_user_items = {}
            break

        # 统计 item->users，及当前物品集合
        item_users: Dict[int, set] = {}
        all_items_set = set()
        for u, items in current_user_items.items():
            uniq_items = set(items)
            all_items_set.update(uniq_items)
            for i in uniq_items:
                item_users.setdefault(i, set()).add(u)

        n_items = len(all_items_set)
        if n_items == 0:
            current_user_items = {}
            break

        items_to_drop = {
            i for i, us in item_users.items()
            if len(us) < min_item_interactions
        }
        users_to_drop = set()
        for u, items in current_user_items.items():
            deg = len(set(items))
            if deg < min_user_interactions:
                users_to_drop.add(u)

        # 生成新集合
        prev_user_count = len(current_user_items)
        prev_items_set = set(all_items_set)

        new_user_items: Dict[int, List[int]] = {}
        for u, items in current_user_items.items():
            if u in users_to_drop:
                continue
            kept = [i for i in items if i not in items_to_drop]
            if kept:
                new_user_items[u] = kept

        new_items_set = set(i for lst in new_user_items.values() for i in set(lst))

        removed_users_iter = prev_user_count - len(new_user_items)
        removed_items_iter = len(prev_items_set - new_items_set)

        total_removed_users += removed_users_iter
        total_removed_items += removed_items_iter

        if removed_users_iter == 0 and removed_items_iter == 0:
            # 稳定，结束
            current_user_items = new_user_items
            break
        else:
            # 继续迭代
            current_user_items = new_user_items

    # 过滤 ratings 到最终用户与物品集合
    allowed_users = set(current_user_items.keys())
    allowed_items = set(i for lst in current_user_items.values() for i in set(lst))
    new_ratings: Dict[Tuple[int, int], float] = {}
    for (u, i), r in ratings.items():
        if u in allowed_users and i in allowed_items:
            new_ratings[(u, i)] = r

    print(
        f"过滤规则: 删除(<{min_item_interactions}交互)的物品 与 "
        f"(<{min_user_interactions}交互)的用户；"
        f"最终移除物品={total_removed_items}，移除用户={total_removed_users}"
    )
    return current_user_items, new_ratings, total_removed_items, total_removed_users

def split_and_save(
    input_file: str,
    delimiter: str = '\t',
    rating_threshold: float = 3.5,
    min_interactions: int = 10,
    tune_num: int = 0,
    test_num: int = 4,
    train_min: int = 2,
    seed: int = 42,
    train_out: str = None,
    tune_out: str = None,
    test_out: str = None
) -> Tuple[str, str, str]:
    """
    从 input_file 加载 -> 划分 -> 清洗 -> ID打乱映射 -> 写入文件
    """
    user_items, ratings, total = load_data(input_file, delimiter=delimiter, rating_threshold=rating_threshold)

    # 新增：加载后过滤
    user_items, ratings, removed_items, removed_users = filter_popular_items_and_active_users(
        user_items, ratings,
        min_user_interactions=min_interactions, min_item_interactions=min_interactions
    )

    # 调整：统计改为过滤后的数据
    n_users = len(user_items)
    n_items = len({i for items in user_items.values() for i in items})
    n_inters = sum(len(lst) for lst in user_items.values())
    if n_users > 0 and n_items > 0:
        sparsity = 1 - (n_inters / (n_users * n_items))
        print(f"过滤后数据统计: 用户数={n_users}, 物品数={n_items}, 交互数={n_inters}, 稀疏度={sparsity*100:.4f}%, 移除物品={removed_items}, 移除用户={removed_users}")
    else:
        print(f"过滤后数据统计: 用户数={n_users}, 物品数={n_items}, 交互数={n_inters}, 稀疏度=NA, 移除物品={removed_items}, 移除用户={removed_users}")

    train, tune, test = split_train_test(
        user_items, ratings,
        tune_num=tune_num,
        test_num=test_num,
        train_min=train_min,
        seed=seed
    )

    # 在保存前对用户/物品ID进行shuffle并映射
    all_data = train + tune + test
    user_id_map, item_id_map = create_id_mappings(all_data, seed=seed)
    train = [(user_id_map[u], item_id_map[i], r) for (u, i, r) in train]
    tune = [(user_id_map[u], item_id_map[i], r) for (u, i, r) in tune]
    test = [(user_id_map[u], item_id_map[i], r) for (u, i, r) in test]

    # 新增：映射后统计（针对清洗+划分后的数据）
    mapped_all = train + tune + test
    mapped_users = len({u for u, _, _ in mapped_all})
    mapped_items = len({i for _, i, _ in mapped_all})
    mapped_inters = len(mapped_all)
    if mapped_users > 0 and mapped_items > 0:
        mapped_sparsity = 1 - (mapped_inters / (mapped_users * mapped_items))
        print(f"映射后数据统计: 用户数={mapped_users}, 物品数={mapped_items}, 交互数={mapped_inters}, 稀疏度={mapped_sparsity*100:.4f}%")
    else:
        print(f"映射后数据统计: 用户数={mapped_users}, 物品数={mapped_items}, 交互数={mapped_inters}, 稀疏度=NA")

    base, _ = os.path.splitext(input_file)
    train_out = train_out or f"{base}_train.tsv"
    tune_out = tune_out or f"{base}_tune.tsv"
    test_out = test_out or f"{base}_test.tsv"

    save_data(train_out, train)
    save_data(tune_out, tune)
    save_data(test_out, test)

    print(
        f"已保存: 训练集 -> {os.path.basename(train_out)} ({len(train)} 行), "
        f"验证集 -> {os.path.basename(tune_out)} ({len(tune)} 行), "
        f"测试集 -> {os.path.basename(test_out)} ({len(test)} 行)"
    )
    return train_out, tune_out, test_out

if __name__ == "__main__":

    args = {
        "dataset_name": "MovieLens-1m",
        "input_file": "ratings.dat",
        'delimiter': '::',
        "rating_threshold": 3.5,
        "min_interactions": 10,
        "tune_num": 5,
        "test_num": 10,
        "train_min": 5,
        "seed": 42,
        "train_out": "q_35_train.txt",
        "tune_out": "q_35_tune.txt",
        "test_out": "q_35_test.txt"
    }
    

    split_and_save(
        input_file=args["dataset_name"] + '/' + args["input_file"],
        delimiter=args["delimiter"],
        rating_threshold=args["rating_threshold"],
        min_interactions=args["min_interactions"],
        tune_num=args["tune_num"],
        test_num=args["test_num"],
        train_min=args["train_min"],
        seed=args["seed"],
        train_out=args["dataset_name"] + '/' + args["train_out"],
        tune_out=args["dataset_name"] + '/' + args["tune_out"],
        test_out=args["dataset_name"] + '/' + args["test_out"]
    )