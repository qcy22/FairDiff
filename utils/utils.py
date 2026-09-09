import random
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def set_seed(seed: int) -> None:
    """设置全局随机种子确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def run_in_parallel(func, tasks, num_threads, desc="并行任务"):
    """通用的多线程执行函数"""
    if len(tasks) < 200:  # 小任务直接单线程
        results = [func(task) for task in tqdm(tasks, desc=desc)]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_task = {executor.submit(func, task): task for task in tasks}
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc=desc):
                results.append(future.result())
    return results
