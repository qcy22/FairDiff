"""
主程序入口
FairDiff 推荐系统主训练脚本
"""
import torch
import time
from tqdm import tqdm
import argparse

from utils.sigdatasets import myDatasetNew
from utils.configs import (
    SPLIT,parse_args_and_config, create_model_instances, EvaluationConfig
)
from utils.utils import set_seed
from utils.evaluating import tune_new,print_best_results


def main():

    parser = argparse.ArgumentParser(description='FairDiff 推荐系统')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--model', type=str, default=None, help='模型名称')
    parser.add_argument('--dataset', type=str, default=None, help='数据集名称')
    parser.add_argument('--active_strategies', nargs='+', default=[], help='传入一个或多个策略名称')

    args = parser.parse_args()

    config = parse_args_and_config(args)
    print(f"使用模型: {config.model}")
    print(f"参数配置: {config}")
    
    set_seed(config.seed)
    device = torch.device("cuda" )
    
    print(f"训练设备: {device}")
    print(f"数据集: {config.dataset}, 模型: {config.model}")
    print(SPLIT)
    
    # 加载数据
    print("加载数据集...")
    dataset = myDatasetNew(
        dataset=config.dataset, 
        dataset_prefix='data',
        user_tiers=config.user_tiers
    )

    # sample_generator = dataset.instance_bpr_train_loader(config.batch_size,num_neg_per_pos=3)

    print("数据加载成功!")

    set_seed(config.seed)

    model_instances = create_model_instances(config, device, dataset)

    train_data = dataset.get_train_data()
    for model_instance in model_instances:
        model_instance.strategy.build_dataloader(train_data, config.batch_size)

    eval_config = EvaluationConfig(
        model_instances=model_instances,
        batch_size=config.eval_batch_size,
        user_tiers=config.user_tiers,
        recording_epoch=config.recording_epoch,
    )

    tune_data=dataset.instance_tune_test_data('tune', device)
    test_data=dataset.instance_tune_test_data('test', device)

    if config.freeze_embedding_path is not None:
        print(f"加载冻结的嵌入向量: {config.freeze_embedding_path}")

        for model_instance in model_instances:
            model_instance.strategy.model.load_embedding(config.freeze_embedding_path, device)
        

    tune_new(0, tune_data, eval_config)
    
    print("开始训练...")
    for epoch in tqdm(range(1, config.epochs+1)):

        try:
            print(f"{'=' * 60}\n训练轮次: {epoch}")
            for model_instance in model_instances:
                model_instance.strategy.train_epoch(epoch, device)

            tune_new(epoch, tune_data, eval_config)

            if epoch == config.epochs:
                print_best_results(model_instances, test_data, eval_config)
        except KeyboardInterrupt:
            print(f"训练中断，进行最后的评估，当前轮次: {epoch}")
            print_best_results(model_instances, test_data, eval_config)
            break
        
if __name__ == "__main__":
    main()
