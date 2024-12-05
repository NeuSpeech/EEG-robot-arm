import sys
import os
import argparse
import time

import numpy as np
import json
import pandas as pd


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    # 添加参数
    parser.add_argument('--config_path', type=str,)
    parser.add_argument('--run_training', type=str, default='False',)
    parser.add_argument('--model_names', type=str, nargs='+', default=None,)
    parser.add_argument('--seed_numbers', type=int, default=10,)
    args = parser.parse_args()
    config_path = args.config_path
    args.run_training = args.run_training.lower() in ['true']
    print(args)
    acc_results = {model_name: [] for model_name in args.model_names}
    f1_results = {model_name: [] for model_name in args.model_names}

    for model_name in args.model_names:
        for i in range(args.seed_numbers):
            if args.run_training:
                os.system(f'python train.py --config_path={config_path} --model_name={model_name} --seed={i}')
            result_path = f'fig/{os.path.basename(config_path)[:-5]}/seed{i}/{model_name}/results.json'
            with open(result_path, 'r') as f:
                data = json.load(f)
                acc_results[model_name].append(data['test_accuracy'])
                f1_results[model_name].append(data['test_f1'])

    results_data = []
    for model_name in args.model_names:
        # 将结果乘以100
        acc_results[model_name] = [x * 100 for x in acc_results[model_name]]
        f1_results[model_name] = [x * 100 for x in f1_results[model_name]]

        acc_mean = round(np.mean(acc_results[model_name]), 2)
        acc_std = round(np.std(acc_results[model_name]), 2)
        f1_mean = round(np.mean(f1_results[model_name]), 2)
        f1_std = round(np.std(f1_results[model_name]), 2)
        
        # 计算最低值和最高值
        acc_min = round(np.min(acc_results[model_name]), 2)
        acc_max = round(np.max(acc_results[model_name]), 2)
        f1_min = round(np.min(f1_results[model_name]), 2)
        f1_max = round(np.max(f1_results[model_name]), 2)

        results_data.append({
            'Model Name': model_name,
            'Acc': f'{acc_mean:.2f} ± {acc_std:.2f}',
            'F1 Score': f'{f1_mean:.2f} ± {f1_std:.2f}',
            'Acc Mean': acc_mean,
            'Acc Std': acc_std,
            'F1 Mean': f1_mean,
            'F1 Std': f1_std,
            'Acc Min': acc_min,
            'Acc Max': acc_max,
            'F1 Min': f1_min,
            'F1 Max': f1_max,
        })

    # 将结果保存为Excel文件
    df = pd.DataFrame(results_data)
    excel_path = f'fig/{os.path.basename(config_path)[:-5]}/model_performance.xlsx'
    df.to_excel(excel_path, index=False)
    print(f'Results saved to {excel_path}')

