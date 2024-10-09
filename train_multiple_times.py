import sys
import os
import argparse
import time

import numpy as np
import json


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    # 添加参数
    parser.add_argument('--config_path', type=str,)
    parser.add_argument('--run_training', type=bool, default=False,)
    args = parser.parse_args()
    config_path = args.config_path
    print(args)
    acc_results=[]
    f1_results=[]
    # 训练十次用不同的seed
    for i in range(10):
        if args.run_training is True:
            print(f'args.run_training:{args.run_training} type:{type(args.run_training)}')
            os.system(f'python train.py --config_path={config_path} --seed={i}')
        result_path=f'fig/{os.path.basename(config_path)[:-5]}/seed{i}/results.json'
        print(result_path)
        time.sleep(1)
        with open(result_path,'r') as f:
            data=json.load(f)
            # print(data)
            acc_results.append(data['test_accuracy'])
            f1_results.append(data['test_f1'])
    # 训练完了，去每个输出的文件下看结果并合并最终的结果
    acc=np.mean(acc_results)
    f1=np.mean(f1_results)
    print(acc,f1)