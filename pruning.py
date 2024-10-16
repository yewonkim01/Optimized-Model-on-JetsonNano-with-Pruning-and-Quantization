import sys
sys.path.append('/home/ywkim/anaconda3/lib/python3.10/site-packages')

import torch
import torch_pruning as tp
from test import test
import pandas as pd

def pruning(model, ratio, device, test_loader):
    example_inputs = torch.randn(1, 1, 32, 32)
    example_inputs = example_inputs.to(device)
    acc_list = []
    param_list = []
    mmac_list = []
    inference_time_list = []

    # pruning 기준: 가중치의 절댓값 기준 작은 채널부터 ratio에 따라 제거
    imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')

    ignored_layers = []
    for m in model.modules():  # 마지막 fc layer는 pruning에서 제외
        if isinstance(m, torch.nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)

    iterative_steps = 5
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        global_pruning=False,
        importance=imp,
        iterative_steps=iterative_steps,  # iterative하게 조금씩 pruning
        pruning_ratio=ratio,  # pruning ratio
        ignored_layers=ignored_layers,
    )

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    base_acc, base_inference_time = test(model, device, test_loader)

    acc_list.append(base_acc.cpu().numpy())
    param_list.append(base_nparams)
    mmac_list.append(int(base_macs))
    inference_time_list.append(base_inference_time)

    for i in range(iterative_steps):
        pruner.step()

        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        acc, inference_time = test(model, device, test_loader)


        acc_list.append(acc.cpu().numpy())
        param_list.append(nparams)
        mmac_list.append(int(macs))
        inference_time_list.append(inference_time)

    df = pd.DataFrame({'acc(%)': acc_list,
                       'n_params': param_list,
                       'MACs': mmac_list,
                       'inference time(μs)': inference_time_list},
                      index = ['Base'] + ['step ' + str(i) for i in range(1,6)])

    return model, acc_list, param_list, mmac_list, inference_time_list, df

