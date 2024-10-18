import sys
#sys.path.append('/home/ywkim/anaconda3/lib/python3.10/site-packages')

import torch
import torch_pruning as tp
from finetuning import fine_tuning
from test import test

import pandas as pd


def experi_pruning_finetuning(model, ratio, device, test_loader, step, lr, word):
    acc_list = []
    param_list = []
    mmac_list = []
    inference_time_list = []

    #기존 pruning
    example_inputs = torch.randn(1, 1, 32, 32)
    example_inputs = example_inputs.to(device)

    imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')

    ignored_layers = []

    for m in model.modules():
        if word == 'except_last_linear_layer':
            if isinstance(m, torch.nn.Linear) and m.out_features == 10:
                ignored_layers.append(m)

        if word == 'except_all_linear_layer':
            if isinstance(m, torch.nn.Linear):
                ignored_layers.append(m)

        if word == 'except_all_conv_layer':
            if isinstance(m, torch.nn.Conv2d):
                ignored_layers.append(m)
            if isinstance(m, torch.nn.Linear) and m.out_features == 10:
                ignored_layers.append(m)


    iterative_steps = step
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        global_pruning=False,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=ratio,
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


        # fine-tuning after pruning
        model = fine_tuning(model, device, test_loader, lr)

        acc, inference_time = test(model, device, test_loader)
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)



        acc_list.append(acc.cpu().numpy())
        param_list.append(nparams)
        mmac_list.append(int(macs))
        inference_time_list.append(inference_time)

    df = pd.DataFrame({'acc(%)': acc_list,
                       'n_params': param_list,
                       'MACs': mmac_list,
                       'inference time(μs)': inference_time_list},
                      index=['Base'] + ['step ' + str(i+1) for i in range(len(acc_list)-1)])

    return model, acc_list, param_list, mmac_list, inference_time_list, df