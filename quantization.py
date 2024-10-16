import sys
sys.path.append('/home/ywkim/anaconda3/lib/python3.10/site-packages')

import torch
import torch_pruning as tp

def quantize_qint8(net):
    example_inputs = torch.randn(1, 1, 32, 32)
    model_dynamic_quantized = torch.quantization.quantize_dynamic(
        net.to('cpu'), qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
    ).to('cpu')
    #quantize_macs, quantize_nparams = tp.utils.count_ops_and_params(model_dynamic_quantized, example_inputs.to('cpu'))



    return model_dynamic_quantized