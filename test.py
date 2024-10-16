import torch
from time import time

def test(model, device, data):
    correct = 0
    total_samples = 0
    total_time = 0
    model.eval()

    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(data):
            imgs = imgs.to(device)
            labels = labels.to(device)
            start_time = time()

            output = model(imgs)
            end_time = time()

            elapsed_time = end_time - start_time
            total_time += elapsed_time

            _, indices = output.max(1)
            correct += (labels == indices).sum()
            total_samples += indices.size(0)

        acc = correct / total_samples * 100
        avg_time_per_sample = total_time / total_samples * 1e6

    return acc, avg_time_per_sample