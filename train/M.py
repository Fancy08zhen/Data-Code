import torch
import argparse

import numpy as np
from tqdm import tqdm
from thop import profile
from model1 import MobileNetV2 as init_model



parser = argparse.ArgumentParser()
args = parser.parse_args()
model = init_model(num_classes=60)
device = torch.device("cuda:0")
model.to(device)
model.eval()


dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings = np.zeros((repetitions, 1))

for _ in tqdm(range(10)):
    _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in tqdm(range(repetitions)):
        starter.record()
        # _= inference_model(model, dummy_input)
        _ = model(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
mean_fps = 1000. / mean_syn
print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn,
                                                                                     mean_fps=mean_fps))
print(mean_syn)

# 计算FLOPs和Params
flops, params = profile(model, inputs=(dummy_input,))
print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
print('Params = ' + str(params / 1000 ** 2) + 'M')