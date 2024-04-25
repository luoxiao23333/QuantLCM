'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-02-16 16:20:21
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-02-16 19:54:24
FilePath: /hyou37/xluo/LCM/module_latency/conv_latency_compare.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%A
'''
import numpy as np
from collections import defaultdict

def get_latencies(filename):
    o_file = open(filename, "r")
    o_latency = defaultdict(list)

    begin = False
    for line in o_file.readlines():
        if 'No. 4'in line:
            begin = True
        if not begin:
            continue
        if 'class' in line:
            l = float(line.split(">")[1].split(" ")[2])
            o_latency[line.split(">")[0].split(" ")[1][1:-1]].append(l)


    print("latency data amount is", len(o_latency))


    total = defaultdict(lambda: 0)
    for layer, latencies in o_latency.items():
        for latency in latencies:
            total[layer] += latency

    # for layer in o_latency.keys():
    #     print(f"Layer: {layer}, Amount: {len(o_latency[layer])}, Total: {total[layer]:.3f} ms, Avg: {total[layer]/len(o_latency[layer]):.3f} ms")
    return o_latency

def get_ordered_latencies(filename):
    o_file = open(filename, "r")
    latencies = []

    begin = False
    for line in o_file.readlines():
        if 'No. 4'in line:
            begin = True
        if not begin:
            continue
        if 'class' in line:
            latencies.append((line.split(">")[0].split(" ")[1][1:-1], float(line.split(">")[1].split(" ")[2])))
    return latencies


o_latencies = get_latencies("ounet.txt")
i_latencies = get_latencies("iunet.txt")

keywords = ["SiLU", "GroupNorm", "LayerNorm", "GEGLU"]
latency_diff = {}
compute = {}
amount = {}
for o_k in o_latencies.keys():
    for k in keywords:
        if k in o_k:
            latency_diff[k] = sum(o_latencies[o_k])
            compute[k] = sum(o_latencies[o_k])
            amount[k] = len(o_latencies[o_k])

for i_k in i_latencies.keys():
    for k in keywords:
        if k+"Q" in i_k:
            latency_diff[k] = sum(i_latencies[i_k]) - latency_diff[k]

# print("data movement", latency_diff)
# print("compute", compute)
# print("amount", amount)
i_attn = {}
o_attn = {}
for key, value in i_latencies.items():
    if "Attention" in key:
        i_attn[key.split(".")[-1]] = sum(i_latencies[key])
    if "Conv" in key:
        i_attn[key.split(".")[-1]] = sum(i_latencies[key])
    if "Linear" in key:
        i_attn[key.split(".")[-1]] = sum(i_latencies[key])

for key, value in o_latencies.items():
    if "Attention" in key:
        o_attn[key.split(".")[-1]] = sum(o_latencies[key])
    if "Conv" in key:
        o_attn[key.split(".")[-1]] = sum(o_latencies[key])
    if "Linear" in key:
        o_attn[key.split(".")[-1]] = sum(o_latencies[key])

print(i_attn)
print(o_attn)

# i_ordered_latencies = get_ordered_latencies("iunet.txt")
# print(i_ordered_latencies)
# exit()

def draw_latency_distribution_per_layer_type():
    from matplotlib import pyplot as plt
    for key, value in i_latencies.items():
        if "diffuser" in key or "torch.nn" in key:
            continue

        plt.cla()

        plt.title(f"{key} Latency")
        plt.xlabel("Layer Index")
        plt.ylabel("Latency (ms)")

        plt.plot(value)
        plt.scatter([list(range(len(value)))], value)

        # if any(keyword in key for keyword in keywords):
        assert len(value) % 4 == 0, f"{key} has {len(value)} layers in 4 steps"
        step_delimeter = [int(len(value)*0.25), len(value)//2, int(len(value)*0.75)]
        plt.scatter(step_delimeter, [value[index] for index in step_delimeter], c="r")

        plt.savefig(f"latency_distribution/{key}.png")


def print_latency_distribution_per_layer_order():
    downblock_keys = ["INTCrossAttnDownBlock2D", "INTDownBlock2D"]
    midblock_keys = ["INTUNetMidBlock2DCrossAttn"]
    upblock_keys = ["INTCrossAttnUpBlock2D", "INTUpBlock2D"]

    downblock_latency = 0
    midblock_latency = 0
    upblock_latency = 0

    for key, value in i_latencies.items():
        if any(keyword == key.split(".")[-1] for keyword in downblock_keys):
            downblock_latency += sum(value)
        if any(keyword == key.split(".")[-1] for keyword in midblock_keys):
            midblock_latency += sum(value)
        if any(keyword == key.split(".")[-1] for keyword in upblock_keys):
            upblock_latency += sum(value)

    print(f"downblocks latency is {downblock_latency} ms, midblock latency is {midblock_latency} ms,"
          f"upblock latency is {upblock_latency} ms")
    

print_latency_distribution_per_layer_order()
print("GEGLUQ VS GEGLU", sum(i_latencies["torch_int.nn.fused.GEGLUQ"]), sum(o_latencies["diffusers.models.activations.GEGLU"]))

#print(sum(o_latencies["diffusers.models.activations.GEGLU"]))
# draw_latency_distribution_per_layer_type()
