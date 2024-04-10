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

print("data movement", latency_diff)
print("compute", compute)
print("amount", amount)
