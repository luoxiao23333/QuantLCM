'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-02-16 16:20:21
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-02-16 19:54:24
FilePath: /hyou37/xluo/LCM/module_latency/conv_latency_compare.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%A
'''
import numpy as np

o_file = open("ounet.txt", "r")
i_file = open("iunet.txt", "r")
o_keyword = "LoRACompatibleConv"
i_keyword = "TestW8A8B8O8Conv2D16"
o_latency = []
i_latency=  []

for line in o_file.readlines():
    if o_keyword in line:
        l = float(line.split(">")[1].split(" ")[2])
        if l<=500:
            o_latency.append(l)

for line in i_file.readlines():
    if i_keyword in line:
        l = float(line.split(">")[1].split(" ")[2])
        if l<=500:
            i_latency.append(l)

print("latency data amount is (original VS Quant)", len(o_latency), len(i_latency))

o_latency = np.array(o_latency)
i_latency = np.array(i_latency)

print(f"latency avg (original VS Quant): {o_latency.mean()} : {i_latency.mean()}")
print(f"latency std (original VS Quant): {o_latency.std()} : {i_latency.std()}")
print(f"latency median (original VS Quant): {np.median(o_latency)} : {np.median(i_latency)}")

'''
latency data amount is (original VS Quant) 1536 1536
latency avg (original VS Quant): 0.3827221516985446 : 0.18156648366129957
latency std (original VS Quant): 0.6133643225770182 : 0.11872348482721759
latency median (original VS Quant): 0.272553414106369 : 0.15253201127052307
'''