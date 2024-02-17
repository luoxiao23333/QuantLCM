import numpy as np

o_file = open("int8_text_encoder.txt", "r")
i_file = open("text_encoder.txt", "r")
o_keyword = "Linear"
i_keyword = "Linear"
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
