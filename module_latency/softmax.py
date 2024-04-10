f = open("softmax.txt", "r")
data_mov = []
compute = []
for line in f.readlines():
    data_mov.append(int(line.split(";")[0].split(" ")[0]))
    compute.append(int(line.split(";")[1].split("μs")[0]))

print(f"For softmax, data movement is {sum(data_mov)/1000} ms")
print(f"For softmax, compute is {sum(compute)/1000} ms")