import torch
import time

# copy attributes from obj1 to obj2, except those already existed in obj2
def copy_and_report_attributes(obj1, obj2):
    # 获取obj1和obj2的所有属性名称
    obj1_attrs = set(dir(obj1))
    obj2_attrs = set(dir(obj2))

    # 找出obj1中独有的属性
    unique_to_obj1 = obj1_attrs - obj2_attrs

    print("attr copied: ")
    # 复制这些独有属性到obj2
    for attr in unique_to_obj1:
        setattr(obj2, attr, getattr(obj1, attr))
        print(attr)

    # 找出和obj2中同名的属性
    common_attrs = obj1_attrs.intersection(obj2_attrs)

    # 打印这些同名属性
    print("attr with the same name:")
    for attr in common_attrs:
        print(attr)


class LatencyLogger:
    stat = []
    __first_open = True
    __counter_dict = {}
    __file_dict = {}

    @staticmethod
    def put(info):
        LatencyLogger.stat.append(info)

    @staticmethod
    def write(filename, clear=True):
        if filename not in LatencyLogger.__file_dict:
            file = open(filename, "w")
            LatencyLogger.__file_dict[filename] = file
            LatencyLogger.__counter_dict[filename] = 1
        else:
            file = LatencyLogger.__file_dict[filename]
        file.write("-"*20+"\n")
        file.write(f"No. {LatencyLogger.__counter_dict[filename]}\n")
        LatencyLogger.__counter_dict[filename] += 1
        for info in LatencyLogger.stat:
            file.write(info+"\n")
        if clear:
            LatencyLogger.stat = []


import time
def time_forward(
        self,
        *args,
        **kwargs):
        _time_forward_start_time = time.perf_counter()

        ans = self.original_forward(
            *args, **kwargs
        )
        torch.cuda.synchronize()

        # print(f"{self.__class__} take {time.perf_counter()-_time_forward_start_time} secs for forwarding")
        LatencyLogger.put(f"{self.__class__} take {1000*(time.perf_counter()-_time_forward_start_time)} ms for forwarding")
        return ans


import types
def replace_with_time_forward(model: torch.nn.Module, first=True):
    if first:
        model.original_forward = types.MethodType(type(model).forward, model)
        model.forward = types.MethodType(time_forward, model)

    for name, child in model.named_children():
        if hasattr(child, 'forward'):
            
            child.original_forward = types.MethodType(type(child).forward, child)
            child.forward = types.MethodType(time_forward, child)

        if isinstance(child, torch.nn.Module):
            replace_with_time_forward(child, False)