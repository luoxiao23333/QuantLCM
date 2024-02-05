
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