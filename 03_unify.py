import numpy as np
import json

file = 'music18_all_features/000055.json'
#file = json.load(open(file))

#first_level = file
#
#def get_nested_values(val):
#    print(val)
#    if val == 'dict':
#        return list(val.values())
#    else:
#        return val
#
#second_level = list(map(get_nested_values, first_level))
#print(second_level)

# with open(file, 'r') as f:
#     f = f.read()
#     #print(f)
#     f = f.replace("-nan",  "0")
#     res = json.loads(f) 
#     print((res))
#     #f.close

def fix_nan_bug(files):
    with open(file, 'r') as f:
        f = f.read()
        f = f.replace("-nan",  "0")
        fixed_file = json.loads(f)
        print(fixed_file)
    return fixed_file

fix_nan_bug(file)

