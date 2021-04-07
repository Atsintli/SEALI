#%%
#integrator all
import json
from utils import*

rootDir = 'music18_all_features/'

def fix_nan_bug(files):
    with open(files, 'r') as f:
        f = f.read()
        f = f.replace("-nan",  "0")
        fixed_file = json.loads(f)
        first_level = fixed_file.values()
    return first_level

def exec_fix_nan_bug(files):
    	return list(map(fix_nan_bug, files))

def get_nested_values(val):
    if 'dict' == val.__class__.__name__:
        return list(val.values())
    else:
        return val

def flatten_without_rec(non_flat): 
    flat = []
    while non_flat: #runs until the given list is empty.
            e = non_flat.pop()
            if type(e) == list: #checks the type of the poped item.
                    non_flat.extend(e) #if list extend the item to given list.
            else:
                    flat.append(e) #if not list then add it to the flat list. 
    return flat

def fisrtLevel(files):
    first_level = fix_nan_bug(files)
    return first_level

#%%

#first_level = exec_fix_nan_bug(sorted(glob.glob(rootDir + "*.json")[0:5]))
first_level = fix_nan_bug("music18_all_features/000024.json")
print(first_level)
#first_level = list(map(fisrtLevel, sorted(glob.glob(rootDir + "*.json"))[0:5]))

second_level = list(map(get_nested_values, first_level))
print(second_level)

features = flatten_without_rec(list(map(get_nested_values, flatten_without_rec(second_level))))
print(features)

#save_descriptors_as_matrix('features_all_2.csv', features)
# %%
