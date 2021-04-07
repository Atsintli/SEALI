#integrator all
import json
from utils import*

rootDir = 'music18_all_features/'

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

def get_all_values(path_to_json):
    with open(path_to_json, 'r') as file:
        file = file.read()
        file = file.replace("-nan",  "0")
    j = json.loads(file)
    first_level = j.values()
    second_level = list(map(get_nested_values, first_level))
    third_level = flatten_without_rec(list(map(get_nested_values, flatten_without_rec(second_level))))
    return list(filter(lambda x: not isinstance(x, str), third_level))

features = list(map(get_all_values, sorted(glob.glob(rootDir + "*.json")[0:10])))
        
save_descriptors_as_matrix('features_all_2.csv', features)
#save_matrix_array('features_all_2.csv', features)