import json
from utils import *

f = "music18_all_features/000024.json"
j = json.load(open(f))

first_level = j.values()
#print(first_level)

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

#flatten_without_rec([[0, 1], [[5]], [6, 7]])
second_level = list(map(get_nested_values, first_level))
#print(second_level)

features = flatten_without_rec(list(map(get_nested_values, flatten_without_rec(second_level))))
print(features)

save_descriptors_as_matrix('features_all_2.csv', features)
