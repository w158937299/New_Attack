import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
import Parser
import time

def get_needed_cut(num, file_path):
    needed_group = []
    finish_group = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line[1: -2].split(', ')

            if len(line) >= num:
                needed_group.append(line)
            else:
                finish_group.append(line)
    return needed_group, finish_group


# for i in range(len(needed_group)):
#     print(needed_group[i])
#
#
# print(len(needed_group))