import random

import numpy as np

def boostrap_sample(data):
   len = np.size(data, 0)
   mask = np.array([random.random() > 0.5 for i in range(len)])
   return data[mask, ...]

def bootstrap_statistic(data, stats_fn, num_samples: int):
    return [stats_fn(boostrap_sample(data)) for i in range(num_samples)]

