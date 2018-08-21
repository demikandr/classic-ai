import requests
import time
import numpy as np
from tqdm import tqdm
poets = ["pushkin", "esenin", "mayakovskij", "blok", "tyutchev"]

for poet in poets:
    times = []
    for i in range(100):
        begin_time = time.time()
        requests.post('http://localhost:8000/generate/pushkin', json={'seed': 'регрессия глубокими нейронными сетями'})
        end_time = time.time()
        times.append(end_time - begin_time)
    print(poet, '\t', np.mean(times), '\t', np.max(times))
