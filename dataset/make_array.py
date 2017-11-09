import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lights_energy_low', default=1)
parser.add_argument('--lights_energy_high', default=5)
parser.add_argument('--lights_pos_low', default=[-5, -2.5, -1])
parser.add_argument('--lights_pos_high', default=[5, -3.5, 5])
parser.add_argument('--size', default=20000)
parser.add_argument('--save_path', default='arrays/shader.npy')
args = parser.parse_args()

def random(low, high):
    if type(high) == list:
        params = [np.random.uniform(low=low[ind], high=high[ind]) for ind in range(len(high))]
    else:
        params = np.random.uniform(low=low, high=high)
    return params

low = [args.lights_energy_low] + args.lights_pos_low
high = [args.lights_energy_high] + args.lights_pos_high

params = [random(low, high) for i in range(args.size)]

np.save(args.save_path, params)


