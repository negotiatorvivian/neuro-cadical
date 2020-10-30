import matplotlib.pyplot as plt
import sys
import json
import subprocess
import os
from collections import defaultdict
from python.util import files_with_extension
from python.config import SATENV_PATH

def get_latest_from_index(ckpt_dir):
    index = files_with_extension(ckpt_dir, "index")[0]
    with open(index, "r") as f:
        cfg_dict = json.load(f)
    return cfg_dict["latest"]


def load_latest_ckpt(root_dir):
    try:
        # print(ckpt_dir, '/'.join(ckpt_dir.split('/')[:-2]))
        # root_dir = '/'.join(ckpt_dir.split('/')[:-2])
        ckpt_path = get_latest_from_index(root_dir)
    except IndexError:
        print("NO INDEX FOUND")
        return None
    logpath = os.path.join('/'.join(ckpt_path.split('/')[:-2]), 'logs', 'log.txt')
    return logpath


def parse_log(logpath):
    res_dict = defaultdict(float)
    frequency = defaultdict(int)
    with open(logpath, 'r') as f:
        while True:
            line = f.readline()
            if len(line) == 0 or line is None:
                break
            line.replace(' ', '')
            line = line.split(':')
            if line[0] == 'name':
                name = line[1].split(',')[0]
                res_dict[name] = float(line[2])
                frequency[name] += 1
    for item in res_dict:
        res_dict[item] = res_dict[item]/frequency[item]
    return res_dict


def parse_pure_file(file_path):
    res_dict = defaultdict(float)
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                continue
            elif line.startswith('DONE'):
                break
            else:
                if not line.startswith('c-'):
                    name = line.strip()
                else:
                    prop = line[2:].split(' ')[0]
                    if prop == 'total-real-time-since-initialization':
                        res_dict[name] = float(line[2:].split(' ')[1])
    return res_dict


def plot_pic(model_result, pure_result):
    x = [item for item in model_result]
    y1 = [model_result[item] for item in x]
    y2 = [pure_result[item] for item in x]
    y1_axis = 
    print(x, y1, y2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y1, y1, '-', label = 'Swdown')
    ax.plot(y2, y2, '-', label = 'Rn')




if __name__ == '__main__':
    logpath = load_latest_ckpt(sys.argv[1])
    model_result = parse_log(logpath)
    command = [os.path.join(SATENV_PATH, 'run-pure-cadical.sh')]
    command += [sys.argv[2]]
    command += [os.path.dirname(logpath)]
    # subprocess.run(command, stdout = subprocess.PIPE)
    pure_result_path = os.path.join(os.path.dirname(logpath), 'pure-results.txt')
    pure_result = parse_pure_file(pure_result_path)
    print(pure_result)
    plot_pic(model_result, pure_result)


