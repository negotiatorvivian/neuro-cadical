from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import sys
import json
import re
import subprocess
import os
import random
from collections import defaultdict
import numpy as np
# from python.util import files_with_extension
# from python.config import SATENV_PATH
from util import files_with_extension
from config import SATENV_PATH


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
    res_dict = defaultdict(list)
    frequency = defaultdict(int)
    with open(logpath, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if len(line) == 0 or line is None:
                break
            line.replace(' ', '')
            line = line.strip().split(':')
            if line[0] == 'total_time':
                j = i + 1
                while lines[j].replace(' ', '').split(':')[0] != ';;name':
                    j += 1
                next_line = lines[j].replace(' ', '').split(':')
                name = next_line[1].split(';')[0].strip()
                if len(res_dict[name]) == 0:
                    res_dict[name] = [float(line[1]), float(next_line[2].split(';')[0])]
                else:
                    res_dict[name][0] += float(line[1])
                    res_dict[name][1] += float(next_line[2].split(';')[0])
                frequency[name] += 1
    for item in res_dict:
        res_dict[item][0] = res_dict[item][0]/frequency[item]
        res_dict[item][1] = res_dict[item][1] / frequency[item]
    return res_dict


def parse_pure_file(file_path):
    res_dict = defaultdict(float)
    clause_dict = defaultdict(int)
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                continue
            elif line.startswith('DONE'):
                break
            else:
                if line.startswith('s') or line.startswith('c ') or line.startswith('v'):
                    continue
                if not line.startswith('c-'):
                    if not re.match("[\d]+", line):
                        name = line.strip()
                    else:
                        clause_dict[name] = int(line) - 1

                else:
                    prop = line[2:].split(' ')[0]
                    if prop == 'total-real-time-since-initialization':
                        res_dict[name] = float(line[2:].split(' ')[1])
    print(f'res_dict: {res_dict}')
    return res_dict, clause_dict


# def plot_pic(model_result, pure_result, clauses):
#     sorted_result = sorted(clauses, key = lambda clauses: clauses[1])
#     x = [item for item in sorted_result]
#     x1 = [str(clauses[name]) for name in x]
#     print(f'x: {x}, length: {len(x)}')
#     y1 = [model_result[item] for item in x]
#     y2 = [pure_result[item]if pure_result[item] > 0 else random.random() for item in x]
#     print(f'y2:{y2}')
#     host = host_subplot(111, axes_class = AA.Axes)
#     plt.subplots_adjust(right = 0.75)
#     host.set_xlim(1, len(x))
#     host.set_ylim(min(y1), max(y1))
#     par1 = host.twinx()
#     par1.axis["right"].toggle(all = True)
#     par1.set_ylim(min(y2), max(y2))
#     host.set_xlabel("CNF")
#     host.set_ylabel("model-real-time")
#     par1.set_ylabel("pure-solver-real-time")
#     print(f'x1: {x1}')
#     plt.xticks(range(1, len(x) + 1), x1, rotation = '45')
#
#     p1, = host.plot([i + 1 for i in range(len(x))], y1, label = "model-real-time")
#     p2, = par1.plot([i + 1 for i in range(len(x))], y2, label = "pure-solver-real-time")
#     host.legend()
#     host.axis["left"].label.set_color(p1.get_color())
#     par1.axis["right"].label.set_color(p2.get_color())
#
#     plt.draw()
#     plt.show()


def merge_result(model_result, pure_result, clauses):
    model_merge_temp = defaultdict(list)
    pure_merge_temp = defaultdict(list)
    clause_dict = defaultdict(str)
    for index, i in enumerate(clauses):
        length = clauses[i]
        if i not in model_result:
            continue
        model_merge_temp[index].append(model_result[i])
        pure_merge_temp[index].append(pure_result[i])
        clause_dict[index] = i
    model_merge = defaultdict(list)
    pure_merge = defaultdict(float)
    for i in model_merge_temp:
        model_merge[i] = [model_merge_temp[i][0][0], model_merge_temp[i][0][1]]
        pure_merge[i] = sum(pure_merge_temp[i])/len(pure_merge_temp[i])
    return model_merge, pure_merge, clause_dict


def plot_table_pic(model_result, pure_result, clauses):
    res1, res2, clause_dict = merge_result(model_result, pure_result, clauses)
    # sorted_result = sorted(clauses, key = lambda clauses: clauses[1])
    x = [[clause_dict[item], clauses[clause_dict[item]]] for item in clause_dict]  # cnf name
    print(x)
    x_index = [i + 1 for i in range(len(res1.keys()))]
    x1 = [str(i) for i in res1]     # clause num
    y1_model = [res1[i][0] for i in res1]
    y1_solver = [res1[i][1] for i in res1]
    y1_all = [res1[i][0] + res1[i][1] for i in res1]
    y2 = [res2[i] for i in res2]
    # print(f'y1_model:{y1_model}, y1_solver:{y1_solver},y2: {y2}')
    min_value = min(y1_all + y2)
    max_value = max(y1_all + y2)
    y = [y1_model, y1_solver, y1_all, y2]
    rows = ['model', 'solver', 'RLSP-total', 'cadical']
    # colors = plt.cm.BuPu(np.linspace(0, 1, len(rows)))
    n_rows = len(y)
    colors = ['deepskyblue', 'dodgerblue', 'tomato', 'orange']
    index = np.arange(len(x1)) + 1
    plt.ylabel('CPU Time(s)')
    # plt.xlabel('Number of clauses', labelpad = 38)
    plt.ylim(min_value, max_value)
    # print(f'x: {x}, x1: {x1}, y1: {y1}, y2: {y2}')

    cell_text = []
    for row in range(n_rows):
        plt.plot(index, y[row], color = colors[row])
        cell_text.append(['%.4f' % x for x in y[row]])
    # print(f'x: {x}, y: {y}, cell_text: {cell_text}')
    # colors = colors[::-1]
    # cell_text.reverse()
    x = [['', ''], ['', ''], ['', ''], ['name', 'clause number']] + x
    print(f'x: {[item[0] for item in x]}')
    label_table = plt.table(cellText = x,
                            rowLabels = ['model', 'cadical', '',''] + x_index,
                            colLabels = ['name', 'clause number'],
                            loc = 'bottom',
                            cellLoc = 'center',
                            rowLoc = 'center',
                            )
    label_table.auto_set_font_size(False)
    label_table.set_fontsize(10)

    the_table = plt.table(cellText = cell_text,
                          rowLabels = rows,
                          rowColours = colors,
                          colLabels = x_index,
                          loc = 'bottom',
                          cellLoc = 'center',
                          rowLoc = 'center',
                          # colWidths = [0.2] * len(x)
                          )
    # print(cell_text, rows, x_index)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)

    plt.subplots_adjust(left = 0.2, bottom = 0.45)
    plt.xticks([])
    plt.title('Total real time')
    plt.show()


if __name__ == '__main__':
    logpath = load_latest_ckpt(sys.argv[1])
    model_result = parse_log(logpath)
    command = [os.path.join(SATENV_PATH, 'run-pure-cadical.sh')]
    command += [sys.argv[2]]
    command += [os.path.dirname(logpath)]
    path = sys.argv[2].split('/')
    answer_path = sys.argv[2]
    if path[-1] == 'SAT' or path[-1] == 'UNSAT':
        answer_path = '/'.join(path[:-1])
    pure_result_path = os.path.join(answer_path + '-answers', 'pure-results.txt')
    if not os.path.exists(pure_result_path):
        subprocess.run(command, stdout = subprocess.PIPE)
    pure_result, clauses = parse_pure_file(pure_result_path)
    # print(f'pure_result: {pure_result}, model_result: {model_result}')
    plot_table_pic(model_result, pure_result, clauses)


