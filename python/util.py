import os
import re
import tempfile
from satenv import SatEnv
from uuid import uuid4

import torch
from torch.autograd import Variable

import datetime
from collections import defaultdict
import numpy as np


def timestamp():
    return str(datetime.datetime.now().strftime("%b-%d-%y--%H-%M-%S"))


def check_make_path(path):
    path = os.path.dirname(path)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            print("[WARNING] os.makedirs failed")


def files_with_extension(dir_path, ext = None):
    if ext is None:
        dataset_files = [os.path.join(dir_path, x) for x in os.listdir(dir_path)]
    else:
        dataset_files = [os.path.join(dir_path, x) for x in os.listdir(dir_path) if re.search(r"." + ext, x)]
    return dataset_files


def recursively_get_files(folder, exts, forbidden = []):
    files = []
    for r, d, f in os.walk(folder):
        for x in f:
            if any([re.search(r"." + ext, x) for ext in exts]):
                if any([re.search(r"." + ext, x) for ext in forbidden]):
                    continue
                else:
                    files.append(os.path.join(r, x))
    return files


def get_adj_list(G):
    clause_size = G.size()[0]
    variables = G.size()[1]
    indices = G._indices()
    values = G._values()
    adj_lists = defaultdict(list)
    node_lists = defaultdict(list)
    for j in range(clause_size):
        nodes = indices[0][indices[1] == j].cpu().numpy()
        for i, v in enumerate(nodes):
            for k in nodes:
                node_lists[i].append(k)
                if k == v:
                    continue
                adj_lists[v].append(k)
    return adj_lists, node_lists


def load_data(G, clause_values, batch_size = 256):
    adj_list, node_lists = get_adj_list(G)
    [clause_size, num_nodes] = G.size()
    rand_indices = np.random.permutation(num_nodes)
    train_nodes = list(rand_indices)
    clause_rand_indices = np.random.permutation(clause_size)
    train_cls = list(clause_rand_indices)
    # print([np.array(train_nodes)], np.array(clause_values)[np.array(train_nodes)])

    return adj_list, node_lists, train_nodes[:batch_size], train_cls[:batch_size], Variable(torch.LongTensor(np.array(clause_values)[np.array(train_nodes[:batch_size])]))


def set_env(from_cnf = None, from_file = None):
    if from_cnf is not None:
        td = tempfile.TemporaryDirectory()
        cnf_path = os.path.join(td.name, str(uuid4()) + ".cnf")
        from_cnf.to_file(cnf_path)
        try:
            env = SatEnv(cnf_path)
            return env
        except RuntimeError as e:
            print("BAD CNF:", cnf_path)
        raise e
    elif from_file is not None:
        try:
            env = SatEnv(from_file)
            return env
        except RuntimeError as e:
            print("BAD CNF:", from_file)
            raise e
    else:
        raise Exception("must set env with CNF or file")
