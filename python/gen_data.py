"""
Generate data for supervised training.
"""
import io
import time
import glob
import datetime
import tempfile
import uuid
import os
import sys
from pysat.solvers import Solver
from pysat.formula import CNF
from uuid import uuid4
import numpy as np
import random
import subprocess
import h5py as h5
import shutil
import torch.nn as nn
from types import SimpleNamespace
import util as util
from data_util import *
from config import *


def lemma_occ(tsr):
    n_vars = tsr.shape[0]
    result = np.zeros(shape = [n_vars])
    for idx in range(n_vars):
        result[idx] = np.sum(tsr[idx, 0, :].numpy())
    return result


def del_occ(tsr):
    n_vars = tsr.shape[0]
    result = np.zeros(shape = [n_vars])
    for idx in range(n_vars):
        result[idx] = np.sum(tsr[idx, 1, :])
    return result


class CNFDataset(td.IterableDataset):
    def __init__(self):
        raise Exception("abstract method")

    def gen_formula(self):
        raise Exception("abstract method")

    def __iter__(self):
        def _gen_formula():
            while True:
                try:
                    yield self.gen_formula()
                except StopIteration:
                    return

        return _gen_formula()


class CNFDirDataset:
    def __init__(self, data_dir, batch_size, type = 'lbdp'):
        self.data_dir = data_dir
        self.files = util.files_with_extension(self.data_dir, "cnf")
        self.file_index = 0
        self.batch_size = batch_size
        self.type = type
        self.lbdps = self.gen_formula()

    def __len__(self):
        return len(self.files)

    def __next__(self):
        if self.batch_size is None:
            return self._mk_iter()
        else:
            return batch_iterator(self._mk_iter(), self.batch_size)

    def _mk_iter(self):
        for f in self.files:
            cnf = CNF(from_file = f)
            td = tempfile.TemporaryDirectory()
            if self.type == 'lbdp':
                yield gen_lbdp(td, cnf)
            else:
                yield gen_nmsdp(td, cnf)

    def gen_formula(self):
        try:
            cnfs = []
            for i in range(len(self.files)):
                cnf = CNF(from_file = self.files[i])
                self.file_index = i
                cnfs.append(cnf)
            return cnfs
        except IndexError:
            raise StopIteration  # self.file_index += 1  # return gen_lbdp(td, cnf)


class Logger:
    def __init__(self):
        raise Exception("Abstract method")

    def write(self):
        raise Exception("Abstract method")


class SimpleLogger(Logger):
    def __init__(self, logfile):
        self.logfile = logfile
        util.check_make_path(logfile)

    def write(self, *args, verbose = True):
        with open(self.logfile, "a") as f:
            if verbose:
                print(f"({datetime.datetime.now()}):", *args)
            print(f"({datetime.datetime.now()}):", *args, file = f)


class DummyLogger(Logger):
    def __init__(self, verbose = False):
        self.verbose = verbose

    def write(self, *args, verbose = True, **kwargs):
        if self.verbose and verbose:
            print(*args)


def coo(fmla):
    """
  Returns sparse indices of a CNF object, as two numpy arrays.
  """
    C_result = []
    L_result = []
    clause_values = [0] * len(fmla.clauses)
    edge_features = []
    for cls_idx in range(len(fmla.clauses)):
        for lit in fmla.clauses[cls_idx]:
            if lit > 0:
                edge_features.append(1)
                clause_values[cls_idx] = 1
                lit_enc = lit - 1
            else:
                edge_features.append(0)
                lit_enc = fmla.nv + abs(lit) - 1

            C_result.append(cls_idx)
            L_result.append(lit_enc)
    variable_ind = np.abs(np.array(L_result, dtype = np.int32))
    function_ind = np.abs(np.array(C_result, dtype = np.int32))
    unbatch_variable_ind = np.zeros(len(variable_ind), dtype = np.int32)
    unbatch_variable_ind[variable_ind < fmla.nv] = variable_ind[variable_ind < fmla.nv]
    unbatch_variable_ind[variable_ind > fmla.nv - 1] = variable_ind[variable_ind > fmla.nv - 1] - fmla.nv
    graph_map = np.stack((unbatch_variable_ind, function_ind))
    mask = torch.sparse_coo_tensor(graph_map, (torch.FloatTensor(edge_features)).squeeze().float(),
                                   [fmla.nv, len(fmla.clauses)]).unsqueeze(1).to_dense()
    if fmla.is_sat:
        core_var_mask = np.sign(fmla.answers)
        core_clause_mask = np.ones(len(fmla.clauses), dtype = np.int32)
    else:
        core_var_mask = np.ones(fmla.nv, dtype = np.int32)
        core_clause_mask = np.array(clause_values, dtype = "int32")
    return function_ind, variable_ind, np.array(clause_values, dtype = "int32"), mask, core_var_mask, core_clause_mask


def lbdcdl(cnf_dir, cnf, llpath, dump_dir = None, dumpfreq = 50e3, timeout = None, clause_limit = 1e6):
    """
  Args: CNF object, optional timeout and dump flags
  Returns: nothing
  """
    cnf_path = os.path.join(cnf_dir, str(uuid.uuid4()) + ".cnf.gz")
    cnf.to_file(cnf_path, compress_with = "gzip")
    cadical_command = [CADICAL_PATH]
    cadical_command += ["-ll", llpath]
    if dump_dir is not None:
        cadical_command += ["--dump"]
        cadical_command += ["-dd", dump_dir]
        cadical_command += [f"--dumpfreq={int(dumpfreq)}"]
    if timeout is not None:
        cadical_command += ["-t", str(int(timeout))]
    if clause_limit is not None:
        cadical_command += [f"--clauselim={int(clause_limit)}"]
    cadical_command += [f"--seed={int(np.random.choice(int(10e5)))}"]
    cadical_command += [cnf_path]
    # print(cadical_command)

    subprocess.run(cadical_command, stdout = subprocess.PIPE)


def gen_lbdp(td, cnf, is_train = True, logger = DummyLogger(verbose = True), dump_dir = None, dumpfreq = 50e3,
             timeout = None, clause_limit = 1e6):
    clause_limit = int(clause_limit)
    fmla = cnf
    counts = np.zeros(fmla.nv)
    n_vars = fmla.nv
    n_clauses = len(fmla.clauses)
    name = str(uuid.uuid4())
    with td as td:
        llpath = os.path.join(td, name + ".json")
        lbdcdl(td, fmla, llpath, dump_dir = dump_dir, dumpfreq = dumpfreq, timeout = timeout,
               clause_limit = clause_limit)
        with open(llpath, "r") as f:
            for idx, line in enumerate(f):
                counts[idx] = int(line.split()[1])

    C_idxs, L_idxs, clause_values, _, _, _ = coo(fmla)
    n_clauses = len(fmla.clauses)

    lbdp = LBDP(dp_id = name, is_train = np.array([is_train], dtype = "bool"),
                n_vars = np.array([n_vars], dtype = "int32"), n_clauses = np.array([n_clauses], dtype = "int32"),
                C_idxs = np.array(C_idxs), L_idxs = np.array(L_idxs), clause_values = np.array(clause_values),
                glue_counts = counts)

    return lbdp


def gen_nmsdp(td, cnf, is_train = True, logger = DummyLogger(verbose = True), dump_dir = None, dumpfreq = 50e3,
              timeout = None, clause_limit = 1e6):
    clause_limit = int(clause_limit)
    fmla = cnf
    counts = np.zeros(fmla.nv)
    is_sat = fmla.is_sat
    n_vars = fmla.nv
    n_clauses = len(fmla.clauses)
    name = str(uuid.uuid4())
    with td as td:
        llpath = os.path.join(td, name + ".json")
        lbdcdl(td, fmla, llpath, dump_dir = dump_dir, dumpfreq = dumpfreq, timeout = timeout,
               clause_limit = clause_limit)
        with open(llpath, "r") as f:
            for idx, line in enumerate(f):
                counts[idx] = int(line.split()[1])

    C_idxs, L_idxs, clause_values, mask, core_var_mask, core_clause_mask = coo(fmla)
    var_lemma_counts = lemma_occ(mask)

    nmsdp = NMSDP(dp_id = name, is_train = np.array([is_train], dtype = "bool"),
                  is_sat = np.array([is_sat], dtype = "bool"), n_vars = np.array([n_vars], dtype = "int32"),
                  n_clauses = np.array([n_clauses], dtype = "int32"), C_idxs = np.array(C_idxs),
                  L_idxs = np.array(L_idxs), core_var_mask = core_var_mask, core_clause_mask = core_clause_mask,
                  var_lemma_counts = var_lemma_counts)

    return nmsdp


def data_to_cnf(data, td, unsat = False):
    clauses = []
    for index in range(data.shape[0]):
        cls = ((np.argwhere(data[index] != 0) + 1) * data[index][np.argwhere(data[index] != 0)]).cpu().numpy().astype(np.int32)[0]
        clauses.append(list(cls))

    cnf = CNF(from_clauses = clauses)
    uid = str(uuid4())[:10]
    if unsat:
        # temp_path = os.path.join('/home/ziwei/Workspace/neuro-cadical/data/unsat_core', str(uuid4())[:10] + '-unsatcore' + str(int(time.time())) + ".cnf")
        # cnf.to_file(temp_path)
        # print(temp_path)
        name = uid + '-unsatcore' + str(int(time.time()))
        cnf_path = os.path.join(td.name, name + ".cnf")
        cnf.to_file(cnf_path)
    else:
        name = uid + str(int(time.time()))
        cnf_path = os.path.join(td.name, name + ".cnf")
        cnf.to_file(cnf_path)
    # print(f'cnf name: {name}')


class CNFProcessor:
    def __init__(self, cnfdataset, tmpdir = None, use_glue_counts = True, timeout = None):
        if tmpdir is None:
            self.tmpdir = tempfile.TemporaryDirectory()
        else:
            self.tmpdir = tmpdir
        self.cnfdataset = cnfdataset
        self.use_glue_counts = use_glue_counts
        self.timeout = timeout

    def _mk_nmsdp_gen(self):
        for cnf in self.cnfdataset:
            if not self.use_glue_counts:
                nmsdp = gen_nmsdp(self.tmpdir, cnf)
            else:
                nmsdp = gen_lbdp(self.tmpdir, cnf, timeout = self.timeout)
                if np.sum(nmsdp.glue_counts) <= 50:
                    continue
            self.tmpdir = tempfile.TemporaryDirectory()
            yield nmsdp

    def __iter__(self):
        return self._mk_nmsdp_gen()

    def clean(self):
        print("[CNF PROCESSOR]: CLEANING TMPDIR")
        self.tmpdir.cleanup()
        util.check_make_path(self.tmpdir.name)

    def __del__(self):
        self.clean()


def mk_CNFDataloader(data_dir, batch_size, num_workers):
    cnf_dataset = CNFDirDataset(data_dir, batch_size)
    return td.DataLoader(cnf_dataset, batch_size = 1, num_workers = num_workers, worker_init_fn = h5_worker_init_fn,
                         pin_memory = True)


def parse_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", action = "store")
    parser.add_argument("dest_dir", action = "store")
    parser.add_argument("--num_workers", dest = "num_workers", action = "store", default = 8, type = int)
    parser.add_argument("--batch_size", dest = "batch_size", action = "store", type = int, default = 16)
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    cfg = parse_main()
    cnf_dataset = CNFDirDataset(cfg.data_dir, cfg.batch_size, 'nmsdp')
    index = 0
    processor = CNFProcessor(cnf_dataset.lbdps, use_glue_counts = False)
    name = str(uuid.uuid4())
    with h5py.File(os.path.join(cfg.dest_dir, name + '.h5'), 'a') as f:
        for ldbp in processor:
            print(ldbp.dp_id)
            serialize_lbdp(ldbp, f)