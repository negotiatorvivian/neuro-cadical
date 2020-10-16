import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import tempfile
import json
import datetime
import multiprocessing
import time
import yaml
import torch.multiprocessing as mp

from python.util import check_make_path, files_with_extension, load_data
from python.gnn import *
from python.batch import Batcher
from python.data_util import H5Dataset, BatchedIterable, mk_H5DataLoader
from sp.factorgraph.dataset import DynamicBatchDivider
from sp.nn.util import SatLossEvaluator
from sp.trainer import Perceptron
from sp.nn import solver


def compute_softmax_kldiv_loss(logitss, probss):
    """
    Args:
      logitss: a list of 1D tensors
      probss: a list of 1D tensors, each of which is a valid probability distribution (entries sum to 1)

    Returns:
      averaged KL divergence loss
    """
    result = 0
    # loss = nn.KLDivLoss(reduction="sum")
    for logits, probs in zip(logitss, probss):
        probs = probs.squeeze()  # required because of DataLoader magic
        # print("LOGITS", logits)
        # print("PROBS", probs)
        # print(logits.size())
        logits = F.log_softmax(logits, dim = 0)  # must be log-probabilities
        cl = F.kl_div(input = logits.view([1, logits.size(0)]), target = probs.view([1, probs.size(0)]),
                      reduction = "sum")
        # print("CL", cl)
        result += cl  # result += F.kl_div(logits, probs, reduction="none")
    result = result / float(len(probss))
    return result


def compute_mask_loss(V_logitss, masks):
    """
    Computes softmax KL-divergence loss with respect to target uniform distributions represented by multi-hot masks
    """
    targets = [(x.float() * 1 / x.sum()).view([x.size(0)]) for x in masks]
    # print("MASK LOSS TARGET EXAMPLE", targets[0])
    return compute_softmax_kldiv_loss(V_logitss, targets)


def compute_softmax_kldiv_loss_from_logits(V_logitss, target_logits, tau = 4.0):
    softmax = nn.Softmax(dim = 1)
    # print(target_logits[0])
    target_probs = []
    for logits in target_logits:
        logits = (logits - logits.mean())
        logits = logits / (logits.std() + 1e-10)
        logits = tau * logits
        target_probs.append(softmax((logits.view((1,) + logits.size()))).view(logits.size()))

    # target_probs = [softmax(tau * (logits.view((1,) + logits.size()))).view(logits.size()) for logits in target_logits]
    # # print("TARGET PROBS", target_probs[0])
    return compute_softmax_kldiv_loss(V_logitss, target_probs)


def NMSDP_to_sparse(nmsdp):
    C_idxs = torch.from_numpy(nmsdp.C_idxs)
    L_idxs = torch.from_numpy(nmsdp.L_idxs)
    indices = torch.stack([C_idxs.type(torch.long), L_idxs.type(torch.long)])
    values = torch.ones(len(C_idxs), device = indices.device)
    size = [nmsdp.n_clauses[0], 2 * nmsdp.n_vars[0]]
    return torch.sparse.FloatTensor(indices = indices, values = values, size = size)


def NMSDP_to_sparse2(nmsdp):  # needed because of some magic tensor coercion done by td.DataLoader constructor
    C_idxs = nmsdp.C_idxs[0]
    L_idxs = nmsdp.L_idxs[0]
    indices = torch.stack([C_idxs.type(torch.long), L_idxs.type(torch.long)])
    values = torch.ones(len(C_idxs), device = indices.device)
    size = [nmsdp.n_clauses[0], 2 * nmsdp.n_vars[0]]
    return torch.sparse.FloatTensor(indices = indices, values = values, size = size)


def NMSDP_to_line(nmsdp):
    C_idxs = nmsdp.C_idxs[0]
    L_idxs = nmsdp.L_idxs[0]
    signs = np.array([1] * (2 * nmsdp.n_vars[0]))
    signs[nmsdp.n_vars[0]:] = -1
    edge_features = signs[L_idxs]
    L_idxs = L_idxs.cpu().numpy()
    n_vars = nmsdp.n_vars[0].cpu().numpy()[0]
    n_cls = nmsdp.n_clauses[0].cpu().numpy()[0]
    L_idxs[L_idxs > n_vars - 1] -= n_vars
    indices = torch.stack([torch.from_numpy(L_idxs).type(torch.long), C_idxs.type(torch.long)])
    result = True
    # return torch.sparse.FloatTensor(indices = indices, values = torch.from_numpy(edge_features), size = size)
    return (n_vars, n_cls, indices, edge_features, None, float(result), [])


def train_step(model, batcher, optim, prediction, nmsdps, device = torch.device("cpu"), CUDA_FLAG = False,
        use_NMSDP_to_sparse2 = True, use_glue_counts = True):
    # the flag use_NMSDP_to_sparse2 should be True when we use mk_H5DataLoader instead of iterating over the H5Dataset directly, because DataLoader
    # does magic conversions from numpy arrays to torch tensors
    optim.zero_grad()
    Gs = []
    var_lemma_countss = []
    core_var_masks = []
    core_clause_masks = []
    glue_countss = []

    def maybe_non_blocking(tsr):
        if CUDA_FLAG:
            return tsr.cuda(non_blocking = True)
        else:
            return tsr

    # print("USE GLUE COUNTS: ", use_glue_counts)
    for nmsdp in nmsdps:
        if not use_glue_counts:
            if not use_NMSDP_to_sparse2:
                Gs.append(maybe_non_blocking(NMSDP_to_sparse(nmsdp)))
                var_lemma_countss.append(
                    maybe_non_blocking(torch.from_numpy(nmsdp.var_lemma_counts).type(torch.float32).squeeze()).to(
                        device))
                core_var_masks.append(
                    maybe_non_blocking(torch.from_numpy(nmsdp.core_var_mask).type(torch.bool).squeeze()).to(device))
                core_clause_masks.append(
                    maybe_non_blocking(torch.from_numpy(nmsdp.core_clause_mask).type(torch.bool).squeeze()).to(device))
            else:
                Gs.append(maybe_non_blocking(NMSDP_to_sparse2(nmsdp)))
                var_lemma_countss.append(maybe_non_blocking(nmsdp.var_lemma_counts.type(torch.float32)[0]).to(device))
                core_var_masks.append(maybe_non_blocking(nmsdp.core_var_mask.type(torch.bool)[0]).to(device))
                core_clause_masks.append(maybe_non_blocking(nmsdp.core_clause_mask.type(torch.bool)[0]).to(device))
        else:
            if not use_NMSDP_to_sparse2:
                Gs.append(maybe_non_blocking(NMSDP_to_sparse(nmsdp)))
                glue_countss.append(
                    maybe_non_blocking(torch.from_numpy(nmsdp.glue_counts).type(torch.float32).squeeze()).to(device))
            else:
                Gs.append(maybe_non_blocking(NMSDP_to_sparse2(nmsdp)))
                glue_countss.append(maybe_non_blocking(nmsdp.glue_counts.type(torch.float32)[0]).to(device))

    G, clause_values = batcher(Gs)
    G.to(device)
    batched_V_drat_logits, batched_V_core_logits, batched_C_core_logits = (
        lambda x: (x[0].view([x[0].size(0)]), x[1].view([x[1].size(0)]), x[2].view([x[2].size(0)])))(model(G))
    # batched_V_drat_logits, batched_V_core_logits = (model(G))

    V_drat_logitss = batcher.unbatch(batched_V_drat_logits, mode = "variable")
    if not use_glue_counts:
        V_core_logitss = batcher.unbatch(batched_V_core_logits, mode = "variable")
        C_core_logitss = batcher.unbatch(batched_C_core_logits, mode = "clause")

    # print("UNBATCHED DRAT LOGITS", [x.shape for x in V_drat_logitss])
    # print("DRAT LABELS", [x.shape for x in var_lemma_countss])

    # print("ok")

    # breakpoint()

    if use_glue_counts:
        drat_loss = compute_softmax_kldiv_loss_from_logits(V_drat_logitss, glue_countss, tau = 1.0)
        core_loss = 0
        core_clause_loss = 0
    else:
        drat_loss = compute_softmax_kldiv_loss_from_logits(V_drat_logitss, var_lemma_countss, tau = 1.0)
        # core_loss = compute_mask_loss(torch.cat(V_core_logitss).unsqueeze(1), [prediction[0].detach()])
        # core_loss = compute_mask_loss(V_core_logitss, core_var_masks)
        core_clause_loss = compute_mask_loss(C_core_logitss, core_clause_masks)
    # core_loss = 0
    # core_clause_loss = 0

    l2_loss = 0.0

    for param in model.parameters():
        l2_loss += (param ** 2).sum()
    # core_loss = 0
    # core_clause_loss = 0

    l2_loss = l2_loss * 1e-9

    # print("EXAMPLE CORE CLAUSE MASK", core_clause_masks[0])
    # print("EXAMPLE DRAT VAR COUNT", var_lemma_countss[0])

    # loss = drat_loss + 0.1 * core_loss + 0.01 * core_clause_loss + l2_loss
    loss = drat_loss + 0.01 * core_clause_loss + l2_loss
    print('loss:', loss)
    # loss = drat_loss
    loss.backward()

    nn.utils.clip_grad_value_(model.parameters(), 100)
    nn.utils.clip_grad_norm_(model.parameters(), 10)

    x = 0
    for name, param in model.named_parameters():
        # print(name, param.grad)
        try:
            g = param.grad
            x += g.norm()

            num_g_entries = torch.prod(torch.tensor(list(g.size())), 0)
            num_g_nonzero_entries = torch.nonzero(g).size(0)
            if not num_g_entries == num_g_nonzero_entries:
                print("G SIZE", num_g_entries, "LEN NONZEROS", num_g_nonzero_entries, "OH NO ZERO GRAD AT", name, g,
                      "AHHHHHHHH")
        except AttributeError:
            pass

    # for k, v in optim.state_dict().items():
    #   print(f"OPTIM KEY: {k} || OPTIM VALUE: {v}")

    optim.step()

    return drat_loss, core_clause_loss, loss, x, l2_loss


def train_step2(model, optim, nmsdps, device = torch.device("cpu"), CUDA_FLAG = False, use_NMSDP_to_sparse2 = False):
    optim.zero_grad()
    Gs = []
    core_var_masks = []
    var_lemma_countss = []

    def maybe_non_blocking(tsr):
        if CUDA_FLAG:
            return tsr.cuda(non_blocking = True)
        else:
            return tsr

    for nmsdp in nmsdps:
        if not use_NMSDP_to_sparse2:
            G = NMSDP_to_sparse(nmsdp)
            Gs.append(maybe_non_blocking(G))
            core_var_masks.append(
                maybe_non_blocking(torch.from_numpy(nmsdp.core_var_mask).type(torch.bool).squeeze()).to(device))
            var_lemma_countss.append(
                maybe_non_blocking(torch.from_numpy(nmsdp.var_lemma_counts).type(torch.float32).squeeze()).to(device))
        else:
            G = NMSDP_to_sparse2(nmsdp)
            Gs.append(maybe_non_blocking(G))
            core_var_masks.append(maybe_non_blocking(nmsdp.core_var_mask.type(torch.bool).squeeze()).to(device))
            var_lemma_countss.append(
                maybe_non_blocking(nmsdp.var_lemma_counts.type(torch.float32).squeeze()).to(device))

    V_drat_logitss, V_core_logitss = model(Gs)

    drat_loss = compute_softmax_kldiv_loss_from_logits(V_drat_logitss, var_lemma_countss)
    core_loss = compute_mask_loss(V_core_logitss, core_var_masks)

    loss = core_loss + drat_loss
    loss.backward()
    optim.step()

    return drat_loss, core_loss, loss


class TrainLogger:
    def __init__(self, logdir):
        self.logdir = logdir
        self.logfile = os.path.join(self.logdir, "log.txt")
        self.writer = SummaryWriter(log_dir = self.logdir)
        check_make_path(self.logdir)

    def write_scalar(self, name, value, global_step):
        print(name, value, global_step)
        self.writer.add_scalar(name, value, global_step)

    def write_log(self, *args):
        # print(f"{datetime.datetime.now()}:", *args)
        with open(self.logfile, "a") as f:
            print(*args, file = f)  # print(f"{datetime.datetime.now()}:", *args, file = f)


class Trainer:
    """
    v1 Trainer object. Only works for a single device.

    Args:
      model: nn.Module object
      dataset: iterable which yields batches of data
      optimizer: pytorch optimizer object
      ckpt dir: path to checkpoint directory
      ckpt_freq: number of gradient updates (i.e. batches) between saving checkpoints

    The `logger` attribute is a TrainLogger object, responsible for writing to training logs _and_ TensorBoard summaries.
    """

    def __init__(self, model, dataset, config, lr, root_dir, ckpt_freq, restore = False, n_steps = -1, n_epochs = -1,
            index = 0, limit = 4000000, batch_replication = 1):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.ckpt_dir = os.path.join(root_dir, "weights/")
        self.logger = TrainLogger(os.path.join(root_dir, "logs/"))
        self.ckpt_freq = ckpt_freq
        self.save_counter = 0
        self.GLOBAL_STEP_COUNT = 0
        self.n_steps = None if n_steps == -1 else n_steps
        self.n_epochs = 1 if n_epochs == -1 else n_epochs
        self.CUDA_FLAG = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.CUDA_FLAG else "cpu")
        self.model.to(self.device)
        self.lr = lr
        self._loss_evaluator = SatLossEvaluator(alpha = self.config['exploration'], device = self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, betas = (0.9, 0.98))
        self.batch_divider = DynamicBatchDivider(limit // batch_replication, self.config['hidden_dim'])
        self.model_list = [self._set_device(model) for model in self._build_graph(self.config)]
        util.check_make_path(self.ckpt_dir)
        if restore:
            try:
                self.load_latest_ckpt()
            except IndexError:
                pass

    def _to_cuda(self, data):
        if isinstance(data, list):
            return data

    def _set_device(self, model):
        "Sets the CPU/GPU device."

        if self.CUDA_FLAG:
            return nn.DataParallel(model).cuda(self.device)
        return model.cpu()

    def _module(self, model):
        return model.module if isinstance(model, nn.DataParallel) else model

    def _build_graph(self, config):
        model_list = []

        if config['model_type'] == 'np-nd-np':
            model_list += [solver.NeuralPropagatorDecimatorSolver(device = self.device, name = config['model_name'],
                                                                  edge_dimension = config['edge_feature_dim'],
                                                                  meta_data_dimension = config['meta_feature_dim'],
                                                                  propagator_dimension = config['hidden_dim'],
                                                                  decimator_dimension = config['hidden_dim'],
                                                                  mem_hidden_dimension = config['mem_hidden_dim'],
                                                                  agg_hidden_dimension = config['agg_hidden_dim'],
                                                                  mem_agg_hidden_dimension = config[
                                                                      'mem_agg_hidden_dim'],
                                                                  prediction_dimension = config['prediction_dim'],
                                                                  variable_classifier = Perceptron(config['hidden_dim'],
                                                                                                   config[
                                                                                                       'classifier_dim'],
                                                                                                   config[
                                                                                                       'prediction_dim']),
                                                                  function_classifier = None,
                                                                  dropout = config['dropout'],
                                                                  local_search_iterations = config[
                                                                      'local_search_iteration'],
                                                                  epsilon = config['epsilon'])]

        elif config['model_type'] == 'p-nd-np':
            model_list += [solver.NeuralSurveyPropagatorSolver(device = self.device, name = config['model_name'],
                                                               edge_dimension = config['edge_feature_dim'],
                                                               meta_data_dimension = config['meta_feature_dim'],
                                                               decimator_dimension = config['hidden_dim'],
                                                               mem_hidden_dimension = config['mem_hidden_dim'],
                                                               agg_hidden_dimension = config['agg_hidden_dim'],
                                                               mem_agg_hidden_dimension = config['mem_agg_hidden_dim'],
                                                               prediction_dimension = config['prediction_dim'],
                                                               variable_classifier = Perceptron(config['hidden_dim'],
                                                                                                config[
                                                                                                    'classifier_dim'],
                                                                                                config[
                                                                                                    'prediction_dim']),
                                                               function_classifier = None, dropout = config['dropout'],
                                                               local_search_iterations = config[
                                                                   'local_search_iteration'],
                                                               epsilon = config['epsilon'])]

        elif config['model_type'] == 'np-d-np':
            model_list += [solver.NeuralSequentialDecimatorSolver(device = self.device, name = config['model_name'],
                                                                  edge_dimension = config['edge_feature_dim'],
                                                                  meta_data_dimension = config['meta_feature_dim'],
                                                                  propagator_dimension = config['hidden_dim'],
                                                                  decimator_dimension = config['hidden_dim'],
                                                                  mem_hidden_dimension = config['mem_hidden_dim'],
                                                                  agg_hidden_dimension = config['agg_hidden_dim'],
                                                                  mem_agg_hidden_dimension = config[
                                                                      'mem_agg_hidden_dim'],
                                                                  classifier_dimension = config['classifier_dim'],
                                                                  dropout = config['dropout'],
                                                                  tolerance = config['tolerance'],
                                                                  t_max = config['t_max'],
                                                                  local_search_iterations = config[
                                                                      'local_search_iteration'],
                                                                  epsilon = config['epsilon'])]

        elif config['model_type'] == 'p-d-p':
            model_list += [solver.SurveyPropagatorSolver(device = self.device, name = config['model_name'],
                                                         tolerance = config['tolerance'], t_max = config['t_max'],
                                                         local_search_iterations = config['local_search_iteration'],
                                                         epsilon = config['epsilon'])]

        elif config['model_type'] == 'walk-sat':
            model_list += [solver.WalkSATSolver(device = self.device, name = config['model_name'],
                                                iteration_num = config['local_search_iteration'],
                                                epsilon = config['epsilon'])]

        elif config['model_type'] == 'reinforce':
            model_list += [solver.ReinforceSurveyPropagatorSolver(device = self.device, name = config['model_name'],
                                                                  pi = config['pi'], decimation_probability = config[
                    'decimation_probability'], local_search_iterations = config['local_search_iteration'],
                                                                  epsilon = config['epsilon'])]

        if config['verbose']:
            self.logger.write_log("The model parameter count is %d." % model_list[0].parameter_count())
            self.logger.write_log("The model list is %s." % model_list)

        return model_list

    def save_model(self, model, optimizer, ckpt_path):  # TODO(jesse): implement a CheckpointManager
        torch.save({
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "save_counter": self.save_counter, "GLOBAL_STEP_COUNT": self.GLOBAL_STEP_COUNT
        }, ckpt_path)

    def load_model(self, model, optimizer, ckpt_path):
        """
        Loads `model` and `optimizer` state from checkpoint at `ckpt_path`

        Args:
          model: nn.Module object
          optimizer: PyTorch optimizer object
          ckpt_path: path to checkpoint containing serialized model state and optimizer state

        Returns:
          Nothing.
        """
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.save_counter = ckpt["save_counter"]
        self.GLOBAL_STEP_COUNT = ckpt["GLOBAL_STEP_COUNT"]
        print(f"LOADED MODEL, SAVE COUNTER: {self.save_counter}, GLOBAL_STEP_COUNT = {self.GLOBAL_STEP_COUNT}")

    def get_latest_from_index(self, ckpt_dir):
        """
        Args:
          ckpt_dir: checkpoint directory

        Returns:
          a dict cfg_dict such that cfg_dict["latest"] is the path to the latest checkpoint
        """
        index = files_with_extension(ckpt_dir, "index")[0]
        with open(index, "r") as f:
            cfg_dict = json.load(f)
        return cfg_dict["latest"]

    def update_index(self, ckpt_path):
        """
        Dump a JSON to a `.index` file, pointing to the most recent checkpoint.
        """
        ckpt_dir = os.path.dirname(ckpt_path)
        index_files = files_with_extension(ckpt_dir, "index")
        if len(index_files) == 0:
            index = os.path.join(ckpt_dir, "latest.index")
        else:
            assert len(index_files) == 1
            index = index_files[0]
        with open(index, "w") as f:
            cfg_dict = {"latest": ckpt_path}
            f.write(json.dumps(cfg_dict, indent = 2))

    def maybe_save_ckpt(self, GLOBAL_STEP_COUNT, type = "best", force_save = False):
        """
        Saves the model if GLOBAL_STEP_COUNT is at the ckpt_freq.

        Returns a bit indicating whether or not the model was saved.
        """
        if (int(GLOBAL_STEP_COUNT) % self.ckpt_freq == 0) or force_save:
            ckpt_path = os.path.join(self.ckpt_dir, f"ckpt_{type}.pth")
            self.save_model(self.model, self.optimizer, ckpt_path)
            self.update_index(ckpt_path)
            self.logger.write_log(f"[TRAIN LOOP]{GLOBAL_STEP_COUNT} Wrote checkpoint to {ckpt_path}.")
            self.save_counter += 1
            return True
        return False

    def load_latest_ckpt(self):
        ckpt_path = self.get_latest_from_index(self.ckpt_dir)
        self.load_model(self.model, self.optimizer, ckpt_path)
        self.logger.write_log(f"[TRAIN LOOP] Loaded weights from {ckpt_path}.")

    def transform_data(self, nmsdps):
        # graph_map, batch_variable_map, batch_function_map,edge_feature, graph_feat, label
        result = []
        for nmsdp in nmsdps:
            data = NMSDP_to_line(nmsdp)
            result.append(data)
        vn, fn, gm, ef, gf, l, md = zip(*result)
        variable_num, function_num, graph_map, edge_feature, graph_feat, label, misc_data = self.batch_divider.divide(
            vn, fn, gm, ef, gf, l, md)
        segment_num = len(variable_num)

        graph_feat_batch = []
        graph_map_batch = []
        batch_variable_map_batch = []
        batch_function_map_batch = []
        edge_feature_batch = []
        label_batch = []

        for i in range(segment_num):

            # Create the graph features batch
            graph_feat_batch += [
                None if graph_feat[i][0] is None else torch.from_numpy(np.stack(graph_feat[i])).float()]

            # Create the edge feature batch
            edge_feature_batch += [torch.from_numpy(np.expand_dims(np.concatenate(edge_feature[i]), 1)).float()]

            # Create the label batch
            label_batch += [torch.from_numpy(np.expand_dims(np.array(label[i]), 1)).float()]

            # Create the graph map, variable map and function map batches
            g_map_b = np.zeros((2, 0), dtype = np.int32)
            v_map_b = np.zeros(0, dtype = np.int32)
            f_map_b = np.zeros(0, dtype = np.int32)
            variable_ind = 0
            function_ind = 0

            for j in range(len(graph_map[i])):
                graph_map[i][j][0, :] += variable_ind
                graph_map[i][j][1, :] += function_ind
                g_map_b = np.concatenate((g_map_b, graph_map[i][j]), axis = 1)

                v_map_b = np.concatenate((v_map_b, np.tile(j, variable_num[i][j])))
                f_map_b = np.concatenate((f_map_b, np.tile(j, function_num[i][j])))

                variable_ind += variable_num[i][j]
                function_ind += function_num[i][j]

            graph_map_batch += [torch.from_numpy(g_map_b).int()]
            batch_variable_map_batch += [torch.from_numpy(v_map_b).int()]
            batch_function_map_batch += [torch.from_numpy(f_map_b).int()]

        yield graph_map_batch, batch_variable_map_batch, batch_function_map_batch, edge_feature_batch, graph_feat_batch, label_batch, misc_data

    def _compute_loss(self, model, loss, prediction, label, graph_map, batch_variable_map, batch_function_map,
            edge_feature, meta_data):
        "Computes the loss function."

        return loss(variable_prediction = prediction[0], label = label, graph_map = graph_map,
                    batch_variable_map = batch_variable_map, batch_function_map = batch_function_map,
                    edge_feature = edge_feature, meta_data = meta_data, global_step = model._global_step,
                    eps = 1e-8 * torch.ones(1, device = self.device), max_coeff = 10.0,
                    loss_sharpness = self.config['loss_sharpness'])

    def get_parameter_list(self):
        "Returns list of dictionaries with models' parameters."
        return [{'params': filter(lambda p: p.requires_grad, model.parameters())} for model in self.model_list]

    def train_sp(self, *train_data):
        total_example_num = 0
        total_loss = np.zeros(len(self.model_list), dtype = np.float32)
        print(self.get_parameter_list())
        optimizer = optim.Adam(self.get_parameter_list(), lr = self.config['learning_rate'],
                               weight_decay = self.config['weight_decay'])
        prediction = None
        for (j, data) in enumerate(train_data):
            segment_num = len(data[0])
            for i in range(segment_num):
                (graph_map, batch_variable_map, batch_function_map, edge_feature, graph_feat, label, _) = [d[i] for d in
                    data]
                total_example_num += (batch_variable_map.max() + 1)
                prediction = self._train_sp_batch(total_loss, optimizer, graph_map, batch_variable_map,
                                                  batch_function_map, edge_feature, graph_feat, label)

                del graph_map
                del batch_variable_map
                del batch_function_map
                del edge_feature
                del graph_feat
                del label

            for model in self.model_list:
                self._module(model)._global_step += 1

        return prediction

    def _train_sp_batch(self, total_loss, optimizer, graph_map, batch_variable_map, batch_function_map, edge_feature,
            graph_feat, label):

        optimizer.zero_grad()
        lambda_value = torch.tensor([self.config['lambda']], dtype = torch.float32, device = self.device)
        prediction = None

        for (i, model) in enumerate(self.model_list):

            state = self._module(model).get_init_state(graph_map, batch_variable_map, batch_function_map, edge_feature,
                                                       graph_feat, self.config['randomized'])

            loss = torch.zeros(1, device = self.device)

            for t in torch.arange(self.config['train_outer_recurrence_num'], dtype = torch.int32, device = self.device):

                prediction, state = model(init_state = state, graph_map = graph_map,
                                          batch_variable_map = batch_variable_map,
                                          batch_function_map = batch_function_map, edge_feature = edge_feature,
                                          meta_data = graph_feat, is_training = True,
                                          iteration_num = self.config['train_inner_recurrence_num'])

                loss += self._compute_loss(model = self._module(model), loss = self._loss_evaluator,
                                           prediction = prediction, label = label, graph_map = graph_map,
                                           batch_variable_map = batch_variable_map,
                                           batch_function_map = batch_function_map, edge_feature = edge_feature,
                                           meta_data = graph_feat) * lambda_value.pow(
                    (self.config['train_outer_recurrence_num'] - t - 1).float())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.config['clip_norm'])
            total_loss[i] += loss.detach().cpu().numpy()

            for s in state:
                del s

        optimizer.step()
        return prediction

    def train(self):
        self.logger.write_log(f"[TRAIN LOOP] HYPERPARAMETERS: LR {self.lr}")
        self.logger.write_log(f"[TRAIN LOOP] NUM_EPOCHS: {self.n_epochs}")
        check_make_path(self.ckpt_dir)
        self.logger.write_log("[TRAIN LOOP] Starting training.")
        batcher = Batcher(device = self.device)
        saved = False
        for epoch_count in range(self.n_epochs):
            self.logger.write_log(f"[TRAIN LOOP] STARTING EPOCH {epoch_count}")
            min_loss = np.zeros(1)
            for nmsdps in self.dataset:
                prediction = torch.ones(1)
                for data in self.transform_data(nmsdps):
                    prediction = self.train_sp(data)

                drat_loss, core_clause_loss, loss, grad_norm, l2_loss = train_step(self.model, batcher, self.optimizer,
                                                                                   prediction, nmsdps,
                                                                                   device = self.device,
                                                                                   CUDA_FLAG = self.CUDA_FLAG,
                                                                                   use_NMSDP_to_sparse2 = True,
                                                                                   use_glue_counts = False)

                if epoch_count % 10 == 0 and self.GLOBAL_STEP_COUNT % 10 == 0:
                    self.logger.write_scalar("drat_loss", drat_loss, self.GLOBAL_STEP_COUNT)
                    # self.logger.write_scalar("core_loss", core_loss, self.GLOBAL_STEP_COUNT)
                    self.logger.write_scalar("core_clause_loss", core_clause_loss, self.GLOBAL_STEP_COUNT)
                    self.logger.write_scalar("total_loss", loss, self.GLOBAL_STEP_COUNT)
                    self.logger.write_scalar("LAST GRAD NORM", grad_norm, self.GLOBAL_STEP_COUNT)
                    self.logger.write_scalar("l2_loss", l2_loss, self.GLOBAL_STEP_COUNT)
                    self.logger.write_log(f"[TRAIN LOOP] Finished global step {self.GLOBAL_STEP_COUNT}."
                                          f" Loss: {loss}.")
                self.GLOBAL_STEP_COUNT += 1
                if min_loss == 0 or min_loss > loss.detach().cpu().numpy():
                    saved = self.maybe_save_ckpt(self.GLOBAL_STEP_COUNT)
                    min_loss = loss.detach().cpu().numpy()
                if self.n_steps is not None:
                    if self.GLOBAL_STEP_COUNT >= self.n_steps:
                        break
        if not saved:
            self.maybe_save_ckpt(self.GLOBAL_STEP_COUNT, "last",
                                 force_save = True)  # save at the end of every epoch regardless


# def gen_nmsdp_batch(k):
#     nmsdps = []
#     D = RandKCNFDataset(3, 40)
#     D_gen = D.__iter__()
#     for _ in range(k):
#         cnf = next(D_gen)
#         with tempfile.TemporaryDirectory() as tmpdir:
#             nmsdp = gen_nmsdp(tmpdir, cnf, is_train = True, logger = DummyLogger())
#         nmsdps.append(nmsdp)
#     return nmsdps


def _parse_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type = str, dest = "cfg", action = "store")
    parser.add_argument("--sp-cfg", type = str, dest = "sp_cfg", action = "store")
    parser.add_argument("--lr", type = float, dest = "lr", action = "store")
    parser.add_argument("--data-dir", type = str, dest = "data_dir", action = "store")
    parser.add_argument("--batch-size", type = int, dest = "batch_size", action = "store")
    parser.add_argument("--n-data-workers", type = int, dest = "n_data_workers", action = "store")
    parser.add_argument("--ckpt-dir", type = str, dest = "ckpt_dir", action = "store")
    parser.add_argument("--ckpt-freq", type = int, dest = "ckpt_freq", action = "store", default = 30)
    parser.add_argument("--n-steps", type = int, dest = "n_steps", action = "store", default = -1)
    parser.add_argument("--n-epochs", type = int, dest = "n_epochs", action = "store", default = 100)
    parser.add_argument("--forever", action = "store_true")
    parser.add_argument("--index", action = "store", default = 0, type = int)
    opts = parser.parse_args()
    opts.ckpt_dir = os.path.join(opts.ckpt_dir, time.strftime("%Y%m%d-%H%M", time.localtime()))

    return opts


def _main_train1(cfg = None, opts = None):
    if opts is None:
        opts = _parse_main()
        opts.n_data_workers = opts.n_data_workers if opts.n_data_workers else multiprocessing.cpu_count()

    if cfg is None:
        # cfg = defaultGNN1Cfg
        with open(opts.cfg, "r") as f:
            cfg = json.load(f)
        with open(opts.sp_cfg, 'r') as f:
            sp_cfg = yaml.load(f)

    model = GNN1(**cfg)

    dataset = mk_H5DataLoader(opts.data_dir, opts.batch_size, opts.n_data_workers, 'nmsdp')
    trainer = Trainer(model, dataset, sp_cfg, opts.lr, root_dir = opts.ckpt_dir, ckpt_freq = opts.ckpt_freq,
                      restore = False, n_steps = opts.n_steps, n_epochs = opts.n_epochs, index = opts.index)

    if opts.forever is True:
        while True:
            trainer.train()
    else:
        trainer.train()


def _test_trainer(opts = None):
    if opts is None:
        opts = _parse_main()
    model = GNN1(**defaultGNN1Cfg)
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    dataset = mk_H5DataLoader("./train_data/", batch_size = 16, num_workers = 2)
    trainer = Trainer(model, dataset, optimizer, ckpt_dir = "./test_weights/", ckpt_freq = 10, restore = True,
                      n_steps = opts.n_steps)
    for _ in range(5):
        trainer.train()  # trainer.load_latest_ckpt()


GNN1Cfg0 = {
    "clause_dim": 64, "lit_dim": 16, "n_hops": 1, "n_layers_C_update": 0, "n_layers_L_update": 0, "n_layers_score": 1,
    "activation": "leaky_relu"
}

if __name__ == "__main__":
    _main_train1()
