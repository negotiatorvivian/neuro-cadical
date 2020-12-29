import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import itertools
import tempfile

from python.sp import trainer
from python.sp.factorgraph import dataset, base
from python.sp.nn import solver
from python.sp.nn.util import SatCNFEvaluator, SatLossEvaluator
from python.gen_data import data_to_cnf
from python.train1 import compute_softmax_kldiv_loss_from_logits, compute_mask_loss
from python.gen_data import lemma_occ
from python.util import get_granularity


def _module(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def decode_activation(activation):
    if activation == "relu":
        return nn.ReLU
    elif activation == "leaky_relu":
        return nn.LeakyReLU
    elif activation == "relu6":
        return nn.ReLU6
    elif activation == "elu":
        return nn.ELU
    else:
        raise Exception("unsupported activation")


class BasicMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation, bias_at_end = True, p_dropout = 0.1, **kwargs):
        super(BasicMLP, self).__init__(**kwargs)
        layers = []
        for k in range(len(hidden_dims) + 1):
            if k == 0:
                d_in = input_dim
            else:
                d_in = hidden_dims[k - 1]

            if k == len(hidden_dims):
                d_out = output_dim
            else:
                d_out = hidden_dims[k]

            layers.append(nn.Linear(in_features = d_in, out_features = d_out, bias = (
                True if ((k == len(hidden_dims) and bias_at_end) or k < len(hidden_dims)) else False)))

            if not (k == len(hidden_dims)):
                layers.append(decode_activation(activation)())
                layers.append(nn.Dropout(p_dropout))

        self.main = nn.Sequential(*layers)

    def forward(self, z):
        return self.main(z)


defaultGNN1Cfg = {
    "clause_dim": 64, "lit_dim": 16, "n_hops": 2, "n_layers_C_update": 0, "n_layers_L_update": 0, "n_layers_score": 1,
    "activation": "relu"
}


def swap_polarity(G):
    indices = G.coalesce().indices()
    values = G.coalesce().values()
    size = G.size()

    pivot = size[1] / 2 - 0.5

    indices[1] = (2.0 * pivot) - indices[1]

    return torch.sparse.FloatTensor(indices = indices, values = values, size = size)


def flop(L_logits):
    return torch.cat([L_logits[0:int(L_logits.size()[0] / 2)], L_logits[int(L_logits.size()[0] / 2):]], dim = 1)


def flip(L_logits):
    return torch.cat([L_logits[int(L_logits.size()[0] / 2):], L_logits[0:int(L_logits.size()[0] / 2)]], dim = 0)


class GNN1(nn.Module):
    def __init__(self, clause_dim, lit_dim, n_hops, n_layers_C_update, n_layers_L_update, n_layers_score, activation,
                 average_pool = False, normalize = True, **kwargs):
        super(GNN1, self).__init__(**kwargs)
        self.L_layer_norm = nn.LayerNorm(lit_dim)  # LayerNorm(16, )
        self.L_init = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty([1, lit_dim])),
                                   requires_grad = True)  # (1,16)
        self.C_update = BasicMLP(input_dim = 2 * lit_dim, hidden_dims = [2 * lit_dim for _ in range(n_layers_C_update)],
                                 output_dim = clause_dim, activation = activation, p_dropout = 0.05)
        self.L_update = BasicMLP(input_dim = (clause_dim), hidden_dims = [clause_dim for _ in range(n_layers_L_update)],
                                 output_dim = lit_dim, activation = activation, p_dropout = 0.05)
        self.V_score_drat = BasicMLP(input_dim = 2 * lit_dim,
                                     hidden_dims = [2 * lit_dim for _ in range(n_layers_score)], output_dim = 1,
                                     activation = activation, bias_at_end = True, p_dropout = 0.15)
        self.V_score_core = BasicMLP(input_dim = 2 * lit_dim,
                                     hidden_dims = [2 * lit_dim for _ in range(n_layers_score)], output_dim = 1,
                                     activation = activation, p_dropout = 0.05)
        self.C_score_core = BasicMLP(input_dim = clause_dim, hidden_dims = [clause_dim for _ in range(n_layers_score)],
                                     output_dim = 1, activation = activation, p_dropout = 0.05)

        self.n_hops = n_hops
        self.lit_dim = lit_dim
        self.clause_dim = clause_dim
        self.average_pool = average_pool
        self.normalize = normalize
        if not self.normalize:
            self.C_layer_norm = nn.LayerNorm(clause_dim)

    def forward(self, G):
        n_clauses, n_lits = G.size()
        L = self.L_init.repeat(n_lits, 1)  # (n_lits, 16)
        if not (G.device == L.device):
            L = L.to(G.device)

        for T in range(self.n_hops):
            L_flip = torch.cat([L[int(L.size()[0] / 2):], L[0:int(L.size()[0] / 2)]], dim = 0)  # (n_lits, 16)
            if self.average_pool:
                C_pre_msg = torch.cat([L, L_flip, torch.ones(G.size()[1], 1, dtype = torch.float32, device = G.device)],
                                      dim = 1)  # (n_lits, 33)
            else:
                C_pre_msg = torch.cat([L, L_flip], dim = 1)  # (n_lits, 32)
            C_msg = torch.sparse.mm(G, C_pre_msg)  # (n_clauses, 32)

            if self.average_pool:
                C_neighbor_counts = C_msg[:, -1:]
                C_msg = C_msg[:, :-1]
                C_msg = C_msg / torch.max(C_neighbor_counts,
                                          torch.ones(C_neighbor_counts.size()[0], C_neighbor_counts.size()[1],
                                                     device = G.device))

            C = self.C_update(C_msg)  # (n_clauses, 64)
            if self.normalize:
                C = C - C.mean(dim = 0)
                C = C / (C.std(dim = 0) + 1e-10)
            else:
                C = self.C_layer_norm(C)
            if self.average_pool:
                L_pre_msg = torch.cat([C, torch.ones(G.size()[0], 1, dtype = torch.float32, device = G.device)],
                                      dim = 1)
            else:
                L_pre_msg = C
            L_msg = torch.sparse.mm(G.t(), L_pre_msg)
            if self.average_pool:
                L_neighbor_counts = L_msg[:, -1:]
                L_msg = L_msg[:, :-1]
                L_msg = L_msg / torch.max(L_neighbor_counts,
                                          torch.ones(L_neighbor_counts.size()[0], L_neighbor_counts.size()[1],
                                                     device = G.device))
            L = self.L_update(L_msg) + (0.1 * L)
            L = self.L_layer_norm(L)
        V = torch.cat([L[0:int(L.size()[0] / 2)], L[int(L.size()[0] / 2):]], dim = 1)
        return self.V_score_drat(V), self.V_score_core(V), self.C_score_core(C)


class GNN1_drat(nn.Module):  # deploy the drat head only
    def __init__(self, clause_dim, lit_dim, n_hops, n_layers_C_update, n_layers_L_update, n_layers_score, activation,
                 average_pool = False, normalize = True, **kwargs):
        super(GNN1_drat, self).__init__(**kwargs)
        self.L_layer_norm = nn.LayerNorm(lit_dim)
        self.L_init = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty([1, lit_dim])), requires_grad = True)

        self.C_update = BasicMLP(input_dim = 2 * lit_dim, hidden_dims = [2 * lit_dim for _ in range(n_layers_C_update)],
                                 output_dim = clause_dim, activation = activation)
        self.L_update = BasicMLP(input_dim = (clause_dim), hidden_dims = [clause_dim for _ in range(n_layers_L_update)],
                                 output_dim = lit_dim, activation = activation)

        self.V_score_drat = BasicMLP(input_dim = 2 * lit_dim,
                                     hidden_dims = [2 * lit_dim for _ in range(n_layers_score)], output_dim = 1,
                                     activation = activation, bias_at_end = True)

        self.n_hops = n_hops
        self.lit_dim = lit_dim
        self.clause_dim = clause_dim
        self.average_pool = average_pool
        self.normalize = normalize

    def forward(self, G):
        n_clauses, n_lits = G.shape
        n_vars = n_lits / 2
        L = self.L_init.repeat(n_lits, 1)
        if not (G.device == L.device):
            L = L.to(G.device)

        for T in range(self.n_hops):
            L_flip = torch.cat([L[int(L.size()[0] / 2):], L[0:int(L.size()[0] / 2)]], dim = 0)
            if self.average_pool:
                C_pre_msg = torch.cat([L, L_flip, torch.ones(G.size()[1], 1, dtype = torch.float32, device = G.device)],
                                      dim = 1)
            else:
                C_pre_msg = torch.cat([L, L_flip], dim = 1)
            C_msg = torch.sparse.mm(G, C_pre_msg)

            if self.average_pool:
                C_neighbor_counts = C_msg[:, -1:]
                C_msg = C_msg[:, :-1]
                C_msg = C_msg / torch.max(C_neighbor_counts,
                                          torch.ones(C_neighbor_counts.size()[0], C_neighbor_counts.size()[1],
                                                     device = G.device))

            C = self.C_update(C_msg)
            if self.normalize:
                C = C - C.mean(dim = 0)
                C = C / (C.std(dim = 0) + 1e-10)
            if self.average_pool:
                L_pre_msg = torch.cat([C, torch.ones(G.size()[0], 1, dtype = torch.float32, device = G.device)],
                                      dim = 1)
            else:
                L_pre_msg = C
            L_msg = torch.sparse.mm(G.t(), L_pre_msg)
            if self.average_pool:
                L_neighbor_counts = L_msg[:, -1:]
                L_msg = L_msg[:, :-1]
                L_msg = L_msg / torch.max(L_neighbor_counts,
                                          torch.ones(L_neighbor_counts.size()[0], L_neighbor_counts.size()[1],
                                                     device = G.device))
            L = self.L_update(L_msg) + (0.1 * L)
            L = self.L_layer_norm(L)
        V = torch.cat([L[0:int(L.size()[0] / 2)], L[int(L.size()[0] / 2):]], dim = 1)
        return self.V_score_drat(V)


class Base(base.FactorGraphTrainerBase):
    def __init__(self, model_cfg, config, device, batcher, batch_replication = 1, average_pool = False,
                 normalize = True, **kwargs):
        super(Base, self).__init__(config = config, has_meta_data = False, error_dim = config['error_dim'], loss = None,
                                   evaluator = nn.L1Loss(), use_cuda = not kwargs['cpu'], logger = kwargs['logger'])
        self.gnn = rl_GNN1(**model_cfg)
        self.config = config
        self._device = device
        self._eps = 1e-8 * torch.ones(1, device = self._device)
        self._loss_evaluator = SatLossEvaluator(alpha = self.config['exploration'], device = self._device)
        self._cnf_evaluator = SatCNFEvaluator(device = self._device)
        self._counter = 0
        self._max_coeff = 10.0
        self.batch_divider = dataset.DynamicBatchDivider(self.config['train_batch_limit'] // batch_replication,
                                                         self.config['hidden_dim'])
        self.model_list.append(self.gnn)
        self.parameters = self.get_parameter_list()
        self.batcher = batcher
        self.logger = kwargs['logger']
        # self.temp_dir = tempfile.TemporaryDirectory()

    def transform_data(self, data):
        # graph_map, batch_variable_map, batch_function_map,edge_feature, graph_feat, label
        vn, fn, gm, ef, gf, l, md = zip(*data)
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

    def load_state_dict(self, ckpt, strict = False):
        for i, model in enumerate(self.model_list):
            _module(model).load_state_dict(ckpt[i], strict)

    def train(self, G, actions, *train_data):
        total_example_num = 0
        total_loss = np.zeros(len(self.model_list), dtype = np.float32)
        optimizer = optim.Adam(self.get_parameter_list(), lr = self.config['learning_rate'],
                               weight_decay = self.config['weight_decay'])
        for (j, data) in enumerate(train_data):
            segment_num = len(data[0])
            for i in range(segment_num):
                (graph_map, batch_variable_map, batch_function_map, edge_feature, graph_feat, label, _) = [d[i] for d in
                                                                                                           data]
                total_example_num += (batch_variable_map.max() + 1)
                self.train_batch(total_loss, optimizer, graph_map,
                                 batch_variable_map, batch_function_map,
                                 edge_feature, graph_feat, label, G, actions)

                del graph_map
                del batch_variable_map
                del batch_function_map
                del edge_feature
                del graph_feat
                del label

            for model in self.model_list:
                base._module(model)._global_step += 1

    def train_batch(self, total_loss, optimizer, graph_map, batch_variable_map, batch_function_map, edge_feature,
                    graph_feat, label, G, actions):

        optimizer.zero_grad()
        lambda_value = torch.tensor([self.config['lambda']], dtype = torch.float32, device = self._device)
        prediction = None
        for (i, model) in enumerate(self.model_list[:-1]):

            state = base._module(model).get_init_state(graph_map, batch_variable_map, batch_function_map, edge_feature,
                                                       graph_feat, self.config['randomized'])

            loss = torch.zeros(1, device = self._device)
            if G is not None:
                # print(f'graph_map: {graph_map}, {graph_map.shape}, {G.shape}')
                mask = torch.sparse.FloatTensor(graph_map.long(), edge_feature.squeeze(1),
                                                torch.Size([int(G.shape[1] / 2), G.shape[0]]), device = self._device).unsqueeze(1).to_dense()
                var_lemma_counts = lemma_occ(mask)
                if np.all(label.cpu().numpy()) and graph_feat.shape[1] > 0:
                    core_var_masks = graph_feat.view([1, -1]).squeeze()
                    core_clause_masks = torch.ones(G.shape[0])
                else:
                    core_var_masks = torch.from_numpy(np.ones(int(G.shape[1] / 2), dtype = np.int32))
                    edge_num = graph_map.size()[1]
                    edge_num_range = torch.arange(edge_num, dtype = torch.int64, device = self._device)
                    variable_sparse_ind = torch.stack([graph_map[0, :].long(), edge_num_range])
                    variable_mask = torch.sparse.FloatTensor(variable_sparse_ind, edge_feature.squeeze(1),
                                                             torch.Size([int(G.shape[1] / 2), edge_num]), device = self._device).t()
                    edge_function_mask = torch.sparse.mm(variable_mask, mask.squeeze())
                    unsigned_edge_features = torch.abs(edge_feature).t()
                    input_num = torch.sparse.mm(unsigned_edge_features, edge_function_mask).squeeze()
                    function_eval = torch.sparse.mm(edge_feature.t(), edge_function_mask).squeeze()
                    core_clause_masks = (function_eval > -input_num).float()

                for t in torch.arange(self.config['train_outer_recurrence_num'], dtype = torch.int32,
                                      device = self._device):
                    prediction, state = model(init_state = state, graph_map = graph_map,
                                              batch_variable_map = batch_variable_map,
                                              batch_function_map = batch_function_map, edge_feature = edge_feature,
                                              meta_data = None, is_training = True,
                                              iteration_num = self.config['train_inner_recurrence_num'])

                    loss += self._compute_loss(model = base._module(model), loss = self._loss_evaluator,
                                               prediction = prediction, label = label, graph_map = graph_map,
                                               batch_variable_map = batch_variable_map,
                                               batch_function_map = batch_function_map, edge_feature = edge_feature,
                                               meta_data = None) * lambda_value.pow(
                        (self.config['train_outer_recurrence_num'] - t - 1).float())
                    batched_V_drat_logits, batched_v_pre_logits, batched_V_core_logits, batched_C_core_logits = self.gnn(G)
                    # (lambda x: (
                    #     x[0].view([x[0].size(0)]), x[1].view([x[1].size(0)]), x[2].view([x[2].size(0)], x[3].view([x[3].size(0)]))))(self.gnn(G))
                    # print(f'core_var_masks: {core_var_masks.shape}, {core_clause_masks}')
                    loss += self.model_list[-1]._compute_loss(self.batcher, batched_V_drat_logits, batched_V_core_logits,
                                                              batched_C_core_logits,
                                                              torch.from_numpy(np.array([var_lemma_counts])).type(
                                                                  torch.float32).squeeze(), core_var_masks,
                                                              core_clause_masks)
            print('loss:', loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.config['clip_norm'])
            total_loss[i] += loss.detach().cpu().numpy()

            for s in state:
                del s

        optimizer.step()

    def predict(self, G, actions, *train_data):
        paths = []
        actionss = []
        with torch.no_grad():

            for (j, data) in enumerate(train_data, 1):
                segment_num = len(data[0])

                for i in range(segment_num):
                    (graph_map, batch_variable_map, batch_function_map,
                     edge_feature, graph_feat, label, misc_data) = [self._to_cuda(d[i]) for d in data]
                    path, new_actions = self.predict_batch(graph_map, batch_variable_map, batch_function_map,
                                                           edge_feature, graph_feat, label, misc_data, actions, G, None, batch_replication = 1)
                    paths.append(path)
                    actionss.append(new_actions)
                    del graph_map
                    del batch_variable_map
                    del batch_function_map
                    del edge_feature
                    del graph_feat
                    del label

        return paths, actionss

    def predict_batch(self, graph_map, batch_variable_map, batch_function_map,
                      edge_feature, graph_feat, label, misc_data, actions, G, post_processor, batch_replication = 1):
        print(f'label: {label.cpu()}')
        new_actions = []

        for (j, model) in enumerate(self.model_list[:-1]):

            state = _module(model).get_init_state(graph_map, batch_variable_map, batch_function_map,
                                                  edge_feature, graph_feat, randomized = False, batch_replication = batch_replication)

            prediction, _ = model(
                init_state = state, graph_map = graph_map, batch_variable_map = batch_variable_map,
                batch_function_map = batch_function_map, edge_feature = edge_feature,
                meta_data = graph_feat, is_training = False, iteration_num = self.config['test_recurrence_num'],
                check_termination = self._check_recurrence_termination, batch_replication = batch_replication)

            if post_processor is not None and callable(post_processor):
                message = post_processor(_module(model), prediction, graph_map,
                                         batch_variable_map, batch_function_map, edge_feature, graph_feat, label, misc_data)
                print(message)

            res = self._cnf_evaluator(variable_prediction = prediction[0], graph_map = graph_map,
                                      batch_variable_map = batch_variable_map, batch_function_map = batch_function_map,
                                      edge_feature = edge_feature, meta_data = graph_feat)
            output, unsat_clause_num = [a.detach().cpu().numpy() for a in res]
            if unsat_clause_num == 0:
                print('-------True-------')

            actions = np.array(actions).squeeze()
            self.logger.write_log(f'all actions: {list(actions)}; {len(list(actions))}')
            temp_dir = tempfile.TemporaryDirectory()
            if np.all(label.cpu().numpy()):
                if isinstance(actions[0], int) or isinstance(actions[0], np.int64):
                    self._update_solution(prediction, model.sat_problem, actions, temp_dir)
                    new_actions.append(list(actions))
                    for i in range(int(len(actions)/5)):
                        indices = np.random.choice(range(len(actions)), 3)
                        self._update_solution(prediction, model.sat_problem, actions[indices], temp_dir)
                        new_actions.append(list(actions[indices]))
                else:
                    for i in range(len(actions)):
                        self._update_solution(prediction, model.sat_problem, actions[i], temp_dir)
                        new_actions.append(list(actions[i]))
                        for k in range(len(actions[i])):
                            indices = np.random.choice(range(len(actions)), 2)
                            self._update_solution(prediction, model.sat_problem, actions[i][indices], temp_dir)
                            new_actions.append(list(actions[i][indices]))

            else:
                _, _, _, clause_logits = self.model_list[-1](G)
                new_actions = self._unsat_core(prediction, clause_logits, model.sat_problem, actions, temp_dir)

            for p in prediction:
                del p

            for s in state:
                del s
        return temp_dir, new_actions

    def _compute_loss(self, model, loss, prediction, label, graph_map, batch_variable_map, batch_function_map,
                      edge_feature, meta_data):

        return self._loss_evaluator(variable_prediction = prediction[0], label = label, graph_map = graph_map,
                                    batch_variable_map = batch_variable_map, batch_function_map = batch_function_map,
                                    edge_feature = edge_feature, meta_data = meta_data,
                                    global_step = model._global_step, eps = self._eps, max_coeff = self._max_coeff,
                                    loss_sharpness = self.config['loss_sharpness'])

    def _build_graph(self, config):
        model_list = []

        if config['model_type'] == 'np-nd-np':
            model_list += [solver.NeuralPropagatorDecimatorSolver(device = self._device, name = config['model_name'],
                                                                  edge_dimension = config['edge_feature_dim'],
                                                                  meta_data_dimension = config['meta_feature_dim'],
                                                                  propagator_dimension = config['hidden_dim'],
                                                                  decimator_dimension = config['hidden_dim'],
                                                                  mem_hidden_dimension = config['mem_hidden_dim'],
                                                                  agg_hidden_dimension = config['agg_hidden_dim'],
                                                                  mem_agg_hidden_dimension = config[
                                                                      'mem_agg_hidden_dim'],
                                                                  prediction_dimension = config['prediction_dim'],
                                                                  variable_classifier = trainer.Perceptron(
                                                                      config['hidden_dim'], config['classifier_dim'],
                                                                      config['prediction_dim']),
                                                                  function_classifier = None,
                                                                  dropout = config['dropout'],
                                                                  local_search_iterations = config[
                                                                      'local_search_iteration'],
                                                                  epsilon = config['epsilon'])]

        elif config['model_type'] == 'p-nd-np':
            model_list += [solver.NeuralSurveyPropagatorSolver(device = self._device, name = config['model_name'],
                                                               edge_dimension = config['edge_feature_dim'],
                                                               meta_data_dimension = config['meta_feature_dim'],
                                                               decimator_dimension = config['hidden_dim'],
                                                               mem_hidden_dimension = config['mem_hidden_dim'],
                                                               agg_hidden_dimension = config['agg_hidden_dim'],
                                                               mem_agg_hidden_dimension = config['mem_agg_hidden_dim'],
                                                               prediction_dimension = config['prediction_dim'],
                                                               variable_classifier = trainer.Perceptron(
                                                                   config['hidden_dim'], config['classifier_dim'],
                                                                   config['prediction_dim']),
                                                               function_classifier = None, dropout = config['dropout'],
                                                               local_search_iterations = config[
                                                                   'local_search_iteration'],
                                                               epsilon = config['epsilon'])]

        elif config['model_type'] == 'np-d-np':
            model_list += [solver.NeuralSequentialDecimatorSolver(device = self._device, name = config['model_name'],
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
            model_list += [solver.SurveyPropagatorSolver(device = self._device, name = config['model_name'],
                                                         tolerance = config['tolerance'], t_max = config['t_max'],
                                                         local_search_iterations = config['local_search_iteration'],
                                                         epsilon = config['epsilon'])]

        return model_list

    def _update_solution(self, prediction, sat_problem, actions, temp_dir):
        if prediction[0] is None:
            return
        solution = torch.zeros(sat_problem._solution.shape[0], device = self._device)
        active_variables = sat_problem._active_variables.clone()
        solution[actions - 1] = 2 * prediction[0].squeeze()[actions - 1] - 1
        flag = True
        _, vf_map_transpose, _, signed_vf_map_transpose = sat_problem._vf_mask_tuple
        # while True:
        #     assignment = solution * active_variables
        #     input_num = torch.mm(vf_map_transpose, assignment.abs())
        #     function_eval = torch.mm(signed_vf_map_transpose, assignment)
        #     # Compute the de-activated functions -> deactivated_functions表示为真的子句
        #     deactivated_functions = (function_eval > -input_num).float() * active_functions
        #     print(f'deactivated_functions: {deactivated_functions.shape}')
        #     indices = np.argwhere(deactivated_functions[:, 0] == 1)
        #     active_functions[indices, 0] = 0
        #     size = indices.shape[1]
        #     if size == 0 and flag:
        #         print(f'removed clauses: {size}, change action: {solution[actions - 1]} to {-solution[actions - 1]}')
        #         solution[actions - 1] = -solution[actions - 1]
        #         flag = not flag
        #     else:
        #         break
        indices = []
        for action in actions:
            temp = np.argwhere(
                torch.mm(signed_vf_map_transpose.to_dense()[:, [action - 1]], torch.tensor([solution[action - 1]]).unsqueeze(1)) > 0)

            indices += list(temp.numpy()[0])
        indices = list(set(indices))
        deactivated_functions = torch.ones(signed_vf_map_transpose.shape[0])
        # deactivated_functions = (
        #             torch.mm(signed_vf_map_transpose.to_dense()[:, [actions - 1]].squeeze(), solution[actions - 1].unsqueeze(1)) > 0).float()
        active_variables[actions - 1] = 0
        deactivated_functions[indices] = 0
        # deactivated_sat = torch.index_select(signed_vf_map_transpose.to_dense(), 0,
        #                                      np.argwhere(deactivated_functions == 0).squeeze()[0])
        deactivated_sat = torch.index_select(signed_vf_map_transpose.to_dense(), 0,
                                             np.argwhere(deactivated_functions != 0).squeeze())
        # deactivated_sat[:, actions - 1] = 0
        deactivated_sat = torch.index_select(deactivated_sat, 1,
                                             np.argwhere(active_variables == 1).squeeze()[0])
        print(f'''action value: {actions}, {solution[actions - 1]}, removed clauses： {len(
            indices)}, cnf_size:{signed_vf_map_transpose.shape} to {deactivated_sat.shape}''')
        self.logger.write_log(f'''removed clauses： {len(
            indices)}; cnf_size:{signed_vf_map_transpose.shape} to {deactivated_sat.shape}''')
        data_to_cnf(deactivated_sat, temp_dir)

    def _unsat_core(self, prediction, clause_logits, sat_problem, actions, temp_dir):
        if prediction[0] is None:
            return

        non_negative_indices = np.argwhere(clause_logits.squeeze(1) > 0)
        _, vf_map_transpose, _, signed_vf_map_transpose = sat_problem._vf_mask_tuple
        '''remove variables & related clauses'''
        parts = []
        granularity = get_granularity(len(actions))
        for i in range(int(len(actions) / granularity)):
            part = np.array(list(set(np.random.choice(actions, int(len(actions) - i * granularity)))))
            parts.append(part)
        parts.append(actions)

        clauses_indices = []

        for part_actions in parts:
            deactivated_functions = np.argwhere((signed_vf_map_transpose.to_dense()[:, [part_actions - 1]] != 0).squeeze() == False)
            # print(f'deactivated_functions: {deactivated_functions}, {deactivated_functions.shape}')
            if len(deactivated_functions.shape) == 1:
                deactivated_functions = deactivated_functions.unsqueeze(1)
            # actions_signs = torch.ones(len(part_actions), 1)

            # for i in range(granularity):
            #     deactivated_indices = np.random.choice(range(shape), min(granularity - i, shape))
            #     deactivated_functions_temp = torch.mm(deactivated_functions, actions_signs)
            #     deactivated_sat = torch.index_select(signed_vf_map_transpose.to_dense(), 0,
            #                                          np.argwhere(deactivated_functions_temp[deactivated_indices] == 0).squeeze()[0])
            #     clauses_indices.append(deactivated_indices)
            #     data_to_cnf(deactivated_sat, temp_dir)

            # deactivated_indices = np.random.choice(range(shape), min(granularity - i, shape))
            # deactivated_functions_temp = torch.mm(deactivated_functions, actions_signs)
            deactivated_indices = np.union1d(non_negative_indices.squeeze(1), deactivated_functions[0])
            if len(deactivated_indices) == 0:
                continue
            print(f'''part_actions: {part_actions}, deactivated_indices: {len(deactivated_indices)},
             negative_indices: {non_negative_indices.shape}''')
            all_indices = np.ones(clause_logits.shape[0])
            # all_indices[deactivated_functions[0]] = 0
            all_indices[deactivated_indices] = 0
            deactivated_sat = torch.index_select(signed_vf_map_transpose.to_dense(), 0, torch.from_numpy(np.argwhere(all_indices > 0).squeeze(1)))
            unsat_core = torch.index_select(signed_vf_map_transpose.to_dense(), 0, torch.from_numpy(np.argwhere(all_indices == 0).squeeze(1)))

            clauses_indices.append(np.array(np.unique(deactivated_indices)) + 1)
            clauses_indices.append(np.array(np.unique(deactivated_indices)) + 1)

            data_to_cnf(deactivated_sat, temp_dir)
            data_to_cnf(unsat_core, temp_dir, True)

        return clauses_indices

    def validate(self, train_data, batch_replication = 1):
        predictions = []
        for (j, data) in enumerate(self.transform_data(train_data)):
            segment_num = len(data[0])
            for i in range(segment_num):
                (graph_map, batch_variable_map, batch_function_map, edge_feature, graph_feat, label, _) = [d[i] for d in data]
                prediction = self.validate_batch(graph_map, batch_variable_map, batch_function_map,
                                                 edge_feature, graph_feat, label, batch_replication)
                predictions.append(prediction)

                del graph_map
                del batch_variable_map
                del batch_function_map
                del edge_feature
                del graph_feat
                del label

        return predictions

    def validate_batch(self, graph_map, batch_variable_map, batch_function_map, edge_feature, graph_feat, label, batch_replication):
        prediction = None
        for (i, model) in enumerate(self.model_list[:-1]):
            state = base._module(model).get_init_state(graph_map, batch_variable_map, batch_function_map, edge_feature,
                                                       graph_feat, False, 1)
            prediction, _ = model(
                init_state = state, graph_map = graph_map, batch_variable_map = batch_variable_map,
                batch_function_map = batch_function_map, edge_feature = edge_feature,
                meta_data = graph_feat, is_training = False, iteration_num = self.config['test_recurrence_num'],
                check_termination = None, batch_replication = batch_replication)

        return prediction


class rl_GNN1(nn.Module):
    def __init__(self, clause_dim, lit_dim, n_hops, n_layers_C_update, n_layers_L_update, n_layers_score, activation,
                 average_pool = False, normalize = True, **kwargs):
        super(rl_GNN1, self).__init__(**kwargs)
        self.L_layer_norm = nn.LayerNorm(lit_dim)
        self.L_init = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty([1, lit_dim])), requires_grad = True)

        self.C_update = BasicMLP(input_dim = 2 * lit_dim, hidden_dims = [2 * lit_dim for _ in range(n_layers_C_update)],
                                 output_dim = clause_dim, activation = activation, p_dropout = 0.05)
        self.L_update = BasicMLP(input_dim = (clause_dim), hidden_dims = [clause_dim for _ in range(n_layers_L_update)],
                                 output_dim = lit_dim, activation = activation, p_dropout = 0.05)

        self.V_score = BasicMLP(input_dim = 2 * lit_dim, hidden_dims = [2 * lit_dim for _ in range(n_layers_score)],
                                output_dim = 1, activation = activation, bias_at_end = True, p_dropout = 0.15)
        self.V_vote = BasicMLP(input_dim = 2 * lit_dim, hidden_dims = [2 * lit_dim for _ in range(n_layers_score)],
                               output_dim = 1, activation = activation, bias_at_end = True, p_dropout = 0.15)
        self.C_score_core = BasicMLP(input_dim = clause_dim, hidden_dims = [clause_dim for _ in range(n_layers_score)],
                                     output_dim = 1, activation = activation, p_dropout = 0.05)
        self.V_score_core = BasicMLP(input_dim = 2 * lit_dim,
                                     hidden_dims = [2 * lit_dim for _ in range(n_layers_score)], output_dim = 1,
                                     activation = activation, p_dropout = 0.05)

        self.n_hops = n_hops
        self.lit_dim = lit_dim
        self.clause_dim = clause_dim
        self.average_pool = average_pool
        self.normalize = normalize
        self._global_step = nn.Parameter(torch.tensor([0], dtype = torch.float), requires_grad = False)
        self._name = 'rl_learner'
        if not self.normalize:
            self.C_layer_norm = nn.LayerNorm(clause_dim)

    def forward(self, G):
        n_clauses, n_lits = G.size()
        n_vars = n_lits / 2
        L = self.L_init.repeat(n_lits, 1)
        if not (G.device == L.device):
            L = L.to(G.device)

        for T in range(self.n_hops):
            L_flip = torch.cat([L[int(L.size()[0] / 2):], L[0:int(L.size()[0] / 2)]], dim = 0)
            if self.average_pool:
                C_pre_msg = torch.cat([L, L_flip, torch.ones(G.size()[1], 1, dtype = torch.float32, device = G.device)],
                                      dim = 1)
            else:
                C_pre_msg = torch.cat([L, L_flip], dim = 1)
            C_msg = torch.sparse.mm(G, C_pre_msg)

            if self.average_pool:
                C_neighbor_counts = C_msg[:, -1:]

                C_msg = C_msg[:, :-1]

                C_msg = C_msg / torch.max(C_neighbor_counts,
                                          torch.ones(C_neighbor_counts.size()[0], C_neighbor_counts.size()[1],
                                                     device = G.device))

            C = self.C_update(C_msg)
            if self.normalize:
                C = C - C.mean(dim = 0)
                C = C / (C.std(dim = 0) + 1e-10)
            else:
                C = self.C_layer_norm(C)
            if self.average_pool:
                L_pre_msg = torch.cat([C, torch.ones(G.size()[0], 1, dtype = torch.float32, device = G.device)],
                                      dim = 1)
            else:
                L_pre_msg = C
            L_msg = torch.sparse.mm(G.t(), L_pre_msg)
            if self.average_pool:
                L_neighbor_counts = L_msg[:, -1:]
                L_msg = L_msg[:, :-1]
                L_msg = L_msg / torch.max(L_neighbor_counts,
                                          torch.ones(L_neighbor_counts.size()[0], L_neighbor_counts.size()[1],
                                                     device = G.device))
            L = self.L_update(L_msg) + (0.1 * L)
            L = self.L_layer_norm(L)

        V = torch.cat([L[0:int(L.size()[0] / 2)], L[int(L.size()[0] / 2):]], dim = 1)

        # return policy logits and value logits before averaging (for unbatching)
        return self.V_score(V), self.V_vote(V), self.V_score_core(V), self.C_score_core(C)

    def _compute_loss(self, batcher, batched_V_drat_logits, batched_V_core_logits, batched_C_core_logits,
                      batched_var_lemma_counts, batched_core_var_masks, batched_core_clause_masks):
        V_drat_logitss = batcher.unbatch(batched_V_drat_logits, mode = "variable")
        V_core_logitss = batcher.unbatch(batched_V_core_logits, mode = "variable")
        C_core_logitss = batcher.unbatch(batched_C_core_logits, mode = "clause")
        var_lemma_counts = batcher.unbatch(batched_var_lemma_counts, mode = "variable")
        core_var_masks = batcher.unbatch(batched_core_var_masks, mode = "variable")
        core_clause_masks = batcher.unbatch(batched_core_clause_masks, mode = "clause")

        drat_loss = compute_softmax_kldiv_loss_from_logits(V_drat_logitss, var_lemma_counts, tau = 1.0)
        # core_loss = compute_mask_loss(torch.cat(V_core_logitss).unsqueeze(1), [prediction[0].detach()])
        core_loss = compute_mask_loss(V_core_logitss, core_var_masks)
        core_clause_loss = compute_mask_loss(C_core_logitss, core_clause_masks)

        l2_loss = 0.0

        for param in self.parameters():
            l2_loss += (param ** 2).sum()
        # core_loss = 0
        # core_clause_loss = 0

        l2_loss = l2_loss * 1e-9

        # print("EXAMPLE CORE CLAUSE MASK", core_clause_masks[0])
        # print("EXAMPLE DRAT VAR COUNT", var_lemma_countss[0])

        loss = drat_loss + 0.1 * core_loss + 0.01 * core_clause_loss + l2_loss
        # loss = drat_loss + 0.01 * core_clause_loss + l2_loss
        return loss  # loss.backward()  # nn.utils.clip_grad_value_(self.parameters(), 100)  # nn.utils.clip_grad_norm_(self.parameters(), 10)