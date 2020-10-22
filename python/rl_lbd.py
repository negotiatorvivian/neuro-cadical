import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy.signal
from pysat.formula import CNF
import time
import os
import yaml
import json
import ray
from ray.util import ActorPool
import random
import queue
import tempfile
from uuid import uuid4

from satenv import SatEnv
from gnn import defaultGNN1Cfg, Base
from gnn import rl_GNN1 as GNN1
from batch import Batcher
from util import files_with_extension, recursively_get_files, load_data, set_env
from data_util import coo
from train1 import TrainLogger, Trainer
from solver import cadical_fn


def discount_cumsum(x, discount = 1):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    计算向量的折现累积和
    input:
    vector x,
      [x0,
       x1,
       x2]
    output,
      [x0 + discount * x1 + discount^2 * x2,
       x1 + discount * x2,
       x2]
    """
    # print(x)
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis = 0)[::-1]


def mk_G(CL_idxs):
    # print(CL_idxs.C_idxs, CL_idxs.L_idxs)
    C_idxs = np.array(CL_idxs.C_idxs, dtype = "int32")
    L_idxs = np.array(CL_idxs.L_idxs, dtype = "int32")
    indices = torch.stack([torch.as_tensor(C_idxs).to(torch.long), torch.as_tensor(L_idxs).to(torch.long)])
    values = torch.ones(len(C_idxs), device = indices.device)
    size = [CL_idxs.n_clauses, 2 * CL_idxs.n_vars]
    return torch.sparse.FloatTensor(indices = indices, values = values, size = size)


def softmax_sample_from_logits(logits):
    probs = torch.softmax(logits, 0)
    return int(torch.multinomial(probs, 1)[0])


def softmax_all_from_logits(logits, length):
    probs = torch.softmax(logits, 0)
    return torch.multinomial(probs, length).cpu().numpy()


def sample_trajectory(agent, env, cnf, logger):
    """
    Samples a trajectory from the environment and then resets it. This assumes the environment has been initialized.
    """
    Gs = []
    mu_logitss = []
    actions = []
    rewards = []
    value_estimates = []
    cnfs = [cnf]

    CL_idxs = env.render()
    terminal_flag = False
    while not terminal_flag:
        G = mk_G(CL_idxs)
        mu_logits, value_estimate = agent.act(G)
        action = (softmax_sample_from_logits(mu_logits) + 1)  # torch multinomial zero-indexes
        CL_idxs, reward, terminal_flag = env.step((np.random.choice([1, -1])) * action)
        Gs.append(G)
        mu_logitss.append(mu_logits)
        actions.append(action)
        rewards.append(reward)
        value_estimates.append(value_estimate)
        # cnfs.append(cnf)
    # logger.write_log(f"actions: {actions}, rewards: {rewards}, value_estimate: {value_estimates}")
    env.reset()
    return Gs, mu_logitss, actions, rewards, value_estimates, cnfs


def cnf_to_data(cnfs):
    results = []
    for cnf in cnfs:
        n_vars = cnf.nv
        n_cls = len(cnf.clauses)
        C_idxs, L_idxs = coo(cnf)
        signs = np.array([1] * (2 * n_vars))
        signs[cnf.nv:] = -1
        edge_features = signs[L_idxs]
        L_idxs = L_idxs
        L_idxs[L_idxs > n_vars - 1] -= n_vars
        indices = torch.stack([torch.from_numpy(L_idxs).type(torch.long), torch.from_numpy(C_idxs).type(torch.long)])
        result = cnf.is_sat
        answers = cnf.answers
        results.append((n_vars, n_cls, indices, edge_features, answers, float(result), []))
    # return torch.sparse.FloatTensor(indices = indices, values = torch.from_numpy(edge_features), size = size)
    return results


def process_trajectory(Gs, mu_logitss, actions, rewards, vals, cnfs, last_val = 0, gam = 1.0, lam = 1.0):
    gs = discount_cumsum(rewards, int(gam))
    deltas = np.append(rewards, last_val)[:-1] + gam * np.append(vals, last_val)[1:] - np.append(vals, last_val)[
                                                                                       :-1]  # future advantage
    adv = discount_cumsum(deltas, int(gam * lam))
    data = cnf_to_data(cnfs)
    return Gs, mu_logitss, actions, (gs + 1.0) / 2.0, adv, gs[0], data  # note, gs[0] is total value


class Agent:
    def __init__(self):
        pass

    def act(self, G):  # returns action and value estimate; agent handles softmax-sampling
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, seed = None):
        if seed is not None:
            torch.manual_seed(seed)

    def act(self, G):
        n_vars = int(G.size()[1] / 2)
        return torch.rand(n_vars), 0


class NeuroAgent(Agent):
    def __init__(self, model_cfg = defaultGNN1Cfg, model_state_dict = None):
        self.model = GNN1(**model_cfg)
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)

    def act(self, G):
        p_logits, v_pre_logits, _, _ = self.model(G)
        return p_logits.squeeze().detach(), torch.sigmoid(v_pre_logits.mean()).detach()

    def set_weights(self, model_state_dict):
        self.model.load_state_dict(model_state_dict)


class EpisodeWorker:  # buf is a handle to a ReplayBuffer object
    def __init__(self, buf, weight_manager, logdir, model_cfg = defaultGNN1Cfg, model_state_dict = None,
                 from_cnf = None, from_file = None, seed = None, sync_freq = 10, restore = True):
        self.buf = buf
        self.weight_manager = weight_manager
        self.logger = TrainLogger(logdir = logdir)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        if from_cnf is not None or from_file is not None:
            self.set_env(from_cnf, from_file)

        self.sync_freq = sync_freq
        self.ckpt_rank = 0
        self.trajectory_count = 0
        self.agent = NeuroAgent(model_cfg = model_cfg)
        if restore:
            self.try_update_weights()
        print("WORKER INITIALIZED")

    def set_env(self, from_cnf = None, from_file = None):
        paths = from_file.split('/')
        from_cnf = os.path.join('/'.join(paths[:-1]) + '-answers', paths[-1])
        if from_cnf is not None:
            # self.td = tempfile.TemporaryDirectory()
            # cnf_path = os.path.join(self.td.name, str(uuid4()) + ".cnf")
            # from_cnf.to_file(cnf_path)
            try:
                # self.env = SatEnv(cnf_path)
                self.cnf = CNF(from_file = from_cnf)
                print(f'self.cnf: {self.cnf}')
            except RuntimeError as e:
                print("BAD CNF:", from_cnf)
                raise e
        if from_file is not None:
            try:
                self.env = SatEnv(from_file)

            except RuntimeError as e:
                print("BAD CNF:", from_file)
                raise e  # else:  #     raise Exception("must set env with CNF or file")

    def sample_trajectory(self):
        tau = sample_trajectory(self.agent, self.env, self.cnf, self.logger)
        self.buf.ingest_trajectory.remote(process_trajectory(*tau))
        print(f"SAMPLED TRAJECTORY OF LENGTH {len(tau[0])}")
        self.trajectory_count += 1
        if self.trajectory_count % self.sync_freq == 0:
            self.try_update_weights()

    def set_weights(self, model_state_dict, new_rank):
        self.agent.set_weights(model_state_dict)
        self.ckpt_rank = new_rank

    def try_update_weights(self):
        status, new_state_dict1, new_state_dict2, new_rank = ray.get(self.weight_manager.sync_weights.remote(
            self.ckpt_rank))
        if status:
            print(f"SYNCING WEIGHTS: {self.ckpt_rank} -> {new_rank}")
            self.set_weights([new_state_dict1, new_state_dict2], new_rank)

    def __del__(self):
        try:
            self.td.cleanup()
        except AttributeError:
            pass


class ReplayBuffer:
    def __init__(self, root_dir, limit = 100000):
        self.root_dir = root_dir
        self.logdir = os.path.join(self.root_dir, "logs/")
        self.ckpt_dir = os.path.join(self.logdir, "returns/")
        self.episode_count = 0
        self.writer = SummaryWriter(log_dir = self.logdir)
        self.logger = TrainLogger(logdir = self.logdir)
        self.queue = queue.Queue()
        self.sample_count = 0
        self.limit = limit

    def set_episode_count(self, episode_count):
        self.episode_count = episode_count

    def get_episode_count(self):
        return self.episode_count

    def ingest_trajectory(self, tau):
        Gs, mu_logitss, actions, gs, advs, total_return, cnfs = tau
        # print(gs)
        self.writer.add_scalar("total return", total_return, self.episode_count)
        for G, mu_logits, action, g, adv, cnf in zip(Gs, mu_logitss, actions, gs, advs, cnfs):
            self.queue.put((G, mu_logits, actions, g, adv, cnf))

        print(f"TOTAL RETURN: {gs[0]}")
        self.episode_count += 1

    def batch_ready(self, batch_size):
        return batch_size <= self.queue.qsize()

    def get_batch(self, batch_size, sample_randomly = False):
        if sample_randomly:
            raise Exception("unsupported")
        else:
            Gs = []
            mu_logitss = []
            actions = []
            gs = []
            advs = []
            cnfs = []

            for _ in range(batch_size):
                G, mu_logits, action, g, adv, cnf = self.queue.get()
                Gs.append(G)
                mu_logitss.append(mu_logits)
                actions.append(action)
                gs.append(g)
                advs.append(adv)
                cnfs.append(cnf)
                self.logger.write_log(f'get batch: action:{action}, g:{g}, adv:{adv}')
            return Gs, mu_logitss, actions, gs, advs, cnfs

    def get_sp_batch(self, batch_size):
        Gs = []
        mu_logitss = []
        actions = []
        gs = []
        advs = []
        cnfs = []

        for _ in range(batch_size):
            G, mu_logits, action, g, adv, cnf = self.queue.get()
            Gs.append(G)
            actions.append(action)
            # mu_logitss.append(mu_logits)
            # gs.append(g)
            # advs.append(adv)
            cnfs.append(cnf)
            self.logger.write_log(f'get batch: action:{action}')
        return Gs, mu_logitss, actions, gs, advs, cnfs


def train_step(model, optim, batcher, G, batch_size, graphsage, nodes, labels, mu_logitss, actions, gs, advs, cnfs,
               device = torch.device("cpu")):
    """
    Calculate loss and perform a single gradient update. Returns a dictionary of statistics.

    model: G -> policy_logits, value_estimate

    Gs: list of bipartite adjacency matrices
    mu_logitss: list of action probabilities
    actions: list of actions taken
    gs: list of returns
    advs: list of advantages
    """
    agg_loss = None
    if graphsage is not None:
        agg_loss = graphsage.loss(nodes, labels)

    pre_policy_logitss, pre_unreduced_value_logitss = model(G, cnfs)
    policy_logitss = batcher.unbatch(pre_policy_logitss, mode = "variable")
    actions = torch.as_tensor(np.array(actions, dtype = "int32")).to(device)
    advs = torch.as_tensor(np.array(advs, dtype = "float32")).to(device)

    policy_distribs = [Categorical(logits = x.squeeze().to(device)) for x in policy_logitss]
    mu_distribs = [Categorical(logits = x.squeeze().to(device)) for x in mu_logitss]
    rhos = torch.stack([torch.exp(x.log_prob(actions[i] - 1) - y.log_prob(actions[i] - 1)) for i, (x, y) in
                        enumerate(zip(policy_distribs, mu_distribs))])

    psis = advs * rhos
    print(f'advs:{advs}, rhos: {rhos}')
    log_probs = torch.empty(batch_size).to(device)
    for i in range(batch_size):
        log_probs[i] = policy_distribs[i].log_prob(actions[i] - 1)

    p_loss = -(log_probs * psis).mean()
    # v^
    vals = torch.sigmoid(
        torch.stack([x.mean() for x in batcher.unbatch(pre_unreduced_value_logitss, mode = "variable")]).to(device))
    # print(f'vals:{vals}, gs: {gs}')

    v_loss = F.mse_loss(vals, torch.as_tensor(np.array(gs, dtype = "float32")).to(device))
    loss = p_loss + 0.1 * v_loss
    if agg_loss:
        loss += agg_loss

    loss.backward()
    print('total_loss', loss.detach().cpu().numpy())

    nn.utils.clip_grad_value_(model.parameters(), 100)
    nn.utils.clip_grad_norm_(model.parameters(), 10)

    optim.step()

    return {"p_loss": p_loss.detach().cpu().numpy(), "v_loss": v_loss.detach().cpu().numpy()}


def train_batch(model, G, batch_size, graphsage, nodes, labels, actions, cnfs, logger,
                device = torch.device("cpu")):
    for data in model.transform_data(cnfs):

        agg_loss = None
        if graphsage is not None:
            agg_loss = graphsage.loss(nodes, labels)
        tempdir = model.train(G, actions, data)
        logger.write_log(f'actions: {actions}')
        validate(tempdir, logger)

        # policy_logitss = batcher.unbatch(pre_policy_logitss, mode = "variable")
        # actions = torch.as_tensor(np.array(actions, dtype = "int32")).to(device)
        # advs = torch.as_tensor(np.array(advs, dtype = "float32")).to(device).clone()
        #
        # policy_distribs = [Categorical(logits = x.squeeze().to(device).clone()) for x in policy_logitss]
        # mu_distribs = [Categorical(logits = x.squeeze().to(device).clone()) for x in mu_logitss]
        # rhos = torch.stack([torch.exp(x.log_prob(actions[i] - 1) - y.log_prob(actions[i] - 1)) for i, (x, y) in
        #                        enumerate(zip(policy_distribs, mu_distribs))])
        #
        # psis = advs * rhos
        # print(f'advs:{advs}, rhos: {rhos}')
        # log_probs = torch.empty(batch_size).to(device)
        # for i in range(batch_size):
        #     log_probs[i] = policy_distribs[i].log_prob(actions[i] - 1)
        #
        # p_loss = -(log_probs * psis).mean()
        # # v^
        # vals = torch.sigmoid(
        #     torch.stack([x.mean().clone() for x in batcher.unbatch(pre_unreduced_value_logitss, mode = "variable")]).to(device))
        # # print(f'vals:{vals}, gs: {gs}')
        #
        # v_loss = F.mse_loss(vals, torch.as_tensor(np.array(gs, dtype = "float32")).to(device))
        # loss = p_loss + 0.1 * v_loss
        # if agg_loss:
        #     loss += agg_loss
        #
        # loss.backward()
        # print('total_loss', loss.detach().cpu().numpy())
        #
        # nn.utils.clip_grad_value_(model.parameters(), 100)
        # nn.utils.clip_grad_norm_(model.parameters(), 10)
        #
        # optim.step()
    # return {"p_loss": p_loss.detach().cpu().numpy(), "v_loss": v_loss.detach().cpu().numpy()}
    return None


def validate(td, logger):
    files = recursively_get_files(td.name, ['cnf'])

    for file in files:
        res = cadical_fn(file, gpu = True)
        logger.write_log(res)
        print(res)



def predict_step(model, batcher, G, batch_size, graphsage, nodes, labels, device = torch.device("cpu")):
    # CL_idxs = env.render()
    # G = mk_G(CL_idxs)
    print('------predict start------')
    pre_policy_logitss, pre_unreduced_value_logitss = model(G)
    # policy_logitss = batcher.unbatch(pre_policy_logitss, mode = "variable")
    actions = softmax_all_from_logits(pre_policy_logitss.squeeze().detach(), batch_size) + 1
    actions = torch.as_tensor(np.array(actions, dtype = "int32")).to(device)
    vals = torch.sigmoid(
        torch.stack([x.mean() for x in batcher.unbatch(pre_unreduced_value_logitss, mode = "variable")]).to(device))
    print(f'actions:{actions}, vals: {vals}')


class WeightManager:
    def __init__(self, ckpt_dir):
        self.ckpt_dir = ckpt_dir
        self.save_counter = 0
        self.model_state_dict1 = {}
        self.model_state_dict2 = {}
        self.optim_state_dict = {}
        self.GLOBAL_STEP_COUNT = 0

    def get_latest_from_index(self, ckpt_dir):
        index = files_with_extension(ckpt_dir, "index")[0]
        with open(index, "r") as f:
            cfg_dict = json.load(f)
        return cfg_dict["latest"]

    def load_latest_ckpt(self):
        try:
            root_dir = '/'.join(self.ckpt_dir.split('/')[:-2])
            ckpt_path = self.get_latest_from_index(root_dir)
            print(root_dir, ckpt_path)
        except IndexError:
            print("[WEIGHT MANAGER] NO INDEX FOUND")
            return None
        ckpt = torch.load(ckpt_path)
        self.load_ckpt(ckpt)
        print(f"[WEIGHT MANAGER] RESTORING FROM {ckpt_path}")
        return ckpt

    def load_ckpt(self, ckpt):
        self.model_state_dict1 = json.loads(ckpt["model_state_dict1"])
        self.model_state_dict2 = json.loads(ckpt["model_state_dict2"])
        self.optim_state_dict = ckpt["optim_state_dict"]
        self.save_counter = ckpt["save_counter"]
        self.GLOBAL_STEP_COUNT = ckpt["GLOBAL_STEP_COUNT"]

    def sync_weights(self, rank):
        status = False
        model_state_dict1 = None
        model_state_dict2 = None

        new_rank = rank
        if rank < self.save_counter:
            status = True
            model_state_dict1 = dict()
            model_state_dict2 = dict()
            for k, v in self.model_state_dict1.items():
                model_state_dict1[k] = v
            for k, v in self.model_state_dict2.items():
                model_state_dict2[k] = v
            new_rank = self.save_counter
        return status, model_state_dict1, model_state_dict2, new_rank

    def update_index(self, ckpt_path):
        ckpt_dir = os.path.dirname(ckpt_path)
        root_dir = '/'.join(ckpt_dir.split('/')[:-2])
        index_files = files_with_extension(root_dir, "index")
        if len(index_files) == 0:
            index = os.path.join(root_dir, "latest.index")
        else:
            assert len(index_files) == 1
            index = index_files[0]
        with open(index, "w") as f:
            cfg_dict = {"latest": ckpt_path}
            f.write(json.dumps(cfg_dict, indent = 2))

    def save_ckpt(self, model_state_dict1, model_state_dict2, optim_state_dict, save_counter, GLOBAL_STEP_COUNT,
                  episode_count, name = 'best'):
        self.model_state_dict1 = model_state_dict1
        self.model_state_dict2 = model_state_dict2
        self.optim_state_dict = optim_state_dict
        self.save_counter = save_counter
        self.GLOBAL_STEP_COUNT = GLOBAL_STEP_COUNT
        ckpt_path = os.path.join(self.ckpt_dir, f"ckpt_{name}.pth")
        torch.save({
            "model_state_dict1": model_state_dict1, "model_state_dict2": model_state_dict2,
            "optim_state_dict": optim_state_dict, "save_counter": save_counter, "GLOBAL_STEP_COUNT": GLOBAL_STEP_COUNT,
            "episode_count": episode_count
        }, ckpt_path)
        self.update_index(
            ckpt_path)  # print(f"SAVED CHECKPOINT TO ckpt_{name}.pth")  # print(f"GLOBAL_STEP_COUNT: {GLOBAL_STEP_COUNT}\nEPISODE_COUNT: {episode_count}")


# let's try single-GPU training for now
class Learner:
    def __init__(self, encode_dim, feature_dim, num_samples, buf, weight_manager, batch_size, log_dir, ckpt_dir,
                 ckpt_freq, lr, sp_config, restore = True, model_cfg = defaultGNN1Cfg):
        self.encode_dim = encode_dim
        self.feature_dim = feature_dim
        self.num_samples = num_samples
        self.buf = buf
        self.weight_manager = weight_manager
        self.batch_size = batch_size
        self.ckpt_dir = ckpt_dir
        self.logdir = log_dir
        self.ckpt_freq = ckpt_freq
        self.lr = lr
        self.writer = SummaryWriter(log_dir = self.ckpt_dir)
        self.logger = TrainLogger(logdir = self.logdir)
        self.GLOBAL_STEP_COUNT = 0
        self.save_counter = 0
        self.batcher = Batcher()
        if torch.cuda.is_available():
            print("[LEARNER] USING GPU")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # self.model = GNN1(model_cfg, {'config': sp_config, 'cpu': True, 'logger': self.logger})
        self.model = Base(model_cfg, sp_config, self.device, self.batcher, cpu = True, logger = self.logger)
        # self.rl_model = GNN1(**model_cfg)
        self.optim = torch.optim.Adam(self.model.parameters, lr = self.lr)
        if restore:
            ckpt = ray.get(self.weight_manager.load_latest_ckpt.remote())
            if ckpt is not None:
                self.set_weights(ckpt)

    def write_stats(self, stats):
        for name, value in stats.items():
            self.writer.add_text(name, str(value), self.GLOBAL_STEP_COUNT)
            self.writer.add_scalar(name, value, self.GLOBAL_STEP_COUNT)

    def data_loader(self):
        yield ray.get(self.buf.get_sp_batch.remote(self.batch_size))

    def train_batch(self, Gs, mu_logitss, actions, gs, advs, cnfs):
        batch_size = len(Gs)
        G, clause_values = self.batcher.batch(Gs)
        self.logger.write_log(f"G: {G.size()}")
        print(f"G: {G.size()}")

        # x_dim = G.size()[0]
        # y_dim = G.size()[1]
        # features = nn.Embedding(x_dim, y_dim)
        # features.weight = nn.Parameter(torch.FloatTensor(G.to_dense()), requires_grad = False)
        # adj_lists, node_lists, nodes, clauses, labels = load_data(G, clause_values)
        #
        # agg1 = MeanAggregator(features, cuda = True)
        # enc1 = Encoder(features, y_dim, self.encode_dim, adj_lists, agg1, gcn = True, cuda = False)
        # agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda = False)
        # enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, self.encode_dim, adj_lists, agg2,
        #                base_model = enc1, gcn = True, cuda = False)
        # enc1.num_samples = self.num_samples
        # enc2.num_samples = self.num_samples
        # graphsage = SupervisedGraphSage(self.feature_dim, enc2)
        # self.model_list = [PropagatorSolverBase(graphsage, self.model, self.device)]
        # self.model_list = [model._set_device() for model in self.model_list]
        # self.optim = torch.optim.Adam(
        #     [{'params': filter(lambda p: p.requires_grad, model.parameters())} for model in self.model_list],
        #     lr = self.lr)
        # stats = train_step(graphsage, self.model, self.optim, self.batcher, G, batch_size, nodes, labels, mu_logitss, actions, gs,
        #                    advs, device = self.device)
        train_batch(self.model, G, batch_size, None, None, None, actions, cnfs, self.logger, device = self.device)
        # return stats.get('p_loss') + stats.get('v_loss')
        return None

    def sync_weights(self, w):
        w.set_weights.remote(self.get_weights())
        print(f"SYNCED WEIGHTS TO WORKER {w}")

    def get_weights(self):
        return self.model.model_list[-1].state_dict()

    def set_weights(self, ckpt):
        self.model.load_state_dict([ckpt["model_state_dict1"], ckpt["model_state_dict2"]], strict = False)
        self.optim.load_state_dict(ckpt["optim_state_dict"])
        self.GLOBAL_STEP_COUNT = ckpt["GLOBAL_STEP_COUNT"]
        self.save_counter = ckpt["save_counter"]
        ray.get(self.buf.set_episode_count.remote(ckpt["episode_count"]))

    def save_ckpt(self, best = False):
        episode_count = ray.get(self.buf.get_episode_count.remote())
        name = 'best' if best else 'last'
        # models = [model.state_dict() for model in self.model.model_list]
        # model_names = [model._name for model in self.model.model_list]
        # model_state = dict(zip(model_names, models))
        self.weight_manager.save_ckpt.remote(self.model.model_list[0], self.model.model_list[1],
                                             self.optim.state_dict(), self.save_counter + 1, self.GLOBAL_STEP_COUNT,
                                             episode_count = episode_count, name = name)
        self.save_counter += 1

    def train(self, step_limit = None, time_limit = None, synchronous = False):
        start = time.time()
        try:
            loss = np.zeros(1)
            while True:
                if step_limit is not None:
                    if self.GLOBAL_STEP_COUNT > step_limit:
                        print("STEP LIMIT", step_limit, "REACHED, STOPPING")
                        break
                elif time_limit is not None:
                    elapsed = time.time() - start
                    if elapsed > time_limit:
                        print("TIME LIMIT", time_limit, "REACHED, STOPPING")
                        break
                else:
                    pass

                if ray.get(self.buf.batch_ready.remote(self.batch_size)):
                    pass
                else:
                    if synchronous:
                        break
                    else:
                        SLEEP_INTERVAL = 0.5
                        print(f"Replay buffer not ready. Sleeping for {SLEEP_INTERVAL}")
                        time.sleep(SLEEP_INTERVAL)
                        continue

                batch = ray.get(self.buf.get_batch.remote(self.batch_size))
                temp = self.train_batch(*batch)
                if temp is None:
                    del batch
                    self.GLOBAL_STEP_COUNT += 1
                    continue
                if loss == 0:
                    loss = temp
                    self.save_ckpt(False)
                elif loss > temp:
                    self.save_ckpt(True)
                    loss = temp
                else:
                    self.save_ckpt(False)
                    loss = temp
                del batch
                self.GLOBAL_STEP_COUNT += 1  # if self.GLOBAL_STEP_COUNT % self.ckpt_freq == 0:  #     self.save_ckpt()
        finally:
            print("Finishing up and saving checkpoint.")
            if self.optim:
                self.save_ckpt(False)

    def predict(self):
        Gs, mu_logitss, actions, gs, advs = ray.get(self.buf.get_batch.remote(self.batch_size))
        batch_size = len(Gs)
        i = 0
        while i < batch_size:
            G, clause_values = self.batcher.batch(Gs)
            predict_step(self.model, self.batcher, G, batch_size, None, None, None, self.device)
            i += 1


def _parse_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", dest = "n_workers", action = "store", default = 1, type = int)
    parser.add_argument("--n-epochs", dest = "n_epochs", action = "store", default = 10, type = int)
    parser.add_argument("--num-samples", dest = "num_samples", action = "store", default = 5, type = int)
    parser.add_argument("--encode-dim", dest = "encode_dim", action = "store", default = 128, type = int)
    parser.add_argument("--feature-dim", dest = "feature_dim", action = "store", default = 2, type = int)
    parser.add_argument("--cnfs", dest = "cnfs", action = "store")
    parser.add_argument("--time-limit", dest = "time_limit", action = "store", type = float)
    parser.add_argument("--lr", dest = "lr", type = float, action = "store", default = 1e-4)
    parser.add_argument("--root-dir", dest = "root_dir", action = "store")
    parser.add_argument("--ckpt-freq", dest = "ckpt_freq", action = "store", type = int, default = 10)
    parser.add_argument("--batch-size", dest = "batch_size", action = "store", type = int, default = 2)
    parser.add_argument("--object-store", dest = "object_store", action = "store", default = None)
    parser.add_argument("--eps-per-worker", dest = "eps_per_worker", action = "store", default = 25, type = int)
    parser.add_argument("--model-cfg", dest = "model_cfg", action = "store", default = None)
    parser.add_argument("--sp-cfg", dest = "sp_cfg", action = "store", default = None)

    opts = parser.parse_args()
    opts.root_dir = os.path.join(opts.root_dir, time.strftime("%Y%m%d-%H%M", time.localtime()))
    opts.ckpt_dir = os.path.join(opts.root_dir, 'weights')
    opts.log_dir = os.path.join(opts.root_dir, 'logs')

    if not os.path.exists(opts.root_dir):
        os.makedirs(opts.ckpt_dir)
        os.makedirs(opts.log_dir)
    with open(opts.sp_cfg, 'r') as f:
        config = yaml.load(f)
        opts.sp_config = config

    return opts


def _main(is_train = True):
    opts = _parse_main()
    if not os.path.exists(opts.log_dir):
        os.makedirs(opts.log_dir)
    if not os.path.exists(opts.ckpt_dir):
        os.makedirs(opts.ckpt_dir)
    files = recursively_get_files(opts.cnfs, exts = ["cnf", "gz"], forbidden = ["bz2"])
    print(f"TRAINING WITH {len(files)} CNFS")

    try:
        ray.init(address = "auto", redis_password = '5241590000000000') if opts.object_store is None else ray.init(
            address = "auto", redis_password = '5241590000000000', object_store_memory = int(opts.object_store))
    except:
        print("[WARNING] FALLING BACK ON SINGLE MACHINE CLUSTER")
        ray.init()

    buf = ray.remote(ReplayBuffer).remote(root_dir = opts.root_dir)
    weight_manager = ray.remote(num_gpus = (0 if torch.cuda.is_available() else 0))(WeightManager).remote(
        ckpt_dir = opts.ckpt_dir)

    if opts.model_cfg is not None:
        with open(opts.model_cfg, "r") as f:
            model_cfg = json.load(f)
    else:
        model_cfg = defaultGNN1Cfg

    learner = ray.remote(num_gpus = (0 if torch.cuda.is_available() else 0))(Learner).remote(
        encode_dim = opts.encode_dim, feature_dim = opts.feature_dim, num_samples = opts.num_samples, buf = buf,
        weight_manager = weight_manager, batch_size = opts.batch_size, log_dir = opts.log_dir,
        ckpt_freq = opts.ckpt_freq, ckpt_dir = opts.ckpt_dir, lr = opts.lr, sp_config = opts.sp_config,
        restore = not is_train, model_cfg = model_cfg)
    # TODO: to avoid oom, either dynamically batch or preprocess the formulas beforehand to ensure that they
    # TODO:  are under a certain size -- this will requre some changes throughout to avoid a fixed batch size
    workers = [ray.remote(EpisodeWorker).remote(buf = buf, weight_manager = weight_manager, logdir = opts.log_dir,
                                                model_cfg = model_cfg) for _ in range(opts.n_workers)]
    pool = ActorPool(workers)

    def shuffle_environments(ws):
        for w in ws:
            ray.get(w.set_env.remote(from_file = random.choice(files)))

    for k_epoch in range(opts.n_epochs):
        waiting = 0
        completed = 0
        shuffle_environments(workers)
        for _ in pool.map_unordered((lambda a, v: a.sample_trajectory.remote()),
                                    range(opts.eps_per_worker * opts.n_workers)):
            pass

    if is_train:
        ray.get(learner.train.remote(step_limit = 5000, synchronous = True))
    else:
        ray.get(learner.predict.remote())


if __name__ == "__main__":
    _main(True)