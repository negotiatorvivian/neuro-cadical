import torch
import torch.nn as nn
import numpy as np
import random
import os


class PropagatorSolverBase(nn.Module):
    def __init__(self, aggregator, gnn, device):
        super(PropagatorSolverBase, self).__init__()
        self.aggregator = aggregator
        self.gnn = gnn
        self.module_list = nn.ModuleList()
        self.module_list.append(self.aggregator)
        self.module_list.append(self.gnn)
        self.device = device

    def save(self, export_path_base):
        torch.save(self.state_dict(), os.path.join(export_path_base, self._name))

    def load(self, import_path_base):
        self.load_state_dict(torch.load(os.path.join(import_path_base, self._name)))

    def _set_device(self):
        "Sets the CPU/GPU device."

        if torch.cuda.is_available():
            return nn.DataParallel(self.module_list).cuda(self.device)
        return self.module_list.cpu()



