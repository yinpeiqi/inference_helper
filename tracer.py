import torch
from torch.fx import Tracer


class ProhibitCallModuleTracer(Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str):
        return True
