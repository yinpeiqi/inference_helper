import dgl
import torch
from torch.fx import GraphModule, Interpreter


class CostEvaluater(Interpreter):
    def __init__(self, gm: GraphModule):
        super().__init__(gm)

    def eval(self, *args):
        fake_graph = dgl.graph((torch.tensor([0]), torch.tensor([0]))).to('cuda')
        new_args = (fake_graph, torch.zeros((1, 3703)).to('cuda'))
        self.run(*new_args)
