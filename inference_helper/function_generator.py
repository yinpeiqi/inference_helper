import types

import dgl
import torch
import torch.nn as nn
from torch.fx import GraphModule, Graph

from .split_plan import SplitPlanGenerator
from .graph_spliter import GraphSpliter
from .schema import Schema
from .tracer import ProhibitCallModuleTracer
from .utils import inference_helper_getattr
from .graph_rewriter import GraphRewriter
from .constants import FORWARD_CONV


class FunctionGenerator(nn.Module):
    def __init__(self, module: nn.Module, debug):
        super().__init__()
        self.debug = debug
        self.schema = None
        self.funcs = []
        for name in module.__dict__:
            if hasattr(module, name):
                attr = getattr(module, name)
                setattr(self, name, attr)
        self.module_split(module)

    def module_split(self, module: nn.Module):
        traced = GraphModule(module, ProhibitCallModuleTracer().trace(module))
        if self.debug:
            print("-------- Origin forward function -------")
            print(traced.code.strip())
            print("----------------------------------------")

        self.schema = Schema()
        self.schema.record_inputs_and_outputs(traced.graph)
        GraphRewriter.blocks_to_graph(traced.graph)
        GraphRewriter.remove_unused_nodes(traced.graph)
        traced.recompile()

        if self.debug:
            print("------- Modified forward function ------")
            print(traced.code.strip())
            print("----------------------------------------")

        planner = SplitPlanGenerator(traced.graph.nodes)
        split_linenos = planner.get_split_plan()
        
        spliter = GraphSpliter(traced.graph.nodes)
        graphs_list = spliter.split_graph(split_linenos)

        for layer_id, graph in enumerate(graphs_list):
            GraphRewriter.remove_unused_nodes(graph)
            self.register_func_from_graph(graph, layer_id)
            self.schema.create_layer(graph)

    def register_func_from_graph(self, graph: Graph, layer_id: int):
        graph_src = graph.python_code("self").src

        func_name = FORWARD_CONV + str(layer_id)
        graph_src = graph_src.replace("def forward(", "def {}(".format(func_name))
        graph_src = graph_src.replace(" getattr(", " inference_helper_getattr(")
        self.set_function_from_string(graph_src, func_name)

        if self.debug:
            print("--------- Layer {} conv function --------".format(layer_id))
            print(graph_src.strip())
            print("----------------------------------------")

    def set_function_from_string(self, func_src, func_name):
        globals_vals = globals()
        exec(func_src, globals_vals)
        setattr(self, func_name, types.MethodType(globals_vals[func_name], self))
        self.funcs.append(getattr(self, func_name))

    def get_schema(self):
        return self.schema

    def get_funcs(self):
        return self.funcs
