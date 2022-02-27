import math

import torch
import dgl.nn
from torch.fx import Tracer, Proxy, Node, GraphModule
from torch.fx._compatibility import compatibility
from dgl.nn.functional import edge_softmax
from dgl.function.message import BinaryMessageFunction, CopyMessageFunction
from dgl.function.reducer import SimpleReduceFunction


def is_dgl_function(target):
    if isinstance(target, SimpleReduceFunction) \
    or isinstance(target, BinaryMessageFunction) \
    or isinstance(target, CopyMessageFunction):
        return True
    return False

def get_dgl_function_kwargs(func):
    if isinstance(func, CopyMessageFunction):
        return {"target": func.target,
                "in_field": func.in_field,
                "out_field": func.out_field}
    if isinstance(func, BinaryMessageFunction):
        return {"binary_op": func.binary_op,
                "lhs": func.lhs,
                "rhs": func.rhs,
                "lhs_field": func.lhs_field,
                "rhs_field": func.rhs_field,
                "out_field": func.out_field}
    if isinstance(func, SimpleReduceFunction):
        return {"name": func._name,
                "msg_field": func.msg_field,
                "out_field": func.out_field}


class DGLTracer(Tracer):
    @compatibility(is_backward_compatible=True)
    def __init__(self, autowrap_modules = (math, ),
                 autowrap_functions = (),
                 param_shapes_constant = False) -> None:
        self.graph_proxy = None
        self.conv_modules = dgl.nn.conv.__dict__["__all__"]
        autowrap_functions += (edge_softmax,)
        super().__init__(autowrap_modules, autowrap_functions, param_shapes_constant)

    def set_conv_module(self, module):
        self.conv_modules.append(module.__class__.__name__)

    @compatibility(is_backward_compatible=True)
    def proxy(self, node: Node) -> "Proxy":
        if self.graph_proxy is None:
            return DGLGraphProxy(node, self)
        if is_dgl_function(node.target):
            return DGLFunctionProxy(node, self)
        return Proxy(node, self)

    @compatibility(is_backward_compatible=True)
    def create_arg(self, a):
        if is_dgl_function(a):
            proxy = self.create_proxy(
                "call_function", a.__class__, (), get_dgl_function_kwargs(a), a.name)
            return proxy.node
        else:
            return super().create_arg(a)

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str):
        if m.__class__.__name__ in self.conv_modules:
            return True
        return super().is_leaf_module(m, module_qualified_name)


class DGLGraphProxy(Proxy):
    @property
    def is_block(self):
        # TODO
        return True

    def __str__(self):
        return "Graph{}".format(super().__str__())

class DGLFunctionProxy(Proxy):
    def __str__(self):
        return "DGLFunction{}".format(super().__str__())


@compatibility(is_backward_compatible=True)
def symbolic_trace(root, concrete_args=None):
    tracer = DGLTracer()
    graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    gm = GraphModule(tracer.root, graph, name)
    for key in dir(root):
        if not hasattr(gm, key):
            setattr(gm, key, getattr(root, key))
    return gm
