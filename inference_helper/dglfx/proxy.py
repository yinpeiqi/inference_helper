import operator
import builtins

from torch.fx import Proxy, Node, Tracer


class DGLGraphProxy(Proxy):
    def __init__(self, node: Node, tracer: Tracer = None):
        super().__init__(node, tracer)
        node.node_type = "DGLGraph"

    @property
    def is_block(self):
        return True

    def local_var(self):
        return self.tracer.create_proxy("call_method", "local_var", (self,), {}, 
            proxy_factory_fn=self.tracer.dgl_graph_proxy)

    def __getitem__(self, rhs):
        return self.tracer.create_proxy("call_function", operator.getitem, (self, rhs), {}, 
            proxy_factory_fn=self.tracer.dgl_graph_proxy)

    def apply_edges(self, *args, **kwargs):
        return self.tracer.create_proxy("call_method", "apply_edges", (self,) + args, kwargs, 
            proxy_factory_fn=self.tracer.dgl_graph_proxy)

    def update_all(self, *args, **kwargs):
        return self.tracer.create_proxy("call_method", "update_all", (self,) + args, kwargs, 
            proxy_factory_fn=self.tracer.dgl_graph_proxy)

    @property
    def srcdata(self):
        return self.tracer.create_proxy("call_function", builtins.getattr, (self, "srcdata"), {},
            proxy_factory_fn=self.tracer.dgl_graph_attribute)

    @property
    def dstdata(self):
        return self.tracer.create_proxy("call_function", builtins.getattr, (self, "dstdata"), {},
            proxy_factory_fn=self.tracer.dgl_graph_attribute)

    @property
    def ndata(self):
        return self.tracer.create_proxy("call_function", builtins.getattr, (self, "ndata"), {},
            proxy_factory_fn=self.tracer.dgl_graph_attribute)

    @property
    def edata(self):
        return self.tracer.create_proxy("call_function", builtins.getattr, (self, "edata"), {},
            proxy_factory_fn=self.tracer.dgl_graph_attribute)

    def __str__(self):
        return "Graph{}".format(super().__str__())


class DGLGraphAttribute(Proxy):
    def __init__(self, node: Node, tracer: Tracer = None):
        super().__init__(node, tracer)
        node.node_type = "DGLGraphAttribute"

    def update(self, *args, **kwargs):
        return self.tracer.create_proxy("call_method", "update", (self,) + args, kwargs, 
            proxy_factory_fn=self.tracer.dgl_graph_proxy)


class DGLFunctionProxy(Proxy):
    def __init__(self, node: Node, tracer: Tracer = None):
        super().__init__(node, tracer)
        node.node_type = "DGLFunction"

    def __str__(self):
        return "DGLFunction{}".format(super().__str__())


class DGLVoidCallProxy(Proxy):
    def __init__(self, node: Node, tracer: Tracer = None):
        super().__init__(node, tracer)
        node.node_type = "DGLVoidCall"

    def __str__(self):
        return "DGLVoidCall{}".format(super().__str__())
