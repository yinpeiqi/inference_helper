from torch.fx import GraphModule, Graph

from .utils import arg_trace
from .node_relation import NodeRelation
from .constants import CALL_MODULE, PLACEHOLDER


class GraphRearranger():
    def __init__(self, traced: GraphModule):
        self.traced = traced

    def tag_nodes(self):
        graph_name = None
        for i, node in enumerate(self.traced.graph.nodes):
            node.lineno = i
            node.is_message = False
            node.message_degree = -1
            if node.op == PLACEHOLDER:
                if graph_name is None:
                    graph_name = node.name
                node.message_degree = 0
            if node.op == CALL_MODULE and graph_name in arg_trace(node.args):
                node.is_message = True

    def get_lineno2node_map(self):
        lineno2node_map = {}
        for i, node in enumerate(self.traced.graph.nodes):
            lineno2node_map[i] = node
        return lineno2node_map

    def get_start_point(self):
        node_queue = []
        for node in self.traced.graph.nodes:
            if node.op == PLACEHOLDER:
                node_queue.append(node)
        return node_queue

    def rearrange(self):
        node_relation = NodeRelation.get_node_relation(self.traced.graph.nodes)
        lineno2node_map = self.get_lineno2node_map()
        self.tag_nodes()
        
        node_queue = self.get_start_point()
        for node in node_queue:
            for use_lineno in node_relation[node.lineno].use:
                last_node = lineno2node_map[use_lineno]
                if last_node.message_degree == -1:
                    continue
            for use_lineno in node_relation[node.lineno].use:
                last_node = lineno2node_map[use_lineno]
                node.message_degree = max(node.message_degree, last_node.message_degree + node.is_message)
            for be_used_lineno in node_relation[node.lineno].be_used:
                node_queue.append(lineno2node_map[be_used_lineno])

        