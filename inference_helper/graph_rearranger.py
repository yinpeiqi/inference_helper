from torch.fx import GraphModule, Graph, Node

from .utils import arg_trace
from .node_relation import NodeRelation
from .constants import CALL_METHOD, CALL_MODULE, OUTPUT, PLACEHOLDER


class GraphRearranger():
    def __init__(self, traced: GraphModule):
        self.traced = traced
        self.output = None
        self.inputs = []

    def tagging_node(self, lineno, node):
        node.lineno = lineno
        node.is_input = False
        node.is_message = False
        node.message_degree = -1
        node.is_graph_function = False
        node.split_points = []

    def tag_nodes(self):
        graph_name = None
        for i, node in enumerate(self.traced.graph.nodes):
            self.tagging_node(i, node)
            if node.op == PLACEHOLDER:
                self.inputs.append(node)
                node.is_input = True
                if graph_name is None:
                    graph_name = node.name
                node.message_degree = 0
            if node.op == CALL_MODULE and graph_name in arg_trace(node.args):
                node.is_message = True
            if node.op == CALL_METHOD and graph_name in arg_trace(node.args):
                node.is_graph_function = True
            if node.op == OUTPUT:
                self.output = node

    def get_lineno2node_map(self):
        lineno2node_map = {}
        for i, node in enumerate(self.traced.graph.nodes):
            lineno2node_map[i] = node
        return lineno2node_map

    def compute_message_degree(self, node_relation, lineno2node_map):
        node_queue = []
        for node in self.traced.graph.nodes:
            if node.op == PLACEHOLDER:
                node_queue.append(node)
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

        # graph function's output only belongs to one layer
        for node in self.traced.graph.nodes:
            if node.is_graph_function:
                change_list = [node]
                message_layer = node.message_degree
                for next_node in change_list:
                    for lineno in node_relation[next_node.lineno].be_used:
                        next_node = lineno2node_map[lineno]
                        if next_node.is_message:
                            message_layer = next_node.message_degree
                        else:
                            change_list.append(next_node)
                for next_node in change_list:
                    next_node.message_degree = message_layer

    def compute_split_points(self, node_relation, lineno2node_map):
        passing_edges = []
        for node in self.traced.graph.nodes:
            for use_lineno in node_relation[node.lineno].use:
                last_node = lineno2node_map[use_lineno]
                if node.message_degree != last_node.message_degree:
                    passing_edges.append((last_node.lineno, node.lineno))

        for edge in passing_edges:
            curr_edge = edge
            while True:
                src_lineno, dst_lineno = curr_edge
                if len(node_relation[src_lineno].be_used) != 1 or \
                    lineno2node_map[src_lineno].is_message or lineno2node_map[src_lineno].is_input:
                    lineno2node_map[src_lineno].split_points.append(dst_lineno)
                    break
                curr_edge = (node_relation[src_lineno].be_used[0], src_lineno)
        
    def rearrange(self):
        node_relation = NodeRelation.get_node_relation(self.traced.graph.nodes)
        lineno2node_map = self.get_lineno2node_map()
        self.tag_nodes()
        
        self.compute_message_degree(node_relation, lineno2node_map)
        
        self.compute_split_points(node_relation, lineno2node_map)

        for node in self.traced.graph.nodes:
            print(node.lineno, node.name, node.message_degree, node.split_points)
        #TODO: refactor the graph