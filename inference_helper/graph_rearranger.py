from torch.fx import GraphModule

from .utils import arg_trace
from .node_relation import get_node_relation
from .graph_replicator import GraphReplicator
from .constants import CALL_METHOD, CALL_MODULE, OUTPUT, PLACEHOLDER


class GraphRearranger():
    def __init__(self, traced: GraphModule):
        self.traced = traced
        self.output = None
        self.inputs = []
        self.graphs_list = []

    def tagging_node(self, lineno, node):
        node.lineno = lineno
        node.is_input = False
        node.is_message = False
        node.message_degree = -1
        node.is_graph_function = False

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
        self.output.message_degree += 1

    def compute_split_points(self, node_relation, lineno2node_map):
        passing_edges = []
        for node in self.traced.graph.nodes:
            for use_lineno in node_relation[node.lineno].use:
                last_node = lineno2node_map[use_lineno]
                if node.message_degree != last_node.message_degree:
                    passing_edges.append((last_node.lineno, node.lineno))

        for edge in passing_edges:
            curr_edge = edge
            path = []
            message_layer = lineno2node_map[curr_edge[1]].message_degree
            while True:
                src_lineno, dst_lineno = curr_edge
                path.append(dst_lineno)
                if len(node_relation[src_lineno].use) != 1 or \
                    lineno2node_map[src_lineno].is_message or lineno2node_map[src_lineno].is_input:
                    break
                if len(node_relation[src_lineno].be_used) != 1:
                    is_same_source = True
                    for be_used in node_relation[src_lineno].be_used:
                        if be_used != dst_lineno and\
                            lineno2node_map[be_used].message_degree != message_layer:
                            is_same_source = False
                    if not is_same_source:
                        break
                curr_edge = (node_relation[src_lineno].use[0], src_lineno)
            for lineno in path:
                lineno2node_map[lineno].message_degree = message_layer

    def generate_new_graphs(self, node_relation, lineno2node_map):
        message_layers = [[] for _ in range(self.output.message_degree + 1)]
        layers_input = [set() for _ in range(self.output.message_degree + 1)]
        layers_output = [set() for _ in range(self.output.message_degree + 1)]
        travel = [0 for _ in self.traced.graph.nodes]
        for node in self.traced.graph.nodes:
            if node.op == PLACEHOLDER:
                message_layers[0].append(node)

        for i, layer in enumerate(message_layers):
            for node in layer:
                for lineno in node_relation[node.lineno].be_used:
                    travel[lineno] += 1
                    next_node = lineno2node_map[lineno]
                    if i != next_node.message_degree:
                        layers_input[next_node.message_degree].add(node)
                        layers_output[i].add(node)
                    if travel[lineno] == len(node_relation[lineno].use):
                        message_layers[next_node.message_degree].append(next_node)
        
        for i, (inputs, nodes, outputs) in enumerate(zip(layers_input, message_layers, layers_output)):
            curr_graph = GraphReplicator()
            for input_node in inputs:
                curr_graph.insert_input(input_node.name)
            for node in nodes:
                curr_graph.insert_node_copy(node)
            curr_graph.insert_output(outputs)
            curr_graph.lint()
            self.graphs_list.append(curr_graph)

    def get_splited_graphs(self):
        return self.graphs_list[1:-1]

    def rearrange(self):
        node_relation = get_node_relation(self.traced.graph.nodes)
        lineno2node_map = self.get_lineno2node_map()
        self.tag_nodes()
        
        self.compute_message_degree(node_relation, lineno2node_map)
        
        # self.compute_split_points(node_relation, lineno2node_map)

        self.generate_new_graphs(node_relation, lineno2node_map)

        # NodeRelation.print_node_relation(self.traced.graph.nodes)
