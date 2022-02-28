from torch.fx import GraphModule

from .utils import arg_trace
from .node_relation import get_node_relation
from .graph_replicator import GraphReplicator
from .constants import CALL_METHOD, CALL_MODULE, DGL_GRAPH, DGL_GRAPH_DATA, DGL_VOID_CALL, OUTPUT, PLACEHOLDER


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

    def tag_nodes(self, nodes):
        for i, node in enumerate(nodes):
            self.tagging_node(i, node)
            if node.op == PLACEHOLDER:
                self.inputs.append(node)
                node.is_input = True
                node.message_degree = 0
            if node.op == CALL_MODULE:
                for e in node.in_edges:
                    if e.src.node_type == DGL_GRAPH:
                        node.is_message = True
            if node.node_type == DGL_VOID_CALL and node.target == "update_all":
                node.is_message = True
            if node.op == CALL_METHOD and node.node_type == DGL_GRAPH_DATA:
                node.is_graph_function = True
            if node.op == OUTPUT:
                self.output = node

    def compute_message_degree(self, nodes):
        node_queue = []
        for node in nodes:
            if node.op == PLACEHOLDER:
                node_queue.append(node)
        for node in node_queue:
            for ie in node.in_edges:
                if ie.src.message_degree == -1:
                    continue
            for ie in node.in_edges:
                node.message_degree = max(node.message_degree, ie.src.message_degree + node.is_message)
            for oe in node.out_edges:
                node_queue.append(oe.dst)

        # graph function's output only belongs to one layer
        for node in nodes:
            if node.is_graph_function:
                change_list = [node]
                message_layer = node.message_degree
                for next_node in change_list:
                    for oe in next_node.out_edges:
                        if oe.dst.is_message:
                            message_layer = oe.dst.message_degree
                        else:
                            change_list.append(oe.dst)
                for next_node in change_list:
                    next_node.message_degree = message_layer
        self.output.message_degree += 1

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
        self.tag_nodes(node_relation)

        self.compute_message_degree(node_relation)
        for node in node_relation:
            print(node, node.message_degree)

        self.generate_new_graphs(node_relation)
