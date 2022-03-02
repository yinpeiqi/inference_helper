from torch.fx import GraphModule

from .dglfx.node_relation import get_node_relation
from .graph_replicator import GraphReplicator
from .constants import CALL_METHOD, CALL_MODULE, DGL_GRAPH, DGL_GRAPH_DATA, DGL_VOID_CALL, UTIL_DATA, \
    OUTPUT, PLACEHOLDER


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
                if node.node_type != DGL_GRAPH:
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
        for node in nodes:
            for oe in node.out_edges:
                oe.dst.message_degree = max(oe.dst.message_degree, node.message_degree + oe.dst.is_message)
            for ie in node.in_edges:
                if ie.src.message_degree == -1 or ie.src.is_graph_function:
                    ie.src.message_degree = node.message_degree

        # graph function's output only belongs to one layer
        for node in nodes:
            if node.is_graph_function or node.node_type == UTIL_DATA:
                change_list = [node]
                update_count = node.message_degree
                for next_node in change_list:
                    for oe in next_node.out_edges:
                        if oe.dst.message_degree > node.message_degree:
                            update_count = oe.dst.message_degree
                        elif not oe.dst.is_message and oe.dst.message_degree == node.message_degree:
                            change_list.append(oe.dst)
                for next_node in change_list:
                    next_node.message_degree = update_count
        # connect hard links
        travelled = [False for _ in nodes]
        for node in nodes:
            def find(n):
                ret = [n]
                travelled[n.lineno] = True
                for e in n.in_edges:
                    if not travelled[e.src.lineno] and not e.allow_break:
                        ret.extend(find(e.src))
                for e in n.out_edges:
                    if not travelled[e.dst.lineno] and not e.allow_break:
                        ret.extend(find(e.dst))
                return ret
            if not travelled[node.lineno]:
                node_set = find(node)
                max_degree = max(n.message_degree for n in node_set)
                for n in node_set:
                    n.message_degree = max_degree
        # remove the first layer
        for node in nodes:
            if node.message_degree > 0:
                node.message_degree -= 1

    def generate_new_graphs(self, nodes):
        message_layers = [[] for _ in range(self.output.message_degree + 1)]
        layers_input = [set() for _ in range(self.output.message_degree + 1)]
        layers_output = [set() for _ in range(self.output.message_degree + 1)]
        for node in nodes:
            message_layers[node.message_degree].append(node)
            for e in node.out_edges:
                if node.message_degree != e.dst.message_degree:
                    layers_input[e.dst.message_degree].add(node)
                    if node.node_type != DGL_GRAPH:
                        layers_output[node.message_degree].add(node)

        for i, (inputs, nodes, outputs) in enumerate(zip(layers_input, message_layers, layers_output)):
            curr_graph = GraphReplicator()
            for input_node in inputs:
                curr_graph.insert_input(input_node.name)
            for node in nodes:
                curr_graph.insert_node_copy(node.node)
            if i != self.output.message_degree:
                curr_graph.insert_output(outputs)
            curr_graph.lint()
            self.graphs_list.append(curr_graph)

    def get_splited_graphs(self):
        return self.graphs_list

    def rearrange(self):
        node_relation = get_node_relation(self.traced.graph.nodes)
        self.tag_nodes(node_relation)

        self.compute_message_degree(node_relation)

        self.generate_new_graphs(node_relation)
