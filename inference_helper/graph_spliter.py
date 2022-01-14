from .graph_replicator import GraphReplicator
from .node_iterator import TaggedNodeIterator
from .constants import PLACEHOLDER, OUTPUT


class GraphSpliter():
    def __init__(self, node_list):
        self.node_list = node_list
        self.graphs_list = []

    def create_graph(self):
        self.graphs_list.append(GraphReplicator())
        return self.graphs_list[-1]

    def split_graph(self, split_linenos):
        curr_graph = self.create_graph()

        node_iterator = TaggedNodeIterator(self.node_list)

        for lineno, n in enumerate(node_iterator):
            if lineno in split_linenos:
                curr_inputs, curr_outputs = node_iterator.get_curr_input_and_output()
                output_nodes = [curr_graph.env[node_name] for node_name in curr_outputs]
                curr_graph.insert_output(output_nodes)

                curr_graph = self.create_graph()
                curr_graph.insert_inputs(curr_inputs)

            if n.op == PLACEHOLDER:
                curr_graph.insert_input(n.name)
            elif n.op == OUTPUT:
                curr_graph.insert_output(n.args)
            else:
                curr_graph.insert_node_copy(n)

        for graph in self.graphs_list:
            graph.lint()
        return self.graphs_list
