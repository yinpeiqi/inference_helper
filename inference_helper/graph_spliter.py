from operator import getitem

from .node_iterator import TaggedNodeIterator
from .schema import Schema
from .constants import CALL_FUNCTION, PLACEHOLDER, OUTPUT


class GraphSpliter():
    def __init__(self, node_list):
        self.node_list = node_list

    def split_graph(self):
        schema = Schema()
        curr_graph = schema.create_graph()

        node_iterator = TaggedNodeIterator(self.node_list)
        split_linenos = self.get_split_linenos()

        for lineno, n in enumerate(node_iterator):
            if n.op != OUTPUT and n.name not in node_iterator.dep_map:
                continue

            if lineno in split_linenos:
                curr_inputs, curr_outputs = node_iterator.get_curr_input_and_output()
                output_nodes = [curr_graph.env[node_name] for node_name in curr_outputs]

                schema.record_outputs(output_nodes)
                curr_graph.insert_output(output_nodes)

                curr_graph = schema.create_graph()

                schema.record_inputs(curr_inputs)
                curr_graph.insert_inputs(curr_inputs)

            if n.op == CALL_FUNCTION and n.target == getitem and n.args[0].name == schema.blocks_name:
                n.replace_all_uses_with(curr_graph.env[schema.blocks_name])
                continue

            if n.op == PLACEHOLDER:
                schema.record_input(n.name)
                curr_graph.insert_input(n.name)
            elif n.op == OUTPUT:
                schema.record_outputs(n.args)
                curr_graph.insert_output(n.args)
            else:
                curr_graph.insert_node_copy(n)

        for graph in schema.graphs:
            graph.lint()

        return schema

    def get_split_linenos(self):
        blocks_name = None
        graph_layer_set = set()
        self.lineno = 0

        split_linenos = set()
        for lineno, node in enumerate(self.node_list):
            if node.op == CALL_FUNCTION and node.target == getitem and node.args[0].name == blocks_name:
                if node.args[1] not in graph_layer_set and len(graph_layer_set) != 0:
                    split_linenos.add(lineno)
                graph_layer_set.add(node.args[1])

            if node.op == PLACEHOLDER:
                if blocks_name is None:
                    blocks_name = node.name
        return split_linenos
