from operator import getitem
from torch.fx import GraphModule

from utils import arg_trace, arg_transform
from schema import Schema
from node_iterator import TaggedNodeIterator


class GraphSpliter():
    def __init__(self, traced: GraphModule):
        self.node_list = traced.graph.nodes
        self.schema = None

    def generate_conv(self):
        schema = Schema()
        curr_graph = schema.create_graph()

        # get the place we want to split the forward function
        split_linenos = self.get_split_lineno()

        node_iterator = TaggedNodeIterator(self.node_list)
        for lineno, n in enumerate(node_iterator):
            if lineno in split_linenos:
                curr_inputs, curr_outputs = node_iterator.get_curr_input_and_output()
                output_nodes = [curr_graph.env[node_name] for node_name in curr_outputs]

                schema.record_outputs(output_nodes)
                curr_graph.insert_output(output_nodes)

                curr_graph = schema.create_graph()

                schema.record_inputs(curr_inputs)
                curr_graph.insert_inputs(curr_inputs)

            if n.op == 'call_function' and n.target == getitem and n.args[0].name == schema.blocks_name:
                n.replace_all_uses_with(curr_graph.env[schema.blocks_name])
                continue

            if n.op == 'placeholder':
                schema.record_input(n.name)
                curr_graph.insert_input(n.name)
            elif n.op == 'output':
                schema.record_outputs(n.args)
                curr_graph.insert_output(n.args)
            else:
                curr_graph.insert_node_copy(n)

        for graph in schema.graphs:
            graph.lint()

        return schema

    def get_split_lineno(self):
        graphs_name = None
        graph_layer_set = set()
        split_linenos = []
        for lineno, node in enumerate(self.node_list):
            if graphs_name is None and node.op == 'placeholder':
                # we assume the first node is placeholder for graphs
                graphs_name = node.name
            elif node.op == 'call_function' and node.target == getitem and node.args[0].name == graphs_name:
                curr_layer = node.args[1]
                if curr_layer not in graph_layer_set:
                    split_linenos.append(lineno)
                    graph_layer_set.add(curr_layer)
        # remove the first one
        return split_linenos[1:]
