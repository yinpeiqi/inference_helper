from operator import getitem
from utils import arg_trace
from constants import CALL_FUNCTION, PLACEHOLDER, OUTPUT


class TaggedNodeIterator():
    def __init__(self, node_list):
        self.node_list = node_list
        self.dep_map = self.trace_dependence()
        self.curr_outputs = []
        self.curr_inputs = []
        self.lineno = 0

    def __iter__(self):
        blocks_name = None
        graph_layer_set = set()
        self.lineno = 0

        for lineno, node in enumerate(self.node_list):
            if node.op != OUTPUT and node.name not in self.dep_map:
                continue

            is_split = False
            if node.op == CALL_FUNCTION and node.target == getitem and node.args[0].name == blocks_name:
                if node.args[1] not in graph_layer_set and len(graph_layer_set) != 0:
                    is_split = True
                graph_layer_set.add(node.args[1])
            
            yield node, is_split

            self.lineno = lineno
            if node.op == PLACEHOLDER:
                if blocks_name is None:
                    blocks_name = node.name
                self.curr_inputs.append(node.name)
            self.curr_outputs.append(node.name)

    def get_curr_input_and_output(self):
        real_next_inputs = [node_name for node_name in self.curr_outputs if self.dep_map[node_name] > self.lineno]
        real_outputs = [node_name for node_name in real_next_inputs if node_name not in self.curr_inputs]
        self.curr_outputs = real_next_inputs.copy()
        self.curr_inputs = real_next_inputs.copy()
        return real_next_inputs, real_outputs

    def trace_dependence(self):
        dep_map = {}
        for lineno, node in enumerate(self.node_list):
            used_args = arg_trace(node.args)
            for arg_name in used_args:
                dep_map[arg_name] = lineno
        return dep_map
