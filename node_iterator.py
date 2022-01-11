from utils import arg_trace

class TaggedNodeIterator():
    def __init__(self, node_list):
        self.node_list = node_list
        self.dep_map = self.trace_dependence()
        self.curr_outputs = []
        self.curr_inputs = []

    def __iter__(self):
        self.lineno = 0
        for lineno, node in enumerate(self.node_list):
            if node.op != 'output' and node.name not in self.dep_map:
                continue
            yield node
            self.lineno = lineno
            if node.op == 'placeholder':
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
