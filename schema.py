from operator import getitem

from node_iterator import TaggedNodeIterator
from graph_replicator import GraphReplicator
from constants import CALL_FUNCTION, PLACEHOLDER, OUTPUT

def generate_schema(node_list):
    schema = Schema()
    curr_graph = schema.create_graph()

    node_iterator = TaggedNodeIterator(node_list)
    for n, is_split in node_iterator:
        if is_split:
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


class Schema():
    def __init__(self):
        self.layers = []
        self.graphs = []
        self.name2arg_map = {}
        self.blocks_name = None

    def create_graph(self):
        self.layers.append(GraphLayer(self))
        self.graphs.append(GraphReplicator())
        return self.curr_graph

    def get_layer(self, id):
        return self.layers[id]

    def get_graph(self, id):
        return self.graphs[id]

    def record_input(self, name):
        if self.blocks_name is None:
            self.blocks_name = name
        if name not in self.name2arg_map:
            self.name2arg_map[name] = ArgNode(name)
        input_arg = self.name2arg_map[name]
        self.curr_layer.add_input(input_arg)
        input_arg.add_layer(self.curr_layer)

    def record_inputs(self, names):
        for name in names:
            self.record_input(name)

    def record_output(self, name):
        if name in self.name2arg_map:
            raise RuntimeError("The output name is used before!")
        output_arg = ArgNode(name, self.curr_layer)
        self.name2arg_map[name] = output_arg
        self.curr_layer.add_output(output_arg)

    def record_outputs(self, nodes):
        for node in nodes:
            self.record_output(node.name)

    @property
    def curr_layer(self):
        return self.layers[-1]

    @property
    def curr_graph(self):
        return self.graphs[-1]

    @property
    def graphs_count(self):
        return len(self.graphs)


class GraphLayer():
    def __init__(self, schema: Schema):
        super().__init__()
        self.schema = schema
        self.id = schema.graphs_count
        self.inputs: list[ArgNode] = []
        self.outputs: list[ArgNode] = []

    def add_input(self, input_arg):
        self.inputs.append(input_arg)

    def add_output(self, output_arg):
        self.outputs.append(output_arg)


class ArgNode():
    # Arg is always create by layer output, or the init
    def __init__(self, name: str, output_layer: GraphLayer = None):
        self.name = name
        self.input_layers = []
        self.output_layer = output_layer

    def add_layer(self, layer: GraphLayer):
        self.input_layers.append(layer)
    
    def __str__(self):
        return "{}, input: {}, output: {}".format(self.name, [layer.id for layer in self.input_layers], self.output_layer)

    @property
    def last_used_layer(self):
        return self.input_layers[-1]
