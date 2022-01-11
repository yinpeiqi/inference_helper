from graph_replicator import GraphReplicator


class Schema():
    def __init__(self):
        self.layers = []
        self.graphs = []
        self.arg_map = {}
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
        if name not in self.arg_map:
            self.arg_map[name] = ArgNode(name)
        input_arg = self.arg_map[name]
        self.curr_layer.add_input(input_arg)
        input_arg.add_layer(self.curr_layer)

    def record_inputs(self, names):
        for name in names:
            self.record_input(name)

    def record_output(self, name):
        if name in self.arg_map:
            raise RuntimeError("The output name is used before!")
        output_arg = ArgNode(name, self.curr_layer)
        self.arg_map[name] = output_arg
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
