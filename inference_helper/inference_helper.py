import types

import time
import dgl
import torch
import torch.nn as nn
from dgl import DGLHeteroGraph
from torch.fx import GraphModule, Graph, Node
import tqdm

from .schema import generate_schema
from .tracer import ProhibitCallModuleTracer
from .utils import inference_helper_getattr
from .constants import FORWARD_CONV


class InferenceHelper(nn.Module):
    def __init__(self, module: nn.Module, batch_size, device, num_workers = 4, debug = False):
        super().__init__()
        # add a '_' in order not crash with the origin one.
        self._batch_size = batch_size
        self._device = device
        self._num_workers = num_workers
        self._debug = debug
        self._schema = None
        for name in module.__dict__:
            if hasattr(module, name):
                attr = getattr(module, name)
                setattr(self, name, attr)
        self._generate_conv(module)

    def _generate_conv(self, module: nn.Module):
        traced = GraphModule(module, ProhibitCallModuleTracer().trace(module))
        if self._debug:
            print("-------- Origin forward function -------")
            print(traced.code.strip())
            print("----------------------------------------")

        self._schema = generate_schema(traced.graph.nodes)

        for layer_id, graph in enumerate(self._schema.graphs):
            self._register_func_from_graph(graph, layer_id)

    def _register_func_from_graph(self, graph: Graph, layer_id: int):
        graph_src = graph.python_code("self").src

        func_name = FORWARD_CONV + str(layer_id)
        graph_src = graph_src.replace("def forward(", "def {}(".format(func_name))
        graph_src = graph_src.replace(" getattr(", " inference_helper_getattr(")
        self._set_function_from_string(graph_src, func_name)

        if self._debug:
            print("--------- Layer {} conv function --------".format(layer_id))
            print(graph_src.strip())
            print("----------------------------------------")

    def _set_function_from_string(self, func_src, func_name):
        globals_vals = globals()
        exec(func_src, globals_vals)
        setattr(self, func_name, types.MethodType(globals_vals[func_name], self))

    def _get_new_arg_input(self, inputs, arg2val_map, input_nodes, inference_graph):
        new_args = ()
        for arg_node in inputs:
            if isinstance(arg2val_map[arg_node], torch.Tensor):
                new_args += (arg2val_map[arg_node][input_nodes].to(self._device),)
            elif isinstance(arg2val_map[arg_node], DGLHeteroGraph):
                new_args += (inference_graph.int().to(self._device),)
            elif hasattr(arg2val_map[arg_node], "to"):
                new_args += (arg2val_map[arg_node].to(self._device),)
            else:
                new_args += (arg2val_map[arg_node],)
        return new_args

    def _trace_output_shape(self, first_layer_inputs):
        arg2val_map = {}
        if len(first_layer_inputs) != len(self._schema.get_layer(0).inputs):
            raise Exception("layer's input not match with args.")
        for val, arg_node in zip(first_layer_inputs, self._schema.get_layer(0).inputs):
            arg2val_map[arg_node] = val

        ret_shapes = [[] for _ in range(self._schema.graphs_count)]
        for layer in self._schema.layers:
            new_args = self._get_new_arg_input(layer.inputs, arg2val_map, [0], dgl.graph((torch.tensor([0]), torch.tensor([0]))))

            func = getattr(self, FORWARD_CONV + str(layer.id))
            output_vals = func(*new_args)
            output_vals = (output_vals,) if not isinstance(output_vals, tuple) else output_vals
            if len(output_vals) != len(layer.outputs):
                raise Exception("output values not match with layer's output.")
            for val, arg_node in zip(output_vals, layer.outputs):
                if not isinstance(val, torch.Tensor):
                    raise NotImplementedError("only support tensor for output now")
                arg2val_map[arg_node] = val.cpu()
                ret_shapes[layer.id].append(val.size()[1:])
        return ret_shapes

    def inference(self, inference_graph, *args):
        first_layer_inputs = (inference_graph,) + tuple(args)
        ret_shapes = self._trace_output_shape(first_layer_inputs)
        arg2val_map = {}
        for val, arg_node in zip(first_layer_inputs, self._schema.get_layer(0).inputs):
            arg2val_map[arg_node] = val

        for layer in self._schema.layers:
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                inference_graph,
                torch.arange(inference_graph.number_of_nodes()).to(inference_graph.device),
                sampler,
                batch_size=self._batch_size,
                device=self._device if self._num_workers == 0 else None,
                shuffle=False,
                drop_last=False,
                num_workers=self._num_workers)

            rets = []
            for j, _ in enumerate(layer.outputs):
                rets.append(
                    torch.zeros((inference_graph.number_of_nodes(),) + tuple(ret_shapes[layer.id][j]))
                )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                new_args = self._get_new_arg_input(layer.inputs, arg2val_map, input_nodes, blocks[0])

                func = getattr(self, FORWARD_CONV + str(layer.id))
                output_vals = func(*new_args)
                del new_args

                if not isinstance(output_vals, tuple):
                    output_vals = (output_vals,)
                for output_val, ret in zip(output_vals, rets):
                    if isinstance(output_val, torch.Tensor):
                        ret[output_nodes] = output_val.cpu()

            # delete intermediate val
            for arg_node in layer.inputs:
                if arg_node.input_layers[-1] == layer and arg_node.input_layers[0] != self._schema.get_layer(0):
                    del arg2val_map[arg_node]

            for ret, arg_node in zip(rets, layer.outputs):
                arg2val_map[arg_node] = ret

        if len(rets) == 1:
            return rets[0]
        return tuple(rets)
