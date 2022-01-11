import types

import dgl
import torch
import torch.nn as nn
from dgl import DGLHeteroGraph
from torch.fx import GraphModule, Graph, Node
from memory_profiler import profile
import tqdm

from schema import Schema
from conv_generator import GraphSpliter
from tracer import ProhibitCallModuleTracer


class InferenceHelper(nn.Module):

    def __init__(self, module: nn.Module, batch_size, device, debug = False):
        super().__init__()
        # add a '_' in order not crash with the origin one.
        self._layers = []
        self._batch_size = batch_size
        self._device = device
        self._debug = debug
        for name in module.__dict__:
            if hasattr(module, name):
                attr = getattr(module, name)
                setattr(self, name, attr)
        self._generate_conv(module)

    def _generate_conv(self, module: nn.Module):
        # trace the module with DGLTracer
        traced = GraphModule(module, ProhibitCallModuleTracer().trace(module))
        if self._debug:
            print("-------- Origin forward function -------")
            print(traced.code.strip())
            print("----------------------------------------")
        
        graph_spliter = GraphSpliter(traced.graph.nodes)
        schema = graph_spliter.generate_conv()
        # register functions
        for layer_id, graph in enumerate(schema.graphs):
            graph.lint()
            # register function from graph
            self._register_func_from_graph(graph, layer_id)

    def _register_func_from_graph(self, graph: Graph, layer_id: int):
        # get source code
        graph_src = graph.python_code("self").src
        func_name = "forward_conv{}".format(layer_id)
        # replace the function name
        graph_src = graph_src.replace("def forward(", "def {}(".format(func_name))
        # replace the getattr function
        graph_src = graph_src.replace(" getattr(", " InferenceHelper.inference_helper_getattr(")
        self._set_function_from_string(graph_src, func_name)
        if self._debug:
            print("--------- Layer {} conv function --------".format(layer_id))
            print(graph_src.strip())
            print("----------------------------------------")

    @staticmethod
    def inference_helper_getattr(obj, name: str):
        if name.isnumeric():
            return obj[int(name)]
        return getattr(obj, name)

    def _set_function_from_string(self, func_src, func_name):
        globals_vals = globals()
        exec(func_src, globals_vals)
        setattr(self, func_name, types.MethodType(globals_vals[func_name], self))

    def _get_new_arg_input(self, inputs, args_map, input_nodes, inference_graph):
        new_args = ()
        for arg_name in inputs:
            if isinstance(args_map[arg_name], torch.Tensor):
                new_args += (args_map[arg_name][input_nodes].to(self._device),)
            elif isinstance(args_map[arg_name], DGLHeteroGraph):
                new_args += (inference_graph.to(self._device),)
            elif hasattr(args_map[arg_name], "to"):
                new_args += (args_map[arg_name].to(self._device),)
            else:
                new_args += (args_map[arg_name],)
        return new_args

    def _trace_output_shape(self, first_layer_inputs):
        args_map = {}
        if len(first_layer_inputs) != len(self._layers[0].inputs):
            raise Exception("layer's input not match with args.")
        for arg, name in zip(first_layer_inputs, self._layers[0].inputs):
            args_map[name] = arg

        ret_shapes = [[] for _ in self._layers]
        # enumerate layers
        for layer_id, layer in enumerate(self._layers):
            new_args = self._get_new_arg_input(layer.inputs, args_map, [0], dgl.graph((torch.tensor([0]), torch.tensor([0]))))

            func = getattr(self, "forward_conv{}".format(layer_id))
            output_vals = func(*new_args)
            output_vals = (output_vals,) if not isinstance(output_vals, tuple) else output_vals
            if len(output_vals) != len(layer.outputs):
                raise Exception("output values not match with layer's output.")
            for val, name in zip(output_vals, layer.outputs):
                if not isinstance(val, torch.Tensor):
                    raise NotImplementedError("only support tensor for output now")
                args_map[name] = val.cpu()
                ret_shapes[layer_id].append(val.size()[1:])
        return ret_shapes

    @profile
    def inference(self, inference_graph, *args):
        torch.set_grad_enabled(False)
        # prepare for the first layer input
        first_layer_inputs = (inference_graph,) + tuple(args)
        ret_shapes = self._trace_output_shape(first_layer_inputs)
        args_map = {}
        for arg, name in zip(first_layer_inputs, self._layers[0].inputs):
            args_map[name] = arg

        # enumerate layers
        for layer_id, layer in enumerate(self._layers):
            # TODO customize sampler and dataloader
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                inference_graph, torch.arange(inference_graph.number_of_nodes()), sampler,
                batch_size=self._batch_size,
                shuffle=False,
                drop_last=False)

            # define returns
            rets = ()
            for j, _ in enumerate(layer.outputs):
                # TODO determine type
                rets = rets + (torch.zeros(
                    (inference_graph.number_of_nodes(),) + tuple(ret_shapes[layer_id][j])
                ),)

            # for loop prepare data
            for input_nodes, output_nodes, blocks in dataloader:
                new_args = self._get_new_arg_input(layer.inputs, args_map, input_nodes, blocks[0])
                # run the conv function for this layer
                func = getattr(self, "forward_conv{}".format(layer_id))
                output_vals = func(*new_args)

                # write output to rets
                output_vals = (output_vals,) if not isinstance(output_vals, tuple) else output_vals
                for output_val, ret in zip(output_vals, rets):
                    if isinstance(output_val, torch.Tensor):
                        ret[output_nodes] = output_val.cpu()

            # record outputs into args_map
            for ret, name in zip(rets, layer.outputs):
                args_map[name] = ret

        # return
        if len(rets) == 1:
            return rets[0]
        return rets
