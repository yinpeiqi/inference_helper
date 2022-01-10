from collections import namedtuple
from operator import getitem
from typing import Iterator
import types

import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLHeteroGraph
from torch.fx import GraphModule, Tracer, Graph, Node, Proxy


InferenceSchema = namedtuple("InferenceSchema", ["inputs", "outputs"])

def _arg_transform(env, args):
    new_args = ()
    for arg in args:
        if isinstance(arg, Node):
            new_arg = env[arg.name]
        elif isinstance(arg, slice):
            new_arg = slice(
                arg.start if not isinstance(arg.start, Node) else env[arg.start.name],
                arg.step if not isinstance(arg.step, Node) else env[arg.step.name],
                arg.stop if not isinstance(arg.stop, Node) else env[arg.stop.name]
            )
        elif isinstance(arg, tuple):
            new_arg = _arg_transform(env, arg)
        else:
            new_arg = arg
        new_args += (new_arg,)
    return new_args

def _arg_trace(args):
    ret = set()
    for arg in args:
        if isinstance(arg, Node):
            ret.add(arg.name)
        if isinstance(arg, slice):
            ret = ret.union(_arg_trace((arg.start, arg.step, arg.stop)))
        if isinstance(arg, tuple):
            ret = ret.union(_arg_trace(arg))
    return ret


class _ProhibitCallModuleTracer(Tracer):

    def iter(self, obj: 'Proxy') -> Iterator:
        idx = -1
        while True:
            idx += 1
            yield obj[idx]

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str):
        return True


class InferenceHelper(nn.Module):

    def __init__(self, module: nn.Module, debug = False):
        super().__init__()
        # add a '_' in order not crash with the origin one.
        self._layers = []
        self._debug = debug
        for name in module.__dict__:
            if hasattr(module, name):
                attr = getattr(module, name)
                setattr(self, name, attr)
        self._generate_conv(module)


    def _generate_conv(self, module: nn.Module):
        # trace the module with DGLTracer
        traced = GraphModule(module, _ProhibitCallModuleTracer().trace(module))
        if self._debug:
            print("-------- Origin forward function -------")
            print(traced.code.strip())
            print("----------------------------------------")
        # generate a dependency map
        dep_map = self._trace_dependence(traced.graph.nodes)
        # get the place we want to split the forward function
        split_linenos = self._conv_split(traced.graph.nodes)

        # here we assume the first arg in forward() is a list of DGLHeteroGraph.
        graph_layers : list[Graph] = []
        curr_layer = -1
        env = {}
        # prepare for the first conv function
        graphs_node, curr_output = self._init_input_args(traced)
        curr_input = ()

        # enumerate the whole graph
        for lineno, n in enumerate(traced.graph.nodes):
            # unused check
            if n.op != 'output' and n.name not in dep_map:
                continue

            # find a new conv, which likes %getitem = call_function[target=getitem](args = (%blocks, 0), kwargs = {})
            if lineno in split_linenos:
                if (curr_layer >= 0):
                    # output the node still in dependency map, but not in this input
                    output_nodes = [node for node in curr_output if node not in curr_input]
                    graph_layers[-1].output(output_nodes[0] if len(output_nodes) == 1 else tuple(output_nodes))
                    # record the name for inference
                    self._layers[-1].outputs.extend([node.name for node in output_nodes])
                curr_layer += 1
                # generate a new graph
                self._layers.append(InferenceSchema(inputs=[], outputs=[]))
                graph_layers.append(Graph())
                env = {}
                curr_input = curr_output
                # insert the blocks and node still needs
                for node in (graphs_node,) + curr_output:
                    self._layers[-1].inputs.append(node.name)
                    new_node = graph_layers[-1].placeholder(node.name)
                    env[new_node.name] = new_node

            if n.op == 'call_function' and n.target == getitem and n.args[0].name == graphs_node.name:
                n.replace_all_uses_with(env[graphs_node.name])
                continue

            # the last output
            if n.op == 'output':
                self._layers[-1].outputs.extend([node.name for node in curr_output])
                graph_layers[-1].output(curr_output[0] if len(curr_output) == 1 else tuple(curr_output))

            if n.op not in ('output', 'placeholder'):
                # trans the 'owning module' in args
                new_args = _arg_transform(env, n.args)
                # create a new node, the same as the one in traced graph
                created_node = graph_layers[-1].create_node(n.op, n.target, new_args, n.kwargs, n.name)
                env[created_node.name] = created_node
                # remove nodes accourding to the dependency map
                curr_output = tuple(out_node for out_node in curr_output if dep_map[out_node.name] > lineno)
                curr_output = curr_output + (created_node,)

        # register functions
        for layer_id, graph in enumerate(graph_layers):
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

    def _trace_dependence(self, node_list):
        dep_map = {}
        for lineno, node in enumerate(node_list):
            if lineno == 0: # not for blocks
                continue
            used_args = _arg_trace(node.args)
            for arg_name in used_args:
                dep_map[arg_name] = lineno
        return dep_map

    def _conv_split(self, node_list):
        graphs_name = None
        first_not_placeholder = -1
        graph_layer_set = set()
        split_linenos = []
        for lineno, node in enumerate(node_list):
            if graphs_name is None and node.op == 'placeholder':
                # we assume the first node is placeholder for graphs
                graphs_name = node.name
            elif first_not_placeholder == -1 and node.op != 'placeholder':
                first_not_placeholder = lineno
            if node.op == 'call_function' and node.target == getitem and node.args[0].name == graphs_name:
                curr_layer = node.args[1]
                if curr_layer not in graph_layer_set:
                    split_linenos.append(lineno)
                    graph_layer_set.add(curr_layer)
        split_linenos[0] = first_not_placeholder
        return split_linenos

    def _init_input_args(self, traced: GraphModule):
        args = ()
        for lineno, node in enumerate(traced.graph.nodes):
            if lineno == 0:
                graph_node = node
            elif node.op == 'placeholder':
                args = args + (node,)
            else:
                break
        return graph_node, args

    def _set_function_from_string(self, func_src, func_name):
        globals_vals = globals()
        exec(func_src, globals_vals)
        setattr(self, func_name, types.MethodType(globals_vals[func_name], self))

    def _get_new_arg_input(self, inputs, args_map, input_nodes, inference_graph, device):
        new_args = ()
        for arg_name in inputs:
            if isinstance(args_map[arg_name], torch.Tensor):
                new_args += (args_map[arg_name][input_nodes].to(device),)
            elif isinstance(args_map[arg_name], DGLHeteroGraph):
                new_args += (inference_graph.to(device),)
            elif hasattr(args_map[arg_name], "to"):
                new_args += (args_map[arg_name].to(device),)
            else:
                new_args += (args_map[arg_name],)
        return new_args

    def _trace_output_shape(self, first_layer_inputs, device):
        args_map = {}
        if len(first_layer_inputs) != len(self._layers[0].inputs):
            raise Exception("layer's input not match with args.")
        for arg, name in zip(first_layer_inputs, self._layers[0].inputs):
            args_map[name] = arg

        ret_shapes = [[] for _ in self._layers]
        # enumerate layers
        for layer_id, layer in enumerate(self._layers):
            new_args = self._get_new_arg_input(layer.inputs, args_map, [0], dgl.graph((torch.tensor([0]), torch.tensor([0]))), device)

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

    def inference(self, inference_graph, batch_size, device, *args):
        torch.set_grad_enabled(False)
        # prepare for the first layer input
        first_layer_inputs = (inference_graph,) + tuple(args)
        ret_shapes = self._trace_output_shape(first_layer_inputs, device)
        args_map = {}
        for arg, name in zip(first_layer_inputs, self._layers[0].inputs):
            args_map[name] = arg

        # enumerate layers
        for layer_id, layer in enumerate(self._layers):
            # TODO customize sampler and dataloader
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                inference_graph, torch.arange(inference_graph.number_of_nodes()), sampler,
                batch_size=batch_size,
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
                new_args = self._get_new_arg_input(layer.inputs, args_map, input_nodes, blocks[0], device)
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
