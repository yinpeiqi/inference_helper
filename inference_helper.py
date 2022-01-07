from collections import namedtuple
import types

import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLHeteroGraph
from torch.fx import GraphModule, Tracer, Graph, Node


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
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str):
        return True


class InferenceHelper(nn.Module):

    def __init__(self, module: nn.Module, hidden_features, out_features):
        super().__init__()
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.layers = []
        for str in module.__dict__:
            if hasattr(module, str):
                attr = getattr(module, str)
                setattr(self, str, attr)
        self._generate_conv(module)


    def _generate_conv(self, module: nn.Module):
        # trace the module with DGLTracer
        traced = GraphModule(module, _ProhibitCallModuleTracer().trace(module))

        # generate a dependency map
        dep_map = self._trace_dependence(traced.graph.nodes)

        # here we assume the first arg in forward() is a list of DGLHeteroGraph.
        graph_layers : list[Graph] = []
        curr_layer = -1
        env = {}
        # prepare for the first conv function
        graphs_node, curr_output = self._init_input_args(traced)
        curr_input = ()

        # enumerate the whole graph
        for i, n in enumerate(traced.graph.nodes):
            # find a new conv, which likes %getitem = call_function[target=getitem](args = (%blocks, 0), kwargs = {})
            if n.op == 'call_function' and n.target.__name__ == "getitem" and n.args[0].name == graphs_node.name:
                if n.args[1] > curr_layer:
                    if (curr_layer >= 0):
                        # output the node still in dependency map, but not in this input
                        nodes_for_output = [node for node in curr_output if node not in curr_input]
                        graph_layers[-1].output(nodes_for_output[0] if len(nodes_for_output) == 1 else tuple(nodes_for_output))
                        # record the name for inference
                        self.layers[-1].outputs.extend([node.name for node in nodes_for_output])
                    curr_layer += 1
                    # generate a new graph
                    self.layers.append(InferenceSchema(inputs=[], outputs=[]))
                    graph_layers.append(Graph())
                    env = {}
                    curr_input = curr_output
                    # insert the blocks and node still needs
                    for node in (graphs_node,) + curr_output:
                        self.layers[-1].inputs.append(node.name)
                        new_node = graph_layers[-1].placeholder(node.name)
                        env[new_node.name] = new_node
                n.replace_all_uses_with(env[graphs_node.name])
                continue
            # the last output
            if n.op == 'output':
                self.layers[-1].outputs.extend([node.name for node in curr_output])
                graph_layers[-1].output(curr_output[0] if len(curr_output) == 1 else tuple(curr_output))
            # TODO not handle get_attr
            if n.op not in ('placeholder', 'output'):
                # trans the 'owning module' in args
                new_args = _arg_transform(env, n.args)
                # create a new node, the same as the one in traced graph
                created_node = graph_layers[-1].create_node(n.op, n.target, new_args, n.kwargs, n.name)
                env[created_node.name] = created_node
                # remove nodes accourding to the dependency map
                curr_output = tuple(out_node for out_node in curr_output if dep_map[out_node.name] > i)
                curr_output = curr_output + (created_node,)

        # register functions
        for i, layer in enumerate(graph_layers):
            # check lint
            layer.lint()
            # get source code
            layer_src = layer.python_code("self").src
            func_name = "forward_conv{}".format(i)
            layer_src = layer_src.replace("def forward", "def {}".format(func_name))
            self._set_function_from_string(layer_src, func_name)

    def _trace_dependence(self, node_list):
        dep_map = {}
        for i, n in enumerate(node_list):
            if i == 0: # not for blocks
                continue
            used_args = _arg_trace(n.args)
            for arg_name in used_args:
                dep_map[arg_name] = i
        return dep_map

    def _init_input_args(self, traced: GraphModule):
        args = ()
        for i, n in enumerate(traced.graph.nodes):
            if i == 0:
                graph_node = n
            elif n.op == 'placeholder':
                args = args + (n,)
            else:
                break
        return graph_node, args

    def _set_function_from_string(self, func_src, func_name):
        d = dict(locals(), **globals())
        exec(func_src, d)
        setattr(self, func_name, types.MethodType(d[func_name], self))

    def inference(self, inference_graph, batch_size, device, *args):
        # prepare for the first layer input
        next_inputs = (inference_graph,) + tuple(args)
        args_map = {}
        if len(next_inputs) != len(self.layers[0].inputs):
            raise Exception("layer's input not match with args.")
        for arg, name in zip(next_inputs, self.layers[0].inputs):
            args_map[name] = arg
        
        # enumerate layers
        for l, layer in enumerate(self.layers):
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                inference_graph, torch.arange(inference_graph.number_of_nodes()), sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False)
            # define returns
            rets = ()
            for _ in layer.outputs:
                # TODO determine type
                rets = rets + (torch.zeros(inference_graph.number_of_nodes(),
                                self.hidden_features
                                if l != len(self.layers) - 1
                                else self.out_features),)
            # for loop prepare data
            for input_nodes, output_nodes, blocks in dataloader:
                # handle inputs
                new_args = ()
                for arg_name in layer.inputs:
                    if isinstance(args_map[arg_name], torch.Tensor):
                        new_args = new_args + (args_map[arg_name][input_nodes].to(device),)
                    elif isinstance(args_map[arg_name], DGLHeteroGraph):
                        new_args = new_args + (blocks[0].to(device),)
                    elif hasattr(args_map[arg_name], "to"):
                        new_args = new_args + (args_map[arg_name].to(device),)
                    else:
                        new_args = new_args + (args_map[arg_name],)

                # run the conv function for this layer
                func = getattr(self, "forward_conv{}".format(l))
                output_vals = func(*new_args)

                # write output to rets
                output_vals = (output_vals,) if not isinstance(output_vals, tuple) else output_vals
                if len(output_vals) != len(rets):
                    raise Exception("output values not match with rets.")
                for output_val, ret in zip(output_vals, rets):
                    if isinstance(output_val, torch.Tensor):
                        ret[output_nodes] = output_val.cpu()
                    else:
                        # TODO
                        raise NotImplementedError("only support tensor for output now")

            # record outputs into args_map
            if len(output_vals) != len(rets):
                raise Exception("layer's output not match with rets.")
            for ret, name in zip(rets, layer.outputs):
                args_map[name] = ret
        
        # return
        if len(rets) == 1:
            return rets[0]
        return rets
