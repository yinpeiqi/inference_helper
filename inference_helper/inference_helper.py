import dgl
import torch
import torch.nn as nn
from dgl import DGLHeteroGraph
import tqdm

from .function_generator import FunctionGenerator


class InferenceHelper():
    def __init__(self, module: nn.Module, batch_size, device, num_workers = 4, debug = False):
        # add a '_' in order not crash with the origin one.
        self._batch_size = batch_size
        self._device = device
        self._num_workers = num_workers
        self._function_generator = FunctionGenerator(module, debug)
        self._schema = self._function_generator.get_schema()
        self._funcs = self._function_generator.get_funcs()

    def _get_new_arg_input(self, inputs, arg2val_map, input_nodes, inference_graph):
        new_args = ()
        for arg_node in inputs:
            if isinstance(arg2val_map[arg_node], torch.Tensor):
                new_args += (arg2val_map[arg_node][input_nodes].to(self._device),)
            elif isinstance(arg2val_map[arg_node], DGLHeteroGraph):
                new_args += (inference_graph.to(self._device),)
            elif hasattr(arg2val_map[arg_node], "to"):
                new_args += (arg2val_map[arg_node].to(self._device),)
            else:
                new_args += (arg2val_map[arg_node],)
        return new_args

    def _trace_output_shape(self, arg2val_map):
        ret_shapes = [[] for _ in range(self._schema.layers_count)]
        for layer, func in zip(self._schema.layers, self._funcs):
            new_args = self._get_new_arg_input(layer.inputs, arg2val_map, [0], dgl.graph((torch.tensor([0]), torch.tensor([0]))))

            output_vals = func(*new_args)
            if not isinstance(output_vals, tuple):
                output_vals = (output_vals,)
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
        if len(first_layer_inputs) != len(self._schema.first_layer_input):
            raise Exception("layer's input not match with args.")
        arg2val_map = {}
        for val, arg_name in zip(first_layer_inputs, self._schema.first_layer_input):
            arg_node = self._schema.name2arg_map[arg_name]
            arg2val_map[arg_node] = val
        ret_shapes = self._trace_output_shape(arg2val_map)

        for layer, func in zip(self._schema.layers, self._funcs):
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

                output_vals = func(*new_args)
                del new_args

                if not isinstance(output_vals, tuple):
                    output_vals = (output_vals,)
                for output_val, ret in zip(output_vals, rets):
                    if isinstance(output_val, torch.Tensor):
                        if output_val.size()[0] == blocks[0].num_dst_nodes():
                            ret[output_nodes] = output_val.cpu()
                        elif output_val.size()[0] == blocks[0].num_src_nodes():
                            ret[input_nodes] = output_val.cpu()
                        else:
                            raise RuntimeError("Can't determine return's type.")

            # delete intermediate val
            for arg_node in layer.inputs:
                if arg_node.input_layers[-1] == layer and arg_node.input_layers[0] != self._schema.get_layer(0):
                    del arg2val_map[arg_node]

            for ret, arg_node in zip(rets, layer.outputs):
                arg2val_map[arg_node] = ret

        outputs = ()
        for name in self._schema.last_layer_output:
            arg_node = self._schema.name2arg_map[name]
            outputs += (arg2val_map[arg_node],)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
