import dgl
import torch
import torch.nn as nn
import tqdm
import gc

from .auto_turner import AutoTurner
from .function_generator import FunctionGenerator
from .custom_dataloader import CustomDataloader
from .module_silencer import Modulesilencer
from .utils import get_new_arg_input, update_ret_output


class InferenceHelperBase():
    def __init__(self, module: nn.Module, device, silence_modules = [], debug = False):
        # add a '_' in order not crash with the origin one.
        self._device = device
        self._function_generator = FunctionGenerator(module, debug)
        # we re register attributes to function generator.
        self._module_silencer = Modulesilencer(self._function_generator)
        self.silence_modules = silence_modules
        self.silence_modules.append(nn.Dropout)
        self._traced = self._function_generator.traced
        self._schema = self._function_generator.get_schema()
        self._funcs = self._function_generator.get_funcs()

    def _trace_output_shape(self, arg2val_map):
        ret_shapes = [[] for _ in range(self._schema.layers_count)]
        for layer, func in zip(self._schema.layers, self._funcs):
            fake_graph = dgl.graph((torch.tensor([0]), torch.tensor([0])))
            device = self._device if not isinstance(self._device, list) else self._device[0]
            new_args = get_new_arg_input(layer.inputs, arg2val_map, [0], fake_graph, device)
            output_vals = func(*new_args)
            if not isinstance(output_vals, tuple):
                output_vals = (output_vals,)
            if len(output_vals) != len(layer.outputs):
                raise Exception("output values not match with layer's output.")
            for val, arg_node in zip(output_vals, layer.outputs):
                if isinstance(val, torch.Tensor):
                    arg2val_map[arg_node] = val.cpu()
                    ret_shapes[layer.id].append((torch.Tensor, val.size()[1:]))
                else:
                    ret_shapes[layer.id].append((val.__class__, None))
        return ret_shapes

    def compute(self, inference_graph, rets, arg2val_map, layer, func):
        raise NotImplementedError()

    def before_inference(self, graph, *args):
        pass

    def after_inference(self):
        pass

    def inference(self, inference_graph, *args):
        self.before_inference(inference_graph, *args)
        self._module_silencer.silence(self.silence_modules)

        first_layer_inputs = (inference_graph,) + tuple(args)
        if len(first_layer_inputs) != len(self._schema.first_layer_input):
            raise Exception("layer's input not match with args.")
        arg2val_map = {}
        for val, arg_name in zip(first_layer_inputs, self._schema.first_layer_input):
            arg_node = self._schema.name2arg_map[arg_name]
            arg2val_map[arg_node] = val
        ret_shapes = self._trace_output_shape(arg2val_map)

        for layer, func in zip(self._schema.layers, self._funcs):

            rets = []
            for j, _ in enumerate(layer.outputs):
                cls, shape = ret_shapes[layer.id][j]
                if cls == torch.Tensor:
                    rets.append(
                        torch.zeros((inference_graph.number_of_nodes(),) + tuple(shape))
                    )
                else:
                    rets.append(None)

            gc.collect()
            torch.cuda.empty_cache()
            rets = self.compute(inference_graph, rets, arg2val_map, layer, func)

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
        
        self.after_inference()
        self._module_silencer.unsilence()

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


class InferenceHelper(InferenceHelperBase):
    def __init__(self, module: nn.Module, batch_size, device, num_workers = 4, silence_modules = [], debug = False):
        super().__init__(module, device, silence_modules, debug)
        self._batch_size = batch_size
        self._num_workers = num_workers

    def compute(self, graph, rets, arg2val_map, layer, func):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.NodeDataLoader(
            graph,
            torch.arange(graph.number_of_nodes()).to(graph.device),
            sampler,
            batch_size=self._batch_size,
            device=self._device if self._num_workers == 0 else 'cpu',
            shuffle=False,
            drop_last=False,
            num_workers=self._num_workers)

        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            new_args = get_new_arg_input(layer.inputs, arg2val_map, input_nodes, blocks[0], self._device)

            output_vals = func(*new_args)
            del new_args

            rets = update_ret_output(output_vals, rets, input_nodes, output_nodes, blocks)
            del output_vals

        return rets


class EdgeControlInferenceHelper(InferenceHelperBase):
    def __init__(self, module: nn.Module, max_edge_in_batch, device, num_workers = 4,  silence_modules = [], debug = False):
        super().__init__(module, device, silence_modules=silence_modules, debug=debug)
        self._max_edge_in_batch = max_edge_in_batch
        self._num_workers = num_workers

    def compute(self, graph, rets, arg2val_map, layer, func):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = CustomDataloader(
            graph,
            self._max_edge_in_batch,
            sampler,
            device=self._device if self._num_workers == 0 else 'cpu',
            shuffle=False,
            drop_last=False,
            num_workers=self._num_workers)

        pbar = tqdm.tqdm(total=graph.number_of_nodes())
        for input_nodes, output_nodes, blocks in dataloader:
            new_args = get_new_arg_input(layer.inputs, arg2val_map, input_nodes, blocks[0], self._device)

            output_vals = func(*new_args)
            del new_args

            rets = update_ret_output(output_vals, rets, input_nodes, output_nodes, blocks)
            del output_vals
            pbar.update(output_nodes.shape[0])
        pbar.close()

        return rets


class AutoInferenceHelper(InferenceHelperBase):
    def __init__(self, module: nn.Module, device, silence_modules = [], debug = False):
        super().__init__(module, device, silence_modules, debug)

    def compute(self, graph, rets, arg2val_map, layer, func):
        auto_tunner = AutoTurner(10000)
        start_edge_count = auto_tunner.edge_count

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = CustomDataloader(
            graph,
            start_edge_count,
            sampler,
            shuffle=False,
            drop_last=False)

        curr_edge_count = start_edge_count
        pbar = tqdm.tqdm(total=graph.number_of_nodes())
        for input_nodes, output_nodes, blocks in dataloader:
            print(blocks[0])
            try:
                torch.cuda.reset_max_memory_allocated()
                new_args = get_new_arg_input(layer.inputs, arg2val_map, input_nodes, blocks[0], self._device)

                output_vals = func(*new_args)
                print(torch.cuda.max_memory_allocated() // 1024 ** 2)
                del new_args

                rets = update_ret_output(output_vals, rets, input_nodes, output_nodes, blocks)
                del output_vals
                curr_edge_count = auto_tunner.search()
                pbar.update(output_nodes.shape[0])

            except Exception as e:
                print(e)
                curr_edge_count = auto_tunner.break_peak()
                dataloader.reset_batch_node(output_nodes.shape[0])
                gc.collect()
                torch.cuda.empty_cache()

            finally:
                dataloader.modify_edge_count(curr_edge_count)
        pbar.close()

        return rets
