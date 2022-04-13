from torch.fx import Node

import torch
from dgl import DGLHeteroGraph
from dgl.utils import gather_pinned_tensor_rows

def arg_trace(a):
    ret = set()
    if isinstance(a, Node):
        ret.add(a)
    if isinstance(a, dict):
        for _, v in a.items():
            ret = ret.union(arg_trace(v))
    if isinstance(a, tuple) or isinstance(a, list):
        for v in a:
            ret = ret.union(arg_trace(v))
    elif isinstance(a, slice):
        ret = ret.union(arg_trace((a.start, a.step, a.stop)))
    return ret


def get_new_arg_input(inputs, arg2val_map, input_nodes, inference_graph, device, use_uva=False):
    new_args = ()
    for arg_node in inputs:
        if arg_node not in arg2val_map:
            raise RuntimeError("schema not match with output.")
        if isinstance(arg2val_map[arg_node], torch.Tensor):
            if use_uva:
                new_args += (gather_pinned_tensor_rows(arg2val_map[arg_node], input_nodes),)
            else:
                new_args += (arg2val_map[arg_node][input_nodes].to(device),)
        elif isinstance(arg2val_map[arg_node], DGLHeteroGraph):
            new_args += (inference_graph.to(device),)
        elif hasattr(arg2val_map[arg_node], "to"):
            new_args += (arg2val_map[arg_node].to(device),)
        else:
            new_args += (arg2val_map[arg_node],)
    return new_args

def update_ret_output(output_vals, rets, input_nodes, output_nodes, blocks):
    if not isinstance(output_vals, tuple):
        output_vals = (output_vals,)
    for output_val, ret in zip(output_vals, rets):
        if isinstance(output_val, torch.Tensor):
            if ret is None:
                raise RuntimeError("Can't determine return's type.")
            if output_val.size()[0] == blocks[0].num_dst_nodes():
                ret[output_nodes] = output_val.cpu()
            elif output_val.size()[0] == blocks[0].num_src_nodes():
                ret[input_nodes] = output_val.cpu()
            else:
                raise RuntimeError("Can't determine return's type.")
        else:
            ret = output_val
    return rets
