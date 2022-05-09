
from dgl.utils import gather_cached_tensor_rows, scatter_cached_tensor_rows
import torch
import numpy as np


def cache_feat(g, feat, cache_ratio, device):
    assert len(feat.shape) == 2
    n_nodes = g.number_of_nodes()
    n_cached = n_nodes * cache_ratio // 100
    n_host = n_nodes - n_cached
    degs = g.out_degrees()
    indices = np.argpartition(degs.numpy(), n_host - 1)
    # indices = list(range(g.number_of_nodes()))
    # indices.sort(key=lambda x: degs[x])

    host_indices = indices[:n_host]
    cached_indices = indices[n_host:]

    host_feat = feat[host_indices,]
    cached_feat = feat[cached_indices,].to(device)

    node_pos_map = [0] * g.number_of_nodes()
    for i in range(g.number_of_nodes()):
        if i < n_cached:
            node_pos_map[indices[i]] = i
        else:
            node_pos_map[indices[i]] = - (i - n_cached) - 2

    node_pos_map = torch.LongTensor(node_pos_map).to(device)


    return host_feat, cached_feat, node_pos_map

def get_node_pos_map(g, feat_dim, cache_size_gb, device):
    n_nodes = g.number_of_nodes()
    n_cached = min(n_nodes, cache_size_gb * 1024 ** 3 // (feat_dim * 4))

    n_host = n_nodes - n_cached
    print('# Cache: {}, # Host: {}'.format(n_cached, n_host))
    degs = g.out_degrees()
    indices = np.argpartition(degs.numpy(), n_host - 1)

    host_indices = indices[:n_host]
    cached_indices = indices[n_host:]

    node_pos_map = [0] * g.number_of_nodes()
    for i in range(g.number_of_nodes()):
        if i < n_host:
            node_pos_map[indices[i]] = - i - 2
        else:
            node_pos_map[indices[i]] = i - n_host

    node_pos_map = torch.LongTensor(node_pos_map).to(device)
    return host_indices, cached_indices, node_pos_map

def cache_feat_v2(feat, host_indices, cached_indices, device):
    host_feat = feat[host_indices,]
    cached_feat = feat[cached_indices,].to(device)
    return host_feat, cached_feat
    



def read_feat(host_feat, cached_feat, index, node_pos_map):
    return gather_cached_tensor_rows(host_feat, cached_feat, index, node_pos_map)

def write_feat(host_feat, cache_feat, feats, index, node_pos_map):
    scatter_cached_tensor_rows(host_feat, cache_feat, feats, index, node_pos_map)

def get_debug_cache_hit(nodes, node_pos_map):
    nodes = nodes.cpu().numpy()
    ret = 0
    for id in nodes:
        if node_pos_map[id] >= 0:
            ret += 1
    return ret

    