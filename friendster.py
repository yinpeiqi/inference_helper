from __future__ import absolute_import

import scipy.sparse as sp
from scipy.sparse import coo_matrix
import numpy as np
import os

from dgl.data.dgl_dataset import DGLDataset, DGLBuiltinDataset
from dgl.data.utils import _get_dgl_url, generate_mask_tensor, load_graphs, save_graphs, deprecate_property
import dgl.backend as F
from dgl.convert import from_scipy
# from dgl.transforms import reorder_graph
import dgl

class FriendSterDataset(DGLDataset):
  raw_dir = '../friendster/'
  url = '../friendster/temp.txt'

  def __init__(self, raw_dir=None, force_reload=False,
               verbose=False, transform=None):
    self.num_classes = 3
    super(FriendSterDataset, self).__init__(name='friendster',
                                            url=FriendSterDataset.url,
                                            raw_dir=FriendSterDataset.raw_dir,
                                            force_reload=force_reload,
                                            verbose=verbose)

  def get_rand_type(self):
    val = np.random.uniform()
    if val < 0.1:
      return 0
    elif val < 0.4:
      return 1
    return 2

  def process(self):
    row = []
    col = []
    num_nodes = 65608366 
    with open(FriendSterDataset.url, 'r') as f:
      for line in f:
          arr = line.split()
          if arr[0] == '#':
              continue
          src, dst = int(arr[0]), int(arr[1])
        #   if cur_node != src:
        #       num_nodes += 1
        #       cur_node = src
          row.append(src)
          col.append(dst)
    row = np.array(row)
    col = np.array(col)
    graph = dgl.graph((row, col))
    graph = dgl.to_bidirected(graph)
    graph = dgl.to_simple(graph)
    # graph.ndata['node_type'] = F.tensor(node_types, dtype=F.data_type_dict['int32'])
    # features = np.random.rand(num_nodes, 128)
    # labels = np.random.randint(0, self.num_classes, size=num_nodes)
    # train_mask = (node_types == 0)
    # val_mask = (node_types == 1)
    # test_mask = (node_types == 2)
    # graph.ndata['train_mask'] = generate_mask_tensor(train_mask)
    # graph.ndata['val_mask'] = generate_mask_tensor(val_mask)
    # graph.ndata['test_mask'] = generate_mask_tensor(test_mask)
    # graph.ndata['feat'] = F.tensor(features, dtype=F.data_type_dict['float32'])
    # graph.ndata['label'] = F.tensor(labels, dtype=F.data_type_dict['int64'])
    # graph = reorder_graph(graph, node_permute_algo='rcmk', edge_permute_algo='dst', store_ids=False)
    self._graph = graph

  def has_cache(self):
    graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
    print("check cache", graph_path)
    if os.path.exists(graph_path):
      print("using cached graph")
      return True
    return False
  
  def save(self):
    graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
    save_graphs(graph_path, self._graph)

  def load(self):
    print("loading graph")
    graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
    graphs, _ = load_graphs(graph_path)
    self._graph = graphs[0]
    # print(self._graph.ndata)
    # self._graph.ndata['train_mask'] = generate_mask_tensor(self._graph.ndata['train_mask'].numpy())
    # self._graph.ndata['val_mask'] = generate_mask_tensor(self._graph.ndata['val_mask'].numpy())
    # self._graph.ndata['test_mask'] = generate_mask_tensor(self._graph.ndata['test_mask'].numpy())
    print("finish loading graph")

  def __getitem__(self, idx):
    assert idx == 0, "Reddit Dataset only has one graph"
    return self._graph

  def __len__(self):
    return 1

if __name__ == '__main__':
    # with open('../friendster/com-friendster.ungraph.txt', 'r') as f:
    #     num_edges = 0
    #     cur_node = 0
    #     mp = {}
    #     for line in f:
    #         arr = line.split()
    #         if arr[0] == '#':
    #             continue
    #         src, dst = int(arr[0]), int(arr[1])
    #         if src not in mp:
    #             mp[src] = cur_node
    #             cur_node += 1
    #         if dst not in mp:
    #             mp[dst] = cur_node
    #             cur_node += 1
    #         print(mp[src], mp[dst])
    #         num_edges += 1
    dataset = FriendSterDataset()
    print(dataset)
    graph = dataset[0]
    print(graph)
    dataset.save()
    dataset.load()
    print(dataset)
