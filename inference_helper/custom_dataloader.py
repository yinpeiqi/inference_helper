from typing import Generic
import functools

import torch
import dgl
from dgl.dataloading.dataloader import _TensorizedDatasetIter


def _divide_by_worker(dataset):
    num_samples = dataset.shape[0]
    worker_info = torch.utils.data.get_worker_info()
    if worker_info:
        num_samples_per_worker = num_samples // worker_info.num_workers + 1
        start = num_samples_per_worker * worker_info.id
        end = min(start + num_samples_per_worker, num_samples)
        dataset = dataset[start:end]
    return dataset

class CustomDataloader(dgl.dataloading.NodeDataLoader):
    def __init__(self, g, nids, sampler, start_max_node=1000, start_max_edge=10000, device='cpu', shuffle=False, drop_last=False, use_uva=False, num_workers=0):

        custom_dataset = CustomDataset(start_max_node, start_max_edge, g, nids)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        super().__init__(g,
                         custom_dataset,
                         sampler,
                         device=device,
                         use_uva=use_uva,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         num_workers=num_workers)

    def modify_max_edge(self, max_edge):
        self.dataset.max_edge = max_edge
        if self.dataset.curr_iter is not None:
            self.dataset.curr_iter.max_edge = max_edge

    def modify_max_node(self, max_node):
        self.dataset.max_node = max_node
        if self.dataset.curr_iter is not None:
            self.dataset.curr_iter.max_node = max_node

    def reset_batch_node(self, node_count):
        if self.dataset.curr_iter is not None:
            self.dataset.curr_iter.index -= node_count

    def __setattr__(self, __name, __value) -> None:
        return super(Generic, self).__setattr__(__name, __value)

class CustomDataset(dgl.dataloading.TensorizedDataset):
    def __init__(self, max_node, max_edge, g, train_nids, in_degrees=None):
        super().__init__(train_nids, max_node, False)
        self.device = train_nids.device
        self.max_node = max_node
        self.max_edge = max_edge
        self.ori_indegrees = in_degrees
        if in_degrees is None:
            self.ori_indegrees = g.in_degrees(train_nids.to(g.device))
        # move __iter__ to here
        indices = _divide_by_worker(train_nids)
        id_tensor = self._id_tensor[indices.to(self._device)]
        self.in_degrees = [0]
        self.in_degrees.extend(self.ori_indegrees[indices].tolist())
        for i in range(1, len(self.in_degrees)):
            self.in_degrees[i] += self.in_degrees[i - 1]
        self.in_degrees.append(2e18)
        self.curr_iter = CustomDatasetIter(
            id_tensor, self.max_node, self.max_edge, self.in_degrees, self.drop_last, self._mapping_keys)

    def __getattr__(self, attribute_name):
        if attribute_name in CustomDataset.functions:
            function = functools.partial(CustomDataset.functions[attribute_name], self)
            return function
        else:
            return super(Generic, self).__getattr__(attribute_name)

    def __iter__(self):
        return self.curr_iter

class CustomDatasetIter(_TensorizedDatasetIter):
    def __init__(self, dataset, max_node, max_edge, in_degrees, drop_last, mapping_keys):
        super().__init__(dataset, max_node, drop_last, mapping_keys)
        self.max_node = max_node
        self.max_edge = max_edge
        self.in_degrees = in_degrees

    def get_end_idx(self):
        # TODO(Peiqi): change it to logN algorithm
        end_idx = self.index + 1
        while self.in_degrees[end_idx + 1] - self.in_degrees[self.index] < self.max_edge and \
            end_idx - self.index < self.max_node:
            end_idx += 1
        return end_idx

    def _next_indices(self):
        print(self.max_node, self.max_edge, self.index)
        num_items = self.dataset.shape[0]
        if self.index >= num_items:
            raise StopIteration
        end_idx = self.get_end_idx()
        batch = self.dataset[self.index:end_idx]
        self.index = end_idx
        return batch
