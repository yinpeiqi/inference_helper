from typing import Generic
import functools

import torch
import dgl
from dgl.dataloading.dataloader import _divide_by_worker, _TensorizedDatasetIter


class CustomDataloader(dgl.dataloading.NodeDataLoader):
    def __init__(self, g, start_edge_count, sampler, device='cpu', shuffle=False, drop_last=False, num_workers=0):
        in_degrees = g.in_degrees()
        nids = torch.arange(g.number_of_nodes()).to(g.device)
        custom_dataset = CustomDataset(start_edge_count, g, nids, in_degrees)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        super().__init__(g,
                         custom_dataset,
                         sampler,
                         device=device,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         num_workers=num_workers)

    def modify_edge_count(self, edge_count):
        self.dataset.max_mem = edge_count
        if self.dataset.curr_iter is not None:
            self.dataset.curr_iter.max_mem = edge_count

    def reset_batch_node(self, node_count):
        if self.dataset.curr_iter is not None:
            self.dataset.curr_iter.index -= node_count

    def __setattr__(self, __name, __value) -> None:
        return super(Generic, self).__setattr__(__name, __value)

class CustomDataset(dgl.dataloading.TensorizedDataset):
    def __init__(self, max_mem, g, train_nids, in_degrees=None):
        super().__init__(train_nids, 20, False)
        self.device = train_nids.device
        self.max_mem = max_mem
        self.ori_indegrees = in_degrees
        if in_degrees is None:
            self.ori_indegrees = g.in_degrees(train_nids)
        # move __iter__ to here
        indices = _divide_by_worker(self._indices, self.batch_size, self.drop_last)
        id_tensor = self._id_tensor[indices.to(self._device)]
        self.in_degrees = [0]
        self.in_degrees.extend(self.ori_indegrees[indices].tolist())
        for i in range(1, len(self.in_degrees)):
            self.in_degrees[i] += self.in_degrees[i - 1]
        self.in_degrees.append(2e18)
        self.curr_iter = CustomDatasetIter(
            id_tensor, self.max_mem, self.in_degrees, self.drop_last, self._mapping_keys)

    def __getattr__(self, attribute_name):
        if attribute_name in CustomDataset.functions:
            function = functools.partial(CustomDataset.functions[attribute_name], self)
            return function
        else:
            return super(Generic, self).__getattr__(attribute_name)

    def __iter__(self):
        return self.curr_iter

class CustomDatasetIter(_TensorizedDatasetIter):
    def __init__(self, dataset, max_mem, in_degrees, drop_last, mapping_keys):
        super().__init__(dataset, 1, drop_last, mapping_keys)
        self.max_mem = max_mem
        self.in_degrees = in_degrees

    def get_end_idx(self):
        end_idx = self.index + 1
        while self.in_degrees[end_idx + 1] - self.in_degrees[self.index] < self.max_mem:
            end_idx += 1
        return end_idx

    def _next_indices(self):
        num_items = self.dataset.shape[0]
        if self.index >= num_items:
            raise StopIteration
        end_idx = self.get_end_idx()
        batch = self.dataset[self.index:end_idx]
        self.index = end_idx
        return batch
