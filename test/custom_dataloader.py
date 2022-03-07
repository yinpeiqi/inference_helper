import dgl
import torch
from dgl.data import CiteseerGraphDataset, RedditDataset
from dgl.dataloading.dataloader import _divide_by_worker, _TensorizedDatasetIter


class CustomDataset(dgl.dataloading.TensorizedDataset):
    def __init__(self, max_mem, g, train_nids):
        super().__init__(train_nids, 20, False)
        # self.in_degree
        self.device = train_nids.device
        self.max_mem = max_mem
        self.ori_indegrees = g.in_degrees(train_nids)
        self.in_degrees = [0]
        self.in_degrees.extend(g.in_degrees(train_nids).tolist())
        for i in range(1, len(self.in_degrees)):
            self.in_degrees[i] += self.in_degrees[i - 1]
        self.in_degrees.append(2e18)

    def __iter__(self):
        indices = _divide_by_worker(self._indices, self.batch_size, self.drop_last)
        id_tensor = self._id_tensor[indices.to(self._device)]
        self.in_degrees = [0]
        self.in_degrees.extend(self.ori_indegrees[indices].tolist())
        for i in range(1, len(self.in_degrees)):
            self.in_degrees[i] += self.in_degrees[i - 1]
        self.in_degrees.append(2e18)
        return CustomDatasetIter(
            id_tensor, self.max_mem, self.in_degrees, self.drop_last, self._mapping_keys)

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
