import pynvml
import torch


class DataManager:
    def __init__(self, device, use_uva):
        self.arg2val_map = {}
        self.arg_in_gpu = {}
        self.device = device
        self.use_uva = use_uva

    def __getitem__(self, arg_node):
        if arg_node not in self.arg2val_map:
            raise RuntimeError("schema not match with output.")
        return self.arg2val_map[arg_node]

    def __setitem__(self, arg_node, val):
        self.arg2val_map[arg_node] = val

    def __delitem__(self, arg_node):
        del self.arg2val_map[arg_node]


class AutoDataManager(DataManager):
    def __init__(self, device, use_uva):
        super().__init__(device, use_uva)
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        self.tot_free = info.free // 10 # use 10% as storage in GPU
        self.curr = 0

    def auto_device_switch(self, arg_node):
        if not isinstance(self[arg_node], torch.Tensor):
            return
        if arg_node not in self.arg_in_gpu:
            memory_comsuption = 4 # float, TODO
            for dim in self[arg_node].shape:
                memory_comsuption *= dim
            if self.curr + memory_comsuption < self.tot_free:
                self[arg_node] = self[arg_node].to(self.device)
                self.arg_in_gpu[arg_node] = memory_comsuption
                self.curr += memory_comsuption
