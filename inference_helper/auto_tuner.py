import pynvml
import torch
from sklearn.linear_model import LinearRegression

def get_auto_tuner(device, cached=0):
    if isinstance(device, torch.device):
        device = device.type
    if 'cuda' in device:
        return GPUAutoTuner(cached)
    elif 'cpu' in device:
        return CPUAutoTuner()
    else:
        raise NotImplementedError("Not implement Auto Tuner for device: {}.".format(device))


class AutoTunerBase:
    def __init__(self):
        self.free_memory = 0
        self.set_free()

    def set_free(self):
        raise NotImplementedError

    def get_max(self):
        raise NotImplementedError

    def search(self, g):
        raise NotImplementedError

    def break_peak(self, g):
        curr_node = g.number_of_dst_nodes()
        curr_edge = g.num_edges()
        return curr_node // 2, curr_edge // 2


class GPUAutoTuner(AutoTunerBase):
    def __init__(self, cached=0):
        self.cached = cached
        super().__init__()

    def set_free(self):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        self.free_memory = info.free * 0.9

    def get_max(self):
        return torch.cuda.max_memory_allocated() - self.cached

    def search(self, g):
        curr_node = g.number_of_dst_nodes()
        curr_edge = g.num_edges()
        increase_rate = self.free_memory / self.get_max()
        # print(self.free_memory // 1024 ** 2, self.get_max() // 1024 ** 2)
        curr_node = int(curr_node * increase_rate)
        curr_edge = int(curr_edge * increase_rate)
        return curr_node, curr_edge

# TODO: not use yet
class NewGPUAutoTuner(GPUAutoTuner):
    def __init__(self, g):
        super().__init__()
        self.tot_node = g.num_nodes()
        self.src_lin_reg = LinearRegression()
        self.mem_lin_reg = LinearRegression()
        self.num = 0
        self.mem_x = []
        self.mem_y = []
        self.src_x = []
        self.src_y = []
        self.target = 1

    def update_src_lin_reg(self, src_nodes, curr_node, curr_edge):
        self.src_x.append([curr_node, curr_edge])
        self.src_y.append(src_nodes)

    def update_mem_lin_reg(self, src_nodes, curr_node, curr_edge, max_memory_allocated):
        self.mem_x.append([src_nodes, curr_node, curr_edge])
        self.mem_y.append(max_memory_allocated)

    def search(self, g):
        self.num += 1
        src_nodes = g.number_of_src_nodes()
        curr_node = g.number_of_dst_nodes()
        curr_edge = g.num_edges()
        max_memory_allocated = self.get_max() // 1024 ** 2
        self.update_src_lin_reg(src_nodes, curr_node, curr_edge)
        self.update_mem_lin_reg(src_nodes, curr_node, curr_edge, max_memory_allocated)

        increase_rate = self.free_memory / self.get_max()
        if self.num < 5:
            if self.target > 1:
                # increase_rate *= min(self.target / max_memory_allocated, 2)
                print(self.target, increase_rate)
            self.target = self.free_memory // 1024 ** 2
        next_node = int(curr_node * increase_rate)
        next_edge = int(curr_edge * increase_rate)
        if self.num > 5:
            self.src_lin_reg.fit(self.src_x[-3:], self.src_y[-3:])
            self.mem_lin_reg.fit(self.mem_x, self.mem_y)
            pred_src = min(int(self.src_lin_reg.predict([[next_node, next_edge]])), self.tot_node)
            pred_memory = int(self.mem_lin_reg.predict([[pred_src, next_node, next_edge]])[0])
            print("src:", pred_src, pred_memory)
            if pred_memory == self.tot_node:
                next_node += next_node // 2
                next_edge += next_edge // 2
        print("free:", self.free_memory // 1024 ** 2, self.get_max() // 1024 ** 2)
        return next_node, next_edge


class CPUAutoTuner(AutoTunerBase):
    def set_free(self):
        raise NotImplementedError

    def get_max(self):
        raise NotImplementedError
