import torch
import time

class Profiler:
    def __init__(self):
        self.time = []
        self.curr = []
        self.m = {}

    def tag(self):
        torch.cuda.synchronize()
        self.curr.append(time.time())
    
    def record_and_reset(self):
        self.tag()
        for i in range(0, len(self.curr) - 1):
            if len(self.time) <= i:
                self.time.append(0)
            self.time[i] += self.curr[i + 1] - self.curr[i]
        self.curr = []
        self.tag()

    def record_name(self, name, val):
        if name not in self.m:
            self.m[name] = val
        else:
            self.m[name] += val

    def last(self):
        return self.curr[-1] - self.curr[-2]

    def show(self):
        for t in self.time:
            print(t, end=" ")
        tot = sum(self.time)
        print("\ntot time:", tot)
        for k, v in self.m.items():
            print(k, v)
